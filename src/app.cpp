//
// app.cpp
// The integrated platform. Two modes:
//
//   Live:    ./llm_app model.gguf ["prompt"] [--record out.trace] [--predict N]
//            Loads a GGUF, wires the non-invasive Instrumentor as the eval
//            callback, runs generation on a worker thread, and drives the live
//            TUI on the main thread (single ring consumer). Optionally records
//            the consumed packet stream to a .trace file.
//
//   Replay:  ./llm_app --replay run.trace
//            No model. Replays a recorded packet stream into the ring at its
//            original cadence and drives the same TUI.
//
#include "instrumentor.hpp"
#include "telemetry_ui.hpp"
#include "trace_io.hpp"

#include <atomic>
#include <thread>
#include <vector>
#include <string>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <chrono>

#include "llama.h"

using namespace llmtrace;

namespace {

void quiet_logger(enum ggml_log_level, const char*, void*) {}

// Replay a recorded trace into the ring, honoring the original inter-packet
// timing (clamped) so the stream animates like the live run.
void replay_thread(std::string path, PacketRing& ring, std::atomic<bool>& stop) {
    while (!stop.load(std::memory_order_acquire)) {
        TraceReader rd(path.c_str());
        if (!rd.ok()) return;
        LayerPacket p{};
        std::uint64_t prev = 0;
        while (!stop.load(std::memory_order_acquire) && rd.next(p)) {
            if (prev != 0 && p.timestamp_ns > prev) {
                std::uint64_t dns = p.timestamp_ns - prev;
                if (dns > 50'000'000ULL) dns = 50'000'000ULL; // clamp 50ms
                std::this_thread::sleep_for(std::chrono::nanoseconds(dns));
            }
            prev = p.timestamp_ns;
            while (!ring.try_push(p) && !stop.load(std::memory_order_acquire))
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(400)); // pause, then loop
    }
}

// Generation worker: continuously decodes the prompt and greedily samples a few
// tokens, clearing the KV memory between passes so the telemetry stream stays
// live for as long as the UI is open. Each decode fires the eval callback.
void inference_thread(llama_context* ctx, const llama_vocab* vocab,
                      std::vector<llama_token> tokens, int n_predict,
                      std::atomic<bool>& stop) {
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    while (!stop.load(std::memory_order_acquire)) {
        llama_memory_clear(llama_get_memory(ctx), true); // fresh context each pass

        llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
        if (llama_decode(ctx, batch) != 0) break;

        for (int i = 0; i < n_predict && !stop.load(std::memory_order_acquire); ++i) {
            llama_token id = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, id)) break;
            batch = llama_batch_get_one(&id, 1);
            if (llama_decode(ctx, batch) != 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(8)); // pace for watchability
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); // brief pause between passes
    }
    llama_sampler_free(smpl);
}

const char* arg_after(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc - 1; ++i) if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return nullptr;
}
bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

} // namespace

int main(int argc, char** argv) {
    PacketRing        ring;
    CaptureTarget     target;
    AttnSlot          attn;
    std::atomic<bool> stop{false};

    // ---- replay mode -------------------------------------------------------
    if (const char* trace = arg_after(argc, argv, "--replay")) {
        TraceReader probe(trace);
        if (!probe.ok()) { std::fprintf(stderr, "cannot open trace: %s\n", trace); return 1; }
        int n_layers = probe.n_layers();
        std::string model_name = probe.model();

        std::thread worker(replay_thread, std::string(trace), std::ref(ring), std::ref(stop));
        ui::TuiConfig cfg{ring, target, attn, stop, model_name, n_layers, "REPLAY", nullptr};
        ui::run_tui(cfg);

        stop.store(true, std::memory_order_release);
        worker.join();
        return 0;
    }

    // ---- live mode ---------------------------------------------------------
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s model.gguf [\"prompt\"] [--record out.trace] [--predict N]\n"
            "       %s --replay run.trace\n", argv[0], argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    std::string prompt = (argc > 2 && argv[2][0] != '-') ? argv[2] : "The capital of France is";
    const char* rec_path = arg_after(argc, argv, "--record");
    const char* npred    = arg_after(argc, argv, "--predict");
    const int   n_predict = npred ? std::atoi(npred) : 96;

    llama_log_set(quiet_logger, nullptr);
    llama_backend_init();

    // NOTE: recent llama.cpp API names. Older checkouts: llama_load_model_from_file /
    //       llama_new_context_with_model / llama_free_model / llama_n_layer.
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0; // host-resident keeps activation + attention reads cheap
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) { std::fprintf(stderr, "failed to load model: %s\n", model_path); return 1; }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_layers = (int)llama_model_n_layer(model);
    char desc[128] = {0};
    llama_model_desc(model, desc, sizeof(desc));

    // The Instrumentor must outlive the context: it is referenced via cb_eval_user_data.
    Instrumentor inst(ring, &target, &attn);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = 1024;
    cp.cb_eval           = &Instrumentor::eval_callback_thunk;
    cp.cb_eval_user_data = &inst;
    // Attention-matrix capture requires the softmax scores tensor to be
    // materialized, i.e. fused flash attention OFF. Recent API:
    cp.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    // Older llama.cpp instead exposes:  cp.flash_attn = false;

    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "failed to create context\n"); llama_model_free(model); return 1; }

    std::vector<llama_token> tokens(prompt.size() + 8);
    int32_t n = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                               tokens.data(), (int32_t)tokens.size(), true, true);
    if (n < 0) { tokens.resize(-n);
        n = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                           tokens.data(), (int32_t)tokens.size(), true, true); }
    tokens.resize(n);

    std::unique_ptr<TraceWriter> recorder;
    if (rec_path) {
        recorder = std::make_unique<TraceWriter>(rec_path, n_layers, desc);
        if (!recorder->ok()) { std::fprintf(stderr, "cannot open record file: %s\n", rec_path); recorder.reset(); }
    }

    std::thread worker(inference_thread, ctx, vocab, std::move(tokens), n_predict, std::ref(stop));

    ui::TuiConfig cfg{ring, target, attn, stop, desc[0] ? desc : "model",
                      n_layers, "LIVE", recorder.get()};
    ui::run_tui(cfg);

    stop.store(true, std::memory_order_release);
    worker.join();

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
