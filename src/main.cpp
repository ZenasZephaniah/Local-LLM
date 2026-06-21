//
// main.cpp
// End-to-end smoke test for the telemetry pipeline.
//
//   1. Load a small GGUF model with llama.cpp
//   2. Wire our non-invasive Instrumentor as the eval callback
//   3. Spawn a consumer thread that drains the ring buffer and prints packets
//   4. Run a single decode (which fires the callback per node)
//
// This proves the producer (inference) -> ring -> consumer (printer) path works
// before we replace the printer with the FTXUI panels.
//
//   ./llm_trace path/to/model.gguf
//
#include "instrumentor.hpp"

#include <atomic>
#include <thread>
#include <vector>
#include <cstdio>
#include <cstring>
#include <string>

#include "llama.h"

using namespace llmtrace;

namespace {

const char* kind_str(LayerKind k) {
    switch (k) {
        case LayerKind::Embedding: return "Embed";
        case LayerKind::Attention: return "Attn ";
        case LayerKind::Mlp:       return "MLP  ";
        case LayerKind::Norm:      return "Norm ";
        case LayerKind::Output:    return "Out  ";
        case LayerKind::Rope:      return "RoPE ";
        case LayerKind::Other:     return "Other";
        default:                   return "?    ";
    }
}

const char* device_str(Device d) {
    switch (d) {
        case Device::Cuda:  return "CUDA";
        case Device::Metal: return "Metal";
        case Device::Cpu:   return "CPU";
        default:            return "Other";
    }
}

void print_packet(const LayerPacket& p) {
    char shape[40] = {0};
    int off = std::snprintf(shape, sizeof(shape), "[");
    for (std::uint8_t i = 0; i < p.ndim && off < (int)sizeof(shape); ++i) {
        off += std::snprintf(shape + off, sizeof(shape) - off,
                             "%lld%s", (long long)p.shape[i],
                             (i + 1 < p.ndim) ? "," : "");
    }
    std::snprintf(shape + off, sizeof(shape) - off, "]");

    std::printf("#%-5llu %-20s %s L%-3d %-14s %6.3f ms  mean=% .3f max=%.3f sp=%4.1f%% %s%s\n",
                (unsigned long long)p.id,
                p.name,
                kind_str(p.kind),
                p.layer_index,
                shape,
                p.latency_us / 1000.0,
                p.mean, p.max_abs, p.sparsity * 100.0f,
                device_str(p.device),
                (p.anomaly & AnomalyFlag::ActivationBlowup) ? "  [BLOWUP]" :
                (p.anomaly & AnomalyFlag::NonFinite)        ? "  [NAN/INF]" :
                (p.anomaly & AnomalyFlag::SparsityCollapse) ? "  [COLLAPSE]" : "");
}

// Quiet llama.cpp's own logging so the packet stream stays readable.
void quiet_logger(enum ggml_log_level, const char*, void*) {}

void consume(PacketRing& ring, std::atomic<bool>& done) {
    LayerPacket p{};
    while (!done.load(std::memory_order_acquire)) {
        while (ring.try_pop(p)) print_packet(p);
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    while (ring.try_pop(p)) print_packet(p); // final drain after decode returns
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s path/to/model.gguf [prompt]\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const std::string prompt = (argc > 2) ? argv[2] : "The capital of France is";

    llama_log_set(quiet_logger, nullptr);
    llama_backend_init();

    // ---- load model --------------------------------------------------------
    // NOTE: API names below match recent llama.cpp. On older checkouts use
    //   llama_load_model_from_file / llama_new_context_with_model / llama_free_model.
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0; // CPU-only keeps tensors host-resident for easy profiling
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        std::fprintf(stderr, "failed to load model: %s\n", model_path);
        return 1;
    }

    // ---- ring + instrumentor, wired non-invasively onto the context --------
    PacketRing   ring;
    Instrumentor inst(ring);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = 512;
    cp.cb_eval           = &Instrumentor::eval_callback_thunk;
    cp.cb_eval_user_data = &inst;
    // When we add attention-matrix capture, disable fused flash attention here
    // so the softmax tensor is materialized (field name is version-specific:
    // older: cp.flash_attn = false; newer: cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED).

    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // ---- tokenize ----------------------------------------------------------
    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(prompt.size() + 8);
    int32_t n = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                               tokens.data(), (int32_t)tokens.size(),
                               /*add_special=*/true, /*parse_special=*/true);
    if (n < 0) { tokens.resize(-n);
        n = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                           tokens.data(), (int32_t)tokens.size(), true, true);
    }
    tokens.resize(n);

    // ---- run consumer thread, then drive one forward pass ------------------
    std::atomic<bool> done{false};
    std::thread consumer(consume, std::ref(ring), std::ref(done));

    std::printf("=== forward pass: %d tokens ===\n", n);
    llama_batch batch = llama_batch_get_one(tokens.data(), n);
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "llama_decode failed\n");
    }

    done.store(true, std::memory_order_release);
    consumer.join();
    std::printf("=== done. packets dropped on overflow: %llu ===\n",
                (unsigned long long)ring.dropped());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
