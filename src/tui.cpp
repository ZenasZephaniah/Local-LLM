//
// tui.cpp
// Standalone demo entry point. Feeds the shared TUI with a synthetic packet
// stream and a synthetic attention matrix so the whole interface runs with no
// model. Integration just swaps this producer for real inference (see app.cpp).
//
// Build target: llm_tui (FTXUI only, no llama dependency).
//
#include "telemetry_ui.hpp"

#include <atomic>
#include <thread>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdio>

using namespace llmtrace;

namespace {

constexpr int kNLayers = 8;

void synthetic_producer(PacketRing& ring, AttnSlot& attn, std::atomic<bool>& stop) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::uint64_t id = 0;
    const char* ops[]      = {"attn_norm", "attn", "ffn_norm", "ffn"};
    const LayerKind kinds[] = {LayerKind::Norm, LayerKind::Attention, LayerKind::Norm, LayerKind::Mlp};

    while (!stop.load(std::memory_order_acquire)) {
        for (int L = 0; L < kNLayers && !stop.load(std::memory_order_acquire); ++L) {
            for (int o = 0; o < 4; ++o) {
                LayerPacket p{};
                p.id = id++;
                p.timestamp_ns = (std::uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                std::snprintf(p.name, kNameLen, "blk.%d.%s", L, ops[o]);
                p.layer_index = L;
                p.kind   = kinds[o];
                p.device = Device::Cpu;
                p.dtype  = DType::F16;
                p.ndim = 3; p.shape[0] = 1; p.shape[1] = 32; p.shape[2] = 4096;
                p.latency_us = 200.0 + u(rng) * 1200.0;
                p.mean    = (u(rng) - 0.5f) * 0.2f;
                p.max_abs = u(rng) * 8.0f;
                p.sparsity = u(rng);
                if (p.max_abs > 7.0f) { p.max_abs = 35.0f; p.anomaly |= AnomalyFlag::ActivationBlowup; }
                ring.try_push(p);

                // Publish a synthetic causal-looking attention matrix for attn nodes.
                if (kinds[o] == LayerKind::Attention) {
                    AttentionSnapshot s;
                    s.layer_index = L; s.head = 0; s.rows = 24; s.cols = 24;
                    for (int r = 0; r < s.rows; ++r) {
                        float norm = 0.0f;
                        for (int c = 0; c <= r; ++c) {
                            float w = std::exp(-0.25f * (r - c)) * (0.5f + 0.5f * u(rng));
                            s.weights[(std::size_t)r * kAttnTile + c] = w;
                            norm += w;
                        }
                        for (int c = 0; c <= r && norm > 0; ++c)
                            s.weights[(std::size_t)r * kAttnTile + c] /= norm; // row-stochastic
                    }
                    attn.store(s);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
    }
}

} // namespace

int main() {
    PacketRing        ring;
    CaptureTarget     target;
    AttnSlot          attn;
    std::atomic<bool> stop{false};

    std::thread producer(synthetic_producer, std::ref(ring), std::ref(attn), std::ref(stop));

    ui::TuiConfig cfg{ring, target, attn, stop, "llama-3-8b (demo)", kNLayers, "DEMO", nullptr};
    ui::run_tui(cfg);

    stop.store(true, std::memory_order_release);
    producer.join();
    return 0;
}
