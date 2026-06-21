#pragma once
//
// instrumentor.hpp
// Non-invasive hook into a llama.cpp / ggml forward pass.
//
// Wiring (no model source is touched):
//
//   llmtrace::PacketRing  ring;
//   llmtrace::Instrumentor inst(ring);
//   llama_context_params cp = llama_context_default_params();
//   cp.cb_eval           = &llmtrace::Instrumentor::eval_callback_thunk;
//   cp.cb_eval_user_data = &inst;
//   llama_context* ctx   = llama_init_from_model(model, cp);
//
// ggml then invokes the callback for every node in the compute graph. We use
// the two-phase ask/done protocol: phase 1 (ask=true) we declare interest and
// start the latency clock; phase 2 (ask=false) the node has executed and its
// data is readable, so we profile it and push a packet.
//
#include "layer_packet.hpp"
#include "ring_buffer.hpp"
#include "packet_ring.hpp"
#include "capture_target.hpp"
#include "attention_slot.hpp"

#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "ggml.h"
#include "ggml-backend.h"

namespace llmtrace {

class Instrumentor {
public:
    explicit Instrumentor(PacketRing& ring,
                          CaptureTarget* target = nullptr,
                          AttnSlot*      attn   = nullptr,
                          float blowup_threshold = 30.0f,
                          float sparsity_eps     = 1e-6f) noexcept
        : ring_(ring), target_(target), attn_(attn),
          blowup_threshold_(blowup_threshold), eps_(sparsity_eps) {}

    // Matches ggml_backend_sched_eval_callback:
    //   bool (*)(struct ggml_tensor* t, bool ask, void* user_data)
    static bool eval_callback_thunk(struct ggml_tensor* t, bool ask, void* user_data) {
        return static_cast<Instrumentor*>(user_data)->on_node(t, ask);
    }

private:
    static constexpr std::int64_t kSampleCap = 4096; // cap elements scanned per node

    bool on_node(struct ggml_tensor* t, bool ask) noexcept {
        if (ask) {
            // Phase 1. Returning true requests a second call after the node runs.
            // We only opt in for nodes we care about, keeping overhead near zero
            // for the rest of the graph.
            if (!want(t)) return false;
            start_ns_ = now_ns();
            return true;
        }

        // Phase 2. Node computed; t->data / t->buffer are valid for its backend.
        LayerPacket p{};
        p.id           = seq_++;
        p.timestamp_ns = now_ns();
        p.latency_us   = static_cast<double>(p.timestamp_ns - start_ns_) / 1000.0;

        std::strncpy(p.name, t->name, kNameLen - 1);
        p.layer_index = parse_layer_index(t->name);
        p.kind        = classify(t->name);
        p.dtype       = map_dtype(t->type);
        p.device      = map_device(t);
        if (p.device == Device::Cuda || p.device == Device::Metal) gpu_seen_ = true;

        p.ndim = static_cast<std::uint8_t>(ggml_n_dims(t));
        for (int i = 0; i < kMaxDims; ++i) p.shape[i] = t->ne[i];

        profile_activations(t, p);
        flag_anomalies(p);

        ring_.try_push(p); // wait-free; never stalls the inference thread

        maybe_capture_attention(t); // only fires for the user-selected layer
        return true;
    }

    // ---- attention capture -------------------------------------------------
    // The post-softmax scores node ("kq_soft_max") is materialized only when
    // flash attention is disabled. We snapshot head 0 of the selected layer into
    // a fixed tile and publish it; everything is bounded and rate-limited to one
    // capture per forward pass for one layer.
    void maybe_capture_attention(struct ggml_tensor* t) noexcept {
        if (!target_ || !attn_) return;
        if (!std::strstr(t->name, "kq_soft")) return;
        if (t->type != GGML_TYPE_F32) return;

        const std::int32_t layer = parse_layer_index(t->name);
        if (!target_->matches(layer, LayerKind::Attention)) return;

        // ggml layout: ne[0]=keys, ne[1]=queries(tokens), ne[2]=heads.
        const std::int64_t ne0 = t->ne[0];
        const std::int64_t ne1 = t->ne[1];
        const int rows = (int)std::min<std::int64_t>(ne1, kAttnTile);
        const int cols = (int)std::min<std::int64_t>(ne0, kAttnTile);
        if (rows <= 1 || cols <= 0) return;

        AttentionSnapshot snap;
        snap.layer_index = layer;
        snap.head        = 0;
        snap.rows        = rows;
        snap.cols        = cols;

        const bool host = t->buffer && ggml_backend_buffer_is_host(t->buffer);
        for (int r = 0; r < rows; ++r) {
            const std::int64_t base = ((std::int64_t)0 * ne1 + r) * ne0; // head 0, row r
            float* dst = snap.weights.data() + (std::size_t)r * kAttnTile;
            if (host && t->data) {
                std::memcpy(dst, (const float*)t->data + base, cols * sizeof(float));
            } else if (t->buffer) {
                ggml_backend_tensor_get(t, dst, base * sizeof(float), cols * sizeof(float));
            }
        }
        attn_->store(snap);
    }

    // ---- node filtering ----------------------------------------------------

    static bool want(const struct ggml_tensor* t) noexcept {
        if (!t->name[0]) return false;
        const char* n = t->name;
        return std::strstr(n, "attn")   || std::strstr(n, "ffn") ||
               std::strstr(n, "norm")   || std::strstr(n, "embd") ||
               std::strstr(n, "result") || std::strstr(n, "kq");
    }

    // ---- name / type / device mapping --------------------------------------

    // Weight tensors are named "blk.<N>.<op>"; computed graph nodes are named
    // "<op>-<layer>" (e.g. "kq_soft_max_ext-3", "attn_norm-3"). Handle both.
    static std::int32_t parse_layer_index(const char* name) noexcept {
        const char* p = std::strstr(name, "blk.");
        if (p && p[4] >= '0' && p[4] <= '9') {
            p += 4;
            std::int32_t idx = 0;
            while (*p >= '0' && *p <= '9') { idx = idx * 10 + (*p - '0'); ++p; }
            return idx;
        }
        // Fall back to a trailing "-<digits>" suffix on computed nodes.
        const char* dash = std::strrchr(name, '-');
        if (dash && dash[1] >= '0' && dash[1] <= '9') {
            std::int32_t idx = 0;
            for (const char* q = dash + 1; *q >= '0' && *q <= '9'; ++q) idx = idx * 10 + (*q - '0');
            return idx;
        }
        return -1;
    }

    static LayerKind classify(const char* n) noexcept {
        if (std::strstr(n, "attn"))                          return LayerKind::Attention;
        if (std::strstr(n, "ffn") || std::strstr(n, "mlp"))  return LayerKind::Mlp;
        if (std::strstr(n, "norm"))                          return LayerKind::Norm;
        if (std::strstr(n, "embd") || std::strstr(n, "embed")) return LayerKind::Embedding;
        if (std::strstr(n, "result") || std::strstr(n, "output")) return LayerKind::Output;
        if (std::strstr(n, "rope"))                          return LayerKind::Rope;
        return LayerKind::Other;
    }

    static DType map_dtype(enum ggml_type t) noexcept {
        switch (t) {
            case GGML_TYPE_F32: return DType::F32;
            case GGML_TYPE_F16: return DType::F16;
            default:            return DType::Other; // Q4_K, Q8_0, etc.
        }
    }

    static Device map_device(const struct ggml_tensor* t) noexcept {
        if (!t->buffer) return Device::Cpu;
        if (ggml_backend_buffer_is_host(t->buffer)) return Device::Cpu;
        const char* bn = ggml_backend_buffer_name(t->buffer);
        if (bn && std::strstr(bn, "CUDA"))  return Device::Cuda;
        if (bn && std::strstr(bn, "Metal")) return Device::Metal;
        return Device::Other;
    }

    // ---- activation profiling (sampled, bounded cost) ----------------------

    void profile_activations(struct ggml_tensor* t, LayerPacket& p) noexcept {
        // Only float tensors are read directly; quantized weights are skipped.
        if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) return;
        const std::int64_t n = ggml_nelements(t);
        if (n <= 0) return;
        const std::int64_t m = std::min<std::int64_t>(n, kSampleCap);

        float buf[kSampleCap];
        const bool host = t->buffer && ggml_backend_buffer_is_host(t->buffer);

        if (t->type == GGML_TYPE_F32) {
            if (host && t->data) std::memcpy(buf, t->data, m * sizeof(float));
            else if (t->buffer)  ggml_backend_tensor_get(t, buf, 0, m * sizeof(float));
            else return;
        } else { // F16 -> upcast into buf
            ggml_fp16_t hbuf[kSampleCap];
            if (host && t->data) std::memcpy(hbuf, t->data, m * sizeof(ggml_fp16_t));
            else if (t->buffer)  ggml_backend_tensor_get(t, hbuf, 0, m * sizeof(ggml_fp16_t));
            else return;
            for (std::int64_t i = 0; i < m; ++i) buf[i] = ggml_fp16_to_fp32(hbuf[i]);
        }

        double sum = 0.0;
        float  maxa = 0.0f;
        std::int64_t zeros = 0;
        bool nonfinite = false;
        for (std::int64_t i = 0; i < m; ++i) {
            const float v = buf[i];
            if (!std::isfinite(v)) { nonfinite = true; continue; }
            sum += v;
            const float a = std::fabs(v);
            if (a > maxa) maxa = a;
            if (a < eps_) ++zeros;
        }
        p.mean     = static_cast<float>(sum / static_cast<double>(m));
        p.max_abs  = maxa;
        p.sparsity = static_cast<float>(zeros) / static_cast<float>(m);
        if (nonfinite) p.anomaly |= AnomalyFlag::NonFinite;
    }

    void flag_anomalies(LayerPacket& p) const noexcept {
        if (p.max_abs > blowup_threshold_)                       p.anomaly |= AnomalyFlag::ActivationBlowup;
        if (std::fabs(p.mean) < 1e-7f && p.sparsity > 0.99f)     p.anomaly |= AnomalyFlag::SparsityCollapse;
        // A CPU node during an otherwise GPU-backed run is a genuine fallback.
        if (p.device == Device::Cpu && gpu_seen_ && p.layer_index >= 0)
                                                                 p.anomaly |= AnomalyFlag::DeviceFallback;
    }

    static std::uint64_t now_ns() noexcept {
        using namespace std::chrono;
        return static_cast<std::uint64_t>(
            duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
    }

    PacketRing&    ring_;
    CaptureTarget* target_ = nullptr;
    AttnSlot*      attn_   = nullptr;
    float          blowup_threshold_;
    float          eps_;
    bool           gpu_seen_ = false;
    std::uint64_t  seq_      = 0;
    std::uint64_t  start_ns_ = 0;
};

} // namespace llmtrace
