#pragma once
//
// layer_packet.hpp
// One dense, flat, cache-friendly telemetry record per computed graph node.
// Route-agnostic: nothing here depends on ggml or libtorch.
//
#include <cstdint>
#include <array>
#include <type_traits>

namespace llmtrace {

// ggml tensors are at most 4-dimensional.
inline constexpr int kMaxDims = 4;
// Max characters we store for a node name. Inline buffer, no heap.
inline constexpr int kNameLen = 48;

enum class LayerKind : std::uint8_t {
    Unknown = 0, Embedding, Attention, Mlp, Norm, Output, Rope, Other
};

enum class Device : std::uint8_t {
    Cpu = 0, Cuda, Metal, Other
};

enum class DType : std::uint8_t {
    F32 = 0, F16, BF16, Q4, Q8, Other
};

// Bit flags consumed by the Numerical Anomaly Ledger panel.
enum AnomalyFlag : std::uint8_t {
    None             = 0,
    ActivationBlowup = 1 << 0,  // max|x| exceeded threshold (activation explosion)
    SparsityCollapse = 1 << 1,  // mean ~ 0 with near-total sparsity
    DeviceFallback   = 1 << 2,  // node ran on CPU when GPU was expected
    NonFinite        = 1 << 3,  // NaN / Inf seen in the sampled window
};

// One telemetry record for a single computed node.
// POD and trivially copyable => safe to memcpy across the SPSC ring buffer
// and cheap to move by value. No pointers, no ownership, no heap.
struct LayerPacket {
    std::uint64_t id           = 0;    // monotonically increasing sequence number
    std::uint64_t timestamp_ns = 0;    // capture time, steady_clock
    double        latency_us   = 0.0;  // node compute time (ask/done bracket)

    char          name[kNameLen]{};    // e.g. "blk.1.attn_out"
    std::int32_t  layer_index  = -1;   // parsed transformer block index, -1 = n/a

    std::array<std::int64_t, kMaxDims> shape{};
    std::uint8_t  ndim         = 0;

    LayerKind     kind         = LayerKind::Unknown;
    Device        device       = Device::Cpu;
    DType         dtype        = DType::F32;
    std::uint8_t  anomaly      = AnomalyFlag::None;

    // Activation profile, computed on a bounded sample to cap overhead.
    float         mean         = 0.0f;
    float         max_abs      = 0.0f;
    float         sparsity     = 0.0f; // fraction of |x| < epsilon in the sample
};

static_assert(std::is_trivially_copyable_v<LayerPacket>,
              "LayerPacket must stay trivially copyable for lock-free transfer");

// Attention weights are too large to stream per node through the ring buffer.
// You only ever VIEW one layer at a time, so attention is delivered through a
// separate single-slot, double-buffered snapshot (built in a later module),
// not through the packet stream. Shape is the small viewport the TUI renders.
inline constexpr int kAttnTile = 64; // capture up to 64x64, TUI pans an 8x8 window
struct AttentionSnapshot {
    std::int32_t layer_index = -1;
    std::int32_t head        = 0;
    std::int32_t rows        = 0;
    std::int32_t cols        = 0;
    std::array<float, kAttnTile * kAttnTile> weights{}; // row-major
};

} // namespace llmtrace
