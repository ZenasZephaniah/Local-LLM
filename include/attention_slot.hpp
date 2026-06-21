#pragma once
//
// attention_slot.hpp
// Single-slot "latest value wins" transport for the attention matrix.
//
// Unlike the per-node telemetry ring (which is hot and must be lock-free), an
// attention snapshot is published at most once per forward pass for the single
// selected layer, and read at ~30fps by the UI. A tiny mutex-guarded copy is
// the correct tool here: zero concurrency risk, and the critical section is far
// off the per-node hot path. The lock-free constraint applies to the ring; this
// is deliberately not that.
//
#include <mutex>
#include "layer_packet.hpp"

namespace llmtrace {

template <typename T>
class LatestSnapshot {
public:
    void store(const T& v) {
        std::lock_guard<std::mutex> g(m_);
        value_ = v;
        has_   = true;
    }
    bool load(T& out) const {
        std::lock_guard<std::mutex> g(m_);
        if (!has_) return false;
        out = value_;
        return true;
    }

private:
    mutable std::mutex m_;
    T    value_{};
    bool has_ = false;
};

using AttnSlot = LatestSnapshot<AttentionSnapshot>;

} // namespace llmtrace
