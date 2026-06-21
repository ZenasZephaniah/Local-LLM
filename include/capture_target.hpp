#pragma once
//
// capture_target.hpp
// The layer the user has selected in the topology panel. Written by the TUI
// thread, read by the instrumentation thread (to decide which attention matrix
// to snapshot). Two small atomics keep it lock-free and trivially shareable.
//
#include <atomic>
#include <cstdint>
#include "layer_packet.hpp"

namespace llmtrace {

struct CaptureTarget {
    std::atomic<std::int32_t> layer_index{-1};
    std::atomic<std::uint8_t> kind{static_cast<std::uint8_t>(LayerKind::Unknown)};

    void set(std::int32_t layer, LayerKind k) noexcept {
        layer_index.store(layer, std::memory_order_relaxed);
        kind.store(static_cast<std::uint8_t>(k), std::memory_order_release);
    }

    bool matches(std::int32_t layer, LayerKind k) const noexcept {
        return layer_index.load(std::memory_order_relaxed) == layer &&
               kind.load(std::memory_order_acquire) == static_cast<std::uint8_t>(k);
    }
};

} // namespace llmtrace
