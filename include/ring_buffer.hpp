#pragma once
//
// ring_buffer.hpp
// Single-producer / single-consumer wait-free ring buffer.
//
//   Producer = inference thread (the ggml eval callback)
//   Consumer = TUI render thread
//
// The producer NEVER blocks, never allocates, and never touches a mutex, so it
// adds no contention to the model forward pass. On overflow it drops the newest
// packet and bumps a counter, which is the correct trade for telemetry: losing a
// diagnostic sample is acceptable, stalling inference is not. Fixed capacity caps
// RAM, satisfying the "RAM must not shoot up" constraint.
//
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <new>

namespace llmtrace {

// Fixed at 64 (the common cache-line size). We avoid
// std::hardware_destructive_interference_size on purpose: it emits an ABI
// warning under -Werror and can vary with -mtune. 64 is correct for x86-64
// and AArch64 here; adjust if you target an exotic cache geometry.
inline constexpr std::size_t kCacheLine = 64;

template <typename T, std::size_t Capacity>
class SpscRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of two for cheap index masking");
    static_assert(Capacity >= 2, "Need at least two slots");

public:
    // Called only from the producer thread.
    bool try_push(const T& item) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t next = (head + 1) & kMask;
        // tail_ is advanced by the consumer; acquire to observe its progress.
        if (next == tail_.load(std::memory_order_acquire)) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return false;                                  // full
        }
        buffer_[head] = item;                              // sole writer owns this slot
        head_.store(next, std::memory_order_release);      // publish to consumer
        return true;
    }

    // Called only from the consumer thread.
    bool try_pop(T& out) noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return false;                                  // empty
        }
        out = buffer_[tail];
        tail_.store((tail + 1) & kMask, std::memory_order_release);
        return true;
    }

    std::uint64_t dropped() const noexcept {
        return dropped_.load(std::memory_order_relaxed);
    }

private:
    static constexpr std::size_t kMask = Capacity - 1;

    std::array<T, Capacity> buffer_{};

    // Each atomic lives on its own cache line so the producer writing head_ does
    // not invalidate the consumer's cache line holding tail_ (false sharing).
    alignas(kCacheLine) std::atomic<std::size_t>  head_{0};
    alignas(kCacheLine) std::atomic<std::size_t>  tail_{0};
    alignas(kCacheLine) std::atomic<std::uint64_t> dropped_{0};
};

} // namespace llmtrace
