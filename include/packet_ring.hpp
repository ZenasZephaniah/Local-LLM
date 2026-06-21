#pragma once
//
// packet_ring.hpp
// The one canonical telemetry ring type, shared by the instrumentation hook
// (producer) and the TUI (consumer) so the integrated app passes a single
// concrete type between them.
//
#include "layer_packet.hpp"
#include "ring_buffer.hpp"

namespace llmtrace {

inline constexpr std::size_t kRingCapacity = 4096; // power of two
using PacketRing = SpscRingBuffer<LayerPacket, kRingCapacity>;

} // namespace llmtrace
