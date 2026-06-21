# Local LLM Instrumentation, Tracing, and Replay Platform

A lightweight C++20 telemetry and diagnostic tool for local transformer models.
It hooks non-invasively into a `llama.cpp` forward pass, streams per-node
telemetry (shapes, latencies, activation statistics, attention weights) through a
lock-free ring buffer, and renders it in an interactive, vim-driven terminal UI
built on FTXUI. Runs can be recorded to disk and replayed without a model.

## Architecture

```
 inference thread                 lock-free                 UI thread
 (llama_decode)                   SPSC ring                 (FTXUI loop)
 ───────────────                 ───────────                ────────────
 ggml eval callback  ──push──▶   PacketRing   ──pop──▶      TelemetryStore ──▶ panels
 (Instrumentor)                                             (single consumer)
        │                                                          ▲
        └── attention snapshot ── LatestSnapshot<AttentionSnapshot> ┘
```

- `LayerPacket` (`include/layer_packet.hpp`) is a flat, trivially copyable record
  per computed node. No heap, fixed size, so it moves cheaply across threads.
- `SpscRingBuffer` (`include/ring_buffer.hpp`) is a wait-free single-producer /
  single-consumer queue. The producer never blocks or allocates; on overflow it
  drops the newest packet and counts it, so instrumentation never stalls
  inference. Fixed capacity caps RAM.
- `Instrumentor` (`include/instrumentor.hpp`) is the non-invasive hook. It is
  wired via `llama_context_params::cb_eval`, computes latency and a sampled
  activation profile per node, and snapshots the selected layer's attention.
- `LatestSnapshot` (`include/attention_slot.hpp`) carries the large attention
  matrix out of band (mutex-guarded, low rate) instead of through the hot ring.
- `telemetry_ui.hpp` is the shared UI used by every entry point.
- `trace_io.hpp` records and replays the packet stream.

## Build

Quickest path (fetches both dependencies and builds all targets):

```bash
./setup.sh                 # CPU build
./setup.sh -DGGML_CUDA=ON  # NVIDIA GPU
./setup.sh -DGGML_METAL=ON # Apple GPU
```

Or do it manually:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git clone --depth 1 https://github.com/ArthurSonzogni/FTXUI third_party/FTXUI
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Three binaries are produced:

- `llm_app`   the integrated platform (live inference + TUI + record/replay)
- `llm_tui`   the UI alone on a synthetic data producer (no model needed)
- `llm_smoke` a headless pipeline check that prints packets to stdout

## Run

```bash
# UI demo, no model:
./build/llm_tui

# Live instrumentation of a local model (use a small quantized one):
./build/llm_app ~/models/llama-3.2-1b-instruct-q4_k_m.gguf "Once upon a time"

# Record a run, then replay it later with no model:
./build/llm_app model.gguf "prompt" --record run.trace
./build/llm_app --replay run.trace
```

## Controls

- `Tab` cycle focus between panels
- Topology: `j`/`k` move, `Space` select capture target, `l`/`Enter` expand, `h` collapse
- Attention: `h`/`j`/`k`/`l` pan the viewport, `+`/`-` contrast, `f` fullscreen
- `q` or `Esc` quit

## Version notes

This targets a recent `llama.cpp`. If your checkout is older, a few API names
differ and are flagged in `src/app.cpp`:
`llama_model_load_from_file`, `llama_init_from_model`, `llama_model_free`,
`llama_model_n_layer`, and the flash-attention toggle
(`cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED` vs the older
`cp.flash_attn = false`). Attention capture requires flash attention disabled so
the softmax scores tensor is materialized.
