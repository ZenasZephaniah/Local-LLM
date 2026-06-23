# Local LLM Instrumentation, Tracing, and Replay Platform

A lightweight **C++20** telemetry and diagnostic tool for local transformer models.
It hooks **non-invasively** into a `llama.cpp` forward pass, streams per-node
telemetry (tensor shapes, layer latencies, activation statistics, attention
weights) through a **lock-free ring buffer**, and renders it in an interactive,
vim-driven terminal UI built on **FTXUI**. Runs can be recorded to disk and
replayed later without a model.

> GDSC Open Projects submission.

## Team

- **Thirupathi Badavath** — [@thirupathi1918](https://github.com/thirupathi1918)
- **Zenas Zephaniah** — [@ZenasZephaniah](https://github.com/ZenasZephaniah)

## What it shows (the 5 panels)

1. **Model Topology** — collapsible tree of the model's layers and sub-modules.
2. **Live Packet Stream** — real-time table of every computed node: id, time,
   layer, type, compute device, latency.
3. **Attention Matrix** — grayscale/blue heatmap of the selected layer's
   attention, with pan, contrast, and fullscreen.
4. **Runtime Metrics** — shape, dtype, mean/max, sparsity gauge, latency delta
   for the selected node.
5. **Anomaly Ledger** — timestamped log of outliers, sparsity collapse,
   NaN/Inf, and CPU fallbacks.

## Requirements

- A C++20 compiler (`g++` 12+ or `clang` 15+)
- `cmake` (3.20+) and `git`
- Linux or macOS. **On Windows, use WSL2** (Ubuntu) — see the note below.

## Build

Easiest path (fetches both dependencies and builds all targets):

```bash
./setup.sh                 # CPU build
./setup.sh -DGGML_CUDA=ON  # NVIDIA GPU
./setup.sh -DGGML_METAL=ON # Apple GPU
```

Or do it manually:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git clone --depth 1 https://github.com/ArthurSonzogni/FTXUI  third_party/FTXUI

cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF \
  -DFTXUI_BUILD_EXAMPLES=OFF -DFTXUI_BUILD_TESTS=OFF

cmake --build build -j
```

Three binaries are produced in `build/`:

- `llm_app`   — the integrated platform (live inference + TUI + record/replay)
- `llm_tui`   — the UI alone on synthetic data (no model needed)
- `llm_smoke` — a headless pipeline check that prints packets to stdout

### Low-memory machines (and WSL)

Building `llama.cpp` is memory-heavy. If the build is killed with `Terminated`
or your terminal closes, compile one file at a time:

```bash
cmake --build build -j1
```

On **WSL2**, also give Linux more memory by creating a file named `.wslconfig`
in your Windows user folder (`C:\Users\<you>\.wslconfig`) with:

```
[wsl2]
memory=4GB
swap=16GB
```

Then run `wsl --shutdown` in PowerShell, reopen Ubuntu, and build again.

## Get a model

Any GGUF model works. A small, fast one to start with (≈491 MB):

```bash
mkdir -p models
wget -c -O models/qwen.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
```

## Run

```bash
# 1) UI demo on synthetic data (no model needed):
./build/llm_tui

# 2) Live instrumentation of a real model:
./build/llm_app models/qwen.gguf "Once upon a time"

# 3) Read the prompt only (best view of the full attention triangle):
./build/llm_app models/qwen.gguf "The quick brown fox jumps over the lazy dog" --predict 0

# 4) Generate N tokens while instrumenting:
./build/llm_app models/qwen.gguf "Once upon a time" --predict 64

# 5) Record a run, then replay it later with no model:
./build/llm_app models/qwen.gguf "prompt" --record run.trace
./build/llm_app --replay run.trace
```

To see attention: focus the topology panel (`Tab`), move to an `attn` node
(`j`/`k`), and press `Space` to select it. Use `--predict 0` so the model reads
the whole prompt at once and the full triangular attention pattern is visible.

## Controls

| Key | Action |
|-----|--------|
| `Tab` | cycle focus between panels |
| `j` / `k` | move up/down (topology) |
| `Space` | select the capture target (topology) |
| `l` / `Enter` | expand a block (topology) |
| `h` | collapse (topology) |
| `h` / `j` / `k` / `l` | pan the attention viewport |
| `+` / `-` | attention contrast |
| `f` | attention fullscreen |
| `q` / `Esc` | quit |

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

- **`LayerPacket`** (`include/layer_packet.hpp`) — a flat, trivially copyable
  record per computed node. No heap, fixed size, so it moves cheaply between
  threads.
- **`SpscRingBuffer`** (`include/ring_buffer.hpp`) — a wait-free
  single-producer / single-consumer queue. The producer never blocks or
  allocates; on overflow it drops the newest packet and counts it, so
  instrumentation never stalls inference. Fixed capacity caps RAM.
- **`Instrumentor`** (`include/instrumentor.hpp`) — the non-invasive hook,
  wired via `llama_context_params::cb_eval`. Computes latency and a sampled
  activation profile per node, and snapshots the selected layer's attention.
- **`LatestSnapshot`** (`include/attention_slot.hpp`) — carries the large
  attention matrix out of band (mutex-guarded, low rate) instead of through the
  hot ring.
- **`telemetry_ui.hpp`** — the shared TUI used by every entry point.
- **`trace_io.hpp`** — records and replays the packet stream.

## Project layout

```
include/   header-only core (packet, ring buffer, hook, UI, trace I/O)
src/       app.cpp (integrated), tui.cpp (demo), main.cpp (smoke test)
CMakeLists.txt
setup.sh   one-command fetch + build
```

## Version notes

This targets a recent `llama.cpp`. If your checkout is older, a few API names
differ and are flagged in `src/app.cpp`: `llama_model_load_from_file`,
`llama_init_from_model`, `llama_model_free`, `llama_model_n_layer`, and the
flash-attention toggle (`cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED`
vs the older `cp.flash_attn = false`). Attention capture requires flash
attention **disabled** so the softmax scores tensor is materialized.
