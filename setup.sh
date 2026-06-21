#!/usr/bin/env bash
#
# setup.sh - fetch dependencies and build everything.
# Works from a plain download (no git repo required).
#
#   ./setup.sh                 # CPU build
#   ./setup.sh -DGGML_CUDA=ON  # NVIDIA GPU build
#   ./setup.sh -DGGML_METAL=ON # Apple GPU build
#
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p third_party
if [ ! -d third_party/llama.cpp ]; then
  echo ">> fetching llama.cpp"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp third_party/llama.cpp
fi
if [ ! -d third_party/FTXUI ]; then
  echo ">> fetching FTXUI"
  git clone --depth 1 https://github.com/ArthurSonzogni/FTXUI third_party/FTXUI
fi

echo ">> configuring"
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DFTXUI_BUILD_EXAMPLES=OFF -DFTXUI_BUILD_TESTS=OFF \
  "$@"

echo ">> building"
cmake --build build -j"$(nproc 2>/dev/null || echo 4)"

echo
echo "Built:"
echo "  build/llm_app    live inference + TUI + record/replay"
echo "  build/llm_tui    UI on synthetic data (no model needed)"
echo "  build/llm_smoke  headless pipeline check"
echo
echo "Try:   ./build/llm_tui"
echo "Then:  ./build/llm_app /path/to/model.gguf \"Once upon a time\""
