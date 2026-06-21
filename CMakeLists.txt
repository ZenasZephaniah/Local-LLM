cmake_minimum_required(VERSION 3.20)
project(llm_trace LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Both expected as git submodules under third_party/
#   git submodule add https://github.com/ggml-org/llama.cpp third_party/llama.cpp
#   git submodule add https://github.com/ArthurSonzogni/FTXUI third_party/FTXUI
add_subdirectory(third_party/llama.cpp)
add_subdirectory(third_party/FTXUI)

set(FTXUI_LIBS ftxui::screen ftxui::dom ftxui::component)

# ---- llm_app : the integrated platform (live inference + TUI + record/replay)
add_executable(llm_app src/app.cpp)
target_include_directories(llm_app PRIVATE include)
target_link_libraries(llm_app PRIVATE llama ${FTXUI_LIBS})

# ---- llm_tui : standalone UI on a synthetic producer (FTXUI only, no model)
add_executable(llm_tui src/tui.cpp)
target_include_directories(llm_tui PRIVATE include)
target_link_libraries(llm_tui PRIVATE ${FTXUI_LIBS})

# ---- llm_smoke : headless pipeline check, prints packets to stdout
add_executable(llm_smoke src/main.cpp)
target_include_directories(llm_smoke PRIVATE include)
target_link_libraries(llm_smoke PRIVATE llama ${FTXUI_LIBS})

# Enable a GPU backend at configure time if you have one, e.g.:
#   cmake -B build -DGGML_CUDA=ON
#   cmake -B build -DGGML_METAL=ON
