cmake_minimum_required(VERSION 3.20)

project(enyo_nnue_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(fmt REQUIRED)

add_executable(nnue-mix-jsonl tools/cpp/mix_jsonl.cpp)
target_link_libraries(nnue-mix-jsonl PRIVATE fmt::fmt)
target_compile_options(nnue-mix-jsonl PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -O3>
)

add_executable(nnue-jsonl-to-bullet-text tools/cpp/jsonl_to_bullet_text.cpp)
target_link_libraries(nnue-jsonl-to-bullet-text PRIVATE fmt::fmt)
target_compile_options(nnue-jsonl-to-bullet-text PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -O3>
)
