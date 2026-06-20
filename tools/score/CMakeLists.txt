cmake_minimum_required(VERSION 3.20)

project(nnue_score_tools LANGUAGES CXX)

find_package(fmt REQUIRED)

add_executable(count_binpack count_binpack.cpp)
target_compile_features(count_binpack PRIVATE cxx_std_23)
target_link_libraries(count_binpack PRIVATE fmt::fmt)
