cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(ChiaVec LANGUAGES C CXX CUDA)

add_subdirectory(src)
add_subdirectory(examples/c)
add_subdirectory(examples/c++)
 