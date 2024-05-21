add_library(chiavec STATIC
    include/allocators.hpp
    include/rawvecs.hpp
    include/vecs.hpp
    include/chiavecs.h
    allocators.cu
    chiavecs.cu)
target_include_directories(chiavec PUBLIC include)
set_target_properties(chiavec PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
