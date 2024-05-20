add_library(cudavec STATIC
    include/allocators.hpp
    include/rawvecs.hpp
    include/vecs.hpp
    include/chiavecs.h
    allocators.cu
    chiavecs.cu)
target_include_directories(cudavec PUBLIC include)
set_target_properties(cudavec PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
