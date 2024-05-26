add_library(chiavec STATIC
    include/allocators.hpp
    include/rawvecs.hpp
    include/vecs.hpp
    include/chiavecs.h
    include/utilities.hpp
    allocators.cu
    chiavecs.cu
    utilities.cu)
target_include_directories(chiavec PUBLIC include)
target_compile_features(chiavec PUBLIC cxx_std_17)
set_target_properties(chiavec PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(chiavec PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(chiavec PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)