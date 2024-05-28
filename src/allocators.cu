#include "allocators.hpp"

#include <cuda_runtime.h>

namespace ChiaVec {
namespace Memory {
void *DefaultAllocator::operator()(std::size_t size) {
  void *ptr = nullptr;

  if (0 == size) {
    return ptr;
  }

  cudaMallocHost(&ptr, size);
  return ptr;
}

void DefaultAllocator::release(void *ptr) {
  if (!ptr) {
    return;
  }
  cudaFreeHost(ptr);
}

void *DefaultAllocator::resize(void *ptr, std::size_t oldSize,
                               std::size_t newSize) {
  if (newSize > oldSize) {
    void *newPtr = nullptr;
    cudaMallocHost(&newPtr, newSize);
    cudaMemcpy(newPtr, ptr, oldSize, cudaMemcpyHostToHost);
    cudaFreeHost(ptr);
    ptr = newPtr;
  }
  return ptr;
}

void DefaultAllocator::copy(void *dst, const void *src, std::size_t size,
                            bool dstOnHost, bool srcOnHost) {
  if (dstOnHost) {
    if (srcOnHost) {
      cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
    } else {
      cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }
  } else {
    if (srcOnHost) {
      cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    } else {
      cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    }
  }
}

void *DefaultCudaAllocator::operator()(std::size_t size) {
  void *ptr = nullptr;
  if (0 == size) {
    return ptr;
  }
  cudaMalloc(&ptr, size);
  return ptr;
}

void DefaultCudaAllocator::release(void *ptr) {
  if (ptr) {
    cudaFree(ptr);
  }
}

void *DefaultCudaAllocator::resize(void *ptr, std::size_t oldSize,
                                   std::size_t newSize) {
  if (newSize > oldSize) {
    void *newPtr = nullptr;
    cudaMallocHost(&newPtr, newSize);
    cudaMemcpy(newPtr, ptr, oldSize, cudaMemcpyHostToHost);
    cudaFreeHost(ptr);
    ptr = newPtr;
  }
  return ptr;
}

void DefaultCudaAllocator::copy(void *dst, const void *src, std::size_t size,
                                bool dstOnHost, bool srcOnHost) {
  if (dstOnHost) {
    if (srcOnHost) {
      cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
    } else {
      cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }
  } else {
    if (srcOnHost) {
      cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    } else {
      cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    }
  }
}

} // namespace Memory
} // namespace ChiaVec