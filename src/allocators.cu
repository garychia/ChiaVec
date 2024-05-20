#include "allocators.hpp"

#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

namespace ChiaVec
{
    namespace Memory
    {
        void *DefaultAllocator::operator()(std::size_t size)
        {
            return size != 0 ? std::malloc(size) : nullptr;
        }

        void DefaultAllocator::release(void *ptr)
        {
            if (ptr)
            {
                std::free(ptr);
            }
        }

        void *DefaultAllocator::resize(void *ptr, std::size_t oldSize, std::size_t newSize)
        {
            if (newSize > oldSize)
            {
                ptr = std::realloc(ptr, newSize);
            }
            return ptr;
        }

        void DefaultAllocator::copy(void *dst, const void *src, std::size_t size)
        {
            std::memcpy(dst, src, size);
        }

        void *DefaultCudaAllocator::operator()(std::size_t size)
        {
            void *ptr;
            cudaMalloc(&ptr, size);
            return ptr;
        }

        void DefaultCudaAllocator::release(void *ptr)
        {
            if (ptr)
            {
                cudaFree(ptr);
            }
        }

        void *DefaultCudaAllocator::resize(void *ptr, std::size_t oldSize, std::size_t newSize)
        {
            if (newSize > oldSize)
            {
                void *newData = (*this)(newSize);
                cudaMemcpy(newData, ptr, oldSize, cudaMemcpyDeviceToDevice);
                release(ptr);
                ptr = newData;
            }
            return ptr;
        }

        void DefaultCudaAllocator::copy(void *dst, const void *src, std::size_t size)
        {
            cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
        }

        void DefaultCudaAllocator::copyDeviceToHost(void *dst, const void *src, std::size_t size)
        {
            cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
        }

        void DefaultCudaAllocator::copyHostToDevice(void *dst, const void *src, std::size_t size)
        {
            cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
        }
    } // namespace Memory
} // namespace ChiaVec