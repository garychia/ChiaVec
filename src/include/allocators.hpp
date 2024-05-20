#ifndef __CHIAVEC_ALLOCATORS_HPP__
#define __CHIAVEC_ALLOCATORS_HPP__

#include <cstdlib>

namespace ChiaVec
{
    namespace Memory
    {
        struct BaseAllocator
        {
            virtual void *operator()(std::size_t size) = 0;
            virtual void release(void *ptr) = 0;
            virtual void *resize(void *ptr, std::size_t oldSize, std::size_t newSize) = 0;
            virtual void copy(void *dst, const void *src, std::size_t size) = 0;
        };

        struct DefaultAllocator : BaseAllocator
        {
            virtual void *operator()(std::size_t size) override;
            virtual void release(void *ptr) override;
            virtual void *resize(void *ptr, std::size_t oldSize, std::size_t newSize) override;
            virtual void copy(void *dst, const void *src, std::size_t size) override;
        };

        struct DefaultCudaAllocator : BaseAllocator
        {
            virtual void *operator()(std::size_t size) override;
            virtual void release(void *ptr) override;
            virtual void *resize(void *ptr, std::size_t oldSize, std::size_t newSize) override;
            virtual void copy(void *dst, const void *src, std::size_t size) override;
            virtual void copyDeviceToHost(void *dst, const void *src, std::size_t size);
            virtual void copyHostToDevice(void *dst, const void *src, std::size_t size);
        };
    } // namespace Memory
} // namespace ChiaVec

#endif
