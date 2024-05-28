#ifndef __CHIAVEC_ALLOCATORS_HPP__
#define __CHIAVEC_ALLOCATORS_HPP__

#include <cstdlib>

namespace ChiaVec {
namespace Memory {
template <bool OnHost>
struct BaseAllocator {
  static constexpr bool AllocatesOnHost = OnHost;
  virtual void *operator()(std::size_t size) = 0;
  virtual void release(void *ptr) = 0;
  virtual void *resize(void *ptr, std::size_t oldSize, std::size_t newSize) = 0;
  virtual void copy(void *dst, const void *src, std::size_t size,
                    bool dstOnHost, bool srcOnHost) = 0;
};

struct DefaultAllocator : BaseAllocator<true> {
  virtual void *operator()(std::size_t size) override;
  virtual void release(void *ptr) override;
  virtual void *resize(void *ptr, std::size_t oldSize,
                       std::size_t newSize) override;
  virtual void copy(void *dst, const void *src, std::size_t size,
                    bool dstOnHost, bool srcOnHost) override;
};

struct DefaultCudaAllocator : BaseAllocator<false> {
  virtual void *operator()(std::size_t size) override;
  virtual void release(void *ptr) override;
  virtual void *resize(void *ptr, std::size_t oldSize,
                       std::size_t newSize) override;
  virtual void copy(void *dst, const void *src, std::size_t size,
                    bool dstOnHost, bool srcOnHost) override;
};
} // namespace Memory
} // namespace ChiaVec

#endif
