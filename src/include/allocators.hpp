#ifndef __CHIAVEC_ALLOCATORS_HPP__
#define __CHIAVEC_ALLOCATORS_HPP__

#include <cstdlib>

namespace ChiaVec {
namespace Memory {
/**
 * @brief Base class for memory allocators.
 *
 * This class defines the interface for memory allocators that can be used to
 * allocate, release, and resize memory, as well as copy data between host and
 * device memory.
 *
 * @tparam OnHost Indicates whether the allocator allocates memory on the host
 * (CPU) or on the device (GPU).
 */
template <bool OnHost>
struct BaseAllocator {
  /**
   * @brief Indicates whether the allocator allocates memory on the host (CPU)
   * or on the device (GPU).
   */
  static constexpr bool AllocatesOnHost = OnHost;

  /**
   * @brief Allocates memory of the specified size.
   *
   * @param size The size of the memory block to allocate, in bytes.
   * @return A pointer to the allocated memory block.
   */
  virtual void *operator()(std::size_t size) = 0;

  /**
   * @brief Releases the memory block pointed to by the given pointer.
   *
   * @param ptr A pointer to the memory block to be released.
   */
  virtual void release(void *ptr) = 0;

  /**
   * @brief Resizes the memory block pointed to by the given pointer.
   *
   * @param ptr The pointer to the memory block to be resized.
   * @param oldSize The current size of the memory block, in bytes.
   * @param newSize The new size of the memory block, in bytes.
   * @return A pointer to the resized memory block.
   */
  virtual void *resize(void *ptr, std::size_t oldSize, std::size_t newSize) = 0;

  /**
   * @brief Copies data between host and device memory.
   *
   * @param dst The destination pointer for the copy operation.
   * @param src The source pointer for the copy operation.
   * @param size The size of the data to be copied, in bytes.
   * @param dstOnHost Indicates whether the destination memory is on the host.
   * @param srcOnHost Indicates whether the source memory is on the host.
   */
  virtual void copy(void *dst, const void *src, std::size_t size,
                    bool dstOnHost, bool srcOnHost) = 0;
};

/**
 * @brief Default memory allocator that allocates memory on the host (CPU).
 *
 * This class implements the BaseAllocator interface to provide a default
 * memory allocator that allocates memory on the host (CPU).
 */
struct DefaultAllocator : BaseAllocator<true> {
  /**
   * @brief Allocates a block of memory of the specified size on the host.
   *
   * @param size The size of the memory block to allocate, in bytes.
   * @return A pointer to the allocated memory block.
   */
  virtual void *operator()(std::size_t size) override;

  /**
   * @brief Releases the memory block pointed to by the given pointer.
   *
   * @param ptr A pointer to the memory block to be released.
   */
  virtual void release(void *ptr) override;

  /**
   * @brief Resizes the memory block pointed to by the given pointer.
   *
   * @param ptr The pointer to the memory block to be resized.
   * @param oldSize The current size of the memory block, in bytes.
   * @param newSize The new size of the memory block, in bytes.
   * @return A pointer to the resized memory block.
   */
  virtual void *resize(void *ptr, std::size_t oldSize,
                       std::size_t newSize) override;

  /**
   * @brief Copies data between host and device memory.
   *
   * @param dst The destination pointer for the copy operation.
   * @param src The source pointer for the copy operation.
   * @param size The size of the data to be copied, in bytes.
   * @param dstOnHost Indicates whether the destination memory is on the host.
   * @param srcOnHost Indicates whether the source memory is on the host.
   */
  virtual void copy(void *dst, const void *src, std::size_t size,
                    bool dstOnHost, bool srcOnHost) override;
};

/**
 * @brief Default memory allocator that allocates memory on the device (GPU).
 *
 * This class implements the BaseAllocator interface to provide a default
 * memory allocator that allocates memory on the device (GPU).
 */
struct DefaultCudaAllocator : BaseAllocator<false> {
  /**
   * @brief Allocates a block of memory of the specified size on the device.
   *
   * @param size The size of the memory block to allocate, in bytes.
   * @return A pointer to the allocated memory block.
   */
  virtual void *operator()(std::size_t size) override;

  /**
   * @brief Releases the memory block pointed to by the given pointer.
   *
   * @param ptr A pointer to the memory block to be released.
   */
  virtual void release(void *ptr) override;

  /**
   * @brief Resizes the memory block pointed to by the given pointer.
   *
   * @param ptr The pointer to the memory block to be resized.
   * @param oldSize The current size of the memory block, in bytes.
   * @param newSize The new size of the memory block, in bytes.
   * @return A pointer to the resized memory block.
   */
  virtual void *resize(void *ptr, std::size_t oldSize,
                       std::size_t newSize) override;

  /**
   * @brief Copies data between host and device memory.
   *
   * @param dst The destination pointer for the copy operation.
   * @param src The source pointer for the copy operation.
   * @param size The size of the data to be copied, in bytes.
   * @param dstOnHost Indicates whether the destination memory is on the host.
   * @param srcOnHost Indicates whether the source memory is on the host.
   */
  virtual void copy(void *dst, const void *src, std::size_t size,
                    bool dstOnHost, bool srcOnHost) override;
};
} // namespace Memory
} // namespace ChiaVec

#endif
