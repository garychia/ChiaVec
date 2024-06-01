#ifndef __CHIAVEC_RAWVECS_HPP__
#define __CHIAVEC_RAWVECS_HPP__

#include "allocators.hpp"

namespace ChiaVec {
namespace Raw {
/**
 * @brief A simple vector-like class that manages raw memory.
 *
 * The `RawVec` class is a simple vector-like class that manages raw memory
 * using a custom memory allocator. It provides basic functionality for
 * allocating, resizing, and copying memory, as well as some convenience
 * methods for working with the raw memory.
 *
 * @tparam T The type of elements stored in the vector.
 * @tparam Allocator The memory allocator to use for managing the raw memory.
 *                   Defaults to `Memory::DefaultAllocator`.
 */
template <class T, class Allocator = Memory::DefaultAllocator>
class RawVec {
private:
  /**
   * @brief Pointer to the raw memory block.
   */
  T *elements;

  /**
   * @brief The length of the raw memory block, in number of elements.
   */
  size_t length;

  /**
   * @brief Releases the memory occupied by the raw memory block.
   */
  void releaseMemory() {
    Allocator allocator;
    allocator.release(elements);
  }

  /**
   * @brief Deletes the elements in the raw memory block and sets the pointer to
   * `nullptr`.
   */
  void deleteElements() {
    releaseMemory();
    elements = nullptr;
  }

  /**
   * @brief Clears all the elements in the raw memory block and sets the length
   * to 0.
   */
  void clearAll() {
    deleteElements();
    length = 0;
  }

  /**
   * @brief Copies a single element to the raw memory block.
   *
   * @param element The element to copy.
   * @param index The index in the raw memory block where the element should be
   *              copied.
   * @param elementOnHost Indicates whether the source element is located on the
   *                      host (CPU) or the device (GPU).
   */
  virtual void copyElement(const T &element, std::size_t index,
                           bool elementOnHost) {
    Allocator allocator;
    allocator.copy(&elements[index], &element, sizeof(T),
                   Allocator::AllocatesOnHost, elementOnHost);
  }

public:
  /**
   * @brief Constructs an empty `RawVec` object.
   */
  RawVec() : elements(nullptr), length(0) {}

  /**
   * @brief Constructs a `RawVec` object with the specified length.
   *
   * @param length The initial length of the raw memory block.
   */
  RawVec(std::size_t length) : length(length) {
    Allocator allocator;
    elements = static_cast<T *>(allocator(sizeof(T) * length));
  }

  /**
   * @brief Constructs a `RawVec` object from a raw memory block.
   *
   * @param data The raw memory block to copy.
   * @param length The length of the raw memory block.
   * @param onHost Indicates whether the raw memory block is located on the host
   *               (CPU) or the device (GPU).
   */
  RawVec(const T *data, std::size_t length, bool onHost) : RawVec(length) {
    copyMemory(data, length, onHost);
  }

  /**
   * @brief Constructs a `RawVec` object by copying another `RawVec` object.
   *
   * @param other The `RawVec` object to copy.
   */
  template <class OtherAllocator>
  RawVec(const RawVec<T, OtherAllocator> &other)
      : RawVec(other.elements, other.length, OtherAllocator::AllocatesOnHost) {}

  /**
   * @brief Constructs a `RawVec` object by moving another `RawVec` object.
   *
   * @param other The `RawVec` object to move.
   */
  RawVec(RawVec<T, Allocator> &&other)
      : elements(other.elements), length(other.length) {
    other.elements = nullptr;
    other.length = 0;
  }

  /**
   * @brief Destroys the `RawVec` object and releases the raw memory block.
   */
  ~RawVec() { releaseMemory(); }

  RawVec<T, Allocator> &operator=(const RawVec<T, Allocator> &other) {
    copyFrom(other);
    return *this;
  }

  /**
   * @brief Assigns the contents of another `RawVec` object to this `RawVec`
   * object.
   *
   * @param other The `RawVec` object to assign.
   * @return A reference to the modified `RawVec` object.
   */
  template <class OtherAllocator>
  RawVec<T, Allocator> &operator=(const RawVec<T, OtherAllocator> &other) {
    copyFrom(other);
    return *this;
  }

  /**
   * @brief Assigns the contents of another `RawVec` object to this `RawVec`
   * object by moving.
   *
   * @param other The `RawVec` object to move.
   * @return A reference to the modified `RawVec` object.
   */
  RawVec<T, Allocator> &operator=(RawVec<T, Allocator> &&other) {
    clearAll();
    elements = other.elements;
    other.elements = nullptr;
    length = other.length;
    other.length = 0;
    return *this;
  }

  /**
   * @brief Returns a pointer to the raw memory block.
   *
   * @return A pointer to the raw memory block.
   */
  T *ptr() { return elements; }

  /**
   * @brief Returns a const pointer to the raw memory block.
   *
   * @return A const pointer to the raw memory block.
   */
  const T *ptr() const { return elements; }

  /**
   * @brief Returns the length of the raw memory block, in number of elements.
   *
   * @return The length of the raw memory block.
   */
  std::size_t len() const { return length; }

  /**
   * @brief Resizes the raw memory block to the specified length.
   *
   * @param newLength The new length of the raw memory block.
   */
  void resize(std::size_t newLength) {
    if (0 == newLength) {
      deleteElements();
    } else if (newLength > length) {
      Allocator allocator;
      elements = static_cast<T *>(allocator.resize(elements, sizeof(T) * length,
                                                   sizeof(T) * newLength));
    }
    length = newLength;
  }

  /**
   * @brief Copies the elements from an iterator range to the raw memory block.
   *
   * @tparam Itr The iterator type.
   * @param iter The beginning of the iterator range to copy.
   * @param length The number of elements to copy.
   * @param onHost Indicates whether the source elements are located on the host
   *               (CPU) or the device (GPU).
   */
  template <class Itr>
  void copy(Itr iter, std::size_t length, bool onHost) {
    resize(length);
    for (std::size_t i = 0; i < length; i++) {
      copyElement(*iter, i, onHost);
      iter++;
    }
  }

  /**
   * @brief Copies the elements from a raw memory block to the raw memory block
   * of this `RawVec` object.
   *
   * @param ptr The pointer to the raw memory block to copy.
   * @param length The length of the raw memory block to copy.
   * @param onHost Indicates whether the source memory block is located on the
   *               host (CPU) or the device (GPU).
   */
  void copyMemory(const T *ptr, std::size_t length, bool onHost) {
    Allocator allocator;
    resize(length);
    allocator.copy(elements, ptr, sizeof(T) * length,
                   Allocator::AllocatesOnHost, onHost);
  }

  /**
   * @brief Copies the contents of this `RawVec` object to another `RawVec`
   * object.
   *
   * @tparam OtherAllocator The allocator type of the other `RawVec` object.
   * @param other The `RawVec` object to copy the contents to.
   */
  template <class OtherAllocator>
  void copyTo(RawVec<T, OtherAllocator> &other) const {
    other.resize(length);
    other.copyMemory(elements, length, Allocator::AllocatesOnHost);
  }

  /**
   * @brief Copies the contents of another `RawVec` object to this `RawVec`
   * object.
   *
   * @tparam OtherAllocator The allocator type of the other `RawVec` object.
   * @param other The `RawVec` object to copy the contents from.
   */
  template <class OtherAllocator>
  void copyFrom(const RawVec<T, OtherAllocator> &other) {
    resize(other.len());
    copyMemory(other.elements, other.len(), OtherAllocator::AllocatesOnHost);
  }

  /**
   * @brief Creates a clone of this `RawVec` object using a different allocator.
   *
   * @tparam OtherAllocator The allocator type to use for the cloned `RawVec`
   *                        object.
   * @return A new `RawVec` object that is a clone of this `RawVec` object.
   */
  template <class OtherAllocator = Allocator>
  RawVec<T, OtherAllocator> clone() const {
    RawVec<T, OtherAllocator> v(length);
    v.copyMemory(elements, length, Allocator::AllocatesOnHost);
    return v;
  }

  template <class U, class OtherAllocator>
  friend class RawVec;
};

template <class T, class CudaAllocator = Memory::DefaultCudaAllocator>
using CudaRawVec = RawVec<T, CudaAllocator>;
} // namespace Raw
} // namespace ChiaVec

#endif