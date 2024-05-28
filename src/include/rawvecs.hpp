#ifndef __CHIAVEC_RAWVECS_HPP__
#define __CHIAVEC_RAWVECS_HPP__

#include "allocators.hpp"

namespace ChiaVec {
namespace Raw {
template <class T, class Allocator = Memory::DefaultAllocator>
class RawVec {
private:
  T *elements;
  size_t length;

  void releaseMemory() {
    Allocator allocator;
    allocator.release(elements);
  }

  void deleteElements() {
    releaseMemory();
    elements = nullptr;
  }

  void clearAll() {
    deleteElements();
    length = 0;
  }

  virtual void copyElement(const T &element, std::size_t index,
                           bool elementOnHost) {
    Allocator allocator;
    allocator.copy(&elements[index], &element, sizeof(T),
                   Allocator::AllocatesOnHost, elementOnHost);
  }

public:
  RawVec() : elements(nullptr), length(0) {}

  RawVec(std::size_t length) : length(length) {
    Allocator allocator;
    elements = static_cast<T *>(allocator(sizeof(T) * length));
  }

  RawVec(const T *data, std::size_t length, bool onHost) : RawVec(length) {
    copyMemory(data, length, onHost);
  }

  template <class OtherAllocator>
  RawVec(const RawVec<T, OtherAllocator> &other)
      : RawVec(other.elements, other.length, OtherAllocator::AllocatesOnHost) {}

  RawVec(RawVec<T, Allocator> &&other)
      : elements(other.elements), length(other.length) {
    other.elements = nullptr;
    other.length = 0;
  }

  ~RawVec() { releaseMemory(); }

  RawVec<T, Allocator> &operator=(const RawVec<T, Allocator> &other) {
    Allocator allocator;
    resize(other.length);
    allocator.copy(elements, other.elements, sizeof(T) * length,
                   Allocator::AllocatesOnHost, Allocator::AllocatesOnHost);
    return *this;
  }

  template <class OtherAllocator>
  RawVec<T, Allocator> &operator=(const RawVec<T, OtherAllocator> &other) {
    Allocator allocator;
    resize(other.length);
    allocator.copy(elements, other.elements, sizeof(T) * length,
                   Allocator::AllocatesOnHost, OtherAllocator::AllocatesOnHost);
    return *this;
  }

  RawVec<T, Allocator> &operator=(RawVec<T, Allocator> &&other) {
    clearAll();
    elements = other.elements;
    other.elements = nullptr;
    length = other.length;
    other.length = 0;
    return *this;
  }

  T *ptr() { return elements; }

  const T *ptr() const { return elements; }

  std::size_t len() const { return length; }

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

  template <class Itr>
  void copy(Itr iter, std::size_t length, bool onHost) {
    resize(length);
    for (std::size_t i = 0; i < length; i++) {
      copyElement(*iter, i, onHost);
      iter++;
    }
  }

  void copyMemory(const T *ptr, std::size_t length, bool onHost) {
    Allocator allocator;
    resize(length);
    allocator.copy(elements, ptr, sizeof(T) * length,
                   Allocator::AllocatesOnHost, onHost);
  }

  template <class OtherAllocator>
  void copyTo(RawVec<T, OtherAllocator> &other) const {
    other.resize(length);
    other.copyMemory(elements, length, Allocator::AllocatesOnHost);
  }

  template <class OtherAllocator>
  void copyFrom(const RawVec<T, OtherAllocator> &other) {
    resize(other.len());
    copyMemory(other.elements, other.len(), OtherAllocator::AllocatesOnHost);
  }

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