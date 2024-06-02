#ifndef __CHIAVEC_VECS_HPP__
#define __CHIAVEC_VECS_HPP__

#include "rawvecs.hpp"

#include "types.hpp"
#include "utilities.hpp"

#include <algorithm>
#include <optional>

namespace ChiaVec {
/**
 * @brief A generic vector data structure that supports different memory
 * allocators and storage types.
 *
 * @tparam T The type of elements stored in the vector.
 * @tparam Allocator The memory allocator used for the vector's storage.
 * Defaults to `Memory::DefaultAllocator`.
 * @tparam Storage The storage type used for the vector's data. Defaults to
 * `Raw::RawVec<T, Allocator>`.
 */
template <class T, class Allocator = Memory::DefaultAllocator,
          class Storage = Raw::RawVec<T, Allocator>>
class Vec {
private:
  Storage data;       // The underlying storage for the vector's data.
  std::size_t length; // The current length of the vector.

  /**
   * @brief Reserves additional space in the vector's storage if needed.
   *
   * @param extraElements The number of additional elements to reserve space
   * for.
   */
  void reserve(std::size_t extraElements) {
    if (length + extraElements > data.len()) {
      expand(extraElements);
    }
  }

  /**
   * @brief Expands the vector's storage to accommodate more elements.
   *
   * @param extraElements The number of additional elements to expand the
   * storage for.
   */
  void expand(std::size_t extraElements) {
    std::size_t capacity = std::max(data.len() * 2, data.len() + extraElements);
    data.resize(capacity);
  }

public:
  /**
   * @brief Constructs an empty vector.
   */
  Vec() : data(), length(0) {}

  /**
   * @brief Constructs a vector with a specified initial capacity.
   *
   * @param capacity The initial capacity of the vector.
   */
  Vec(std::size_t capacity) : data(capacity), length(0) {}

  /**
   * @brief Constructs a vector from a raw data pointer and length.
   *
   * @param data The raw data pointer.
   * @param length The length of the data.
   * @param onHost Indicates whether the data is located on the host (true) or
   * device (false).
   */
  Vec(const T *data, std::size_t length, bool onHost)
      : data(data, length, onHost), length(length) {}

  /**
   * @brief Constructs a vector from an initializer list.
   *
   * @param l The initializer list.
   */
  Vec(std::initializer_list<T> l) : data(l.size()), length(l.size()) {
    data.copy(l.begin(), l.size());
  }

  /**
   * @brief Constructs a vector from another vector.
   *
   * @tparam OtherAllocator The memory allocator used for the other vector's
   * storage.
   * @tparam OtherStorage The storage type used for the other vector's data.
   * @param other The other vector.
   */
  template <class OtherAllocator, class OtherStorage>
  Vec(const Vec<T, OtherAllocator, OtherStorage> &other)
      : data(other.data), length(other.length) {}

  /**
   * @brief Move constructor.
   *
   * Constructs a new vector by moving the contents of the `other` vector. After
   * the construction, the `other` vector will be in a valid but unspecified
   * state, and its length will be set to 0.
   *
   * @param other The vector to move from.
   */
  Vec(Vec<T, Allocator, Storage> &&other)
      : data(std::move(other.data)), length(other.length) {
    other.length = 0;
  }

  /**
   * @brief Copy assignment operator.
   *
   * @param other The vector to copy from.
   * @return A reference to the current vector after the assignment.
   */
  Vec<T, Allocator, Storage> &
  operator=(const Vec<T, Allocator, Storage> &other) {
    this->data = other.data;
    this->length = other.length;
    return *this;
  }

  /**
   * @brief Copy assignment operator for vectors with different allocators and
   * storage types.
   *
   * Assigns the contents of the `other` vector to the current vector. This
   * operation performs a deep copy, where the underlying storage is also
   * copied.
   *
   * @tparam OtherAllocator The memory allocator used for the `other` vector.
   * @tparam OtherStorage The storage type used for the `other` vector.
   * @param other The vector to copy from.
   * @return A reference to the current vector after the assignment.
   */
  template <class OtherAllocator, class OtherStorage>
  Vec<T, Allocator, Storage> &
  operator=(const Vec<T, OtherAllocator, OtherStorage> &other) {
    this->data = other.data;
    this->length = other.length;
    return *this;
  }

  /**
   * @brief Move assignment operator.
   *
   * Moves the contents of the `other` vector to the current vector, effectively
   * transferring ownership of the underlying storage. After the assignment, the
   * `other` vector will be in a valid but unspecified state.
   *
   * @param other The vector to move from.
   * @return A reference to the current vector after the assignment.
   */
  Vec<T, Allocator, Storage> &operator=(Vec<T, Allocator, Storage> &&other) {
    this->data = std::move(other.data);
    this->length = other.length;
    return *this;
  }

  /**
   * @brief Returns a reference to the element at the specified index.
   *
   * @param index The index of the element.
   * @return A reference to the element.
   */
  virtual T &operator[](std::size_t index) { return data.ptr()[index]; }

  /**
   * @brief Returns a reference to the element at the specified index.
   *
   * @param index The index of the element.
   * @return A reference to the element.
   */
  virtual const T &operator[](std::size_t index) const {
    return data.ptr()[index];
  }

  /**
   * @brief Returns the length (number of elements) of the vector.
   *
   * @return The length of the vector.
   */
  std::size_t len() const { return length; }

  /**
   * @brief Retrieves a pointer to the element at the specified index.
   *
   * If the index is within the bounds of the vector, this function returns a
   * `std::optional` containing a pointer to the element at the specified index.
   * Otherwise, it returns a `std::nullopt`.
   *
   * @param index The index of the element to retrieve.
   * @return A `std::optional` containing a pointer to the element at the
   * specified index, or `std::nullopt` if the index is out of bounds.
   */
  std::optional<T *> get(std::size_t index) {
    return index < length ? std::optional<T *>(&this->data.ptr()[index])
                          : std::nullopt;
  }

  /**
   * @brief Retrieves a const pointer to the element at the specified index.
   *
   * If the index is within the bounds of the vector, this function returns a
   * `std::optional` containing a const pointer to the element at the specified
   * index. Otherwise, it returns a `std::nullopt`.
   *
   * @param index The index of the element to retrieve.
   * @return A `std::optional` containing a const pointer to the element at the
   * specified index, or `std::nullopt` if the index is out of bounds.
   */
  std::optional<const T *> getConst(std::size_t index) const {
    return index < length ? std::optional<const T *>(&this->data.ptr()[index])
                          : std::nullopt;
  }

  /**
   * @brief Adds a new element to the end of the vector.
   *
   * This function appends the given `element` to the end of the vector. If the
   * vector's capacity needs to be increased to accommodate the new element, the
   * function will automatically resize the underlying storage.
   *
   * The `onHost` parameter specifies whether the `element` is located on the
   * host (CPU) or the device (GPU). If `onHost` is true and the allocator is
   * set to allocate on the host, the element is directly assigned to the
   * vector's storage. Otherwise, the element is copied using the allocator's
   * `copy()` function.
   *
   * @tparam U The type of the element to be added. This type must be
   * convertible to `T`.
   * @param element The element to be added to the vector.
   * @param onHost Indicates whether the `element` is located on the host (CPU)
   * or the device (GPU).
   */
  template <class U>
  void push(U &&element, bool onHost) {
    reserve(1);
    if (Allocator::AllocatesOnHost && onHost) {
      this->data.ptr()[length] = std::forward<U &&>(element);
    } else {
      Allocator allocator;
      allocator.copy(this->data.ptr() + length, &element, sizeof(T),
                     Allocator::AllocatesOnHost, onHost);
    }
    length++;
  }

  /**
   * @brief Removes and returns the last element from the vector.
   *
   * This function removes and returns the last element from the vector. If the
   * vector is empty, it returns `std::nullopt`.
   *
   * If the allocator is configured to allocate on the host, the function
   * directly returns the last element. Otherwise, it copies the last element to
   * a temporary buffer and returns it.
   *
   * @return A `std::optional` containing the last element of the vector, or
   * `std::nullopt` if the vector is empty.
   */
  virtual std::optional<T> pop() {
    if (length != 0) {
      length--;
      if (Allocator::AllocatesOnHost) {
        return std::optional<T>(std::move(data.ptr()[length]));
      } else {
        Allocator allocator;
        T last[1];
        allocator.copy(last, data.ptr() + length, sizeof(T), true, false);
        return std::optional<T>(std::move(last[0]));
      }
    }
    return std::nullopt;
  }

  /**
   * @brief Copies the contents of the current vector to another vector.
   *
   * This function copies the contents of the current vector to the `vec`
   * vector, which may have a different allocator and storage type. The function
   * performs a deep copy, where the underlying storage is also copied.
   *
   * After the copy operation, the `vec` vector will have the same length as the
   * current vector.
   *
   * @tparam OtherAllocator The memory allocator used for the `vec` vector.
   * @tparam OtherStorage The storage type used for the `vec` vector.
   * @param vec The vector to copy the contents to.
   */
  template <class OtherAllocator, class OtherStorage>
  void copyTo(Vec<T, OtherAllocator, OtherStorage> &vec) const {
    this->data.copyTo(vec.data);
    vec.length = this->length;
  }

  /**
   * @brief Copies the contents of another vector to the current vector.
   *
   * This function copies the contents of the `vec` vector to the current
   * vector. The function performs a deep copy, where the underlying storage is
   * also copied. After the copy operation, the current vector will have the
   * same length and contents as the `vec` vector.
   *
   * @tparam OtherAllocator The memory allocator used for the `vec` vector.
   * @tparam OtherStorage The storage type used for the `vec` vector.
   * @param vec The vector to copy the contents from.
   */
  template <class OtherAllocator, class OtherStorage>
  void copyFrom(const Vec<T, OtherAllocator, OtherStorage> &vec) {
    this->data.copyFrom(vec.data);
    this->length = vec.length;
  }

  /**
   * @brief Creates a new vector that is a deep copy of the current vector.
   *
   * This function creates a new vector that is a deep copy of the current
   * vector. The new vector can have a different allocator and storage type than
   * the current vector.
   *
   * @tparam OtherAllocator The memory allocator to use for the new vector.
   * @tparam OtherStorage The storage type to use for the new vector.
   * @return A new vector that is a deep copy of the current vector.
   */
  template <class OtherAllocator = Memory::DefaultAllocator,
            class OtherStorage = Raw::RawVec<T, OtherAllocator>>
  Vec<T, OtherAllocator, OtherStorage> clone() const {
    Vec<T, OtherAllocator, OtherStorage> vec(this->length);
    this->data.copyTo(vec.data);
    vec.length = this->length;
    return vec;
  }

  template <class U, class OtherAllocator, class OtherStorage>
  friend class Vec;

  template <class U, class OtherAllocator, class OtherStorage>
  friend class CudaVec;
};

/**
 * @brief A specialized vector class for CUDA-based computations.
 *
 * @tparam T The type of elements stored in the vector.
 * @tparam CudaAllocator The CUDA memory allocator used for the vector's
 * storage. Defaults to `Memory::DefaultCudaAllocator`.
 * @tparam Storage The storage type used for the vector's data. Defaults to
 * `Raw::CudaRawVec<T, CudaAllocator>`.
 */
template <class T, class CudaAllocator = Memory::DefaultCudaAllocator,
          class Storage = Raw::CudaRawVec<T, CudaAllocator>>
class CudaVec : public Vec<T, CudaAllocator, Storage> {
public:
  using Vec<T, CudaAllocator, Storage>::Vec;

  /**
   * @brief Performs element-wise operations between two CudaVec instances and
   * returns a new CudaVec with the result.
   *
   * @param other The other CudaVec instance to perform the operation with.
   * @param op The operation to perform (e.g., addition, subtraction).
   * @return A new CudaVec instance with the result of the operation.
   */
  CudaVec(const CudaVec &other) : Vec<T, CudaAllocator, Storage>(other) {}

  /**
   * @brief Performs element-wise operations between the current CudaVec
   * instance and another CudaVec instance in-place.
   *
   * @param other The other CudaVec instance to perform the operation with.
   * @param type The data type of the elements.
   * @param op The operation to perform (e.g., addition, subtraction).
   */
  CudaVec(CudaVec &&other) : Vec<T, CudaAllocator, Storage>(std::move(other)) {}

  /**
   * @brief Copy assignment operator for CudaVec.
   *
   * @param other The CudaVec object to copy from.
   * @return A reference to the current CudaVec object after the assignment.
   */
  CudaVec<T, CudaAllocator, Storage> &
  operator=(const CudaVec<T, CudaAllocator, Storage> &other) {
    return static_cast<CudaVec<T, CudaAllocator, Storage> &>(
        Vec<T, CudaAllocator, Storage>::operator=(other));
  }

  /**
   * @brief Move assignment operator for CudaVec.
   *
   * @param other The CudaVec object to move from.
   * @return A reference to the current CudaVec object after the assignment.
   */
  CudaVec<T, CudaAllocator, Storage> &
  operator=(CudaVec<T, CudaAllocator, Storage> &&other) {
    return static_cast<CudaVec<T, CudaAllocator, Storage> &>(
        Vec<T, CudaAllocator, Storage>::operator=(std::move(other)));
  }

  /**
   * @brief Performs an element-wise operation on two CudaVec objects.
   *
   * This function takes two CudaVec objects and an operation type, and returns
   * a new CudaVec object containing the result of applying the operation to the
   * corresponding elements of the input vectors.
   *
   * The length of the result vector is set to the minimum of the lengths of the
   * two input vectors.
   *
   * @param other The second CudaVec object to perform the operation with.
   * @param op The operation to perform on the elements of the input vectors.
   * @return A new CudaVec object containing the result of the operation.
   */
  CudaVec<T, CudaAllocator, Storage>
  calculate(const CudaVec<T, CudaAllocator, Storage> &other,
            Types::Operator op) const {
    std::size_t length = std::min(this->len(), other.len());
    CudaVec<T, CudaAllocator, Storage> result(length);

    Utilities::deviceArrayCalculate(result.data.ptr(), this->data.ptr(),
                                    other.data.ptr(), length,
                                    Types::ToDataType<T>::value, op);
    result.length = length;
    return result;
  }

  /**
   * @brief Performs an in-place element-wise operation on the current CudaVec.
   *
   * This function takes another CudaVec object, an operation type, and a data
   * type, and applies the operation to the corresponding elements of the
   * current CudaVec and the other CudaVec. The operation is performed in-place,
   * modifying the current CudaVec.
   *
   * The operation is performed on the minimum length of the two input vectors.
   *
   * @param other The second CudaVec object to perform the operation with.
   * @param type The data type of the elements in the input vectors.
   * @param op The operation to perform on the elements of the input vectors.
   */
  void calculateInplace(const CudaVec<T, CudaAllocator, Storage> &other,
                        Types::DataType type, Types::Operator op) {
    Utilities::deviceArrayCalculate(
        this->data.ptr(), this->data.ptr(), other.data.ptr(),
        std::min(this->length, other.length), type, op);
  }
};
} // namespace ChiaVec
#endif