#ifndef __CHIAVEC_TYPES_HPP__
#define __CHIAVEC_TYPES_HPP__

#include <cstdint>

#define __CHIAVEC_TO_DATA_TYPE(type, target)                                   \
  template <>                                                                  \
  struct ToDataType<type> {                                                    \
    static constexpr DataType value = target;                                  \
  };

#define __CHIAVEC_TO_CONCRETE_TYPE(type, target)                               \
  template <>                                                                  \
  struct ToConcreteType<type> {                                                \
    typedef target value;                                                      \
  };

#define __CHIAVEC_CAST_TO(type, target)                                        \
  template <>                                                                  \
  struct CastTo<type> {                                                        \
    template <class T>                                                         \
    target *operator()(T *ptr) {                                               \
      return static_cast<target *>(ptr);                                       \
    }                                                                          \
                                                                               \
    template <class T>                                                         \
    target *operator()(const T *ptr) {                                         \
      return static_cast<const target *>(ptr);                                 \
    }                                                                          \
  };

namespace ChiaVec {
namespace Types {
/**
 * @brief Enumeration of supported data types.
 *
 * The supported data types are:
 *
 * - `i8`: Signed 8-bit integer
 * - `u8`: Unsigned 8-bit integer
 * - `i16`: Signed 16-bit integer
 * - `u16`: Unsigned 16-bit integer
 * - `i32`: Signed 32-bit integer
 * - `u32`: Unsigned 32-bit integer
 * - `i64`: Signed 64-bit integer
 * - `u64`: Unsigned 64-bit integer
 * - `Float`: 32-bit floating-point number
 * - `Double`: 64-bit floating-point number
 *
 * These data types can be used to represent various types of data in the
 * application, such as scalar values, vector components, and matrix elements.
 */
enum class DataType {
  i8,
  u8,
  i16,
  u16,
  i32,
  u32,
  i64,
  u64,
  Float,
  Double,
};

/**
 * @brief Enumeration of supported arithmetic operators.
 *
 * The supported operators are:
 *
 * - `Pls`: Addition
 * - `Sub`: Subtraction
 * - `Mul`: Multiplication
 * - `Div`: Division
 *
 * These operators can be used to perform element-wise arithmetic operations on
 * vectors.
 */
enum class Operator {
  Pls,
  Sub,
  Mul,
  Div,
};

/**
 * @brief Converts a C++ type to a supported `DataType`.
 *
 * This struct template is used to convert a C++ type to the corresponding
 * `DataType` enumeration value. It provides a static member `value` that
 * holds the `DataType` value for the given type.
 *
 * The specializations of this struct template cover the basic integral and
 * floating-point types supported by the application.
 *
 * @tparam T The C++ type to be converted to a `DataType`.
 */
template <class T>
struct ToDataType;

__CHIAVEC_TO_DATA_TYPE(int8_t, DataType::i8);
__CHIAVEC_TO_DATA_TYPE(uint8_t, DataType::u8);
__CHIAVEC_TO_DATA_TYPE(int16_t, DataType::i16);
__CHIAVEC_TO_DATA_TYPE(uint16_t, DataType::u16);
__CHIAVEC_TO_DATA_TYPE(int32_t, DataType::i32);
__CHIAVEC_TO_DATA_TYPE(uint32_t, DataType::u32);
__CHIAVEC_TO_DATA_TYPE(int64_t, DataType::i64);
__CHIAVEC_TO_DATA_TYPE(uint64_t, DataType::u64);
__CHIAVEC_TO_DATA_TYPE(float, DataType::Float);
__CHIAVEC_TO_DATA_TYPE(double, DataType::Double);

/**
 * @brief Provides a concrete type for a given `DataType`.
 *
 * This struct template maps a `DataType` enumeration value to a corresponding
 * C++ type. It provides a `value` member alias that represents the concrete
 * type for the given `DataType`.
 *
 * @tparam Type The `DataType` to be mapped to a concrete C++ type.
 */
template <DataType Type>
struct ToConcreteType;

__CHIAVEC_TO_CONCRETE_TYPE(DataType::i8, int8_t);
__CHIAVEC_TO_CONCRETE_TYPE(DataType::u8, uint8_t);
__CHIAVEC_TO_CONCRETE_TYPE(DataType::i16, int16_t);
__CHIAVEC_TO_CONCRETE_TYPE(DataType::u16, uint16_t);
__CHIAVEC_TO_CONCRETE_TYPE(DataType::i32, int32_t);
__CHIAVEC_TO_CONCRETE_TYPE(DataType::u32, uint32_t);
__CHIAVEC_TO_CONCRETE_TYPE(DataType::i64, int64_t);
__CHIAVEC_TO_CONCRETE_TYPE(DataType::u64, uint64_t);
__CHIAVEC_TO_CONCRETE_TYPE(DataType::Float, float);
__CHIAVEC_TO_CONCRETE_TYPE(DataType::Double, double);

/**
 * @brief Provides a casting mechanism for a given `DataType`.
 *
 * This struct template defines a `cast` static member function that can be used
 * to cast a value of the corresponding concrete type to another concrete type.
 *
 * The specializations of this struct template cover all the `DataType` values,
 * providing the necessary casting logic for each data type.
 *
 * @tparam Type The `DataType` to provide the casting mechanism for.
 */
template <DataType Type>
struct CastTo;

__CHIAVEC_CAST_TO(DataType::i8, int8_t);
__CHIAVEC_CAST_TO(DataType::u8, uint8_t);
__CHIAVEC_CAST_TO(DataType::i16, int16_t);
__CHIAVEC_CAST_TO(DataType::u16, uint16_t);
__CHIAVEC_CAST_TO(DataType::i32, int32_t);
__CHIAVEC_CAST_TO(DataType::u32, uint32_t);
__CHIAVEC_CAST_TO(DataType::i64, int64_t);
__CHIAVEC_CAST_TO(DataType::u64, uint64_t);
__CHIAVEC_CAST_TO(DataType::Float, float);
__CHIAVEC_CAST_TO(DataType::Double, double);
} // namespace Types
} // namespace ChiaVec

#endif
