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

enum class Operator {
  Pls,
  Sub,
  Mul,
  Div,
};

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
