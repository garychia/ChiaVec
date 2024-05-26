#include "utilities.hpp"

#define IMPLEMENT_CALCULATE_KERNEL(func_name, operator)                               \
    template <class T>                                                                \
    __global__ void func_name(T *dst, const T *op1, const T *op2, std::size_t length) \
    {                                                                                 \
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;                       \
        if (i < length)                                                               \
        {                                                                             \
            dst[i] = op1[i] operator op2[i];                                          \
        }                                                                             \
    }

IMPLEMENT_CALCULATE_KERNEL(arrayAdd, +);
IMPLEMENT_CALCULATE_KERNEL(arraySub, -);
IMPLEMENT_CALCULATE_KERNEL(arrayMul, *);
IMPLEMENT_CALCULATE_KERNEL(arrayDiv, /);

namespace ChiaVec
{
    namespace Utilities
    {
        template <class T>
        void calculate(T *dst, const T *op1, const T *op2, std::size_t length, Types::Operator op)
        {
            constexpr size_t threads = 512;
            std::size_t blocks = (length + threads - 1) / threads;
            switch (op)
            {
            case Types::Operator::Pls:
                arrayAdd<<<blocks, threads>>>(dst, op1, op2, length);
                break;
            case Types::Operator::Sub:
                arraySub<<<blocks, threads>>>(dst, op1, op2, length);
                break;
            case Types::Operator::Mul:
                arrayMul<<<blocks, threads>>>(dst, op1, op2, length);
                break;
            case Types::Operator::Div:
                arrayDiv<<<blocks, threads>>>(dst, op1, op2, length);
                break;
            }
        }

        void deviceArrayCalculate(void *dst, const void *operand1, const void *operand2, std::size_t length, Types::DataType type, Types::Operator op)
        {
            switch (type)
            {
            case Types::DataType::i8:
                calculate(static_cast<int8_t *>(dst), static_cast<const int8_t *>(operand1), static_cast<const int8_t *>(operand2), length, op);
                break;
            case Types::DataType::u8:
                calculate(static_cast<uint8_t *>(dst), static_cast<const uint8_t *>(operand1), static_cast<const uint8_t *>(operand2), length, op);
                break;
            case Types::DataType::i16:
                calculate(static_cast<int16_t *>(dst), static_cast<const int16_t *>(operand1), static_cast<const int16_t *>(operand2), length, op);
                break;
            case Types::DataType::u16:
                calculate(static_cast<uint16_t *>(dst), static_cast<const uint16_t *>(operand1), static_cast<const uint16_t *>(operand2), length, op);
                break;
            case Types::DataType::i32:
                calculate(static_cast<int32_t *>(dst), static_cast<const int32_t *>(operand1), static_cast<const int32_t *>(operand2), length, op);
                break;
            case Types::DataType::u32:
                calculate(static_cast<uint32_t *>(dst), static_cast<const uint32_t *>(operand1), static_cast<const uint32_t *>(operand2), length, op);
                break;
            case Types::DataType::i64:
                calculate(static_cast<int64_t *>(dst), static_cast<const int64_t *>(operand1), static_cast<const int64_t *>(operand2), length, op);
                break;
            case Types::DataType::u64:
                calculate(static_cast<uint64_t *>(dst), static_cast<const uint64_t *>(operand1), static_cast<const uint64_t *>(operand2), length, op);
                break;
            case Types::DataType::Float:
                calculate(static_cast<float *>(dst), static_cast<const float *>(operand1), static_cast<const float *>(operand2), length, op);
                break;
            default:
                calculate(static_cast<double *>(dst), static_cast<const double *>(operand1), static_cast<const double *>(operand2), length, op);
                break;
            }
        }
    } // namespace Utilities
} // namespace ChiaVec