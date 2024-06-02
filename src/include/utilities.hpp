#ifndef __CHIAVEC_UTILITIES_HPP__
#define __CHIAVEC_UTILITIES_HPP__

#include "types.hpp"

namespace ChiaVec {
namespace Utilities {
/**
 * @brief Performs an element-wise arithmetic operation on two device arrays.
 *
 * This function takes two input device arrays, `operand1` and `operand2`, and
 * performs the specified arithmetic operation on their corresponding elements.
 * The result is stored in the `dst` device array.
 *
 * The operation is performed on the first `length` elements of the input
 * arrays. The `type` parameter specifies the data type of the elements in the
 * input arrays, and the `op` parameter specifies the arithmetic operation to
 * perform.
 *
 * @param dst The destination device array to store the result.
 * @param operand1 The first input device array.
 * @param operand2 The second input device array.
 * @param length The number of elements to process.
 * @param type The data type of the elements in the input arrays.
 * @param op The arithmetic operation to perform.
 */
void deviceArrayCalculate(void *dst, const void *operand1, const void *operand2,
                          std::size_t length, Types::DataType type,
                          Types::Operator op);
} // namespace Utilities
} // namespace ChiaVec

#endif