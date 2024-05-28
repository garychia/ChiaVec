#ifndef __CHIAVEC_UTILITIES_HPP__
#define __CHIAVEC_UTILITIES_HPP__

#include "types.hpp"

namespace ChiaVec {
namespace Utilities {
void deviceArrayCalculate(void *dst, const void *operand1, const void *operand2,
                          std::size_t length, Types::DataType type,
                          Types::Operator op);
} // namespace Utilities
} // namespace ChiaVec

#endif