#include "vecs.hpp"

#include <iostream>

template <class T>
void printVec(const ChiaVec::Vec<T> &v) {
  std::cout << "[";
  for (std::size_t i = 0; i < v.len(); i++) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << v[i];
  }
  std::cout << "]\n";
}

int main(void) {
  constexpr std::size_t length = 4096;
  int nums[length]; // Contains numbers from 0 to 4095.

  // Populate the array.
  for (int i = 0; i < length; i++) {
    nums[i] = i;
  }

  ChiaVec::CudaVec<int> v1(nums, length, true), v2;

  // v2 has the same elements as v1.
  for (int i = 0; i < length; i++) {
    v2.push(i, true);
  }

  // v3[i] = v1[i] + v2[i] for i in [0..4095].
  ChiaVec::CudaVec<int> v3 = v1.calculate(v2, ChiaVec::Types::Operator::Pls);

  // Print the elements in v1, v2 and v3.
  printVec(v1.clone());
  printVec(v2.clone());
  printVec(v3.clone());
  return 0;
}