#include "vecs.hpp"

#include <iostream>

template <class T>
__global__ void vec_add(T *dst, const T *op1, const T *op2, std::size_t length)
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < length)
    {
        dst[i] = op1[i] + op2[i];
    }
}

template <class T>
void printVec(const ChiaVec::Vec<T> &v)
{
    std::cout << "[";
    for (std::size_t i = 0; i < v.len(); i++)
    {
        if (i != 0)
        {
            std::cout << ", ";
        }
        std::cout << v[i];
    }
    std::cout << "]\n";
}

int main(void)
{
    constexpr std::size_t length = 4096;
    int nums[length];

    for (int i = 0; i < length; i++)
    {
        nums[i] = i;
    }

    ChiaVec::CudaVec<int> v1(nums, length, true), v2;

    for (int i = 0; i < length; i++)
    {
        v2.push(i, true);
    }

    ChiaVec::CudaVec<int> v3 = v1.calculate(v2, ChiaVec::Types::Operator::Pls);

    printVec(v1.clone());
    printVec(v2.clone());
    printVec(v3.clone());
    return 0;
}