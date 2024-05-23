#include "chiavecs.h"
#include "vecs.hpp"

#define CHIAVEC_KERNEL_IMPLEMENTATION(func_name, operator)                            \
    template <class T>                                                                \
    __global__ void func_name(T *dst, const T *op1, const T *op2, std::size_t length) \
    {                                                                                 \
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;                       \
        if (i < length)                                                               \
        {                                                                             \
            dst[i] = op1[i] operator op2[i];                                          \
        }                                                                             \
    }

#define CHIAVEC_CUDAVEC_OPERATION_IMPLEMENTATION(type, operation, kernel)                                                        \
    void CudaVec_##type##_##operation(CudaVec_##type *result, const CudaVec_##type *op1, const CudaVec_##type *op2, size_t size) \
    {                                                                                                                            \
        ChiaVec::CudaVec<type> &resultVec = *static_cast<ChiaVec::CudaVec<type> *>(result->_ptr);                                \
        const ChiaVec::CudaVec<type> &v1 = *static_cast<const ChiaVec::CudaVec<type> *>(op1->_ptr);                              \
        const ChiaVec::CudaVec<type> &v2 = *static_cast<const ChiaVec::CudaVec<type> *>(op2->_ptr);                              \
        auto f = [](type *dst, const type *op1, const type *op2, size_t length) {                                                \
            constexpr size_t threads = 512;                                                                                      \
            std::size_t blocks = (length + threads - 1) / threads;                                                               \
            kernel<<<blocks, threads>>>(dst, op1, op2, length);                                                                  \
        };                                                                                                                       \
        resultVec = v1.calculate(v2, f);                                                                                         \
    }

#define CUDAVEC_VECS_IMPLEMENTATION(type)                                                                         \
    void Vec_##type##_init(Vec_##type *vec)                                                                       \
    {                                                                                                             \
        vec->_ptr = new ChiaVec::Vec<type>();                                                                     \
    }                                                                                                             \
    void CudaVec_##type##_init(CudaVec_##type *vec)                                                               \
    {                                                                                                             \
        vec->_ptr = new ChiaVec::CudaVec<type>();                                                                 \
    }                                                                                                             \
    void Vec_##type##_init_with_values(Vec_##type *vec, const type *values, size_t size, uint8_t on_host)         \
    {                                                                                                             \
        vec->_ptr = new ChiaVec::Vec<type>(values, size, on_host);                                                \
    }                                                                                                             \
    void CudaVec_##type##_init_with_values(CudaVec_##type *vec, const type *values, size_t size, uint8_t on_host) \
    {                                                                                                             \
        vec->_ptr = new ChiaVec::CudaVec<type>(values, size, on_host);                                            \
    }                                                                                                             \
    void Vec_##type##_copy(Vec_##type *dst, const Vec_##type *src)                                                \
    {                                                                                                             \
        ChiaVec::Vec<type> &dstVec = *static_cast<ChiaVec::Vec<type> *>(dst->_ptr);                               \
        const ChiaVec::Vec<type> &srcVec = *static_cast<const ChiaVec::Vec<type> *>(src->_ptr);                   \
        dstVec = srcVec;                                                                                          \
    }                                                                                                             \
    void CudaVec_##type##_copy(CudaVec_##type *dst, const CudaVec_##type *src)                                    \
    {                                                                                                             \
        ChiaVec::CudaVec<type> &dstVec = *static_cast<ChiaVec::CudaVec<type> *>(dst->_ptr);                       \
        const ChiaVec::CudaVec<type> &srcVec = *static_cast<const ChiaVec::CudaVec<type> *>(src->_ptr);           \
        dstVec = srcVec;                                                                                          \
    }                                                                                                             \
    void Vec_##type##_copy_from_device(struct Vec_##type *dst, const struct CudaVec_##type *src)                  \
    {                                                                                                             \
        ChiaVec::Vec<type> &dstVec = *static_cast<ChiaVec::Vec<type> *>(dst->_ptr);                               \
        const ChiaVec::CudaVec<type> &srcVec = *static_cast<const ChiaVec::CudaVec<type> *>(src->_ptr);           \
        srcVec.copyTo(dstVec);                                                                                    \
    }                                                                                                             \
    void CudaVec_##type##_copy_from_host(struct CudaVec_##type *dst, const struct Vec_##type *src)                \
    {                                                                                                             \
        ChiaVec::CudaVec<type> &dstVec = *static_cast<ChiaVec::CudaVec<type> *>(dst->_ptr);                       \
        const ChiaVec::Vec<type> &srcVec = *static_cast<const ChiaVec::Vec<type> *>(src->_ptr);                   \
        dstVec.copyFrom(srcVec);                                                                                  \
    }                                                                                                             \
    void Vec_##type##_destroy(Vec_##type *vec)                                                                    \
    {                                                                                                             \
        if (vec->_ptr)                                                                                            \
        {                                                                                                         \
            delete static_cast<ChiaVec::Vec<type> *>(vec->_ptr);                                                  \
        }                                                                                                         \
        vec->_ptr = nullptr;                                                                                      \
    }                                                                                                             \
    void CudaVec_##type##_destroy(CudaVec_##type *vec)                                                            \
    {                                                                                                             \
        if (vec->_ptr)                                                                                            \
        {                                                                                                         \
            delete static_cast<ChiaVec::CudaVec<type> *>(vec->_ptr);                                              \
        }                                                                                                         \
        vec->_ptr = nullptr;                                                                                      \
    }                                                                                                             \
    size_t Vec_##type##_len(const struct Vec_##type *vec)                                                         \
    {                                                                                                             \
        ChiaVec::Vec<type> &v = *static_cast<ChiaVec::Vec<type> *>(vec->_ptr);                                    \
        return v.len();                                                                                           \
    }                                                                                                             \
    size_t CudaVec_##type##_len(const struct CudaVec_##type *vec)                                                 \
    {                                                                                                             \
        ChiaVec::CudaVec<type> &v = *static_cast<ChiaVec::CudaVec<type> *>(vec->_ptr);                            \
        return v.len();                                                                                           \
    }                                                                                                             \
    type *Vec_##type##_get(struct Vec_##type *vec, size_t index)                                                  \
    {                                                                                                             \
        ChiaVec::Vec<type> &v = *static_cast<ChiaVec::Vec<type> *>(vec->_ptr);                                    \
        return index < v.len() ? &v[index] : nullptr;                                                             \
    }                                                                                                             \
    type *CudaVec_##type##_get(struct CudaVec_##type *vec, size_t index)                                          \
    {                                                                                                             \
        return nullptr;                                                                                           \
    }                                                                                                             \
    const type *Vec_##type##_get_const(const struct Vec_##type *vec, size_t index)                                \
    {                                                                                                             \
        const ChiaVec::Vec<type> &v = *static_cast<const ChiaVec::Vec<type> *>(vec->_ptr);                        \
        return index < v.len() ? &v[index] : nullptr;                                                             \
    }                                                                                                             \
    const type *CudaVec_##type##_get_const(const struct CudaVec_##type *vec, size_t index)                        \
    {                                                                                                             \
        return nullptr;                                                                                           \
    }                                                                                                             \
    void Vec_##type##_push(struct Vec_##type *vec, const type *value, uint8_t on_host)                            \
    {                                                                                                             \
        ChiaVec::Vec<type> &v = *static_cast<ChiaVec::Vec<type> *>(vec->_ptr);                                    \
        v.push(*value, on_host);                                                                                  \
    }                                                                                                             \
    void CudaVec_##type##_push(struct CudaVec_##type *vec, const type *value, uint8_t on_host)                    \
    {                                                                                                             \
        ChiaVec::CudaVec<type> &v = *static_cast<ChiaVec::CudaVec<type> *>(vec->_ptr);                            \
        v.push(*value, on_host);                                                                                  \
    }                                                                                                             \
    void Vec_##type##_pop(struct Vec_##type *vec, type *dst)                                                      \
    {                                                                                                             \
        ChiaVec::Vec<type> &v = *static_cast<ChiaVec::Vec<type> *>(vec->_ptr);                                    \
        std::optional<type> last_optional = v.pop();                                                              \
        if (last_optional.has_value())                                                                            \
        {                                                                                                         \
            *dst = last_optional.value();                                                                         \
        }                                                                                                         \
    }                                                                                                             \
    void CudaVec_##type##_pop(struct CudaVec_##type *vec, type *dst)                                              \
    {                                                                                                             \
        ChiaVec::CudaVec<type> &v = *static_cast<ChiaVec::CudaVec<type> *>(vec->_ptr);                            \
        std::optional<type> last_optional = v.pop();                                                              \
        if (last_optional.has_value())                                                                            \
        {                                                                                                         \
            *dst = last_optional.value();                                                                         \
        }                                                                                                         \
    }                                                                                                             \
    CHIAVEC_CUDAVEC_OPERATION_IMPLEMENTATION(type, add, vec_add);                                                 \
    CHIAVEC_CUDAVEC_OPERATION_IMPLEMENTATION(type, sub, vec_sub);                                                 \
    CHIAVEC_CUDAVEC_OPERATION_IMPLEMENTATION(type, mul, vec_mul);                                                 \
    CHIAVEC_CUDAVEC_OPERATION_IMPLEMENTATION(type, div, vec_div);

CHIAVEC_KERNEL_IMPLEMENTATION(vec_add, +);
CHIAVEC_KERNEL_IMPLEMENTATION(vec_sub, -);
CHIAVEC_KERNEL_IMPLEMENTATION(vec_mul, *);
CHIAVEC_KERNEL_IMPLEMENTATION(vec_div, /);

extern "C"
{
    CUDAVEC_VECS_IMPLEMENTATION(int32_t);
    CUDAVEC_VECS_IMPLEMENTATION(int64_t);
}
