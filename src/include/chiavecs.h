#ifndef __CHIAVEC_CHIAVECS_H__
#define __CHIAVEC_CHIAVECS_H__

#include <stddef.h>
#include <stdint.h>

#define CUDAVEC_DECLEAR_CUDAVEC_OPERATION(type, operation)                     \
  void CudaVec_##type##_##operation(struct CudaVec_##type *result,             \
                                    const struct CudaVec_##type *op1,          \
                                    const struct CudaVec_##type *op2);

#define CUDAVEC_DECLARE_VECS(type)                                             \
  struct Vec_##type {                                                          \
    void *_ptr;                                                                \
  };                                                                           \
  struct CudaVec_##type {                                                      \
    void *_ptr;                                                                \
  };                                                                           \
  void Vec_##type##_init(struct Vec_##type *vec);                              \
  void CudaVec_##type##_init(struct CudaVec_##type *vec);                      \
  void Vec_##type##_init_with_values(struct Vec_##type *vec,                   \
                                     const type *values, size_t size,          \
                                     uint8_t on_host);                         \
  void CudaVec_##type##_init_with_values(struct CudaVec_##type *vec,           \
                                         const type *values, size_t size,      \
                                         uint8_t on_host);                     \
  void Vec_##type##_copy(struct Vec_##type *dst,                               \
                         const struct Vec_##type *src);                        \
  void CudaVec_##type##_copy(struct CudaVec_##type *dst,                       \
                             const struct CudaVec_##type *src);                \
  void Vec_##type##_copy_from_device(struct Vec_##type *dst,                   \
                                     const struct CudaVec_##type *src);        \
  void CudaVec_##type##_copy_from_host(struct CudaVec_##type *dst,             \
                                       const struct Vec_##type *src);          \
  void Vec_##type##_destroy(struct Vec_##type *vec);                           \
  void CudaVec_##type##_destroy(struct CudaVec_##type *vec);                   \
  size_t Vec_##type##_len(const struct Vec_##type *vec);                       \
  size_t CudaVec_##type##_len(const struct CudaVec_##type *vec);               \
  type *Vec_##type##_get(struct Vec_##type *vec, size_t index);                \
  type *CudaVec_##type##_get(struct CudaVec_##type *vec, size_t index);        \
  const type *Vec_##type##_get_const(const struct Vec_##type *vec,             \
                                     size_t index);                            \
  const type *CudaVec_##type##_get_const(const struct CudaVec_##type *vec,     \
                                         size_t index);                        \
  void Vec_##type##_push(struct Vec_##type *vec, const type *value,            \
                         uint8_t on_host);                                     \
  void CudaVec_##type##_push(struct CudaVec_##type *vec, const type *value,    \
                             uint8_t on_host);                                 \
  void Vec_##type##_pop(struct Vec_##type *vec, type *dst);                    \
  void CudaVec_##type##_pop(struct CudaVec_##type *vec, type *dst);            \
  CUDAVEC_DECLEAR_CUDAVEC_OPERATION(type, add);                                \
  CUDAVEC_DECLEAR_CUDAVEC_OPERATION(type, sub);                                \
  CUDAVEC_DECLEAR_CUDAVEC_OPERATION(type, div);                                \
  CUDAVEC_DECLEAR_CUDAVEC_OPERATION(type, mul);

#ifdef __cplusplus
extern "C" {
#endif
CUDAVEC_DECLARE_VECS(int8_t);
CUDAVEC_DECLARE_VECS(int16_t);
CUDAVEC_DECLARE_VECS(int32_t);
CUDAVEC_DECLARE_VECS(int64_t);

CUDAVEC_DECLARE_VECS(uint8_t);
CUDAVEC_DECLARE_VECS(uint16_t);
CUDAVEC_DECLARE_VECS(uint32_t);
CUDAVEC_DECLARE_VECS(uint64_t);

CUDAVEC_DECLARE_VECS(float);
CUDAVEC_DECLARE_VECS(double);
#ifdef __cplusplus
}
#endif

#endif