#include "chiavecs.h"

#include <stdio.h>

// Print the content of the Vec.
void printVec(struct Vec_int32_t *v) {
  size_t length = Vec_int32_t_len(v);
  size_t i;

  printf("[");
  for (i = 0; i < length; i++) {
    if (i != 0) {
      printf(", ");
    }
    printf("%d", *Vec_int32_t_get(v, i));
  }
  printf("]\n");
}

int main(void) {
  // Allocate memory on the device.
  struct CudaVec_int32_t v1, v2, v3;
  struct Vec_int32_t v; // the result needs to be copied to host.
  // 4096 elements in each vector.
  const size_t length = 4096;
  int32_t i;
  int32_t nums[length];

  // Populate `nums` with numbers from 0 to `length` - 1.
  for (i = 0; i < length; i++) {
    nums[i] = i;
  }

  // Populate `v1` with the values of `nums`.
  CudaVec_int32_t_init_with_values(&v1, nums, length, 1);
  // Vectors must be initialized before use.
  CudaVec_int32_t_init(&v2);
  CudaVec_int32_t_init(&v3);
  Vec_int32_t_init(&v);

  // We can also use `push` to populate a vector.
  for (i = 0; i < length; i++) {
    CudaVec_int32_t_push(&v2, &i, 1);
  }

  // Perform the addition of `v1` and `v2` and store the result in `v3`.
  CudaVec_int32_t_add(&v3, &v1, &v2);

  // Copy a vector from the device to the host to output the result.
  Vec_int32_t_copy_from_device(&v, &v1);
  printVec(&v);
  Vec_int32_t_copy_from_device(&v, &v2);
  printVec(&v);
  Vec_int32_t_copy_from_device(&v, &v3);
  printVec(&v);

  // Destroy the vectors.
  CudaVec_int32_t_destroy(&v1);
  CudaVec_int32_t_destroy(&v2);
  CudaVec_int32_t_destroy(&v3);
  Vec_int32_t_destroy(&v);
  return 0;
}
