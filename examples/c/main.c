#include "chiavecs.h"

#include <stdio.h>

void printVec(struct Vec_int32_t *v)
{
    size_t length = Vec_int32_t_len(v);
    size_t i;

    printf("[");
    for (i = 0; i < length; i++)
    {
        if (i != 0)
        {
            printf(", ");
        }
        printf("%d", *Vec_int32_t_get(v, i));
    }
    printf("]\n");
}

int main(void)
{
    struct CudaVec_int32_t v1, v2, v3;
    struct Vec_int32_t v;
    const size_t length = 4096;
    int32_t i;
    int32_t nums[length];

    for (i = 0; i < length; i++)
    {
        nums[i] = i;
    }

    CudaVec_int32_t_init_with_values(&v1, nums, length, 1);
    CudaVec_int32_t_init(&v2);
    CudaVec_int32_t_init(&v3);
    Vec_int32_t_init(&v);

    for (i = 0; i < length; i++)
    {
        CudaVec_int32_t_push(&v2, &i, 1);
    }

    CudaVec_int32_t_add(&v3, &v1, &v2);

    Vec_int32_t_copy_from_device(&v, &v1);
    printVec(&v);
    Vec_int32_t_copy_from_device(&v, &v2);
    printVec(&v);
    Vec_int32_t_copy_from_device(&v, &v3);
    printVec(&v);

    CudaVec_int32_t_destroy(&v1);
    CudaVec_int32_t_destroy(&v2);
    CudaVec_int32_t_destroy(&v3);
    Vec_int32_t_destroy(&v);
    return 0;
}
