#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::{addr_of, addr_of_mut, null_mut};

    #[test]
    fn vec_init() {
        unsafe {
            // Test "Vec_int32_t"
            let mut v1 = Vec_int32_t { _ptr: null_mut() };
            let mut v2 = Vec_int32_t { _ptr: null_mut() };
            let mut values = [0_i32; 1024];

            for i in 0..1024 {
                values[i] = i as i32;
            }

            Vec_int32_t_init(addr_of_mut!(v1));
            Vec_int32_t_init_with_values(addr_of_mut!(v2), values.as_ptr(), values.len(), 1);
            Vec_int32_t_destroy(addr_of_mut!(v1));
            Vec_int32_t_destroy(addr_of_mut!(v2));

            // Test "CudaVec_int32_t"
            let mut v1 = CudaVec_int32_t { _ptr: null_mut() };
            let mut v2 = CudaVec_int32_t { _ptr: null_mut() };
            let mut values = [0_i32; 1024];

            for i in 0..1024 {
                values[i] = i as i32;
            }

            CudaVec_int32_t_init(addr_of_mut!(v1));
            CudaVec_int32_t_init_with_values(addr_of_mut!(v2), values.as_ptr(), values.len(), 1);
            CudaVec_int32_t_destroy(addr_of_mut!(v1));
            CudaVec_int32_t_destroy(addr_of_mut!(v2));
        }
    }

    #[test]
    fn vec_get() {
        unsafe {
            let mut v1 = Vec_int32_t { _ptr: null_mut() };
            let mut v2 = Vec_int32_t { _ptr: null_mut() };
            let mut cuda_v1 = CudaVec_int32_t { _ptr: null_mut() };
            let mut cuda_v2 = CudaVec_int32_t { _ptr: null_mut() };
            const VALUE_LEN: usize = 1024;
            let mut values = [0_i32; VALUE_LEN];

            for i in 0..VALUE_LEN {
                values[i] = i as i32;
            }

            Vec_int32_t_init(addr_of_mut!(v1));
            Vec_int32_t_init_with_values(addr_of_mut!(v2), values.as_ptr(), values.len(), 1);

            CudaVec_int32_t_init(addr_of_mut!(cuda_v1));
            CudaVec_int32_t_init_with_values(
                addr_of_mut!(cuda_v2),
                values.as_ptr(),
                values.len(),
                1,
            );

            let value_from_v1 = Vec_int32_t_get(addr_of_mut!(v1), 0);
            assert!(value_from_v1.is_null());
            let value_from_v1 = Vec_int32_t_get_const(addr_of!(v1), 0);
            assert!(value_from_v1.is_null());

            let value_from_v1 = CudaVec_int32_t_get(addr_of_mut!(cuda_v1), 0);
            assert!(value_from_v1.is_null());
            let value_from_v1 = CudaVec_int32_t_get_const(addr_of!(cuda_v1), 0);
            assert!(value_from_v1.is_null());

            for i in 0..VALUE_LEN {
                let value_from_v2 = Vec_int32_t_get(addr_of_mut!(v2), i);
                assert!(!value_from_v2.is_null());
                assert_eq!(*value_from_v2, i as i32);
                let value_from_v2 = Vec_int32_t_get_const(addr_of!(v2), i);
                assert!(!value_from_v2.is_null());
                assert_eq!(*value_from_v2, i as i32);

                let value_from_v2 = CudaVec_int32_t_get(addr_of_mut!(cuda_v2), i);
                assert!(value_from_v2.is_null());
                let value_from_v2 = CudaVec_int32_t_get_const(addr_of!(cuda_v2), i);
                assert!(value_from_v2.is_null());
            }

            let value_from_v2 = Vec_int32_t_get(addr_of_mut!(v2), VALUE_LEN);
            assert!(value_from_v2.is_null());
            let value_from_v2 = Vec_int32_t_get_const(addr_of!(v2), VALUE_LEN);
            assert!(value_from_v2.is_null());

            Vec_int32_t_destroy(addr_of_mut!(v1));
            Vec_int32_t_destroy(addr_of_mut!(v2));

            CudaVec_int32_t_destroy(addr_of_mut!(cuda_v1));
            CudaVec_int32_t_destroy(addr_of_mut!(cuda_v2));
        }
    }
}
