use chiavec_sys::*;
use std::ops::{Index, IndexMut};
use std::ptr::{addr_of, addr_of_mut, null_mut};

macro_rules! declare_vec {
    (u8) => {
        implement_vec!(
            u8,
            Vecu8,
            Vec_uint8_t,
            Vec_uint8_t_init,
            Vec_uint8_t_init_with_values,
            Vec_uint8_t_destroy,
            Vec_uint8_t_len,
            Vec_uint8_t_push,
            Vec_uint8_t_pop,
            Vec_uint8_t_get_const,
            Vec_uint8_t_get
        );
    };
    (u16) => {
        implement_vec!(
            u16,
            Vecu16,
            Vec_uint16_t,
            Vec_uint16_t_init,
            Vec_uint16_t_init_with_values,
            Vec_uint16_t_destroy,
            Vec_uint16_t_len,
            Vec_uint16_t_push,
            Vec_uint16_t_pop,
            Vec_uint16_t_get_const,
            Vec_uint16_t_get
        );
    };
    (u32) => {
        implement_vec!(
            u32,
            Vecu32,
            Vec_uint32_t,
            Vec_uint32_t_init,
            Vec_uint32_t_init_with_values,
            Vec_uint32_t_destroy,
            Vec_uint32_t_len,
            Vec_uint32_t_push,
            Vec_uint32_t_pop,
            Vec_uint32_t_get_const,
            Vec_uint32_t_get
        );
    };
    (u64) => {
        implement_vec!(
            u64,
            Vecu64,
            Vec_uint64_t,
            Vec_uint64_t_init,
            Vec_uint64_t_init_with_values,
            Vec_uint64_t_destroy,
            Vec_uint64_t_len,
            Vec_uint64_t_push,
            Vec_uint64_t_pop,
            Vec_uint64_t_get_const,
            Vec_uint64_t_get
        );
    };
    (i8) => {
        implement_vec!(
            i8,
            Veci8,
            Vec_int8_t,
            Vec_int8_t_init,
            Vec_int8_t_init_with_values,
            Vec_int8_t_destroy,
            Vec_int8_t_len,
            Vec_int8_t_push,
            Vec_int8_t_pop,
            Vec_int8_t_get_const,
            Vec_int8_t_get
        );
    };
    (i16) => {
        implement_vec!(
            i16,
            Veci16,
            Vec_int16_t,
            Vec_int16_t_init,
            Vec_int16_t_init_with_values,
            Vec_int16_t_destroy,
            Vec_int16_t_len,
            Vec_int16_t_push,
            Vec_int16_t_pop,
            Vec_int16_t_get_const,
            Vec_int16_t_get
        );
    };
    (i32) => {
        implement_vec!(
            i32,
            Veci32,
            Vec_int32_t,
            Vec_int32_t_init,
            Vec_int32_t_init_with_values,
            Vec_int32_t_destroy,
            Vec_int32_t_len,
            Vec_int32_t_push,
            Vec_int32_t_pop,
            Vec_int32_t_get_const,
            Vec_int32_t_get
        );
    };
    (i64) => {
        implement_vec!(
            i64,
            Veci64,
            Vec_int64_t,
            Vec_int64_t_init,
            Vec_int64_t_init_with_values,
            Vec_int64_t_destroy,
            Vec_int64_t_len,
            Vec_int64_t_push,
            Vec_int64_t_pop,
            Vec_int64_t_get_const,
            Vec_int64_t_get
        );
    };
    (f32) => {
        implement_vec!(
            f32,
            Vecf32,
            Vec_float,
            Vec_float_init,
            Vec_float_init_with_values,
            Vec_float_destroy,
            Vec_float_len,
            Vec_float_push,
            Vec_float_pop,
            Vec_float_get_const,
            Vec_float_get
        );
    };
    (f64) => {
        implement_vec!(
            f64,
            Vecf64,
            Vec_double,
            Vec_double_init,
            Vec_double_init_with_values,
            Vec_double_destroy,
            Vec_double_len,
            Vec_double_push,
            Vec_double_pop,
            Vec_double_get_const,
            Vec_double_get
        );
    };
}

macro_rules! implement_vec {
    ($element_type: ident, $name: ident, $internal: ident, $init: ident, $init_with_values: ident, $destroy: ident, $get_len: ident, $push: ident, $pop: ident, $get_const: ident, $get_mut: ident) => {
        pub struct $name {
            data: $internal,
        }

        impl $name {
            fn new() -> Self {
                let mut data = $internal { _ptr: null_mut() };
                unsafe { $init(addr_of_mut!(data)) };
                Self { data }
            }

            fn from_array(values: &[$element_type]) -> Self {
                let mut data = $internal { _ptr: null_mut() };
                unsafe { $init_with_values(addr_of_mut!(data), values.as_ptr(), values.len(), 1) };
                Self { data }
            }

            fn len(&self) -> usize {
                unsafe { $get_len(addr_of!(self.data)) as usize }
            }

            fn is_empty(&self) -> bool {
                self.len() == 0
            }

            fn push(&mut self, element: $element_type) {
                unsafe { $push(addr_of_mut!(self.data), addr_of!(element), 1) };
            }

            fn pop(&mut self) -> Option<$element_type> {
                if self.len() == 0 {
                    return None;
                }

                let mut element = $element_type::default();
                unsafe { $pop(addr_of_mut!(self.data), addr_of_mut!(element)) };
                Some(element)
            }
        }

        impl Index<usize> for $name {
            type Output = $element_type;

            fn index(&self, index: usize) -> &Self::Output {
                assert!(index < self.len());
                unsafe { &*$get_const(addr_of!(self.data), index) }
            }
        }

        impl IndexMut<usize> for $name {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                assert!(index < self.len());
                unsafe { &mut *$get_mut(addr_of_mut!(self.data), index) }
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                unsafe { $destroy(addr_of_mut!(self.data)) };
            }
        }
    };
}

declare_vec!(u8);
declare_vec!(u16);
declare_vec!(u32);
declare_vec!(u64);
declare_vec!(i8);
declare_vec!(i16);
declare_vec!(i32);
declare_vec!(i64);
declare_vec!(f32);
declare_vec!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vecs() {
        let mut vec = Veci32::new();

        assert!(vec.is_empty());

        for i in 1..=10 {
            vec.push(i);
            assert!(vec.len() == i as usize);
            assert!(vec[i as usize - 1] == i);
        }
        for i in (1..=10).rev() {
            assert!(vec.pop().map_or(false, |e| e == i));
        }

        assert!(vec.is_empty());

        let vec = Veci32::from_array(&[1, 2, 3, 4, 5]);
        assert!(!vec.is_empty());
        assert!(vec.len() == 5);

        for i in 1..=5 {
            assert!(vec[i - 1] == i as i32);
        }
    }
}
