use chiavec_sys::*;
use std::ptr::{addr_of, addr_of_mut, null_mut};

macro_rules! declare_vec {
    (u8) => {
        implement_vec!(
            u8,
            Vecu8,
            Vec_uint8_t,
            Vec_uint8_t_init,
            Vec_uint8_t_destroy,
            Vec_uint8_t_len,
            Vec_uint8_t_push,
            Vec_uint8_t_pop
        );
    };
    (u16) => {
        implement_vec!(
            u16,
            Vecu16,
            Vec_uint16_t,
            Vec_uint16_t_init,
            Vec_uint16_t_destroy,
            Vec_uint16_t_len,
            Vec_uint16_t_push,
            Vec_uint16_t_pop
        );
    };
    (u32) => {
        implement_vec!(
            u32,
            Vecu32,
            Vec_uint32_t,
            Vec_uint32_t_init,
            Vec_uint32_t_destroy,
            Vec_uint32_t_len,
            Vec_uint32_t_push,
            Vec_uint32_t_pop
        );
    };
    (u64) => {
        implement_vec!(
            u64,
            Vecu64,
            Vec_uint64_t,
            Vec_uint64_t_init,
            Vec_uint64_t_destroy,
            Vec_uint64_t_len,
            Vec_uint64_t_push,
            Vec_uint64_t_pop
        );
    };
    (i8) => {
        implement_vec!(
            i8,
            Veci8,
            Vec_int8_t,
            Vec_int8_t_init,
            Vec_int8_t_destroy,
            Vec_int8_t_len,
            Vec_int8_t_push,
            Vec_int8_t_pop
        );
    };
    (i16) => {
        implement_vec!(
            i16,
            Veci16,
            Vec_int16_t,
            Vec_int16_t_init,
            Vec_int16_t_destroy,
            Vec_int16_t_len,
            Vec_int16_t_push,
            Vec_int16_t_pop
        );
    };
    (i32) => {
        implement_vec!(
            i32,
            Veci32,
            Vec_int32_t,
            Vec_int32_t_init,
            Vec_int32_t_destroy,
            Vec_int32_t_len,
            Vec_int32_t_push,
            Vec_int32_t_pop
        );
    };
    (i64) => {
        implement_vec!(
            i64,
            Veci64,
            Vec_int64_t,
            Vec_int64_t_init,
            Vec_int64_t_destroy,
            Vec_int64_t_len,
            Vec_int64_t_push,
            Vec_int64_t_pop
        );
    };
    (f32) => {
        implement_vec!(
            f32,
            Vecf32,
            Vec_float,
            Vec_float_init,
            Vec_float_destroy,
            Vec_float_len,
            Vec_float_push,
            Vec_float_pop
        );
    };
    (f64) => {
        implement_vec!(
            f64,
            Vecf64,
            Vec_double,
            Vec_double_init,
            Vec_double_destroy,
            Vec_double_len,
            Vec_double_push,
            Vec_double_pop
        );
    };
}

macro_rules! implement_vec {
    ($element_type: ident, $name: ident, $internal: ident, $init: ident, $destroy: ident, $get_len: ident, $push: ident, $pop: ident) => {
        pub struct $name {
            data: $internal,
        }

        impl $name {
            fn new() -> Self {
                let mut data = $internal { _ptr: null_mut() };
                unsafe { $init(addr_of_mut!(data)) };
                Self { data }
            }

            fn len(&self) -> usize {
                unsafe { $get_len(addr_of!(self.data)) as usize }
            }

            fn push(&mut self, element: $element_type, on_host: bool) {
                unsafe { $push(addr_of_mut!(self.data), addr_of!(element), on_host as u8) };
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
    fn test_vec_new() {
        let _vec = Vecu8::new();
        let _vec = Vecu16::new();
        let _vec = Vecu32::new();
        let _vec = Vecu64::new();
        let _vec = Veci8::new();
        let _vec = Veci16::new();
        let _vec = Veci32::new();
        let _vec = Veci64::new();
        let _vec = Vecf32::new();
        let _vec = Vecf64::new();
    }

    #[test]
    fn test_vec_push() {
        let mut vec = Vecu8::new();
        for i in 1..=10 {
            vec.push(i, true);
            assert!(vec.len() == i as usize);
        }
        for i in (1..=10).rev() {
            assert!(vec.pop().map_or(false, |e| e == i));
        }
    }
}
