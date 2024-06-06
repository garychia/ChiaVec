use chiavec_sys::*;
use paste::paste;
use std::ops::{Index, IndexMut};
use std::ptr::{addr_of, addr_of_mut, null_mut};

macro_rules! implement_vec {
    ($type: ident, $c_type: ident) => {
        paste! {
            implement_vec!(
                $type,
                concat!("A container that stores ", stringify!($type), " elements on the CPU."),
                concat!("Creates an empty ", stringify!([<Vec $type>].)),
                concat!("Creates a ", stringify!([<Vec $type>]), " populated with the elements in a given slice."),
                concat!("Returns the number of elements in the ", stringify!([<Vec $type>].)),
                concat!("Returns a bool indicating whether the ", stringify!([<Vec $type>]), " is empty."),
                concat!("Push (append) a new element into the ", stringify!([<Vec $type>].)),
                concat!("Pop (remove) the last element from the ", stringify!([<Vec $type>].)),
                [<Vec $type>],
                [<Vec_ $c_type>],
                [<Vec_ $c_type _init>],
                [<Vec_ $c_type _init_with_values>],
                [<Vec_ $c_type _destroy>],
                [<Vec_ $c_type _len>],
                [<Vec_ $c_type _push>],
                [<Vec_ $c_type _pop>],
                [<Vec_ $c_type _get_const>],
                [<Vec_ $c_type _get>]
            );
            implement_vec!(
                $type,
                concat!("A container that stores ", stringify!($type), " elements on the GPU."),
                concat!("Creates an empty ", stringify!([<CudaVec $type>].)),
                concat!("Creates a ", stringify!([<CudaVec $type>]), " populated with the elements in a given slice."),
                concat!("Returns the number of elements in the ", stringify!([<CudaVec $type>].)),
                concat!("Returns a bool indicating whether the ", stringify!([<CudaVec $type>]), " is empty."),
                concat!("Push (append) a new element to the ", stringify!([<CudaVec $type>].)),
                concat!("Pop (remove) the last element from the ", stringify!([<CudaVec $type>].)),
                [<CudaVec $type>],
                [<CudaVec_ $c_type>],
                [<CudaVec_ $c_type _init>],
                [<CudaVec_ $c_type _init_with_values>],
                [<CudaVec_ $c_type _destroy>],
                [<CudaVec_ $c_type _len>],
                [<CudaVec_ $c_type _push>],
                [<CudaVec_ $c_type _pop>],
                [<CudaVec_ $c_type _get_const>],
                [<CudaVec_ $c_type _get>]
            );
        }
    };
    ($element_type: ident, $struc_doc: expr, $new_doc: expr, $from_slice_doc: expr, $len_doc: expr, $is_empty_doc: expr, $push_doc: expr, $pop_doc: expr, $name: ident, $internal: ident, $init: ident, $init_with_values: ident, $destroy: ident, $get_len: ident, $push: ident, $pop: ident, $get_const: ident, $get_mut: ident) => {
        #[doc = $struc_doc]
        pub struct $name {
            data: $internal,
        }

        impl $name {
            #[doc = $new_doc]
            pub fn new() -> Self {
                let mut data = $internal { _ptr: null_mut() };
                unsafe { $init(addr_of_mut!(data)) };
                Self { data }
            }

            #[doc = $from_slice_doc]
            pub fn from_slice(values: &[$element_type]) -> Self {
                let mut data = $internal { _ptr: null_mut() };
                unsafe { $init_with_values(addr_of_mut!(data), values.as_ptr(), values.len(), 1) };
                Self { data }
            }

            #[doc = $len_doc]
            pub fn len(&self) -> usize {
                unsafe { $get_len(addr_of!(self.data)) as usize }
            }

            #[doc = $is_empty_doc]
            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }

            #[doc = $push_doc]
            pub fn push(&mut self, element: $element_type) {
                unsafe { $push(addr_of_mut!(self.data), addr_of!(element), 1) };
            }

            #[doc = $pop_doc]
            pub fn pop(&mut self) -> Option<$element_type> {
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

implement_vec!(u8, uint8_t);
implement_vec!(u16, uint16_t);
implement_vec!(u32, uint32_t);
implement_vec!(u64, uint64_t);
implement_vec!(i8, int8_t);
implement_vec!(i16, int16_t);
implement_vec!(i32, int32_t);
implement_vec!(i64, int64_t);
implement_vec!(f32, float);
implement_vec!(f64, double);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vecs() {
        let mut vec = Veci32::new();
        let mut cuda_vec = CudaVeci32::new();

        assert!(vec.is_empty());
        assert!(cuda_vec.is_empty());

        for i in 1..=10 {
            vec.push(i);
            assert!(vec.len() == i as usize);
            assert!(vec[i as usize - 1] == i);

            cuda_vec.push(i);
            assert!(cuda_vec.len() == i as usize);
        }
        for i in (1..=10).rev() {
            assert!(vec.pop().map_or(false, |e| e == i));
            assert!(cuda_vec.pop().map_or(false, |e| e == i));
        }

        assert!(vec.is_empty());
        assert!(cuda_vec.is_empty());

        let vec = Veci32::from_slice(&[1, 2, 3, 4, 5]);
        let mut cuda_vec = CudaVeci32::from_slice(&[1, 2, 3, 4, 5]);
        assert!(!vec.is_empty());
        assert!(vec.len() == 5);
        assert!(!cuda_vec.is_empty());
        assert!(cuda_vec.len() == 5);

        for i in 1..=5 {
            assert!(vec[i - 1] == i as i32);
        }
        for i in (1..=5).rev() {
            assert!(cuda_vec.pop().map_or(false, |e| e == i));
        }
    }
}
