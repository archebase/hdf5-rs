use std::slice;

/// A scalar integer type used by `Dimension` trait for indexing.
pub type Ix = usize;

/// A trait for the shape and index types.
pub trait Dimension {
    fn ndim(&self) -> usize;

    fn dims(&self) -> Vec<Ix>;

    fn size(&self) -> Ix {
        let dims = self.dims();
        if dims.is_empty() {
            1
        } else {
            // Use checked product to avoid overflow on large dimensions
            dims.iter()
                .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
                .expect("Dimension size calculation overflowed")
        }
    }
}

impl<'a, T: Dimension + ?Sized> Dimension for &'a T {
    fn ndim(&self) -> usize {
        Dimension::ndim(*self)
    }

    fn dims(&self) -> Vec<Ix> {
        Dimension::dims(*self)
    }
}

impl Dimension for [Ix] {
    fn ndim(&self) -> usize {
        self.len()
    }

    fn dims(&self) -> Vec<Ix> {
        self.to_vec()
    }
}

impl Dimension for Vec<Ix> {
    fn ndim(&self) -> usize {
        self.len()
    }

    fn dims(&self) -> Vec<Ix> {
        self.clone()
    }
}

macro_rules! count_ty {
    () => { 0 };
    ($_i:ty, $($rest:ty,)*) => { 1 + count_ty!($($rest,)*) }
}

macro_rules! impl_tuple {
    () => (
        impl Dimension for () {
            fn ndim(&self) -> usize { 0 }
            fn dims(&self) -> Vec<Ix> { vec![] }
        }
    );

    (@impl <$tp:ty>, $head:ty, $($tail:ty,)*) => (
        impl Dimension for $tp {
            #[inline]
            fn ndim(&self) -> usize {
                count_ty!($head, $($tail,)*)
            }

            #[inline]
            fn dims(&self) -> Vec<Ix> {
                unsafe {
                    slice::from_raw_parts((self as *const Self).cast(), self.ndim())
                }.iter().cloned().collect()
            }
        }
    );

    ($head:ty, $($tail:ty,)*) => (
        impl_tuple! { @impl <($head, $($tail,)*)>, $head, $($tail,)* }
        impl_tuple! { @impl <[Ix; count_ty!($head, $($tail,)*)]>, $head, $($tail,)* }
        impl_tuple! { $($tail,)* }
    );
}

impl_tuple! { Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, }

impl Dimension for Ix {
    fn ndim(&self) -> usize {
        1
    }

    fn dims(&self) -> Vec<Ix> {
        vec![*self]
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    // compile-time test
    #[allow(dead_code)]
    pub fn slice_as_shape(shape: &[Ix]) {
        let file = crate::File::create("foo.h5").unwrap();
        file.new_dataset::<u8>().shape(shape).create("Test").unwrap();
    }

    #[test]
    pub fn test_unit_ndim() {
        assert_eq!(().ndim(), 0);
    }

    #[test]
    pub fn test_unit_dims() {
        assert_eq!(().dims(), vec![]);
    }

    #[test]
    pub fn test_unit_size() {
        assert_eq!(().size(), 1);
    }

    #[test]
    pub fn test_scalar_ndim() {
        assert_eq!(5usize.ndim(), 1);
    }

    #[test]
    pub fn test_scalar_dims() {
        assert_eq!(42usize.dims(), vec![42]);
    }

    #[test]
    pub fn test_scalar_size() {
        assert_eq!(5usize.size(), 5);
        assert_eq!(1usize.size(), 1);
    }

    #[test]
    pub fn test_slice_ndim() {
        assert_eq!([1, 2, 3].ndim(), 3);
        assert_eq!([42].ndim(), 1);
        assert_eq!([].ndim(), 0);
    }

    #[test]
    pub fn test_slice_dims() {
        assert_eq!([1, 2, 3].dims(), vec![1, 2, 3]);
        assert_eq!([42].dims(), vec![42]);
        assert_eq!([].dims(), vec![]);
    }

    #[test]
    pub fn test_slice_size() {
        assert_eq!([2, 3, 4].size(), 24);
        assert_eq!([1].size(), 1);
        assert_eq!([].size(), 1);
    }

    #[test]
    pub fn test_vec_ndim() {
        assert_eq!(vec![1, 2, 3].ndim(), 3);
        assert_eq!(vec![42].ndim(), 1);
        assert_eq!(Vec::<Ix>::new().ndim(), 0);
    }

    #[test]
    pub fn test_vec_dims() {
        assert_eq!(vec![1, 2, 3].dims(), vec![1, 2, 3]);
        assert_eq!(vec![42].dims(), vec![42]);
        assert_eq!(Vec::<Ix>::new().dims(), vec![]);
    }

    #[test]
    pub fn test_vec_size() {
        assert_eq!(vec![2, 3, 4].size(), 24);
        assert_eq!(vec![1].size(), 1);
        assert_eq!(Vec::<Ix>::new().size(), 1);
    }

    #[test]
    pub fn test_tuple_1_ndim() {
        assert_eq!((5usize,).ndim(), 1);
    }

    #[test]
    pub fn test_tuple_1_dims() {
        assert_eq!((5usize,).dims(), vec![5]);
    }

    #[test]
    pub fn test_tuple_1_size() {
        assert_eq!((5usize,).size(), 5);
    }

    #[test]
    pub fn test_tuple_2_ndim() {
        assert_eq!((2usize, 3usize).ndim(), 2);
    }

    #[test]
    pub fn test_tuple_2_dims() {
        assert_eq!((2usize, 3usize).dims(), vec![2, 3]);
        assert_eq!((10usize, 20usize).dims(), vec![10, 20]);
    }

    #[test]
    pub fn test_tuple_2_size() {
        assert_eq!((2usize, 3usize).size(), 6);
        assert_eq!((10usize, 20usize).size(), 200);
    }

    #[test]
    pub fn test_tuple_3_ndim() {
        assert_eq!((2usize, 3usize, 4usize).ndim(), 3);
    }

    #[test]
    pub fn test_tuple_3_dims() {
        assert_eq!((2usize, 3usize, 4usize).dims(), vec![2, 3, 4]);
    }

    #[test]
    pub fn test_tuple_3_size() {
        assert_eq!((2usize, 3usize, 4usize).size(), 24);
    }

    #[test]
    pub fn test_tuple_4_ndim() {
        assert_eq!((1usize, 2usize, 3usize, 4usize).ndim(), 4);
    }

    #[test]
    pub fn test_tuple_4_size() {
        assert_eq!((1usize, 2usize, 3usize, 4usize).size(), 24);
    }

    #[test]
    pub fn test_tuple_6_size() {
        assert_eq!((1usize, 2usize, 3usize, 4usize, 5usize, 6usize).size(), 720);
    }

    #[test]
    pub fn test_tuple_12_ndim() {
        assert_eq!(
            (
                1usize, 2usize, 3usize, 4usize, 5usize, 6usize, 7usize, 8usize, 9usize, 10usize,
                11usize, 12usize,
            )
                .ndim(),
            12
        );
    }

    #[test]
    pub fn test_tuple_12_size() {
        assert_eq!(
            (
                1usize, 1usize, 1usize, 1usize, 1usize, 1usize, 1usize, 1usize, 1usize, 1usize,
                1usize, 1usize
            )
                .size(),
            1
        );
    }

    #[test]
    pub fn test_reference_ndim() {
        let arr = [1, 2, 3];
        assert_eq!((&arr).ndim(), 3);
        let scalar = 5usize;
        assert_eq!((&scalar).ndim(), 1);
    }

    #[test]
    pub fn test_reference_dims() {
        let arr = [1, 2, 3];
        assert_eq!((&arr).dims(), vec![1, 2, 3]);
        let scalar = 5usize;
        assert_eq!((&scalar).dims(), vec![5]);
    }

    #[test]
    pub fn test_reference_size() {
        let arr = [2, 3, 4];
        assert_eq!((&arr).size(), 24);
        let scalar = 5usize;
        assert_eq!((&scalar).size(), 5);
    }

    #[test]
    pub fn test_size_large_dimensions() {
        assert_eq!([1000usize, 1000].size(), 1_000_000);
        assert_eq!([1024usize, 1024, 1024].size(), 1_073_741_824);
    }

    #[test]
    #[should_panic(expected = "overflow")]
    pub fn test_size_overflow() {
        // Dimensions that would overflow usize
        let huge = usize::MAX;
        let _ = [huge, 2usize].size();
    }

    #[test]
    pub fn test_size_empty_slice() {
        assert_eq!([].size(), 1);
    }

    #[test]
    pub fn test_size_zero_element() {
        // Array with zero element
        assert_eq!([0usize, 10usize].size(), 0);
        assert_eq!([10usize, 0usize].size(), 0);
    }

    #[test]
    pub fn test_size_mixed_with_zero() {
        // Arrays with zeros should result in zero size
        assert_eq!([2usize, 0usize, 5usize].size(), 0);
        assert_eq!([0usize, 0usize].size(), 0);
    }

    #[test]
    pub fn test_array_3_ndim() {
        assert_eq!([1usize, 2usize, 3usize].ndim(), 3);
    }

    #[test]
    pub fn test_array_3_dims() {
        assert_eq!([1usize, 2usize, 3usize].dims(), vec![1, 2, 3]);
    }

    #[test]
    pub fn test_array_3_size() {
        assert_eq!([1usize, 2usize, 3usize].size(), 6);
    }

    #[test]
    pub fn test_array_5_ndim() {
        assert_eq!([1usize, 2usize, 3usize, 4usize, 5usize].ndim(), 5);
    }

    #[test]
    pub fn test_array_5_size() {
        assert_eq!([1usize, 2usize, 3usize, 4usize, 5usize].size(), 120);
    }

    #[test]
    pub fn test_array_slice_consistency() {
        let arr = [1usize, 2usize, 3usize];
        let slice: &[Ix] = &arr;
        assert_eq!(arr.ndim(), slice.ndim());
        assert_eq!(arr.dims(), slice.dims());
        assert_eq!(arr.size(), slice.size());
    }
}
