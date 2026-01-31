//! Rust bindings to the `hdf5` library for reading and writing data to and from storage
#![allow(non_camel_case_types, non_snake_case, dead_code, deprecated)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::cognitive_complexity)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::module_name_repetitions)]
#![cfg_attr(docsrs, feature(doc_cfg))]

macro_rules! extern_static {
    ($dest:ident, $src:ident) => {
        extern "C" {
            static $src: id_t;
        }
        pub static $dest: &'static id_t = unsafe { &$src };
    };
}

#[cfg(all(feature = "mpio", not(feature = "have-parallel")))]
compile_error!("Enabling \"mpio\" feature requires HDF5 library built with MPI support");

#[cfg(all(feature = "mpio", feature = "static"))]
compile_error!("\"mpio\" and \"static\" are incompatible features");

pub mod h5;
pub mod h5a;
pub mod h5ac;
pub mod h5c;
pub mod h5d;
pub mod h5e;
pub mod h5f;
pub mod h5fd;
pub mod h5g;
pub mod h5i;
pub mod h5l;
pub mod h5mm;
pub mod h5o;
pub mod h5p;
pub mod h5r;
pub mod h5s;
pub mod h5t;
pub mod h5vl;
pub mod h5z;

#[cfg(feature = "1.8.15")]
pub mod h5pl;

#[cfg(feature = "1.14.0")]
pub mod h5es;

#[allow(non_camel_case_types)]
mod internal_prelude {
    pub use crate::h5::{
        haddr_t, hbool_t, herr_t, hsize_t, hssize_t, htri_t, H5_ih_info_t, H5_index_t,
        H5_iter_order_t,
    };
    pub use crate::h5i::hid_t;
    pub use crate::h5t::H5T_cset_t;
    pub use libc::{int64_t, off_t, size_t, ssize_t, time_t, uint32_t, uint64_t, FILE};
    #[allow(unused_imports)]
    pub use std::os::raw::{
        c_char, c_double, c_float, c_int, c_long, c_longlong, c_uchar, c_uint, c_ulong,
        c_ulonglong, c_void,
    };
}

#[cfg(test)]
mod tests {
    use super::h5::H5open;
    use super::h5p::H5P_CLS_ROOT;

    #[test]
    pub fn test_smoke() {
        unsafe {
            H5open();
            assert!(*H5P_CLS_ROOT > 0);
        }
    }

    /// HDF5 2.0.0 specific tests
    #[cfg(feature = "2.0.0")]
    mod hdf5_2_0 {
        use crate::h5::H5open;
        use crate::h5i::hid_t;
        use crate::h5t::*;

        #[test]
        fn test_h5t_complex_class_value() {
            // H5T_COMPLEX should be 11 in HDF5 2.0.0
            assert_eq!(H5T_class_t::H5T_COMPLEX as i32, 11);
            assert_eq!(H5T_class_t::H5T_NCLASSES as i32, 12);
        }

        #[test]
        fn test_complex_predefined_types_valid() {
            unsafe {
                H5open();

                // Verify complex predefined types are valid (non-negative) after H5open
                assert!(*H5T_COMPLEX_IEEE_F32LE >= 0 as hid_t);
                assert!(*H5T_COMPLEX_IEEE_F32BE >= 0 as hid_t);
                assert!(*H5T_COMPLEX_IEEE_F64LE >= 0 as hid_t);
                assert!(*H5T_COMPLEX_IEEE_F64BE >= 0 as hid_t);
            }
        }

        #[test]
        fn test_bfloat16_predefined_types_valid() {
            unsafe {
                H5open();

                // Verify bfloat16 predefined types are valid
                assert!(*H5T_FLOAT_BFLOAT16LE >= 0 as hid_t);
                assert!(*H5T_FLOAT_BFLOAT16BE >= 0 as hid_t);
            }
        }

        #[test]
        fn test_fp8_predefined_types_valid() {
            unsafe {
                H5open();

                // Verify FP8 predefined types are valid
                assert!(*H5T_FLOAT_F8E4M3 >= 0 as hid_t);
                assert!(*H5T_FLOAT_F8E5M2 >= 0 as hid_t);
            }
        }

        #[test]
        fn test_h5tcomplex_create_callable() {
            use crate::h5t::{H5Tclose, H5Tcomplex_create, H5Tget_class};

            unsafe {
                H5open();

                // Create a complex type based on native float
                let complex_type = H5Tcomplex_create(*H5T_NATIVE_FLOAT);
                if complex_type >= 0 {
                    // Verify it's a complex type
                    let cls = H5Tget_class(complex_type);
                    assert_eq!(cls, H5T_class_t::H5T_COMPLEX);
                    H5Tclose(complex_type);
                }
            }
        }

        #[test]
        fn test_versioned_api_h5dread_chunk2() {
            use crate::h5d::H5Dread_chunk2;

            // Verify function exists and has correct signature
            let _: unsafe extern "C" fn(
                crate::h5i::hid_t,
                crate::h5i::hid_t,
                *const crate::h5::hsize_t,
                *mut u32,
                *mut std::os::raw::c_void,
                *mut usize,
            ) -> crate::h5::herr_t = H5Dread_chunk2;
        }

        #[test]
        fn test_versioned_api_h5tdecode2() {
            use crate::h5t::H5Tdecode2;

            // Verify function exists and has correct signature
            let _: unsafe extern "C" fn(*const std::os::raw::c_void, usize) -> crate::h5i::hid_t =
                H5Tdecode2;
        }

        #[test]
        fn test_versioned_api_h5iregister_type2() {
            use crate::h5i::{H5I_free_t, H5I_type_t, H5Iregister_type2};

            // Verify function exists and has correct signature
            let _: unsafe extern "C" fn(std::os::raw::c_uint, H5I_free_t) -> H5I_type_t =
                H5Iregister_type2;
        }

        #[test]
        fn test_deprecated_apis_exist() {
            // Verify deprecated v1 APIs still exist for backward compatibility
            #[allow(deprecated)]
            {
                use crate::h5d::H5Dread_chunk1;
                use crate::h5i::H5Iregister_type1;
                use crate::h5t::H5Tdecode1;

                let _: unsafe extern "C" fn(
                    crate::h5i::hid_t,
                    crate::h5i::hid_t,
                    *const crate::h5::hsize_t,
                    *mut u32,
                    *mut std::os::raw::c_void,
                ) -> crate::h5::herr_t = H5Dread_chunk1;

                let _: unsafe extern "C" fn(*const std::os::raw::c_void) -> crate::h5i::hid_t =
                    H5Tdecode1;

                let _: unsafe extern "C" fn(
                    usize,
                    std::os::raw::c_uint,
                    crate::h5i::H5I_free_t,
                ) -> crate::h5i::H5I_type_t = H5Iregister_type1;
            }
        }
    }

    /// Pre-HDF5 2.0.0 tests
    #[cfg(not(feature = "2.0.0"))]
    mod pre_hdf5_2_0 {
        use crate::h5t::H5T_class_t;

        #[test]
        fn test_h5t_nclasses_without_complex() {
            // H5T_NCLASSES should be 11 before HDF5 2.0.0
            assert_eq!(H5T_class_t::H5T_NCLASSES as i32, 11);
        }
    }
}
