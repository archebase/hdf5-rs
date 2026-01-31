//! Tests for HDF5 2.0.0 features
//!
//! These tests are conditionally compiled when linking against HDF5 2.0.0 or later.

mod common;

use self::common::util::new_in_memory_file;

/// Test that the H5T_COMPLEX datatype class is recognized (HDF5 2.0.0+)
#[cfg(feature = "2.0.0")]
mod hdf5_2_0_tests {
    use super::*;

    #[test]
    fn test_is_complex_method_exists() {
        // Test that is_complex() method is available on Datatype
        let dt = hdf5::Datatype::from_type::<f32>().unwrap();
        // A regular float should not be complex
        assert!(!dt.is_complex());
    }

    #[test]
    fn test_integer_not_complex() {
        let dt = hdf5::Datatype::from_type::<i32>().unwrap();
        assert!(!dt.is_complex());
    }

    #[test]
    fn test_float_not_complex() {
        let dt = hdf5::Datatype::from_type::<f64>().unwrap();
        assert!(!dt.is_complex());
    }

    #[test]
    fn test_h5t_class_t_has_complex_variant() {
        // Verify that H5T_COMPLEX is available in the enum
        use hdf5_sys::h5t::H5T_class_t;
        let _complex = H5T_class_t::H5T_COMPLEX;
        // H5T_NCLASSES should be 12 in HDF5 2.0.0+
        assert_eq!(H5T_class_t::H5T_NCLASSES as i32, 12);
    }

    #[test]
    fn test_complex_predefined_types_exist() {
        // Test that the complex predefined types are available
        use hdf5_sys::h5t::*;

        // These should be accessible (they are extern statics)
        // We just verify they exist by referencing them
        let _ = *H5T_COMPLEX_IEEE_F32LE;
        let _ = *H5T_COMPLEX_IEEE_F32BE;
        let _ = *H5T_COMPLEX_IEEE_F64LE;
        let _ = *H5T_COMPLEX_IEEE_F64BE;
        let _ = *H5T_NATIVE_FLOAT_COMPLEX;
        let _ = *H5T_NATIVE_DOUBLE_COMPLEX;
    }

    #[test]
    fn test_bfloat16_predefined_types_exist() {
        // Test that bfloat16 predefined types are available
        use hdf5_sys::h5t::*;

        let _ = *H5T_FLOAT_BFLOAT16LE;
        let _ = *H5T_FLOAT_BFLOAT16BE;
    }

    #[test]
    fn test_fp8_predefined_types_exist() {
        // Test that FP8 predefined types are available
        use hdf5_sys::h5t::*;

        let _ = *H5T_FLOAT_F8E4M3;
        let _ = *H5T_FLOAT_F8E5M2;
    }

    #[test]
    fn test_h5tcomplex_create_exists() {
        // Test that H5Tcomplex_create function is available
        use hdf5_sys::h5t::H5Tcomplex_create;

        // Just verify the function exists - we can't actually call it without a valid base type
        let _fn_ptr: unsafe extern "C" fn(hdf5_sys::h5i::hid_t) -> hdf5_sys::h5i::hid_t =
            H5Tcomplex_create;
    }

    #[test]
    fn test_h5dread_chunk2_exists() {
        // Test that H5Dread_chunk2 function is available
        use hdf5_sys::h5d::H5Dread_chunk2;
        use std::os::raw::c_void;

        // Just verify the function exists
        let _fn_ptr: unsafe extern "C" fn(
            hdf5_sys::h5i::hid_t,
            hdf5_sys::h5i::hid_t,
            *const hdf5_sys::h5::hsize_t,
            *mut u32,
            *mut c_void,
            *mut usize,
        ) -> hdf5_sys::h5::herr_t = H5Dread_chunk2;
    }

    #[test]
    fn test_h5tdecode2_exists() {
        // Test that H5Tdecode2 function is available
        use hdf5_sys::h5t::H5Tdecode2;
        use std::os::raw::c_void;

        // Just verify the function exists
        let _fn_ptr: unsafe extern "C" fn(*const c_void, usize) -> hdf5_sys::h5i::hid_t =
            H5Tdecode2;
    }

    #[test]
    fn test_h5iregister_type2_exists() {
        // Test that H5Iregister_type2 function is available
        use hdf5_sys::h5i::{H5I_free_t, H5I_type_t, H5Iregister_type2};
        use std::os::raw::c_uint;

        // Just verify the function exists
        let _fn_ptr: unsafe extern "C" fn(c_uint, H5I_free_t) -> H5I_type_t = H5Iregister_type2;
    }

    #[test]
    fn test_basic_file_operations_work_on_2_0() {
        // Ensure basic HDF5 operations still work with 2.0
        let file = new_in_memory_file().unwrap();

        // Create a group
        let group = file.create_group("test_group").unwrap();
        assert!(group.name() == "/test_group");

        // Create a dataset
        let ds = file.new_dataset::<i32>().shape([10, 10]).create("test_ds").unwrap();
        assert_eq!(ds.shape(), vec![10, 10]);

        // Write and read data
        let data: Vec<i32> = (0..100).collect();
        let arr = ndarray::Array::from_shape_vec((10, 10), data).unwrap();
        ds.write(&arr).unwrap();

        let read_arr: ndarray::Array2<i32> = ds.read().unwrap();
        assert_eq!(arr, read_arr);
    }

    #[test]
    fn test_datatype_operations_work_on_2_0() {
        // Test datatype operations on HDF5 2.0
        let dt_i32 = hdf5::Datatype::from_type::<i32>().unwrap();
        let dt_f64 = hdf5::Datatype::from_type::<f64>().unwrap();

        // Basic operations should work
        assert_eq!(dt_i32.size(), 4);
        assert_eq!(dt_f64.size(), 8);

        // Conversion checks should work
        assert!(dt_i32.is::<i32>());
        assert!(!dt_i32.is::<f64>());
    }
}

/// Tests that should work on all HDF5 versions
#[cfg(not(feature = "2.0.0"))]
mod pre_hdf5_2_0_tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_h5t_class_t_without_complex() {
        // Verify that H5T_NCLASSES is 11 in pre-2.0 HDF5
        use hdf5_sys::h5t::H5T_class_t;
        assert_eq!(H5T_class_t::H5T_NCLASSES as i32, 11);
    }

    #[test]
    fn test_h5dread_chunk_exists() {
        // Test that the original H5Dread_chunk function is available
        #[cfg(feature = "1.10.3")]
        {
            use hdf5_sys::h5d::H5Dread_chunk;
            use std::os::raw::c_void;

            let _fn_ptr: unsafe extern "C" fn(
                hdf5_sys::h5i::hid_t,
                hdf5_sys::h5i::hid_t,
                *const hdf5_sys::h5::hsize_t,
                *mut u32,
                *mut c_void,
            ) -> hdf5_sys::h5::herr_t = H5Dread_chunk;
        }
    }

    #[test]
    fn test_h5tdecode_exists() {
        // Test that the original H5Tdecode function is available
        use hdf5_sys::h5t::H5Tdecode;
        use std::os::raw::c_void;

        let _fn_ptr: unsafe extern "C" fn(*const c_void) -> hdf5_sys::h5i::hid_t = H5Tdecode;
    }

    #[test]
    fn test_h5iregister_type_exists() {
        // Test that the original H5Iregister_type function is available
        use hdf5_sys::h5i::{H5I_free_t, H5I_type_t, H5Iregister_type};
        use std::os::raw::c_uint;

        let _fn_ptr: unsafe extern "C" fn(usize, c_uint, H5I_free_t) -> H5I_type_t =
            H5Iregister_type;
    }
}

/// Version-agnostic tests
mod version_agnostic_tests {
    use super::*;

    #[test]
    fn test_basic_roundtrip() {
        let file = new_in_memory_file().unwrap();
        let ds = file.new_dataset::<f64>().shape([5]).create("data").unwrap();

        let data = ndarray::arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        ds.write(&data).unwrap();

        let read_data: ndarray::Array1<f64> = ds.read().unwrap();
        assert_eq!(data, read_data);
    }

    #[test]
    fn test_datatype_descriptor_roundtrip() {
        use hdf5::H5Type;

        #[derive(H5Type)]
        #[repr(C)]
        struct TestStruct {
            x: i32,
            y: f64,
        }

        let dt = hdf5::Datatype::from_type::<TestStruct>().unwrap();
        let desc = dt.to_descriptor().unwrap();
        let dt2 = hdf5::Datatype::from_descriptor(&desc).unwrap();
        assert_eq!(dt, dt2);
    }
}
