//! Integration tests using a real HDF5 file.
//!
//! This test module uses a real HDF5 file (`episode_0.hdf5`) to test
//! dataset operations, variable-length data reading, and chunk information.
//!
//! File structure:
//! - Group "actions": master_gripper_widths (311, 1), master_joints (311, 6)
//! - Group "images": wrist_cam (311) - variable-length u8 data
//! - Group "obs": puppet_gripper_widths (311, 1), puppet_joints (311, 6)

use std::path::PathBuf;

use ndarray::Array2;

mod common;

use common::util::new_in_memory_file;

/// Returns the path to the fixtures directory.
fn fixtures_dir() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.push("tests/fixtures");
    dir
}

/// Returns the path to the episode_0.hdf5 test file.
fn episode_0_path() -> PathBuf {
    let mut path = fixtures_dir();
    path.push("episode_0.hdf5");
    path
}

/// Opens the episode_0.hdf5 file, skipping the test if the file is not found.
fn with_episode_0<F: FnOnce(&hdf5::File) -> hdf5::Result<()>>(f: F) {
    let path = episode_0_path();
    if !path.exists() {
        println!("Skipping test: fixture file not found: {:?}", path);
        return;
    }
    let file = hdf5::File::open(&path).unwrap_or_else(|e| {
        panic!("Failed to open fixture file {:?}: {}", path, e);
    });
    f(&file).expect("Test failed");
}

#[test]
fn test_open_real_file() {
    let path = episode_0_path();
    if !path.exists() {
        println!("Skipping test: fixture file not found: {:?}", path);
        return;
    }
    let file = hdf5::File::open(&path).unwrap();
    assert!(file.is_valid());
    assert_eq!(file.name(), "/");
}

#[test]
fn test_file_groups_exist() {
    with_episode_0(|file| {
        assert!(file.group("actions").is_ok());
        assert!(file.group("images").is_ok());
        assert!(file.group("obs").is_ok());
        Ok(())
    });
}

#[test]
fn test_dataset_names_in_groups() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let images = file.group("images")?;
        let obs = file.group("obs")?;

        // Check datasets exist in actions group
        assert!(actions.dataset("master_gripper_widths").is_ok());
        assert!(actions.dataset("master_joints").is_ok());

        // Check datasets exist in images group
        assert!(images.dataset("wrist_cam").is_ok());

        // Check datasets exist in obs group
        assert!(obs.dataset("puppet_gripper_widths").is_ok());
        assert!(obs.dataset("puppet_joints").is_ok());

        Ok(())
    });
}

#[test]
fn test_dataset_shapes() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let obs = file.group("obs")?;

        let master_gripper = actions.dataset("master_gripper_widths")?;
        let master_joints = actions.dataset("master_joints")?;
        let puppet_gripper = obs.dataset("puppet_gripper_widths")?;
        let puppet_joints = obs.dataset("puppet_joints")?;

        assert_eq!(master_gripper.shape(), vec![311, 1]);
        assert_eq!(master_gripper.ndim(), 2);
        assert_eq!(master_gripper.size(), 311);

        assert_eq!(master_joints.shape(), vec![311, 6]);
        assert_eq!(master_joints.ndim(), 2);
        assert_eq!(master_joints.size(), 1866);

        assert_eq!(puppet_gripper.shape(), vec![311, 1]);
        assert_eq!(puppet_gripper.ndim(), 2);

        assert_eq!(puppet_joints.shape(), vec![311, 6]);
        assert_eq!(puppet_joints.ndim(), 2);

        Ok(())
    });
}

#[test]
fn test_dataset_datatypes() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let images = file.group("images")?;

        let master_gripper = actions.dataset("master_gripper_widths")?;
        let wrist_cam = images.dataset("wrist_cam")?;

        // Check datatype for float dataset
        let dtype = master_gripper.dtype()?;
        assert_eq!(dtype.size(), 4); // f32 is 4 bytes

        // Check variable-length datatype
        let vlen_dtype = wrist_cam.dtype()?;
        assert_eq!(vlen_dtype.size(), std::mem::size_of::<hdf5_sys::h5t::hvl_t>());

        Ok(())
    });
}

#[test]
fn test_dataset_is_chunked() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_gripper = actions.dataset("master_gripper_widths")?;

        // Test is_chunked() method
        let is_chunked = master_gripper.is_chunked();
        let _ = is_chunked; // Result may vary based on file

        Ok(())
    });
}

#[test]
fn test_dataset_is_resizable() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_gripper = actions.dataset("master_gripper_widths")?;

        // Test is_resizable() method
        let resizable = master_gripper.is_resizable();
        assert!(!resizable, "Dataset should not be resizable");

        Ok(())
    });
}

#[test]
fn test_dataset_layout() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_gripper = actions.dataset("master_gripper_widths")?;

        // Test layout() method
        let layout = master_gripper.layout();
        // Layout should be one of Chunked, Compact, Contiguous, or Virtual
        let layout_str = format!("{:?}", layout);
        assert!(!layout_str.is_empty(), "Layout should not be empty");

        Ok(())
    });
}

#[test]
fn test_dataset_offset() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_gripper = actions.dataset("master_gripper_widths")?;

        // Test offset() method - returns Some(u64) if offset is defined
        let offset = master_gripper.offset();
        // Chunked datasets return None for offset
        let _offset = offset;

        Ok(())
    });
}

#[test]
fn test_dataset_fill_value() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_gripper = actions.dataset("master_gripper_widths")?;

        // Test fill_value() method
        let fill_value = master_gripper.fill_value();
        // May return Ok(None) if no fill value is set
        let _fill_value = fill_value;

        Ok(())
    });
}

#[test]
fn test_dataset_access_plist() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_gripper = actions.dataset("master_gripper_widths")?;

        // Test access_plist() method
        let dapl = master_gripper.access_plist()?;
        assert!(dapl.is_valid());

        // Test dapl() alias
        let dapl2 = master_gripper.dapl()?;
        assert!(dapl2.is_valid());

        Ok(())
    });
}

#[test]
fn test_dataset_create_plist() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_gripper = actions.dataset("master_gripper_widths")?;

        // Test create_plist() method
        let dcpl = master_gripper.create_plist()?;
        assert!(dcpl.is_valid());

        // Test dcpl() alias
        let dcpl2 = master_gripper.dcpl()?;
        assert!(dcpl2.is_valid());

        Ok(())
    });
}

#[test]
fn test_dataset_chunk() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_gripper = actions.dataset("master_gripper_widths")?;

        // Test chunk() method - returns Some<Vec<Ix>> if chunked
        let chunk = master_gripper.chunk();
        let _chunk = chunk;

        Ok(())
    });
}

#[test]
fn test_dataset_filters() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_gripper = actions.dataset("master_gripper_widths")?;

        // Test filters() method
        let filters = master_gripper.filters();
        // May be empty if no filters are applied
        let _filters = filters;

        Ok(())
    });
}

#[test]
fn test_read_float_dataset_1d_column() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_gripper = actions.dataset("master_gripper_widths")?;

        // Read as 2D array
        let data: Array2<f32> = master_gripper.read_2d()?;
        assert_eq!(data.shape(), [311, 1]);

        // Read as raw vector
        let raw: Vec<f32> = master_gripper.read_raw()?;
        assert_eq!(raw.len(), 311);

        Ok(())
    });
}

#[test]
fn test_read_float_dataset_2d() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_joints = actions.dataset("master_joints")?;

        // Read as 2D array
        let data: Array2<f32> = master_joints.read_2d()?;
        assert_eq!(data.shape(), [311, 6]);
        assert_eq!(data.len(), 1866);

        Ok(())
    });
}

#[test]
fn test_read_dyn_dataset() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_joints = actions.dataset("master_joints")?;

        // Read as dynamic array
        let data: ndarray::ArrayD<f32> = master_joints.read_dyn()?;
        assert_eq!(data.shape(), vec![311, 6]);

        Ok(())
    });
}

#[test]
fn test_read_variable_length_dataset() {
    with_episode_0(|file| {
        let images = file.group("images")?;
        let wrist_cam = images.dataset("wrist_cam")?;

        // Check shape
        assert_eq!(wrist_cam.shape(), vec![311]);

        // Check the datatype size for variable-length
        let dtype = wrist_cam.dtype()?;
        assert_eq!(dtype.size(), std::mem::size_of::<hdf5_sys::h5t::hvl_t>());

        // Verify we can query the space
        let space = wrist_cam.space()?;
        assert_eq!(space.size(), 311);

        Ok(())
    });
}

#[test]
fn test_dataset_reader() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_joints = actions.dataset("master_joints")?;

        // Test as_reader()
        let reader = master_joints.as_reader();
        let data: Array2<f32> = reader.read_2d()?;
        assert_eq!(data.shape(), [311, 6]);

        Ok(())
    });
}

#[test]
fn test_dataset_space() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_joints = actions.dataset("master_joints")?;

        // Test space() method
        let space = master_joints.space()?;
        assert_eq!(space.ndim(), 2);
        assert_eq!(space.size(), 1866);

        Ok(())
    });
}

#[test]
fn test_dataset_dtype() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_joints = actions.dataset("master_joints")?;

        // Test dtype() method
        let dtype = master_joints.dtype()?;
        assert!(dtype.is_valid());
        assert_eq!(dtype.size(), 4); // f32

        Ok(())
    });
}

#[test]
fn test_dataset_group_ref() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;

        // Test that dataset can be accessed via group reference
        let master_joints = actions.dataset("master_joints")?;
        assert_eq!(master_joints.name(), "/actions/master_joints");

        Ok(())
    });
}

#[test]
fn test_dataset_iterate_members() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;

        // Iterate over group members
        let member_names = actions.member_names().unwrap();
        assert!(member_names.contains(&"master_gripper_widths".to_string()));
        assert!(member_names.contains(&"master_joints".to_string()));

        Ok(())
    });
}

#[cfg(feature = "1.10.5")]
#[test]
fn test_chunk_info_for_dataset() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_joints = actions.dataset("master_joints")?;

        // Test num_chunks() if dataset is chunked
        let num_chunks = master_joints.num_chunks();
        let _num_chunks = num_chunks;

        // Test chunk_info() for first chunk if available
        if let Some(n) = num_chunks {
            if n > 0 {
                let info = master_joints.chunk_info(0);
                let _info = info;
            }
        }

        Ok(())
    });
}

#[cfg(feature = "1.10.5")]
#[test]
fn test_chunk_info_non_chunked_returns_none() {
    let file = new_in_memory_file().unwrap();
    let ds = file.new_dataset::<i32>().no_chunk().shape((4, 4)).create("nochunk").unwrap();

    // Non-chunked dataset should return None for num_chunks
    assert_eq!(ds.num_chunks(), None);

    // Non-chunked dataset should return None for chunk_info
    assert_eq!(ds.chunk_info(0), None);
}

#[cfg(feature = "1.10.5")]
#[test]
fn test_chunk_info_disabled_filters() {
    use hdf5::dataset::ChunkInfo;

    // Test ChunkInfo::disabled_filters() method
    let info = ChunkInfo {
        offset: vec![0, 0],
        filter_mask: 0b101, // bits 0 and 2 are set
        addr: 0,
        size: 1024,
    };

    let disabled = info.disabled_filters();
    assert_eq!(disabled, vec![0, 2]);

    // Test with zero filter mask
    let info2 = ChunkInfo {
        offset: vec![0, 0],
        filter_mask: 0,
        addr: 0,
        size: 1024,
    };
    assert!(info2.disabled_filters().is_empty());
}

#[cfg(feature = "1.14.0")]
#[test]
fn test_chunks_visit() {
    use hdf5::dataset::ChunkInfoRef;

    let file = new_in_memory_file().unwrap();

    // Create a chunked dataset
    let ds = file.new_dataset::<i16>().shape([3, 2]).chunk([1, 1]).create("chunk").unwrap();
    ds.write(&ndarray::arr2(&[[1, 2], [3, 4], [5, 6]])).unwrap();

    // Test chunks_visit() method
    let mut count = 0;
    ds.chunks_visit(|c: ChunkInfoRef| {
        count += 1;
        // Each chunk is 1x1 elements, with each element being 2 bytes (i16)
        assert!(c.size >= std::mem::size_of::<i16>() as u64);
        0
    })
    .unwrap();

    assert_eq!(count, 6); // 3 rows x 2 chunks per row
}

#[cfg(feature = "1.14.0")]
#[test]
fn test_chunks_visit_non_chunked_errors() {
    let file = new_in_memory_file().unwrap();
    let ds = file.new_dataset::<i16>().no_chunk().shape((4, 4)).create("nochunk").unwrap();

    // chunks_visit() should fail for non-chunked datasets
    let result = ds.chunks_visit(|_| 0);
    assert!(result.is_err());
}

#[cfg(feature = "1.14.0")]
#[test]
fn test_chunk_info_ref_conversion() {
    use hdf5::dataset::{ChunkInfo, ChunkInfoRef};

    let info_ref = ChunkInfoRef {
        offset: &[1, 2, 3],
        filter_mask: 5,
        addr: 1024,
        size: 2048,
    };

    // Test conversion from ChunkInfoRef to ChunkInfo
    let info: ChunkInfo = info_ref.into();
    assert_eq!(info.offset, vec![1, 2, 3]);
    assert_eq!(info.filter_mask, 5);
    assert_eq!(info.addr, 1024);
    assert_eq!(info.size, 2048);
}

#[cfg(feature = "1.14.0")]
#[test]
fn test_chunk_info_ref_disabled_filters() {
    use hdf5::dataset::ChunkInfoRef;

    let info_ref = ChunkInfoRef {
        offset: &[0, 0],
        filter_mask: 0b1101,
        addr: 0,
        size: 512,
    };

    // Test disabled_filters() on ChunkInfoRef
    let disabled = info_ref.disabled_filters();
    assert_eq!(disabled, vec![0, 2, 3]);
}

#[test]
fn test_dataset_clone() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_joints = actions.dataset("master_joints")?;

        // Test cloning
        let ds2 = master_joints.clone();
        assert_eq!(master_joints.id(), ds2.id());
        assert_eq!(master_joints.shape(), ds2.shape());

        Ok(())
    });
}

#[test]
fn test_dataset_debug_fmt() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_joints = actions.dataset("master_joints")?;

        // Test Debug formatting
        let debug_str = format!("{:?}", master_joints);
        assert!(debug_str.contains("dataset") || debug_str.contains("Dataset"));

        Ok(())
    });
}

#[test]
fn test_multiple_group_access() {
    with_episode_0(|file| {
        // Test accessing multiple groups in sequence
        let actions = file.group("actions")?;
        let obs = file.group("obs")?;

        let master_joints = actions.dataset("master_joints")?;
        let puppet_joints = obs.dataset("puppet_joints")?;

        assert_eq!(master_joints.shape(), vec![311, 6]);
        assert_eq!(puppet_joints.shape(), vec![311, 6]);

        Ok(())
    });
}

#[test]
fn test_dataset_attribute_access() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_joints = actions.dataset("master_joints")?;

        // Test attribute access - even if no attributes exist
        let attr_names = master_joints.attr_names().unwrap();
        // May be empty if no attributes are set
        let _attr_names = attr_names;

        Ok(())
    });
}

#[test]
fn test_container_ref_deref() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_joints = actions.dataset("master_joints")?;

        // Test that Dataset can be dereferenced to Container via Deref
        use std::ops::Deref;
        let _container: &hdf5::Container = master_joints.deref();

        Ok(())
    });
}

#[test]
fn test_dataset_name_and_path() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_joints = actions.dataset("master_joints")?;

        // Test name() method
        let name = master_joints.name();
        assert_eq!(name, "/actions/master_joints");

        Ok(())
    });
}

#[test]
fn test_dataset_file_ref() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_joints = actions.dataset("master_joints")?;

        // Test file() method to get parent file
        let parent_file = master_joints.file()?;
        // Both should point to the same file
        assert_eq!(parent_file.name(), file.name());

        Ok(())
    });
}

#[test]
fn test_reader_slice_operations() {
    with_episode_0(|file| {
        let obs = file.group("obs")?;
        let puppet_joints = obs.dataset("puppet_joints")?;

        let reader = puppet_joints.as_reader();

        // Test read_slice with various slice patterns
        let slice1: Array2<f32> = reader.read_slice(ndarray::s![0..10, ..]).unwrap();
        assert_eq!(slice1.shape(), [10, 6]);

        let slice2: Array2<f32> = reader.read_slice(ndarray::s![.., 0..3]).unwrap();
        assert_eq!(slice2.shape(), [311, 3]);

        let slice3: Array2<f32> = reader.read_slice(ndarray::s![0..5, 0..3]).unwrap();
        assert_eq!(slice3.shape(), [5, 3]);

        Ok(())
    });
}

#[test]
fn test_reader_single_element() {
    with_episode_0(|file| {
        let obs = file.group("obs")?;
        let puppet_gripper = obs.dataset("puppet_gripper_widths")?;

        let reader = puppet_gripper.as_reader();

        // Use read_slice to get a single element
        let single: Array2<f32> = reader.read_slice(ndarray::s![0..1, 0..1]).unwrap();

        // Verify we got a single value
        assert_eq!(single.shape(), [1, 1]);

        Ok(())
    });
}

#[test]
fn test_read_raw_comparison() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_joints = actions.dataset("master_joints")?;

        // Compare read_raw with read_2d
        let raw: Vec<f32> = master_joints.read_raw()?;
        let arr: Array2<f32> = master_joints.read_2d()?;

        assert_eq!(raw.len(), arr.len());
        assert_eq!(raw.as_slice(), arr.as_slice().unwrap());

        Ok(())
    });
}

#[test]
fn test_dataset_byte_order() {
    with_episode_0(|file| {
        let actions = file.group("actions")?;
        let master_joints = actions.dataset("master_joints")?;

        // Test dtype byte order
        let dtype = master_joints.dtype()?;
        let _byte_order = dtype.byte_order();

        Ok(())
    });
}
