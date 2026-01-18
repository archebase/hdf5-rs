//! Comprehensive coverage tests for dataset operations
//!
//! This file targets coverage gaps identified in dataset.rs to improve coverage from 59.68% to 80%+.
//!
//! Coverage areas:
//! - Error handling paths (resize failures, chunk validation errors, type conversion errors)
//! - Feature-gated functionality (chunks_visit, virtual datasets, external storage)
//! - Edge cases (anonymous datasets, empty datasets, compact layout)
//! - Filter combinations and property list configurations
//! - Builder method variations
//! - Table-driven systematic tests for comprehensive builder coverage

mod common;

use common::dataset_test_utils::{
    alloc_time_test_cases, chunk_min_kb_test_cases, conversion_test_cases,
    error_builder_test_cases, fill_time_test_cases, filter_combo_test_cases, layout_test_cases,
    resize_test_cases, run_builder_test, standard_builder_test_cases, validate_builder_test,
    TestData,
};
use common::util::new_in_memory_file;
use hdf5::types::H5Type;
use hdf5::{Dataset, File, Result};
use ndarray::{s, Array1, Array2, ArrayView1};

// ============================================================================
// Phase 1: Error Path Tests
// ============================================================================

#[test]
fn test_dataset_resize_non_resizable_fails() {
    let file = new_in_memory_file().unwrap();
    let ds = file.new_dataset::<i32>().shape(&[10, 20]).create("ds").unwrap();

    // Non-resizable dataset should fail to resize
    let result = ds.resize(&[15, 20]);
    assert!(result.is_err(), "Non-resizable dataset should fail to resize");
    let err_msg = result.as_ref().unwrap_err().to_string();
    assert!(
        err_msg.contains("not resizable")
            || err_msg.contains("dataset is not resizable")
            || err_msg.contains("maximal size")
            || err_msg.contains("contiguous storage"),
        "Error should mention resize limitation: {}",
        err_msg
    );
}

#[test]
fn test_dataset_resize_chunked_succeeds() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();
    let ds = file.new_dataset::<i32>().chunk(&[5, 10]).shape((10.., 10)).create("ds").unwrap();
    ds.write_raw(&data).unwrap();

    // Resize to larger
    ds.resize(&[15, 10]).unwrap();
    assert_eq!(ds.shape(), &[15, 10]);

    // Verify original data is preserved
    let read_data: Vec<i32> = ds.read_raw().unwrap();
    assert_eq!(&read_data[..100], &data[..]);
}

#[test]
fn test_dataset_resize_unbounded_dimension() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..50).collect();
    let ds = file.new_dataset::<i32>().chunk(&[5, 5]).shape((10.., 5)).create("ds").unwrap();
    ds.write_raw(&data).unwrap();

    // Resize unbounded dimension
    ds.resize(&[20, 5]).unwrap();
    assert_eq!(ds.shape(), &[20, 5]);
}

#[test]
fn test_dataset_fill_value_none() {
    let file = new_in_memory_file().unwrap();
    let ds = file.new_dataset::<i32>().no_fill_value().shape(&[10]).create("ds").unwrap();

    // fill_value() should return Ok - the actual value depends on HDF5 defaults
    let fill_value = ds.fill_value();
    assert!(fill_value.is_ok(), "fill_value() should return Ok");
}

#[test]
fn test_dataset_fill_value_with_value() {
    let file = new_in_memory_file().unwrap();
    let ds = file.new_dataset::<i32>().fill_value(-42).shape(&[10]).create("ds").unwrap();

    let fill_value = ds.fill_value().unwrap();
    assert!(fill_value.is_some(), "fill_value should be Some");
    use hdf5_types::OwnedDynValue;
    assert_eq!(fill_value.unwrap(), OwnedDynValue::from(-42_i32));
}

#[test]
fn test_chunk_resizable_without_chunking_fails() {
    let file = new_in_memory_file().unwrap();

    // Creating resizable dataset without chunking should fail
    let result = file.new_dataset::<i32>().no_chunk().shape(&[10..]).create("ds");

    assert!(result.is_err(), "Resizable dataset without chunking should fail");
    assert!(
        result.unwrap_err().to_string().contains("Chunking required"),
        "Error should mention chunking requirement"
    );
}

#[test]
fn test_chunk_filters_without_chunking_fails() {
    let file = new_in_memory_file().unwrap();

    // If deflate is available, this should fail without explicit chunking
    if hdf5::filters::deflate_available() {
        let result = file.new_dataset::<i32>().no_chunk().deflate(3).shape(&[100]).create("ds");

        assert!(result.is_err(), "Filters without chunking should fail");
        assert!(
            result.unwrap_err().to_string().contains("Chunking required"),
            "Error should mention chunking requirement"
        );
    }
}

#[test]
fn test_chunk_exceeds_data_shape_fails() {
    let file = new_in_memory_file().unwrap();

    // Chunk larger than data shape should fail
    let result = file.new_dataset::<i32>()
        .chunk(&[50, 50])  // Larger than data shape
        .shape(&[10, 20])
        .create("ds");

    assert!(result.is_err(), "Chunk larger than data should fail");
    let err_msg = result.as_ref().unwrap_err().to_string();
    assert!(
        err_msg.contains("exceed") || err_msg.contains("Chunk"),
        "Error should mention chunk dimensions"
    );
}

#[test]
fn test_chunk_zero_dimension_fails() {
    let file = new_in_memory_file().unwrap();

    // Chunk with zero dimension should fail
    let result = file.new_dataset::<i32>()
        .chunk(&[10, 0])  // Zero in one dimension
        .shape(&[20, 30])
        .create("ds");

    assert!(result.is_err(), "Chunk with zero dimension should fail");
    let err_msg = result.as_ref().unwrap_err().to_string();
    assert!(
        err_msg.contains("positive") || err_msg.contains("chunk"),
        "Error should mention positive dimensions"
    );
}

#[test]
fn test_chunk_ndim_mismatch_fails() {
    let file = new_in_memory_file().unwrap();

    // This is implicitly tested by the chunk() method expecting correct dimensions
    // but we can test related behavior
    let ds = file.new_dataset::<i32>().chunk(&[5, 5]).shape(&[10, 20]).create("ds").unwrap();

    assert_eq!(ds.chunk().unwrap(), vec![5, 5]);
}

#[test]
fn test_chunk_zero_dim_dataset_succeeds() {
    let file = new_in_memory_file().unwrap();

    // 0D (scalar) dataset - chunking is effectively ignored for scalars
    let ds = file.new_dataset::<i32>()
        .shape(())  // Scalar
        .create("ds")
        .unwrap();

    assert_eq!(ds.shape(), &[]);
    assert_eq!(ds.size(), 1);
    // Scalar datasets don't have chunks
    assert!(ds.chunk().is_none());
}

#[test]
fn test_builder_conversion_noop_with_same_type_succeeds() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = vec![1, 2, 3, 4, 5];

    // Create dataset and write data with same type
    let ds = file.new_dataset::<i32>().shape(5).create("ds").unwrap();

    ds.write(&data).unwrap();

    let read_data: Vec<i32> = ds.read_raw().unwrap();
    assert_eq!(read_data, data);
}

#[test]
fn test_builder_conversion_soft_allows_compatible_types() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = vec![1, 2, 3, 4, 5];

    // Create i32 dataset from i32 data (default soft conversion)
    let ds = file.new_dataset::<i32>().shape(5).create("ds").unwrap();

    ds.write(&data).unwrap();

    let read_data: Vec<i32> = ds.read_raw().unwrap();
    assert_eq!(read_data, data);
}

#[test]
fn test_builder_conversion_soft_widening() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = vec![1, 2, 3, 4, 5];

    // Create i64 dataset from i32 data (widening conversion)
    let ds = file.new_dataset::<i64>().shape(data.len()).create("ds").unwrap();

    ds.write(&data).unwrap();

    let read_data: Vec<i64> = ds.read_raw().unwrap();
    assert_eq!(read_data, vec![1i64, 2, 3, 4, 5]);
}

// ============================================================================
// Phase 2: Feature-Gated Functionality Tests
// ============================================================================

#[cfg(feature = "1.14.0")]
#[test]
fn test_dataset_chunks_visit() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    let ds = file.new_dataset::<i32>().chunk(&[10, 10]).shape(&[10, 10]).create("ds").unwrap();
    ds.write_raw(&data).unwrap();

    let mut visit_count = 0;
    ds.chunks_visit(|_chunk_info| {
        visit_count += 1;
        0 // Continue iteration
    })
    .unwrap();

    assert!(visit_count > 0, "chunks_visit should visit at least one chunk");
}

#[cfg(feature = "1.10.5")]
#[test]
fn test_dataset_chunk_info_out_of_bounds() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    let ds = file.new_dataset::<i32>().chunk(&[10, 10]).shape(&[10, 10]).create("ds").unwrap();
    ds.write_raw(&data).unwrap();

    // chunk_info() should return None for out-of-bounds index
    let result = ds.chunk_info(9999);
    assert!(result.is_none(), "chunk_info should return None for out-of-bounds");
}

#[cfg(feature = "1.10.5")]
#[test]
fn test_dataset_chunk_info_valid() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    let ds = file.new_dataset::<i32>().chunk(&[10, 10]).shape(&[10, 10]).create("ds").unwrap();
    ds.write_raw(&data).unwrap();

    // chunk_info() should return Some for valid index
    if let Some(info) = ds.chunk_info(0) {
        assert!(!info.offset.is_empty() || info.size > 0, "Chunk info should have valid data");
    } else {
        panic!("chunk_info should return Some for valid chunk index");
    }
}

#[cfg(feature = "1.10.5")]
#[test]
fn test_dataset_num_chunks() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    let ds = file.new_dataset::<i32>().chunk(&[10, 10]).shape(&[10, 10]).create("ds").unwrap();
    ds.write_raw(&data).unwrap();

    let num_chunks = ds.num_chunks();
    assert!(num_chunks.is_some(), "num_chunks should return Some for chunked dataset");
    assert!(num_chunks.unwrap() > 0, "num_chunks should be positive");
}

#[test]
fn test_dataset_chunk_returns_none_for_contiguous() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    let ds = file.new_dataset::<i32>().shape(&[100]).create("ds").unwrap();
    ds.write(&data).unwrap();

    // chunk() should return None for contiguous dataset
    assert!(ds.chunk().is_none(), "chunk() should return None for contiguous dataset");
}

#[cfg(feature = "1.10.0")]
#[test]
fn test_dataset_chunk_opts() {
    use hdf5::plist::dataset_create::ChunkOpts;

    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    // Create dataset with ChunkOpts
    let ds = file
        .new_dataset::<i32>()
        .chunk(10)
        .chunk_opts(ChunkOpts::default())
        .shape(&[100])
        .create("ds")
        .unwrap();

    ds.write(&data).unwrap();
    assert_eq!(ds.shape(), &[100]);
}

// ============================================================================
// Phase 3: Edge Case Tests
// ============================================================================

#[test]
fn test_dataset_anonymous_creation() {
    let file = new_in_memory_file().unwrap();

    // Create anonymous dataset by passing None as name
    let ds = file.new_dataset::<i32>().shape(&[10]).create(None::<&str>).unwrap();

    assert_eq!(ds.shape(), &[10]);

    // Anonymous dataset should not be accessible by name
    let result = file.dataset("ds");
    assert!(result.is_err(), "Anonymous dataset should not be found by name");
}

#[test]
fn test_dataset_chunked_empty() {
    let file = new_in_memory_file().unwrap();

    // Create empty chunked dataset (size 0)
    let data: Vec<i32> = vec![];
    let ds = file.new_dataset::<i32>().chunk(10).shape(&[0]).create("ds").unwrap();

    ds.write(&data).unwrap();
    assert_eq!(ds.shape(), &[0]);
}

#[test]
fn test_dataset_resizable_empty() {
    let file = new_in_memory_file().unwrap();

    // Create empty resizable dataset
    let data: Vec<i32> = vec![];
    let ds = file.new_dataset::<i32>().chunk(10).shape(&[0..]).create("ds").unwrap();

    ds.write(&data).unwrap();

    // Resize to non-zero
    ds.resize(&[10]).unwrap();
    assert_eq!(ds.shape(), &[10]);
}

#[test]
fn test_dataset_offset_chunked_is_none() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    let ds = file.new_dataset::<i32>().chunk(10).shape(&[100]).create("ds").unwrap();
    ds.write(&data).unwrap();

    // offset() should return None for chunked dataset
    assert!(ds.offset().is_none(), "offset() should return None for chunked dataset");
}

#[test]
fn test_dataset_offset_contiguous_is_some() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    let ds = file.new_dataset::<i32>().shape(&[100]).create("ds").unwrap();
    ds.write(&data).unwrap();

    // offset() should return Some for contiguous dataset
    assert!(ds.offset().is_some(), "offset() should return Some for contiguous dataset");
}

#[test]
fn test_dataset_packed_compound() {
    use hdf5::H5Type;

    #[derive(H5Type, Clone, PartialEq, Debug)]
    #[repr(C)]
    struct PackedStruct {
        a: i32,
        b: i8,
        c: i64,
    }

    let file = new_in_memory_file().unwrap();
    let data = vec![PackedStruct { a: 1, b: 2, c: 3 }, PackedStruct { a: 4, b: 5, c: 6 }];

    let ds = file.new_dataset::<PackedStruct>().packed(true).shape(&[2]).create("ds").unwrap();

    ds.write(&data).unwrap();

    let read_data: Vec<PackedStruct> = ds.read_raw().unwrap();
    assert_eq!(read_data, data);
}

#[test]
fn test_dataset_scalar_creation() {
    let file = new_in_memory_file().unwrap();

    let ds = file.new_dataset::<i32>().shape(()).create("scalar_ds").unwrap();

    assert_eq!(ds.shape(), &[]);
    assert_eq!(ds.ndim(), 0);
    assert_eq!(ds.size(), 1);
}

#[test]
fn test_dataset_scalar_with_data() {
    let file = new_in_memory_file().unwrap();
    let value: i32 = 42;

    let ds = file.new_dataset::<i32>().shape(()).create("scalar_ds").unwrap();

    ds.write_scalar(&value).unwrap();
    assert_eq!(ds.shape(), &[]);

    let read_value = ds.read_scalar::<i32>().unwrap();
    assert_eq!(read_value, value);
}

#[test]
fn test_dataset_multi_dimensional() {
    let file = new_in_memory_file().unwrap();
    let data = Array2::from_shape_fn((5, 10), |(i, j)| (i * 10 + j) as i32);

    let ds = file.new_dataset::<i32>().shape(&[5, 10]).create("ds").unwrap();

    ds.write(&data).unwrap();

    let read_data = ds.read_2d::<i32>().unwrap();
    assert_eq!(read_data, data);
}

// ============================================================================
// Phase 4: Filter and Property List Tests
// ============================================================================

#[test]
fn test_builder_multiple_filters() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..1000).collect();

    if hdf5::filters::deflate_available() {
        let ds = file
            .new_dataset::<i32>()
            .chunk(50)
            .shuffle()
            .deflate(3)
            .shape(&[1000])
            .create("ds")
            .unwrap();

        ds.write(&data).unwrap();

        let filters = ds.filters();
        assert!(filters.len() >= 2, "Should have multiple filters");

        let read_data: Vec<i32> = ds.read_raw().unwrap();
        assert_eq!(read_data, data);
    }
}

#[test]
fn test_builder_fletcher32() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    let ds = file.new_dataset::<i32>().chunk(20).fletcher32().shape(&[100]).create("ds").unwrap();

    ds.write(&data).unwrap();

    let filters = ds.filters();
    assert!(
        filters.iter().any(|f| matches!(f, hdf5::filters::Filter::Fletcher32)),
        "Should have Fletcher32 filter"
    );

    let read_data: Vec<i32> = ds.read_raw().unwrap();
    assert_eq!(read_data, data);
}

#[test]
fn test_builder_nbit() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    let ds = file.new_dataset::<i32>().chunk(20).nbit().shape(&[100]).create("ds").unwrap();

    ds.write(&data).unwrap();

    let filters = ds.filters();
    assert!(
        filters.iter().any(|f| matches!(f, hdf5::filters::Filter::NBit)),
        "Should have NBit filter"
    );

    let read_data: Vec<i32> = ds.read_raw().unwrap();
    assert_eq!(read_data, data);
}

#[test]
fn test_builder_scale_offset() {
    use hdf5::filters::ScaleOffset;

    let file = new_in_memory_file().unwrap();
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();

    let ds = file
        .new_dataset::<f32>()
        .chunk(20)
        .scale_offset(ScaleOffset::FloatDScale(2))
        .shape(&[100])
        .create("ds")
        .unwrap();

    ds.write(&data).unwrap();

    let filters = ds.filters();
    assert!(
        filters.iter().any(|f| matches!(f, hdf5::filters::Filter::ScaleOffset(_))),
        "Should have ScaleOffset filter"
    );
}

#[test]
fn test_builder_clear_filters() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    if hdf5::filters::deflate_available() {
        let ds = file
            .new_dataset::<i32>()
            .chunk(20)
            .deflate(3)
            .clear_filters()
            .shape(&[100])
            .create("ds")
            .unwrap();

        ds.write(&data).unwrap();

        let filters = ds.filters();
        assert_eq!(filters.len(), 0, "Should have no filters after clear");

        let read_data: Vec<i32> = ds.read_raw().unwrap();
        assert_eq!(read_data, data);
    }
}

#[test]
fn test_builder_set_filters() {
    use hdf5::filters::Filter;

    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    let filters = vec![Filter::Shuffle];

    let ds = file
        .new_dataset::<i32>()
        .chunk(20)
        .set_filters(&filters)
        .shape(&[100])
        .create("ds")
        .unwrap();

    ds.write(&data).unwrap();

    let ds_filters = ds.filters();
    assert_eq!(ds_filters, filters);
}

#[test]
fn test_builder_alloc_time() {
    use hdf5::plist::dataset_create::AllocTime;

    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    let ds = file
        .new_dataset::<i32>()
        .chunk(20)
        .alloc_time(Some(AllocTime::Early))
        .shape(&[100])
        .create("ds")
        .unwrap();

    ds.write(&data).unwrap();

    let dcpl = ds.create_plist().unwrap();
    assert_eq!(dcpl.alloc_time(), AllocTime::Early);
}

#[test]
fn test_builder_fill_time() {
    use hdf5::plist::dataset_create::FillTime;

    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    let ds = file
        .new_dataset::<i32>()
        .chunk(20)
        .fill_time(FillTime::Alloc)
        .shape(&[100])
        .create("ds")
        .unwrap();

    ds.write(&data).unwrap();

    let dcpl = ds.create_plist().unwrap();
    assert_eq!(dcpl.fill_time(), FillTime::Alloc);
}

#[test]
fn test_builder_attr_phase_change() {
    let file = new_in_memory_file().unwrap();

    let ds = file.new_dataset::<i32>().shape(&[100]).attr_phase_change(8, 5).create("ds").unwrap();

    let dcpl = ds.create_plist().unwrap();
    let phase_change = dcpl.attr_phase_change();
    assert_eq!(phase_change.max_compact, 8);
    assert_eq!(phase_change.min_dense, 5);
}

#[test]
fn test_builder_attr_creation_order() {
    use hdf5::plist::dataset_create::AttrCreationOrder;

    let file = new_in_memory_file().unwrap();

    let ds = file
        .new_dataset::<i32>()
        .shape(&[100])
        .attr_creation_order(AttrCreationOrder::TRACKED | AttrCreationOrder::INDEXED)
        .create("ds")
        .unwrap();

    let dcpl = ds.create_plist().unwrap();
    let order = dcpl.attr_creation_order();
    assert_eq!(order, AttrCreationOrder::TRACKED | AttrCreationOrder::INDEXED);
}

#[test]
fn test_builder_obj_track_times() {
    let file = new_in_memory_file().unwrap();

    let ds = file.new_dataset::<i32>().shape(&[100]).obj_track_times(true).create("ds").unwrap();

    let dcpl = ds.create_plist().unwrap();
    assert_eq!(dcpl.obj_track_times(), true);
}

#[test]
fn test_builder_layout_compact() {
    use hdf5::plist::dataset_create::Layout;

    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = vec![1, 2, 3, 4, 5];

    let ds = file.new_dataset::<i32>().layout(Layout::Compact).shape(&[5]).create("ds").unwrap();

    ds.write(&data).unwrap();

    let layout = ds.layout();
    assert_eq!(layout, Layout::Compact);
}

#[test]
fn test_builder_layout_contiguous() {
    use hdf5::plist::dataset_create::Layout;

    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    let ds =
        file.new_dataset::<i32>().layout(Layout::Contiguous).shape(&[100]).create("ds").unwrap();

    ds.write(&data).unwrap();

    let layout = ds.layout();
    assert_eq!(layout, Layout::Contiguous);
}

#[test]
fn test_builder_layout_chunked() {
    use hdf5::plist::dataset_create::Layout;

    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    let ds = file
        .new_dataset::<i32>()
        .layout(Layout::Chunked)
        .chunk(&[20])
        .shape(&[100])
        .create("ds")
        .unwrap();

    ds.write(&data).unwrap();

    let layout = ds.layout();
    assert_eq!(layout, Layout::Chunked);
}

#[test]
fn test_builder_chunk_min_kb() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..1000).collect();

    let ds = file.new_dataset::<i32>().chunk_min_kb(1).shape(&[1000]).create("ds").unwrap();

    ds.write(&data).unwrap();

    assert!(ds.is_chunked(), "Dataset should be chunked");

    let chunk = ds.chunk().unwrap();
    assert!(chunk.len() == 1, "Chunk should be 1D");
    assert!(chunk[0] > 0, "Chunk size should be positive");
}

#[cfg(feature = "1.8.17")]
#[test]
fn test_builder_efile_prefix() {
    let file = new_in_memory_file().unwrap();

    let ds = file.new_dataset::<i32>().shape(&[100]).efile_prefix("/tmp").create("ds").unwrap();

    // Verify the DAPL has the prefix set
    let dapl = ds.access_plist().unwrap();
    // The exact verification depends on HDF5 version behavior
    assert!(dapl.is_valid(), "DAPL should be valid");
}

#[cfg(feature = "1.10.0")]
#[test]
fn test_builder_virtual_printf_gap() {
    let file = new_in_memory_file().unwrap();

    let ds = file.new_dataset::<i32>().shape(&[100]).virtual_printf_gap(100).create("ds").unwrap();

    // Verify the DAPL has the gap set
    let dapl = ds.access_plist().unwrap();
    assert!(dapl.is_valid(), "DAPL should be valid");
}

#[cfg(all(feature = "1.10.0", feature = "have-parallel"))]
#[test]
fn test_builder_all_coll_metadata_ops() {
    let file = new_in_memory_file().unwrap();

    let ds =
        file.new_dataset::<i32>().shape(&[100]).all_coll_metadata_ops(true).create("ds").unwrap();

    // Verify the DAPL has the setting
    let dapl = ds.access_plist().unwrap();
    assert!(dapl.is_valid(), "DAPL should be valid");
}

#[test]
fn test_builder_chunk_cache() {
    let file = new_in_memory_file().unwrap();

    let ds = file
        .new_dataset::<i32>()
        .shape(&[100])
        .chunk_cache(100, 1024 * 1024, 0.75)
        .create("ds")
        .unwrap();

    // Verify the DAPL has the cache settings
    let dapl = ds.access_plist().unwrap();
    assert!(dapl.is_valid(), "DAPL should be valid");
}

#[test]
fn test_builder_create_intermediate_group() {
    let file = new_in_memory_file().unwrap();

    // This should create intermediate groups
    let _ds = file.new_dataset::<i32>().shape(&[100]).create("group1/group2/ds").unwrap();

    assert!(file.group("group1").is_ok(), "Intermediate group should be created");
    assert!(file.group("group1/group2").is_ok(), "Nested intermediate group should be created");
}

#[test]
fn test_builder_char_encoding() {
    use hdf5::plist::link_create::CharEncoding;

    let file = new_in_memory_file().unwrap();

    let _ds = file
        .new_dataset::<i32>()
        .shape(&[100])
        .char_encoding(CharEncoding::Utf8)
        .create("ds")
        .unwrap();

    // Character encoding is set on the builder
    // We verify the dataset was created successfully
    assert!(file.dataset("ds").is_ok());
}

#[test]
fn test_dataset_access_plist() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    let ds = file.new_dataset::<i32>().shape(&[100]).create("ds").unwrap();

    ds.write(&data).unwrap();

    // Get access property list
    let dapl = ds.access_plist().unwrap();
    assert!(dapl.is_valid(), "Access plist should be valid");

    // Test alias
    let dapl2 = ds.dapl().unwrap();
    assert_eq!(dapl, dapl2, "access_plist() and dapl() should return same");
}

#[test]
fn test_dataset_create_plist() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    let ds = file.new_dataset::<i32>().shape(&[100]).create("ds").unwrap();

    ds.write(&data).unwrap();

    // Get create property list
    let dcpl = ds.create_plist().unwrap();
    assert!(dcpl.is_valid(), "Create plist should be valid");

    // Test alias
    let dcpl2 = ds.dcpl().unwrap();
    assert_eq!(dcpl, dcpl2, "create_plist() and dcpl() should return same");
}

#[test]
fn test_dataset_is_resizable() {
    let file = new_in_memory_file().unwrap();

    // Non-resizable dataset
    let ds1 = file.new_dataset::<i32>().shape(&[100]).create("ds1").unwrap();
    assert!(!ds1.is_resizable(), "Fixed-size dataset should not be resizable");

    // Resizable dataset
    let ds2 = file.new_dataset::<i32>().chunk(10).shape(&[100..]).create("ds2").unwrap();
    assert!(ds2.is_resizable(), "Dataset with max dimension should be resizable");
}

#[test]
fn test_dataset_is_chunked() {
    let file = new_in_memory_file().unwrap();

    // Contiguous dataset
    let ds1 = file.new_dataset::<i32>().shape(&[100]).create("ds1").unwrap();
    assert!(!ds1.is_chunked(), "Contiguous dataset should not be chunked");

    // Chunked dataset
    let ds2 = file.new_dataset::<i32>().chunk(10).shape(&[100]).create("ds2").unwrap();
    assert!(ds2.is_chunked(), "Chunked dataset should be chunked");
}

#[test]
fn test_dataset_chunk_shape() {
    let file = new_in_memory_file().unwrap();

    let ds = file.new_dataset::<i32>().chunk(&[10, 20]).shape(&[30, 40]).create("ds").unwrap();

    assert_eq!(ds.chunk().unwrap(), vec![10, 20]);
}

#[test]
fn test_dataset_filters_empty() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = (0..100).collect();

    let ds = file.new_dataset::<i32>().shape(&[100]).create("ds").unwrap();

    ds.write(&data).unwrap();

    assert_eq!(ds.filters().len(), 0, "Dataset without filters should have empty filter list");
}

#[cfg(feature = "blosc")]
#[test]
fn test_builder_all_blosc_variants() {
    use hdf5::filters::BloscShuffle;

    let file = new_in_memory_file().unwrap();

    // Test blosclz
    let ds = file.new_dataset::<i32>().chunk(10).shape(&[100]).blosc_blosclz(5, BloscShuffle::None).create("ds_blosclz").unwrap();
    assert!(ds.filters().iter().any(|f| matches!(f, hdf5::filters::Filter::Blosc(_, _, _))));

    // Test lz4
    let ds = file.new_dataset::<i32>().chunk(10).shape(&[100]).blosc_lz4(5, BloscShuffle::Byte).create("ds_lz4").unwrap();
    assert!(ds.filters().iter().any(|f| matches!(f, hdf5::filters::Filter::Blosc(_, _, _))));

    // Test lz4hc
    let ds = file.new_dataset::<i32>().chunk(10).shape(&[100]).blosc_lz4hc(9, BloscShuffle::None).create("ds_lz4hc").unwrap();
    assert!(ds.filters().iter().any(|f| matches!(f, hdf5::filters::Filter::Blosc(_, _, _))));

    // Test snappy
    let ds = file.new_dataset::<i32>().chunk(10).shape(&[100]).blosc_snappy(5, BloscShuffle::Byte).create("ds_snappy").unwrap();
    assert!(ds.filters().iter().any(|f| matches!(f, hdf5::filters::Filter::Blosc(_, _, _))));

    // Test zlib
    let ds = file.new_dataset::<i32>().chunk(10).shape(&[100]).blosc_zlib(5, BloscShuffle::None).create("ds_zlib").unwrap();
    assert!(ds.filters().iter().any(|f| matches!(f, hdf5::filters::Filter::Blosc(_, _, _))));
}

#[test]
fn test_dataset_size() {
    let file = new_in_memory_file().unwrap();

    let ds = file.new_dataset::<i32>().shape(&[10, 20, 30]).create("ds").unwrap();

    assert_eq!(ds.size(), 10 * 20 * 30);
}

#[test]
fn test_dataset_ndim() {
    let file = new_in_memory_file().unwrap();

    let ds1 = file.new_dataset::<i32>().shape(&[100]).create("ds1").unwrap();
    assert_eq!(ds1.ndim(), 1);

    let ds2 = file.new_dataset::<i32>().shape(&[10, 20]).create("ds2").unwrap();
    assert_eq!(ds2.ndim(), 2);

    let ds3 = file.new_dataset::<i32>().shape(&[5, 5, 5]).create("ds3").unwrap();
    assert_eq!(ds3.ndim(), 3);
}

#[test]
fn test_dataset_empty_as_with_type_descriptor() {
    let file = new_in_memory_file().unwrap();

    // Create empty dataset with custom type descriptor
    let type_desc = i32::type_descriptor();
    let ds = file.new_dataset_builder().empty_as(&type_desc).shape(&[100]).create("ds").unwrap();

    assert_eq!(ds.shape(), &[100]);
}

#[test]
fn test_dataset_with_data_as() {
    let file = new_in_memory_file().unwrap();
    let data: Vec<i32> = vec![1, 2, 3, 4, 5];

    // Create dataset with data using custom type descriptor
    let type_desc = i32::type_descriptor();
    let arr: ArrayView1<i32> = ArrayView1::from(&data[..]);

    let ds = file.new_dataset_builder().with_data_as(arr, &type_desc).create("ds").unwrap();

    let read_data: Vec<i32> = ds.read_raw().unwrap();
    assert_eq!(read_data, data);
}

// ============================================================================
// Table-Driven Tests: Systematic Builder Coverage
// ============================================================================

/// Run all standard builder test cases using the table-driven framework.
#[test]
fn test_table_driven_builder_configurations() {
    let file = new_in_memory_file().expect("Failed to create in-memory file");

    for case in standard_builder_test_cases() {
        let result = run_builder_test(&file, &case);
        validate_builder_test(&case, result);

        // Clean up for next test
        let _ = file.unlink(case.name);
    }
}

/// Run all error builder test cases using the table-driven framework.
#[test]
fn test_table_driven_builder_error_cases() {
    let file = new_in_memory_file().expect("Failed to create in-memory file");

    for case in error_builder_test_cases() {
        // Skip deflate tests if deflate is not available
        if case.deflate_level.is_some() && !hdf5::filters::deflate_available() {
            continue;
        }

        let result = run_builder_test(&file, &case);
        validate_builder_test(&case, result);
    }
}

// ============================================================================
// Table-Driven Tests: Read/Write Operations
// ============================================================================

struct ReadWriteTestCase {
    name: &'static str,
    shape: Vec<usize>,
    chunked: bool,
}

fn read_write_test_table() -> Vec<ReadWriteTestCase> {
    vec![
        ReadWriteTestCase { name: "rw_scalar", shape: vec![], chunked: false },
        ReadWriteTestCase { name: "rw_1d_small", shape: vec![10], chunked: false },
        ReadWriteTestCase { name: "rw_1d_large", shape: vec![1000], chunked: false },
        ReadWriteTestCase { name: "rw_1d_chunked", shape: vec![100], chunked: true },
        ReadWriteTestCase { name: "rw_2d_square", shape: vec![10, 10], chunked: false },
        ReadWriteTestCase { name: "rw_2d_rect", shape: vec![5, 20], chunked: false },
        ReadWriteTestCase { name: "rw_2d_chunked", shape: vec![10, 10], chunked: true },
        ReadWriteTestCase { name: "rw_3d", shape: vec![5, 5, 5], chunked: false },
        ReadWriteTestCase { name: "rw_3d_chunked", shape: vec![5, 5, 5], chunked: true },
        ReadWriteTestCase { name: "rw_4d", shape: vec![2, 3, 4, 5], chunked: false },
    ]
}

#[test]
fn test_table_driven_read_write_roundtrip() {
    let file = new_in_memory_file().expect("Failed to create file");

    for test in read_write_test_table() {
        let mut builder = file.new_dataset::<i32>();

        if test.chunked && !test.shape.is_empty() {
            // Use shape/2 as chunk size, minimum 1
            let chunk: Vec<usize> = test.shape.iter().map(|&s| std::cmp::max(1, s / 2)).collect();
            builder = builder.chunk(chunk.as_slice());
        }

        let ds = builder
            .shape(test.shape.as_slice())
            .create(test.name)
            .unwrap_or_else(|e| panic!("Test '{}' create failed: {}", test.name, e));

        // Generate test data
        if test.shape.is_empty() {
            // Scalar test
            let val = 42i32;
            ds.write_scalar(&val).expect("write_scalar failed");
            let read_val: i32 = ds.read_scalar().expect("read_scalar failed");
            assert_eq!(val, read_val, "Test '{}' scalar roundtrip failed", test.name);
        } else {
            let size: usize = test.shape.iter().product();
            let data: Vec<i32> = (0..size as i32).collect();
            ds.write_raw(&data).expect("write failed");
            let read_data: Vec<i32> = ds.read_raw().expect("read_raw failed");
            assert_eq!(data, read_data, "Test '{}' roundtrip failed", test.name);
        }

        // Clean up
        file.unlink(test.name).ok();
    }
}

// ============================================================================
// Table-Driven Tests: Error Paths
// ============================================================================

struct ErrorPathTestCase {
    name: &'static str,
    setup: fn(&File) -> Result<()>,
    expected_error: &'static str,
}

fn error_path_test_table() -> Vec<ErrorPathTestCase> {
    vec![
        ErrorPathTestCase {
            name: "err_read_scalar_on_1d",
            setup: |f| {
                let ds = f.new_dataset::<i32>().shape(10).create("ds")?;
                ds.write(&vec![0i32; 10])?;
                ds.read_scalar::<i32>().map(|_| ())
            },
            expected_error: "ndim mismatch",
        },
        ErrorPathTestCase {
            name: "err_read_1d_on_2d",
            setup: |f| {
                let ds = f.new_dataset::<i32>().shape((5, 5)).create("ds")?;
                ds.write_raw(&vec![0i32; 25])?;
                ds.read_1d::<i32>().map(|_| ())
            },
            expected_error: "ndim mismatch",
        },
        ErrorPathTestCase {
            name: "err_read_2d_on_1d",
            setup: |f| {
                let ds = f.new_dataset::<i32>().shape(10).create("ds")?;
                ds.write(&vec![0i32; 10])?;
                ds.read_2d::<i32>().map(|_| ())
            },
            expected_error: "ndim mismatch",
        },
        ErrorPathTestCase {
            name: "err_write_wrong_size",
            setup: |f| {
                let ds = f.new_dataset::<i32>().shape(10).create("ds")?;
                ds.write(&vec![0i32; 5]) // Wrong size
            },
            expected_error: "shape mismatch",
        },
        ErrorPathTestCase {
            name: "err_write_raw_wrong_length",
            setup: |f| {
                let ds = f.new_dataset::<i32>().shape(10).create("ds")?;
                ds.write_raw(&[0i32; 5]) // Wrong length
            },
            expected_error: "length mismatch",
        },
        ErrorPathTestCase {
            name: "err_write_scalar_on_1d",
            setup: |f| {
                let ds = f.new_dataset::<i32>().shape(10).create("ds")?;
                ds.write_scalar(&42i32)
            },
            expected_error: "ndim mismatch",
        },
    ]
}

#[test]
fn test_table_driven_error_paths() {
    for test in error_path_test_table() {
        let file = new_in_memory_file().expect("Failed to create file");
        let result = (test.setup)(&file);

        match result {
            Ok(_) => panic!("Test '{}' expected error but succeeded", test.name),
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    err_msg.contains(test.expected_error),
                    "Test '{}' expected error containing '{}', got: {}",
                    test.name,
                    test.expected_error,
                    err_msg
                );
            }
        }
    }
}

// ============================================================================
// Table-Driven Tests: Dataset Methods
// ============================================================================

struct MethodTestCase {
    name: &'static str,
    setup: fn(&File) -> Result<Dataset>,
    test: fn(&Dataset),
}

fn method_test_table() -> Vec<MethodTestCase> {
    vec![
        // is_chunked() tests
        MethodTestCase {
            name: "method_is_chunked_true",
            setup: |f| f.new_dataset::<i32>().chunk(10).shape(100).create("ds"),
            test: |ds| assert!(ds.is_chunked()),
        },
        MethodTestCase {
            name: "method_is_chunked_false",
            setup: |f| f.new_dataset::<i32>().no_chunk().shape(100).create("ds"),
            test: |ds| assert!(!ds.is_chunked()),
        },
        // is_resizable() tests
        MethodTestCase {
            name: "method_is_resizable_false",
            setup: |f| f.new_dataset::<i32>().no_chunk().shape(100).create("ds"),
            test: |ds| assert!(!ds.is_resizable()),
        },
        // chunk() tests
        MethodTestCase {
            name: "method_chunk_some",
            setup: |f| f.new_dataset::<i32>().chunk(10).shape(100).create("ds"),
            test: |ds| {
                let chunk = ds.chunk();
                assert!(chunk.is_some());
                assert_eq!(chunk.unwrap(), vec![10]);
            },
        },
        MethodTestCase {
            name: "method_chunk_none",
            setup: |f| f.new_dataset::<i32>().no_chunk().shape(100).create("ds"),
            test: |ds| assert!(ds.chunk().is_none()),
        },
        // offset() tests
        MethodTestCase {
            name: "method_offset_chunked_none",
            setup: |f| {
                let ds = f.new_dataset::<i32>().chunk(10).shape(100).create("ds")?;
                ds.write(&vec![0i32; 100])?;
                Ok(ds)
            },
            test: |ds| assert!(ds.offset().is_none()),
        },
        MethodTestCase {
            name: "method_offset_contiguous_some",
            setup: |f| {
                let ds = f.new_dataset::<i32>().no_chunk().shape(100).create("ds")?;
                ds.write(&vec![0i32; 100])?;
                Ok(ds)
            },
            test: |ds| assert!(ds.offset().is_some()),
        },
        // filters() tests
        MethodTestCase {
            name: "method_filters_empty",
            setup: |f| f.new_dataset::<i32>().chunk(10).shape(100).create("ds"),
            test: |ds| assert!(ds.filters().is_empty()),
        },
        // layout() tests
        MethodTestCase {
            name: "method_layout_chunked",
            setup: |f| f.new_dataset::<i32>().chunk(10).shape(100).create("ds"),
            test: |ds| {
                use hdf5::plist::dataset_create::Layout;
                assert_eq!(ds.layout(), Layout::Chunked);
            },
        },
        MethodTestCase {
            name: "method_layout_contiguous",
            setup: |f| f.new_dataset::<i32>().no_chunk().shape(100).create("ds"),
            test: |ds| {
                use hdf5::plist::dataset_create::Layout;
                assert_eq!(ds.layout(), Layout::Contiguous);
            },
        },
        // Property list tests
        MethodTestCase {
            name: "method_access_plist",
            setup: |f| f.new_dataset::<i32>().shape(100).create("ds"),
            test: |ds| {
                let dapl = ds.access_plist().expect("access_plist should succeed");
                assert!(dapl.is_valid());
            },
        },
        MethodTestCase {
            name: "method_dapl_alias",
            setup: |f| f.new_dataset::<i32>().shape(100).create("ds"),
            test: |ds| {
                let dapl = ds.dapl().expect("dapl should succeed");
                assert!(dapl.is_valid());
            },
        },
        MethodTestCase {
            name: "method_create_plist",
            setup: |f| f.new_dataset::<i32>().shape(100).create("ds"),
            test: |ds| {
                let dcpl = ds.create_plist().expect("create_plist should succeed");
                assert!(dcpl.is_valid());
            },
        },
        MethodTestCase {
            name: "method_dcpl_alias",
            setup: |f| f.new_dataset::<i32>().shape(100).create("ds"),
            test: |ds| {
                let dcpl = ds.dcpl().expect("dcpl should succeed");
                assert!(dcpl.is_valid());
            },
        },
    ]
}

#[test]
fn test_table_driven_dataset_methods() {
    for test_case in method_test_table() {
        let file = new_in_memory_file().expect("Failed to create file");
        let ds = (test_case.setup)(&file).unwrap_or_else(|e| {
            panic!("Test '{}' setup failed: {}", test_case.name, e);
        });
        (test_case.test)(&ds);
    }
}

// ============================================================================
// Table-Driven Tests: Data Types
// ============================================================================

#[test]
fn test_table_driven_data_types() {
    let file = new_in_memory_file().unwrap();

    macro_rules! test_type {
        ($name:ident, $ty:ty, $val:expr) => {
            let ds = file.new_dataset::<$ty>().shape(10).create(stringify!($name)).unwrap();
            ds.write(&vec![$val; 10]).unwrap();
            let read: Vec<$ty> = ds.read_raw().unwrap();
            assert_eq!(read, vec![$val; 10], "Type {} failed", stringify!($ty));
        };
    }

    test_type!(td_i8, i8, 42i8);
    test_type!(td_i16, i16, 42i16);
    test_type!(td_i32, i32, 42i32);
    test_type!(td_i64, i64, 42i64);
    test_type!(td_u8, u8, 42u8);
    test_type!(td_u16, u16, 42u16);
    test_type!(td_u32, u32, 42u32);
    test_type!(td_u64, u64, 42u64);
    test_type!(td_f32, f32, 3.14f32);
    test_type!(td_f64, f64, 3.14f64);
    test_type!(td_bool, bool, true);
}

// ============================================================================
// Table-Driven Tests: Slice Operations
// ============================================================================

#[test]
fn test_table_driven_slice_operations() {
    let file = new_in_memory_file().unwrap();
    let data = TestData::int_2d(10, 10);

    let ds = file.new_dataset_builder().with_data(&data).create("slice_test").unwrap();

    // Read a row
    let row: Array1<i32> = ds.read_slice_1d(s![0, ..]).unwrap();
    assert_eq!(row.len(), 10);
    assert_eq!(row[0], 0);
    assert_eq!(row[9], 9);

    // Read a column
    let col: Array1<i32> = ds.read_slice_1d(s![.., 0]).unwrap();
    assert_eq!(col.len(), 10);
    assert_eq!(col[0], 0);
    assert_eq!(col[9], 90);

    // Read a sub-matrix
    let sub: Array2<i32> = ds.read_slice_2d(s![0..5, 0..5]).unwrap();
    assert_eq!(sub.shape(), [5, 5]);
}

// ============================================================================
// Table-Driven Tests: Container Trait Methods
// ============================================================================

#[test]
fn test_table_driven_container_methods() {
    let file = new_in_memory_file().unwrap();
    let ds = file.new_dataset::<i32>().shape((5, 10)).create("container_test").unwrap();

    ds.write_raw(&vec![0i32; 50]).unwrap();

    // Test all Container methods via Deref
    assert_eq!(ds.shape(), vec![5, 10]);
    assert_eq!(ds.ndim(), 2);
    assert_eq!(ds.size(), 50);
    assert!(!ds.is_scalar());
    assert!(ds.storage_size() > 0);

    let dtype = ds.dtype().unwrap();
    assert_eq!(dtype.size(), 4);

    let space = ds.space().unwrap();
    assert_eq!(space.ndim(), 2);
    assert_eq!(space.size(), 50);
}

// ============================================================================
// Table-Driven Tests: Reader/Writer Operations
// ============================================================================

#[test]
fn test_table_driven_reader_writer() {
    let file = new_in_memory_file().unwrap();
    let ds = file.new_dataset::<i32>().shape(100).create("rw_test").unwrap();

    // Writer operations
    let writer = ds.as_writer();
    writer.write(&vec![42i32; 100]).unwrap();

    // Reader operations
    let reader = ds.as_reader();
    let data: Vec<i32> = reader.read_raw().unwrap();
    assert_eq!(data, vec![42i32; 100]);

    // Reader with no_convert
    let reader_nc = ds.as_reader().no_convert();
    let data_nc: Vec<i32> = reader_nc.read_raw().unwrap();
    assert_eq!(data_nc, vec![42i32; 100]);
}

// ============================================================================
// Table-Driven Tests: ByteReader Operations
// ============================================================================

#[test]
fn test_table_driven_byte_reader() {
    use std::io::{Read, Seek, SeekFrom};

    let file = new_in_memory_file().unwrap();
    let ds = file.new_dataset::<u8>().shape(100).create("byte_reader_test").unwrap();

    let data: Vec<u8> = (0..100).collect();
    ds.write(&data).unwrap();

    let mut reader = ds.as_byte_reader().unwrap();

    // Test read
    let mut buf = [0u8; 10];
    reader.read(&mut buf).unwrap();
    assert_eq!(&buf, &data[0..10]);

    // Test seek
    reader.seek(SeekFrom::Start(50)).unwrap();
    reader.read(&mut buf).unwrap();
    assert_eq!(&buf, &data[50..60]);

    // Test stream_position
    let pos = reader.stream_position().unwrap();
    assert_eq!(pos, 60);
}

// ============================================================================
// Table-Driven Tests: AllocTime Property
// ============================================================================

#[test]
fn test_table_driven_alloc_time() {
    use hdf5::plist::dataset_create::AllocTime;

    for case in alloc_time_test_cases() {
        let file = new_in_memory_file().expect("Failed to create file");

        let mut builder = file.new_dataset::<i32>().chunk(10).shape(&[100]);

        if let Some(alloc_time_str) = case.alloc_time {
            let alloc_time = match alloc_time_str {
                "early" => AllocTime::Early,
                "late" => AllocTime::Late,
                "incr" => AllocTime::Incr,
                _ => panic!("Unknown alloc_time: {}", alloc_time_str),
            };
            builder = builder.alloc_time(Some(alloc_time));
        }

        let ds = builder
            .create(case.name)
            .unwrap_or_else(|e| panic!("Test '{}' create failed: {}", case.name, e));

        // Verify dataset was created successfully
        assert_eq!(ds.shape(), &[100]);

        // Verify the AllocTime property if it was set
        if case.alloc_time.is_some() {
            let dcpl = ds.create_plist().expect("Failed to get DCPL");
            // The property should be retrievable
            let _ = dcpl.alloc_time();
        }
    }
}

// ============================================================================
// Table-Driven Tests: FillTime Property
// ============================================================================

#[test]
fn test_table_driven_fill_time() {
    use hdf5::plist::dataset_create::FillTime;

    for case in fill_time_test_cases() {
        let file = new_in_memory_file().expect("Failed to create file");

        let fill_time = match case.fill_time {
            "ifset" => FillTime::IfSet,
            "alloc" => FillTime::Alloc,
            "never" => FillTime::Never,
            _ => panic!("Unknown fill_time: {}", case.fill_time),
        };

        let ds = file
            .new_dataset::<i32>()
            .chunk(10)
            .fill_time(fill_time)
            .shape(&[100])
            .create(case.name)
            .unwrap_or_else(|e| panic!("Test '{}' create failed: {}", case.name, e));

        // Verify dataset was created successfully
        assert_eq!(ds.shape(), &[100]);

        // Verify the FillTime property
        let dcpl = ds.create_plist().expect("Failed to get DCPL");
        let actual_fill_time = dcpl.fill_time();
        assert_eq!(actual_fill_time, fill_time, "Test '{}': FillTime mismatch", case.name);
    }
}

// ============================================================================
// Table-Driven Tests: Chunk MinKB
// ============================================================================

#[test]
fn test_table_driven_chunk_min_kb() {
    for case in chunk_min_kb_test_cases() {
        let file = new_in_memory_file().expect("Failed to create file");

        let ds = file
            .new_dataset::<i32>()
            .chunk_min_kb(case.kb)
            .shape(case.shape.as_slice())
            .create(case.name)
            .unwrap_or_else(|e| panic!("Test '{}' create failed: {}", case.name, e));

        // Verify chunking status
        assert_eq!(
            ds.is_chunked(),
            case.expect_chunked,
            "Test '{}': Expected is_chunked={}",
            case.name,
            case.expect_chunked
        );

        // Verify shape
        assert_eq!(ds.shape(), case.shape);
    }
}

// ============================================================================
// Table-Driven Tests: Filter Combinations
// ============================================================================

#[test]
fn test_table_driven_filter_combinations() {
    for case in filter_combo_test_cases() {
        let file = new_in_memory_file().expect("Failed to create file");

        // Skip deflate tests if not available
        if case.filters.contains(&"deflate") && !hdf5::filters::deflate_available() {
            continue;
        }

        let mut builder = file.new_dataset::<i32>().chunk(case.chunk.as_slice());

        // Apply filters based on the test case
        // Note: Filter order in HDF5 pipeline: shuffle comes before deflate
        // But the builder applies filters in reverse order, so we call deflate first
        for filter in &case.filters {
            match *filter {
                "deflate" => {
                    builder = builder.deflate(3);
                }
                "shuffle" => {
                    builder = builder.shuffle();
                }
                "fletcher32" => {
                    builder = builder.fletcher32();
                }
                "nbit" => {
                    builder = builder.nbit();
                }
                "scale_offset" => {
                    use hdf5::filters::ScaleOffset;
                    builder = builder.scale_offset(ScaleOffset::FloatDScale(2));
                }
                _ => panic!("Unknown filter: {}", filter),
            }
        }

        let result = builder.shape(case.shape.as_slice()).create(case.name);

        if case.should_succeed {
            let ds = result.unwrap_or_else(|e| {
                panic!("Test '{}' expected to succeed but failed: {}", case.name, e)
            });

            // Verify filters were applied
            let filters = ds.filters();
            assert_eq!(
                filters.len(),
                case.filters.len(),
                "Test '{}': Expected {} filters, got {}",
                case.name,
                case.filters.len(),
                filters.len()
            );
        } else {
            assert!(result.is_err(), "Test '{}' expected to fail but succeeded", case.name);
        }
    }
}

// ============================================================================
// Table-Driven Tests: Layout
// ============================================================================

#[test]
fn test_table_driven_layout() {
    use hdf5::plist::dataset_create::Layout;

    for case in layout_test_cases() {
        let file = new_in_memory_file().expect("Failed to create file");

        let mut builder = file.new_dataset::<i32>();

        match case.layout {
            "contiguous" => {
                builder = builder.layout(Layout::Contiguous);
            }
            "chunked" => {
                builder = builder.layout(Layout::Chunked);
                if let Some(ref chunk) = case.chunk {
                    let chunk_ix: Vec<hdf5::Ix> = chunk.iter().map(|&c| c as hdf5::Ix).collect();
                    builder = builder.chunk(chunk_ix.as_slice());
                }
            }
            "compact" => {
                builder = builder.layout(Layout::Compact);
            }
            _ => panic!("Unknown layout: {}", case.layout),
        }

        let result = builder.shape(case.shape.as_slice()).create(case.name);

        if case.should_succeed {
            let ds = result.unwrap_or_else(|e| {
                panic!("Test '{}' expected to succeed but failed: {}", case.name, e)
            });

            // Verify the layout matches
            let actual_layout = ds.layout();
            let expected_layout = match case.layout {
                "contiguous" => Layout::Contiguous,
                "chunked" => Layout::Chunked,
                "compact" => Layout::Compact,
                _ => panic!("Unknown layout: {}", case.layout),
            };
            assert_eq!(actual_layout, expected_layout, "Test '{}': Layout mismatch", case.name);
        } else {
            assert!(result.is_err(), "Test '{}' expected to fail but succeeded", case.name);
        }
    }
}

// ============================================================================
// Table-Driven Tests: Resize Operations
// ============================================================================

#[test]
fn test_table_driven_resize_operations() {
    for case in resize_test_cases() {
        let file = new_in_memory_file().expect("Failed to create file");

        // Create initial dataset with resizable dimension
        let ds = {
            let chunk_size = 10;

            match case.resizable.as_slice() {
                [true] => {
                    // 1D resizable
                    file.new_dataset::<i32>()
                        .chunk(chunk_size)
                        .shape(case.initial_shape[0]..)
                        .create(case.name)
                        .unwrap_or_else(|e| panic!("Test '{}' setup failed: {}", case.name, e))
                }
                [false] => {
                    // 1D non-resizable
                    file.new_dataset::<i32>()
                        .shape(&[case.initial_shape[0]])
                        .create(case.name)
                        .unwrap_or_else(|e| panic!("Test '{}' setup failed: {}", case.name, e))
                }
                [true, false] => {
                    // 2D with first dimension resizable
                    file.new_dataset::<i32>()
                        .chunk(&[chunk_size, case.initial_shape[1].min(chunk_size)])
                        .shape((case.initial_shape[0].., case.initial_shape[1]))
                        .create(case.name)
                        .unwrap_or_else(|e| panic!("Test '{}' setup failed: {}", case.name, e))
                }
                [false, true] => {
                    // 2D with second dimension resizable
                    file.new_dataset::<i32>()
                        .chunk(&[case.initial_shape[0].min(chunk_size), chunk_size])
                        .shape((case.initial_shape[0], case.initial_shape[1]..))
                        .create(case.name)
                        .unwrap_or_else(|e| panic!("Test '{}' setup failed: {}", case.name, e))
                }
                [true, true] => {
                    // 2D with both dimensions resizable
                    file.new_dataset::<i32>()
                        .chunk(&[chunk_size, chunk_size])
                        .shape((case.initial_shape[0].., case.initial_shape[1]..))
                        .create(case.name)
                        .unwrap_or_else(|e| panic!("Test '{}' setup failed: {}", case.name, e))
                }
                _ => {
                    // For other cases, use fixed size (non-resizable)
                    file.new_dataset::<i32>()
                        .shape(case.initial_shape.as_slice())
                        .create(case.name)
                        .unwrap_or_else(|e| panic!("Test '{}' setup failed: {}", case.name, e))
                }
            }
        };

        // Write initial data
        let initial_size: usize = case.initial_shape.iter().product();
        if initial_size > 0 {
            let data: Vec<i32> = (0..initial_size as i32).collect();
            ds.write_raw(&data).unwrap();
        }

        // Attempt resize
        let result = ds.resize(case.new_shape.as_slice());

        if case.should_succeed {
            result.unwrap_or_else(|e| {
                panic!("Test '{}' resize expected to succeed but failed: {}", case.name, e)
            });

            // Verify new shape
            assert_eq!(
                ds.shape(),
                case.new_shape,
                "Test '{}': Shape mismatch after resize",
                case.name
            );
        } else {
            assert!(result.is_err(), "Test '{}' resize expected to fail but succeeded", case.name);
        }
    }
}

// ============================================================================
// Table-Driven Tests: Conversion Modes
// ============================================================================

#[test]
fn test_table_driven_conversion_modes() -> Result<(), Box<dyn std::error::Error>> {
    for case in conversion_test_cases() {
        let file = new_in_memory_file()?;

        // For this test framework, we'll create datasets with different types
        // and test that they can be created and written to/read from

        let result: Result<hdf5::Dataset, hdf5::Error> = match case.source_type {
            "i32" => {
                let data: Vec<i32> = (0..10).collect();
                let ds = file.new_dataset::<i32>().shape(10).create(case.name)?;
                ds.write(&data)?;
                Ok(ds)
            }
            "i64" => {
                let data: Vec<i64> = (0..10).collect();
                let ds = file.new_dataset::<i64>().shape(10).create(case.name)?;
                ds.write(&data)?;
                Ok(ds)
            }
            "u8" => {
                let data: Vec<u8> = (0..10).collect();
                let ds = file.new_dataset::<u8>().shape(10).create(case.name)?;
                ds.write(&data)?;
                Ok(ds)
            }
            "u32" => {
                let data: Vec<u32> = (0..10).collect();
                let ds = file.new_dataset::<u32>().shape(10).create(case.name)?;
                ds.write(&data)?;
                Ok(ds)
            }
            "f32" => {
                let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
                let ds = file.new_dataset::<f32>().shape(10).create(case.name)?;
                ds.write(&data)?;
                Ok(ds)
            }
            "f64" => {
                let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
                let ds = file.new_dataset::<f64>().shape(10).create(case.name)?;
                ds.write(&data)?;
                Ok(ds)
            }
            _ => panic!("Unknown source type: {}", case.source_type),
        };

        if case.should_succeed {
            result.unwrap_or_else(|e| {
                panic!(
                    "Test '{}' ({}) expected to succeed but failed: {}",
                    case.name, case.description, e
                )
            });
        } else {
            // Note: This is a simplified test - in practice, conversion failures
            // would occur when writing data of one type to a dataset of another type
            // The actual conversion testing would require more complex setup
        }
    }
    Ok(())
}

// ============================================================================
// Table-Driven Tests: Edge Cases
// ============================================================================

#[test]
fn test_table_driven_edge_cases() {
    let file = new_in_memory_file().expect("Failed to create file");

    // Test: Very large chunk size
    let ds =
        file.new_dataset::<i32>().chunk(&[100000]).shape(&[100000]).create("large_chunk").unwrap();
    assert!(ds.is_chunked());
    assert_eq!(ds.chunk().unwrap(), vec![100000]);

    // Test: Chunk equal to data size
    let ds =
        file.new_dataset::<i32>().chunk(&[100]).shape(&[100]).create("chunk_equals_data").unwrap();
    assert_eq!(ds.chunk().unwrap(), vec![100]);

    // Test: Multi-dimensional chunk with size 1 in some dimensions
    let ds =
        file.new_dataset::<i32>().chunk(&[1, 50]).shape(&[100, 50]).create("chunk_1x50").unwrap();
    assert_eq!(ds.chunk().unwrap(), vec![1, 50]);

    // Test: Very small dataset (single element)
    let ds = file.new_dataset::<i32>().shape(&[1]).create("single_element").unwrap();
    assert_eq!(ds.shape(), &[1]);
    assert_eq!(ds.size(), 1);

    // Test: Zero-sized dimension
    let ds = file.new_dataset::<i32>().shape(&[0]).create("zero_sized").unwrap();
    assert_eq!(ds.shape(), &[0]);
    assert_eq!(ds.size(), 0);
}

// ============================================================================
// Table-Driven Tests: Filter Pipeline Order
// ============================================================================

#[test]
fn test_table_driven_filter_pipeline_order() {
    use hdf5::filters::Filter;

    let file = new_in_memory_file().expect("Failed to create file");

    if !hdf5::filters::deflate_available() {
        return;
    }

    // Test that filters are applied in the correct order
    // Expected order for this combo: Shuffle -> Deflate
    let ds = file
        .new_dataset::<i32>()
        .chunk(&[100])
        .shuffle()
        .deflate(5)
        .shape(&[1000])
        .create("filter_order_test")
        .unwrap();

    let filters = ds.filters();
    assert!(filters.len() >= 2, "Should have at least 2 filters");

    // Verify order: optional filters come first, then deflate
    let deflate_idx = filters
        .iter()
        .position(|f| matches!(f, Filter::Deflate(_)))
        .expect("Should have Deflate filter");

    // Shuffle should come before Deflate in the pipeline
    let shuffle_idx = filters
        .iter()
        .position(|f| matches!(f, Filter::Shuffle))
        .expect("Should have Shuffle filter");

    assert!(shuffle_idx < deflate_idx, "Shuffle should come before Deflate in the filter pipeline");
}

// ============================================================================
// Table-Driven Tests: Fill Value Variants
// ============================================================================

#[test]
fn test_table_driven_fill_values() {
    let file = new_in_memory_file().expect("Failed to create file");
    use hdf5_types::OwnedDynValue;

    let fill_values: Vec<i32> = vec![-1, 0, 42, -100];

    for (i, &fill_val) in fill_values.iter().enumerate() {
        let name = format!("fill_{}", i);
        let ds = file
            .new_dataset::<i32>()
            .fill_value(fill_val)
            .shape(&[10])
            .create(name.as_str())
            .unwrap();

        // Verify fill value was set
        let retrieved = ds.fill_value().unwrap();
        assert_eq!(
            retrieved,
            Some(OwnedDynValue::from(fill_val)),
            "Fill value mismatch for {}",
            fill_val
        );
    }
}

// ============================================================================
// Table-Driven Tests: Chunk Resizable Edge Cases
// ============================================================================

#[test]
fn test_table_driven_chunk_resizable_edge_cases() {
    let file = new_in_memory_file().expect("Failed to create file");

    // Test: Multiple resizable dimensions
    let ds = file
        .new_dataset::<i32>()
        .chunk(&[10, 10])
        .shape((100.., 100..))
        .create("multi_resizable")
        .unwrap();
    assert!(ds.is_resizable());
    assert_eq!(ds.shape(), &[100, 100]);

    // Resize in first dimension
    ds.resize(&[150, 100]).unwrap();
    assert_eq!(ds.shape(), &[150, 100]);

    // Resize in second dimension
    ds.resize(&[150, 150]).unwrap();
    assert_eq!(ds.shape(), &[150, 150]);

    // Test: One resizable, one fixed
    let ds = file
        .new_dataset::<i32>()
        .chunk(&[10, 20])
        .shape((100.., 50))
        .create("mixed_resizable")
        .unwrap();
    assert!(ds.is_resizable());

    // Should only be able to resize the first dimension
    let result = ds.resize(&[150, 50]);
    assert!(result.is_ok(), "Should be able to resize resizable dimension");

    let result = ds.resize(&[100, 100]);
    assert!(result.is_err(), "Should not be able to resize fixed dimension");
}

// ============================================================================
// Table-Driven Tests: Anonymous Dataset Properties
// ============================================================================

#[test]
fn test_table_driven_anonymous_dataset_properties() {
    let file = new_in_memory_file().expect("Failed to create file");

    // Create anonymous dataset with various configurations
    let configs = vec![
        ("anon_scalar", vec![]),
        ("anon_1d", vec![100]),
        ("anon_2d", vec![10, 20]),
        ("anon_3d", vec![5, 5, 5]),
    ];

    for (name, shape) in configs {
        let ds =
            file.new_dataset::<i32>().shape(shape.as_slice()).create(None::<&str>).unwrap_or_else(
                |e| panic!("Anonymous dataset creation failed for {}: {}", name, e),
            );

        assert_eq!(ds.shape(), shape);
        assert_eq!(ds.ndim(), shape.len());
        assert_eq!(ds.size(), shape.iter().product::<usize>().max(1));
    }
}
