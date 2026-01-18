//! Table-driven testing framework for dataset module.
//!
//! This module provides infrastructure for systematic, parameterized testing
//! of dataset operations to achieve comprehensive coverage.

use hdf5::{Dataset, File, Result};
use ndarray::{Array2, ArrayD};
use std::fmt;

// ============================================================================
// Test Data Generators
// ============================================================================

/// Test data generator for common array patterns.
pub struct TestData;

impl TestData {
    /// Create a simple 1D array of integers
    pub fn int_1d(count: usize) -> Vec<i32> {
        (0..count as i32).collect()
    }

    /// Create a simple 2D array of integers
    pub fn int_2d(rows: usize, cols: usize) -> Array2<i32> {
        Array2::from_shape_fn((rows, cols), |(i, j)| (i * cols + j) as i32)
    }

    /// Create a 1D array of floats
    pub fn float_1d(count: usize) -> Vec<f32> {
        (0..count).map(|i| i as f32).collect()
    }

    /// Create a 2D array of floats
    pub fn float_2d(rows: usize, cols: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |(i, j)| (i * cols + j) as f32)
    }

    /// Create sequential data starting from a value
    pub fn sequence_i32(start: i32, count: usize) -> Vec<i32> {
        (start..start + count as i32).collect()
    }

    /// Create constant data
    pub fn constant_i32(value: i32, count: usize) -> Vec<i32> {
        vec![value; count]
    }

    /// Create a dynamic array of given shape
    pub fn int_nd(shape: &[usize]) -> ArrayD<i32> {
        let size: usize = shape.iter().product();
        ArrayD::from_shape_vec(shape.to_vec(), (0..size as i32).collect()).unwrap()
    }
}

// ============================================================================
// Builder Configuration Test Cases
// ============================================================================

/// Describes a dataset builder configuration for testing.
#[derive(Clone)]
pub struct BuilderTestCase {
    pub name: &'static str,
    pub description: &'static str,
    pub shape: Vec<usize>,
    pub chunk: Option<Vec<usize>>,
    pub deflate_level: Option<u8>,
    pub shuffle: bool,
    pub fill_value: Option<i32>,
    pub packed: bool,
    pub no_chunk: bool,
    // Expected outcomes
    pub expect_chunked: Option<bool>,
    pub expect_error: Option<&'static str>,
}

impl Default for BuilderTestCase {
    fn default() -> Self {
        Self {
            name: "default",
            description: "default configuration",
            shape: vec![100],
            chunk: None,
            deflate_level: None,
            shuffle: false,
            fill_value: None,
            packed: false,
            no_chunk: false,
            expect_chunked: None,
            expect_error: None,
        }
    }
}

impl BuilderTestCase {
    pub fn new(name: &'static str) -> Self {
        Self { name, ..Default::default() }
    }

    pub fn description(mut self, desc: &'static str) -> Self {
        self.description = desc;
        self
    }

    pub fn shape(mut self, shape: Vec<usize>) -> Self {
        self.shape = shape;
        self
    }

    pub fn chunk(mut self, chunk: Vec<usize>) -> Self {
        self.chunk = Some(chunk);
        self
    }

    pub fn deflate(mut self, level: u8) -> Self {
        self.deflate_level = Some(level);
        self
    }

    pub fn shuffle(mut self) -> Self {
        self.shuffle = true;
        self
    }

    pub fn fill_value(mut self, value: i32) -> Self {
        self.fill_value = Some(value);
        self
    }

    pub fn packed(mut self) -> Self {
        self.packed = true;
        self
    }

    pub fn no_chunk(mut self) -> Self {
        self.no_chunk = true;
        self
    }

    pub fn expect_chunked(mut self, expected: bool) -> Self {
        self.expect_chunked = Some(expected);
        self
    }

    pub fn expect_error(mut self, pattern: &'static str) -> Self {
        self.expect_error = Some(pattern);
        self
    }
}

impl fmt::Debug for BuilderTestCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BuilderTestCase({}: {})", self.name, self.description)
    }
}

/// Standard builder test cases covering common configurations.
pub fn standard_builder_test_cases() -> Vec<BuilderTestCase> {
    vec![
        // Basic configurations
        BuilderTestCase::new("contiguous_1d")
            .description("1D contiguous dataset")
            .shape(vec![100])
            .no_chunk()
            .expect_chunked(false),
        BuilderTestCase::new("contiguous_2d")
            .description("2D contiguous dataset")
            .shape(vec![10, 10])
            .no_chunk()
            .expect_chunked(false),
        BuilderTestCase::new("chunked_1d")
            .description("1D chunked dataset")
            .shape(vec![100])
            .chunk(vec![10])
            .expect_chunked(true),
        BuilderTestCase::new("chunked_2d")
            .description("2D chunked dataset")
            .shape(vec![100, 100])
            .chunk(vec![10, 10])
            .expect_chunked(true),
        // Compression (requires chunking)
        BuilderTestCase::new("deflate_chunked")
            .description("Deflate compression with chunking")
            .shape(vec![100])
            .chunk(vec![10])
            .deflate(5)
            .expect_chunked(true),
        // Shuffle filter
        BuilderTestCase::new("shuffle_chunked")
            .description("Shuffle filter with chunking")
            .shape(vec![100])
            .chunk(vec![10])
            .shuffle()
            .expect_chunked(true),
        // Fill value
        BuilderTestCase::new("fill_value_contiguous")
            .description("Contiguous with fill value")
            .shape(vec![100])
            .no_chunk()
            .fill_value(42),
        BuilderTestCase::new("fill_value_chunked")
            .description("Chunked with fill value")
            .shape(vec![100])
            .chunk(vec![10])
            .fill_value(-1),
        // Packed storage
        BuilderTestCase::new("packed_contiguous")
            .description("Packed contiguous dataset")
            .shape(vec![100])
            .no_chunk()
            .packed(),
        BuilderTestCase::new("packed_chunked")
            .description("Packed chunked dataset")
            .shape(vec![100])
            .chunk(vec![10])
            .packed(),
        // Multi-dimensional
        BuilderTestCase::new("chunked_3d")
            .description("3D chunked dataset")
            .shape(vec![10, 10, 10])
            .chunk(vec![5, 5, 5])
            .expect_chunked(true),
        BuilderTestCase::new("chunked_4d")
            .description("4D chunked dataset")
            .shape(vec![5, 5, 5, 5])
            .chunk(vec![2, 2, 2, 2])
            .expect_chunked(true),
        // Edge case: scalar
        BuilderTestCase::new("scalar")
            .description("Scalar dataset")
            .shape(vec![])
            .expect_chunked(false),
        // Small datasets
        BuilderTestCase::new("single_element")
            .description("Single element dataset")
            .shape(vec![1])
            .expect_chunked(false),
        BuilderTestCase::new("small_2d")
            .description("Small 2D dataset")
            .shape(vec![2, 2])
            .expect_chunked(false),
    ]
}

/// Error test cases - configurations that should fail.
pub fn error_builder_test_cases() -> Vec<BuilderTestCase> {
    vec![
        BuilderTestCase::new("deflate_no_chunk")
            .description("Deflate without chunking should fail")
            .shape(vec![100])
            .no_chunk()
            .deflate(5)
            .expect_error("Chunking required"),
        BuilderTestCase::new("shuffle_no_chunk")
            .description("Shuffle without chunking should fail")
            .shape(vec![100])
            .no_chunk()
            .shuffle()
            .expect_error("Chunking required"),
    ]
}

// ============================================================================
// Test Runner
// ============================================================================

/// Runs a builder test case and returns the result.
pub fn run_builder_test(file: &File, case: &BuilderTestCase) -> Result<Dataset> {
    let mut builder = file.new_dataset::<i32>();

    // Apply configuration
    if case.packed {
        builder = builder.packed(true);
    }

    if let Some(ref chunk) = case.chunk {
        builder = builder.chunk(chunk.as_slice());
    }

    if case.no_chunk {
        builder = builder.no_chunk();
    }

    if let Some(level) = case.deflate_level {
        if hdf5::filters::deflate_available() {
            builder = builder.deflate(level);
        }
    }

    if case.shuffle {
        builder = builder.shuffle();
    }

    if let Some(fill) = case.fill_value {
        builder = builder.fill_value(fill);
    }

    // Set shape and create
    let shape_tuple: Vec<hdf5::Ix> = case.shape.iter().map(|&s| s as hdf5::Ix).collect();
    let shape_slice = shape_tuple.as_slice();

    builder.shape(shape_slice).create(case.name)
}

/// Validates the test case outcome.
pub fn validate_builder_test(case: &BuilderTestCase, result: Result<Dataset>) {
    match (&case.expect_error, result) {
        (Some(expected_err), Err(actual_err)) => {
            let err_msg = actual_err.to_string();
            assert!(
                err_msg.contains(expected_err),
                "Test '{}': Expected error containing '{}', got: {}",
                case.name,
                expected_err,
                err_msg
            );
        }
        (Some(expected_err), Ok(_)) => {
            panic!(
                "Test '{}': Expected error containing '{}', but succeeded",
                case.name, expected_err
            );
        }
        (None, Err(err)) => {
            panic!("Test '{}': Unexpected error: {}", case.name, err);
        }
        (None, Ok(ds)) => {
            // Validate expected properties
            if let Some(expected_chunked) = case.expect_chunked {
                assert_eq!(
                    ds.is_chunked(),
                    expected_chunked,
                    "Test '{}': Expected is_chunked={}, got {}",
                    case.name,
                    expected_chunked,
                    ds.is_chunked()
                );
            }

            // Validate shape
            let expected_shape: Vec<usize> = case.shape.clone();
            assert_eq!(
                ds.shape(),
                expected_shape,
                "Test '{}': Expected shape {:?}, got {:?}",
                case.name,
                expected_shape,
                ds.shape()
            );

            // Validate data round-trip if non-scalar
            if !case.shape.is_empty() && case.shape.iter().product::<usize>() > 0 {
                let size: usize = case.shape.iter().product();
                let data: Vec<i32> = (0..size as i32).collect();

                // For 2D+ datasets, use write_raw which accepts a flat slice
                ds.write_raw(&data).expect("Write should succeed");
                let read_data: Vec<i32> = ds.read_raw().expect("Read should succeed");
                assert_eq!(data, read_data, "Test '{}': Data round-trip failed", case.name);
            }
        }
    }
}

// ============================================================================
// Dataset Method Test Cases
// ============================================================================

/// Test case for Dataset method testing.
#[derive(Clone)]
pub struct MethodTestCase {
    pub name: &'static str,
    pub description: &'static str,
    pub setup: DatasetSetup,
    pub method: &'static str,
    pub expect_some: Option<bool>,
    pub expect_value: Option<&'static str>,
}

/// Dataset setup configuration.
#[derive(Clone, Default)]
pub struct DatasetSetup {
    pub shape: Vec<usize>,
    pub chunk: Option<Vec<usize>>,
    pub no_chunk: bool,
    pub write_data: bool,
}

impl DatasetSetup {
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape, ..Default::default() }
    }

    pub fn chunked(mut self, chunk: Vec<usize>) -> Self {
        self.chunk = Some(chunk);
        self
    }

    pub fn contiguous(mut self) -> Self {
        self.no_chunk = true;
        self
    }

    pub fn with_data(mut self) -> Self {
        self.write_data = true;
        self
    }
}

/// Standard method test cases.
pub fn dataset_method_test_cases() -> Vec<MethodTestCase> {
    vec![
        // is_chunked tests
        MethodTestCase {
            name: "is_chunked_true",
            description: "Chunked dataset returns true",
            setup: DatasetSetup::new(vec![100]).chunked(vec![10]),
            method: "is_chunked",
            expect_some: None,
            expect_value: Some("true"),
        },
        MethodTestCase {
            name: "is_chunked_false",
            description: "Contiguous dataset returns false",
            setup: DatasetSetup::new(vec![100]).contiguous(),
            method: "is_chunked",
            expect_some: None,
            expect_value: Some("false"),
        },
        // chunk() tests
        MethodTestCase {
            name: "chunk_some",
            description: "Chunked dataset returns Some",
            setup: DatasetSetup::new(vec![100]).chunked(vec![10]),
            method: "chunk",
            expect_some: Some(true),
            expect_value: None,
        },
        MethodTestCase {
            name: "chunk_none",
            description: "Contiguous dataset returns None",
            setup: DatasetSetup::new(vec![100]).contiguous(),
            method: "chunk",
            expect_some: Some(false),
            expect_value: None,
        },
        // offset() tests
        MethodTestCase {
            name: "offset_chunked_none",
            description: "Chunked dataset returns None for offset",
            setup: DatasetSetup::new(vec![100]).chunked(vec![10]).with_data(),
            method: "offset",
            expect_some: Some(false),
            expect_value: None,
        },
        MethodTestCase {
            name: "offset_contiguous_some",
            description: "Contiguous dataset with data returns Some",
            setup: DatasetSetup::new(vec![100]).contiguous().with_data(),
            method: "offset",
            expect_some: Some(true),
            expect_value: None,
        },
        // is_resizable tests
        MethodTestCase {
            name: "is_resizable_fixed",
            description: "Fixed-size dataset is not resizable",
            setup: DatasetSetup::new(vec![100]).contiguous(),
            method: "is_resizable",
            expect_some: None,
            expect_value: Some("false"),
        },
    ]
}

// ============================================================================
// Compute Chunk Shape Test Cases
// ============================================================================

/// Test case for compute_chunk_shape function.
#[derive(Debug, Clone)]
pub struct ChunkShapeTestCase {
    pub name: &'static str,
    pub dims: Vec<(usize, Option<usize>)>, // (current_dim, max_dim)
    pub min_elements: usize,
    pub expected: Vec<usize>,
}

/// Standard chunk shape computation test cases.
pub fn chunk_shape_test_cases() -> Vec<ChunkShapeTestCase> {
    vec![
        ChunkShapeTestCase {
            name: "1x1_min1",
            dims: vec![(1, Some(1)), (1, Some(1))],
            min_elements: 1,
            expected: vec![1, 1],
        },
        ChunkShapeTestCase {
            name: "1x10_min3",
            dims: vec![(1, Some(1)), (10, Some(10))],
            min_elements: 3,
            expected: vec![1, 3],
        },
        ChunkShapeTestCase {
            name: "1x10_min11",
            dims: vec![(1, Some(1)), (10, Some(10))],
            min_elements: 11,
            expected: vec![1, 10],
        },
        ChunkShapeTestCase {
            name: "4x4x4_min12",
            dims: vec![(4, Some(4)), (4, Some(4)), (4, Some(4))],
            min_elements: 12,
            expected: vec![1, 4, 4],
        },
        ChunkShapeTestCase {
            name: "4x4x4_min100",
            dims: vec![(4, Some(4)), (4, Some(4)), (4, Some(4))],
            min_elements: 100,
            expected: vec![4, 4, 4],
        },
        ChunkShapeTestCase {
            name: "4x4x4_min9",
            dims: vec![(4, Some(4)), (4, Some(4)), (4, Some(4))],
            min_elements: 9,
            expected: vec![1, 2, 4],
        },
        ChunkShapeTestCase {
            name: "1x1x100_min51",
            dims: vec![(1, Some(1)), (1, Some(1)), (100, Some(100))],
            min_elements: 51,
            expected: vec![1, 1, 100],
        },
        // Unlimited dimension tests
        ChunkShapeTestCase {
            name: "1x_unlimited_min11",
            dims: vec![(1, Some(1)), (10, None)], // None = unlimited
            min_elements: 11,
            expected: vec![1, 11],
        },
        ChunkShapeTestCase {
            name: "1x_unlimited_min9",
            dims: vec![(1, Some(1)), (10, None)],
            min_elements: 9,
            expected: vec![1, 9],
        },
    ]
}

// ============================================================================
// Read/Write Test Cases
// ============================================================================

/// Test case for read/write operations.
#[derive(Clone)]
pub struct ReadWriteTestCase {
    pub name: &'static str,
    pub shape: Vec<usize>,
    pub test_1d: bool,
    pub test_2d: bool,
    pub test_scalar: bool,
    pub test_raw: bool,
    pub test_dyn: bool,
}

impl Default for ReadWriteTestCase {
    fn default() -> Self {
        Self {
            name: "default",
            shape: vec![100],
            test_1d: true,
            test_2d: false,
            test_scalar: false,
            test_raw: true,
            test_dyn: true,
        }
    }
}

/// Standard read/write test cases.
pub fn read_write_test_cases() -> Vec<ReadWriteTestCase> {
    vec![
        ReadWriteTestCase {
            name: "scalar",
            shape: vec![],
            test_1d: false,
            test_2d: false,
            test_scalar: true,
            test_raw: true,
            test_dyn: true,
        },
        ReadWriteTestCase {
            name: "1d_small",
            shape: vec![10],
            test_1d: true,
            test_2d: false,
            test_scalar: false,
            test_raw: true,
            test_dyn: true,
        },
        ReadWriteTestCase {
            name: "1d_large",
            shape: vec![1000],
            test_1d: true,
            test_2d: false,
            test_scalar: false,
            test_raw: true,
            test_dyn: true,
        },
        ReadWriteTestCase {
            name: "2d_square",
            shape: vec![10, 10],
            test_1d: false,
            test_2d: true,
            test_scalar: false,
            test_raw: true,
            test_dyn: true,
        },
        ReadWriteTestCase {
            name: "2d_rect",
            shape: vec![5, 20],
            test_1d: false,
            test_2d: true,
            test_scalar: false,
            test_raw: true,
            test_dyn: true,
        },
        ReadWriteTestCase {
            name: "3d",
            shape: vec![5, 5, 5],
            test_1d: false,
            test_2d: false,
            test_scalar: false,
            test_raw: true,
            test_dyn: true,
        },
    ]
}

// ============================================================================
// Test Utilities
// ============================================================================

/// Creates a dataset with the given setup.
pub fn create_test_dataset(file: &File, name: &str, setup: &DatasetSetup) -> Result<Dataset> {
    let mut builder = file.new_dataset::<i32>();

    if let Some(ref chunk) = setup.chunk {
        builder = builder.chunk(chunk.as_slice());
    }

    if setup.no_chunk {
        builder = builder.no_chunk();
    }

    let shape_tuple: Vec<hdf5::Ix> = setup.shape.iter().map(|&s| s as hdf5::Ix).collect();
    let ds = builder.shape(shape_tuple.as_slice()).create(name)?;

    if setup.write_data && !setup.shape.is_empty() {
        let size: usize = setup.shape.iter().product();
        if size > 0 {
            let data: Vec<i32> = (0..size as i32).collect();
            ds.write(&data)?;
        }
    }

    Ok(ds)
}

/// Creates a temporary file for testing.
pub fn with_tmp_file<T, F: FnOnce(File) -> T>(func: F) -> T {
    use std::path::PathBuf;
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let path: PathBuf = dir.path().join("test.h5");
    let file = File::create(&path).unwrap();
    func(file)
}

// ============================================================================
// Conversion Mode Test Cases
// ============================================================================

/// Test case for type conversion modes (NoOp, Soft, Hard).
#[derive(Clone, Debug)]
pub struct ConversionTestCase {
    pub name: &'static str,
    pub description: &'static str,
    pub source_type: &'static str,
    pub target_type: &'static str,
    pub conversion: &'static str, // "noop", "soft", "hard"
    pub should_succeed: bool,
    pub expected_error_pattern: Option<&'static str>,
}

impl ConversionTestCase {
    pub fn new(name: &'static str, source_type: &'static str, target_type: &'static str) -> Self {
        Self {
            name,
            description: "",
            source_type,
            target_type,
            conversion: "soft",
            should_succeed: true,
            expected_error_pattern: None,
        }
    }

    pub fn description(mut self, desc: &'static str) -> Self {
        self.description = desc;
        self
    }

    pub fn conversion(mut self, conv: &'static str) -> Self {
        self.conversion = conv;
        self
    }

    pub fn should_fail(mut self, pattern: &'static str) -> Self {
        self.should_succeed = false;
        self.expected_error_pattern = Some(pattern);
        self
    }
}

/// Standard conversion test cases.
pub fn conversion_test_cases() -> Vec<ConversionTestCase> {
    vec![
        // Same type - should always succeed
        ConversionTestCase::new("i32_to_i32_noop", "i32", "i32")
            .description("Same type with NoOp conversion")
            .conversion("noop"),
        ConversionTestCase::new("i32_to_i32_soft", "i32", "i32")
            .description("Same type with Soft conversion")
            .conversion("soft"),
        // Widening conversions - should succeed
        ConversionTestCase::new("i32_to_i64_soft", "i32", "i64")
            .description("Widening conversion i32 to i64")
            .conversion("soft"),
        ConversionTestCase::new("u8_to_u32_soft", "u8", "u32")
            .description("Widening conversion u8 to u32")
            .conversion("soft"),
        ConversionTestCase::new("f32_to_f64_soft", "f32", "f64")
            .description("Widening conversion f32 to f64")
            .conversion("soft"),
        // Narrowing conversions - should fail with Soft, succeed with Hard
        ConversionTestCase::new("i64_to_i32_soft", "i64", "i32")
            .description("Narrowing conversion i64 to i32 with Soft")
            .conversion("soft")
            .should_fail("convertible"),
        ConversionTestCase::new("u32_to_u8_soft", "u32", "u8")
            .description("Narrowing conversion u32 to u8 with Soft")
            .conversion("soft")
            .should_fail("convertible"),
        // Signed/unsigned conversion - should fail
        ConversionTestCase::new("i32_to_u32_soft", "i32", "u32")
            .description("Signed to unsigned conversion")
            .conversion("soft")
            .should_fail("convertible"),
        ConversionTestCase::new("u32_to_i32_soft", "u32", "i32")
            .description("Unsigned to signed conversion")
            .conversion("soft")
            .should_fail("convertible"),
    ]
}

// ============================================================================
// AllocTime Test Cases
// ============================================================================

/// Test case for AllocTime property.
#[derive(Clone, Debug)]
pub struct AllocTimeTestCase {
    pub name: &'static str,
    pub alloc_time: Option<&'static str>, // "early", "late", "incr"
    pub description: &'static str,
}

impl AllocTimeTestCase {
    pub fn new(name: &'static str, alloc_time: Option<&'static str>) -> Self {
        Self { name, alloc_time, description: "" }
    }

    pub fn description(mut self, desc: &'static str) -> Self {
        self.description = desc;
        self
    }
}

/// Standard AllocTime test cases.
pub fn alloc_time_test_cases() -> Vec<AllocTimeTestCase> {
    vec![
        AllocTimeTestCase::new("alloc_none", None).description("Default (no explicit AllocTime)"),
        AllocTimeTestCase::new("alloc_early", Some("early")).description("Allocate space early"),
        AllocTimeTestCase::new("alloc_late", Some("late")).description("Allocate space late"),
        AllocTimeTestCase::new("alloc_incr", Some("incr")).description("Incremental allocation"),
    ]
}

// ============================================================================
// FillTime Test Cases
// ============================================================================

/// Test case for FillTime property.
#[derive(Clone, Debug)]
pub struct FillTimeTestCase {
    pub name: &'static str,
    pub fill_time: &'static str, // "ifset", "alloc", "never"
    pub description: &'static str,
}

impl FillTimeTestCase {
    pub fn new(name: &'static str, fill_time: &'static str) -> Self {
        Self { name, fill_time, description: "" }
    }

    pub fn description(mut self, desc: &'static str) -> Self {
        self.description = desc;
        self
    }
}

/// Standard FillTime test cases.
pub fn fill_time_test_cases() -> Vec<FillTimeTestCase> {
    vec![
        FillTimeTestCase::new("fill_ifset", "ifset")
            .description("Fill only when fill value is set"),
        FillTimeTestCase::new("fill_alloc", "alloc").description("Fill on allocation"),
        FillTimeTestCase::new("fill_never", "never").description("Never fill"),
    ]
}

// ============================================================================
// Chunk MinKB Edge Cases
// ============================================================================

/// Test case for chunk_min_kb edge cases.
#[derive(Clone, Debug)]
pub struct ChunkMinKBTestCase {
    pub name: &'static str,
    pub kb: usize,
    pub dtype_size: usize,
    pub shape: Vec<usize>,
    pub description: &'static str,
    pub expect_chunked: bool,
}

impl ChunkMinKBTestCase {
    pub fn new(name: &'static str, kb: usize, dtype_size: usize, shape: Vec<usize>) -> Self {
        Self { name, kb, dtype_size, shape, description: "", expect_chunked: true }
    }

    pub fn description(mut self, desc: &'static str) -> Self {
        self.description = desc;
        self
    }

    pub fn expect_chunked(mut self, expected: bool) -> Self {
        self.expect_chunked = expected;
        self
    }
}

/// Standard chunk_min_kb test cases.
pub fn chunk_min_kb_test_cases() -> Vec<ChunkMinKBTestCase> {
    vec![
        // Small sizes
        ChunkMinKBTestCase::new("chunk_1kb_i32", 1, 4, vec![1000])
            .description("1KB chunk for i32")
            .expect_chunked(true),
        ChunkMinKBTestCase::new("chunk_64kb_i32", 64, 4, vec![10000])
            .description("64KB chunk for i32")
            .expect_chunked(true),
        // Multi-dimensional
        ChunkMinKBTestCase::new("chunk_1kb_2d", 1, 4, vec![100, 100])
            .description("1KB chunk for 2D dataset")
            .expect_chunked(true),
        ChunkMinKBTestCase::new("chunk_64kb_3d", 64, 8, vec![50, 50, 50])
            .description("64KB chunk for 3D dataset with i64")
            .expect_chunked(true),
        // Very small chunk size
        ChunkMinKBTestCase::new("chunk_min_0kb", 0, 4, vec![100])
            .description("0KB should result in minimum chunk")
            .expect_chunked(true),
        // Large chunk size
        ChunkMinKBTestCase::new("chunk_1024kb", 1024, 4, vec![100000])
            .description("1MB chunk")
            .expect_chunked(true),
    ]
}

// ============================================================================
// Filter Combination Test Cases
// ============================================================================

/// Test case for filter combinations.
#[derive(Clone, Debug)]
pub struct FilterComboTestCase {
    pub name: &'static str,
    pub description: &'static str,
    pub filters: Vec<&'static str>, // "deflate", "shuffle", "fletcher32", "nbit", "scale_offset"
    pub chunk: Vec<usize>,
    pub shape: Vec<usize>,
    pub should_succeed: bool,
}

impl FilterComboTestCase {
    pub fn new(name: &'static str, filters: Vec<&'static str>) -> Self {
        Self {
            name,
            description: "",
            filters,
            chunk: vec![10],
            shape: vec![100],
            should_succeed: true,
        }
    }

    pub fn description(mut self, desc: &'static str) -> Self {
        self.description = desc;
        self
    }

    pub fn chunk(mut self, chunk: Vec<usize>) -> Self {
        self.chunk = chunk;
        self
    }

    pub fn shape(mut self, shape: Vec<usize>) -> Self {
        self.shape = shape;
        self
    }

    pub fn should_fail(mut self) -> Self {
        self.should_succeed = false;
        self
    }
}

/// Standard filter combination test cases.
pub fn filter_combo_test_cases() -> Vec<FilterComboTestCase> {
    vec![
        // Single filters
        FilterComboTestCase::new("filter_deflate_only", vec!["deflate"])
            .description("Deflate filter only"),
        FilterComboTestCase::new("filter_shuffle_only", vec!["shuffle"])
            .description("Shuffle filter only"),
        FilterComboTestCase::new("filter_fletcher32_only", vec!["fletcher32"])
            .description("Fletcher32 checksum filter only"),
        FilterComboTestCase::new("filter_nbit_only", vec!["nbit"]).description("NBit filter only"),
        // Common combinations - shuffle before deflate in pipeline means we list shuffle first
        // The builder applies them in reverse, so this creates the correct pipeline order
        FilterComboTestCase::new("filter_shuffle_deflate", vec!["shuffle", "deflate"])
            .description("Shuffle + Deflate (common combo)")
            .chunk(vec![10])
            .shape(vec![100]),
        FilterComboTestCase::new("filter_fletcher32_deflate", vec!["fletcher32", "deflate"])
            .description("Fletcher32 + Deflate")
            .chunk(vec![10])
            .shape(vec![100]),
        FilterComboTestCase::new("filter_shuffle_fletcher32", vec!["shuffle", "fletcher32"])
            .description("Shuffle + Fletcher32")
            .chunk(vec![10])
            .shape(vec![100]),
        // Triple combinations
        FilterComboTestCase::new(
            "filter_shuffle_fletcher32_deflate",
            vec!["shuffle", "fletcher32", "deflate"],
        )
        .description("Shuffle + Fletcher32 + Deflate")
        .chunk(vec![10])
        .shape(vec![100]),
    ]
}

// ============================================================================
// Resize Test Cases
// ============================================================================

/// Test case for dataset resize operations.
#[derive(Clone, Debug)]
pub struct ResizeTestCase {
    pub name: &'static str,
    pub initial_shape: Vec<usize>,
    pub resizable: Vec<bool>, // Which dimensions are resizable
    pub new_shape: Vec<usize>,
    pub should_succeed: bool,
    pub description: &'static str,
}

impl ResizeTestCase {
    pub fn new(name: &'static str, initial_shape: Vec<usize>, new_shape: Vec<usize>) -> Self {
        let resizable = vec![false; initial_shape.len()];
        Self { name, initial_shape, resizable, new_shape, should_succeed: true, description: "" }
    }

    pub fn resizable(mut self, resizable: Vec<bool>) -> Self {
        self.resizable = resizable;
        self
    }

    pub fn description(mut self, desc: &'static str) -> Self {
        self.description = desc;
        self
    }

    pub fn should_fail(mut self) -> Self {
        self.should_succeed = false;
        self
    }
}

/// Standard resize test cases.
pub fn resize_test_cases() -> Vec<ResizeTestCase> {
    vec![
        // Successful resizes
        ResizeTestCase::new("resize_1d_expand", vec![100], vec![200])
            .description("Expand 1D dataset")
            .resizable(vec![true]),
        ResizeTestCase::new("resize_1d_shrink", vec![200], vec![100])
            .description("Shrink 1D dataset")
            .resizable(vec![true]),
        ResizeTestCase::new("resize_2d_expand_rows", vec![10, 20], vec![20, 20])
            .description("Expand 2D dataset in first dimension")
            .resizable(vec![true, false]),
        ResizeTestCase::new("resize_2d_expand_cols", vec![10, 20], vec![10, 40])
            .description("Expand 2D dataset in second dimension")
            .resizable(vec![false, true]),
        ResizeTestCase::new("resize_2d_expand_both", vec![10, 20], vec![20, 40])
            .description("Expand 2D dataset in both dimensions")
            .resizable(vec![true, true]),
        // Edge cases
        ResizeTestCase::new("resize_to_zero", vec![100], vec![0])
            .description("Resize to zero")
            .resizable(vec![true]),
        ResizeTestCase::new("resize_from_zero", vec![0], vec![100])
            .description("Resize from zero")
            .resizable(vec![true]),
    ]
}

// ============================================================================
// Layout Test Cases
// ============================================================================

/// Test case for dataset layout.
#[derive(Clone, Debug)]
pub struct LayoutTestCase {
    pub name: &'static str,
    pub layout: &'static str, // "contiguous", "chunked", "compact", "virtual"
    pub shape: Vec<usize>,
    pub chunk: Option<Vec<usize>>,
    pub description: &'static str,
    pub should_succeed: bool,
}

impl LayoutTestCase {
    pub fn new(name: &'static str, layout: &'static str, shape: Vec<usize>) -> Self {
        Self { name, layout, shape, chunk: None, description: "", should_succeed: true }
    }

    pub fn chunk(mut self, chunk: Vec<usize>) -> Self {
        self.chunk = Some(chunk);
        self
    }

    pub fn description(mut self, desc: &'static str) -> Self {
        self.description = desc;
        self
    }

    pub fn should_fail(mut self) -> Self {
        self.should_succeed = false;
        self
    }
}

/// Standard layout test cases.
pub fn layout_test_cases() -> Vec<LayoutTestCase> {
    vec![
        LayoutTestCase::new("layout_contiguous_1d", "contiguous", vec![100])
            .description("Contiguous 1D dataset"),
        LayoutTestCase::new("layout_contiguous_2d", "contiguous", vec![10, 20])
            .description("Contiguous 2D dataset"),
        LayoutTestCase::new("layout_chunked_1d", "chunked", vec![100])
            .chunk(vec![10])
            .description("Chunked 1D dataset"),
        LayoutTestCase::new("layout_chunked_2d", "chunked", vec![100, 100])
            .chunk(vec![10, 20])
            .description("Chunked 2D dataset"),
        LayoutTestCase::new("layout_compact_small", "compact", vec![10])
            .description("Compact layout for small dataset"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_test_data_generators() {
        assert_eq!(TestData::int_1d(5), vec![0, 1, 2, 3, 4]);
        assert_eq!(TestData::constant_i32(42, 3), vec![42, 42, 42]);

        let arr2d = TestData::int_2d(2, 3);
        assert_eq!(arr2d.shape(), [2, 3]);
        assert_eq!(arr2d[[0, 0]], 0);
        assert_eq!(arr2d[[1, 2]], 5);
    }

    #[test]
    fn test_builder_test_case_builder() {
        let case = BuilderTestCase::new("test")
            .description("test case")
            .shape(vec![10, 10])
            .chunk(vec![5, 5])
            .deflate(3)
            .expect_chunked(true);

        assert_eq!(case.name, "test");
        assert_eq!(case.shape, vec![10, 10]);
        assert_eq!(case.chunk, Some(vec![5, 5]));
        assert_eq!(case.deflate_level, Some(3));
        assert_eq!(case.expect_chunked, Some(true));
    }

    #[test]
    fn test_dataset_setup_builder() {
        let setup = DatasetSetup::new(vec![100]).chunked(vec![10]).with_data();

        assert_eq!(setup.shape, vec![100]);
        assert_eq!(setup.chunk, Some(vec![10]));
        assert!(setup.write_data);
    }

    #[test]
    fn test_conversion_test_cases_exist() {
        let cases = conversion_test_cases();
        assert!(!cases.is_empty());
        assert_eq!(cases[0].name, "i32_to_i32_noop");
    }

    #[test]
    fn test_alloc_time_test_cases_exist() {
        let cases = alloc_time_test_cases();
        assert!(!cases.is_empty());
        assert_eq!(cases.len(), 4);
    }

    #[test]
    fn test_fill_time_test_cases_exist() {
        let cases = fill_time_test_cases();
        assert!(!cases.is_empty());
        assert_eq!(cases.len(), 3);
    }

    #[test]
    fn test_chunk_min_kb_test_cases_exist() {
        let cases = chunk_min_kb_test_cases();
        assert!(!cases.is_empty());
        assert_eq!(cases[0].name, "chunk_1kb_i32");
    }

    #[test]
    fn test_filter_combo_test_cases_exist() {
        let cases = filter_combo_test_cases();
        assert!(!cases.is_empty());
        assert_eq!(cases[0].name, "filter_deflate_only");
    }

    #[test]
    fn test_resize_test_cases_exist() {
        let cases = resize_test_cases();
        assert!(!cases.is_empty());
        assert_eq!(cases[0].name, "resize_1d_expand");
    }

    #[test]
    fn test_layout_test_cases_exist() {
        let cases = layout_test_cases();
        assert!(!cases.is_empty());
        assert_eq!(cases[0].name, "layout_contiguous_1d");
    }
}
