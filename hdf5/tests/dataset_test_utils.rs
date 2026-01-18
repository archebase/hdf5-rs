//! Testing framework for dataset module.
//!
//! This module provides helper functions, macros, and test fixtures
//! to make testing dataset operations easier and more comprehensive.

use hdf5::{File, Group, Result};
use ndarray::Array2;
use std::ops::Deref;

/// Creates a temporary file for testing.
pub fn with_tmp_file<T, F: FnOnce(File) -> T>(func: F) -> T {
    use std::path::PathBuf;
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let path: PathBuf = dir.path().join("test.h5");
    let file = File::create(&path).unwrap();
    func(file)
}

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
}

/// Dataset configuration for testing - enables fluent builder pattern
/// for common dataset configurations.
pub struct DatasetConfig {
    pub chunk_size: Option<usize>,
    pub compress: bool,
    pub shuffle: bool,
    pub fill_value: Option<i32>,
    pub max_dims: Option<Vec<usize>>,
    pub packed: bool,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            chunk_size: None,
            compress: false,
            shuffle: false,
            fill_value: None,
            max_dims: None,
            packed: false,
        }
    }
}

impl DatasetConfig {
    pub fn chunked(mut self, size: usize) -> Self {
        self.chunk_size = Some(size);
        self
    }

    pub fn compressed(mut self) -> Self {
        self.compress = true;
        self
    }

    pub fn shuffled(mut self) -> Self {
        self.shuffle = true;
        self
    }

    pub fn with_fill(mut self, value: i32) -> Self {
        self.fill_value = Some(value);
        self
    }

    pub fn with_maxdims(mut self, dims: Vec<usize>) -> Self {
        self.max_dims = Some(dims);
        self
    }

    pub fn packed(mut self) -> Self {
        self.packed = true;
        self
    }

    /// Apply this configuration to a dataset builder, returning the configured builder.
    /// This is a convenience method that applies common configurations.
    pub fn apply_to_builder<T, D>(
        &self,
        mut builder: hdf5::DatasetBuilderEmptyShape,
    ) -> hdf5::DatasetBuilderEmptyShape
    where
        T: hdf5::H5Type,
        D: ndarray::Dimension,
    {
        if let Some(chunk) = self.chunk_size {
            builder = builder.chunk(chunk);
        }
        if self.compress {
            builder = builder.deflate(3);
        }
        if self.shuffle {
            builder = builder.shuffle();
        }
        if let Some(fill) = self.fill_value {
            builder = builder.fill_value(fill);
        }
        builder
    }
}

/// Macro to simplify testing dataset builder configurations.
///
/// # Example
/// ```ignore
/// test_dataset_builder! {
///     name: test_chunked_compressed,
///     config: chunked(10).compressed(),
///     shape: 100,
///     data: TestData::int_1d(100),
///     assert: |ds| {
///         assert!(ds.is_chunked());
///         assert!(!ds.filters().is_empty());
///     }
/// }
/// ```
macro_rules! test_dataset_builder {
    (
        $(#[$meta:meta])*
        $test_name:ident {
            $($config:tt)*
        },
        shape: $shape:expr,
        data: $data:expr,
        $(assert: $assert_block:expr)?
    ) => {
        #[test]
        $(#[$meta])*
        fn $test_name() {
            with_tmp_file(|file| {
                // Get the base builder
                let mut builder = file.new_dataset::<i32>();

                // Apply configuration
                $(
                    apply_config!(&mut builder, $($config)*);
                )*

                // Create the dataset
                let ds = builder.shape($shape).create(stringify!($test_name)).unwrap();

                // Write data
                ds.write(&$data).unwrap();

                // Run assertions
                $($assert_block)?
            })
        }
    };
}

/// Helper macro to apply configuration to a builder.
macro_rules! apply_config {
    (builder: $builder:expr, chunked($size:expr)) => {
        let builder = $builder.chunk($size);
    };
    (builder: $builder:expr, compressed()) => {
        if hdf5::filters::deflate_available() {
            let builder = $builder.deflate(3);
        }
    };
    (builder: $builder:expr, shuffled()) => {
        let builder = $builder.shuffle();
    };
    (builder: $builder:expr, no_chunk()) => {
        let builder = $builder.no_chunk();
    };
    (builder: $builder:expr, packed()) => {
        let builder = $builder.packed(true);
    };
    (builder: $builder:expr, fill($val:expr)) => {
        let builder = $builder.fill_value($val);
    };
}

/// Macro for table-driven dataset testing.
///
/// # Example
/// ```ignore
/// test_dataset_configs! {
///     // shape | chunk | compress | shuffle | expected_is_chunked
///     (100,    None,  false,   false,   false),
///     (100,    Some(10), false, false,   true),
///     (100,    Some(10), true,  false,   true),
/// }
/// ```
macro_rules! test_dataset_configs {
    (
        // shape | chunk | compress | shuffle | expected_is_chunked
        $($shape:expr, $chunk:expr, $compress:expr, $shuffle:expr, $expected_is_chunked:expr),* $(,)?
    ) => {
        $(
            paste::paste! {
                #[test]
                fn [<test_config_ $shape _chunk_ $chunk _comp_ $compress _shuffle_ $shuffle>]() {
                    test_dataset_config_impl($shape, $chunk, $compress, $shuffle, $expected_is_chunked);
                }
            }
        )*
    }
}

/// Helper function for table-driven dataset configuration tests.
#[allow(dead_code, clippy::too_many_arguments)]
fn test_dataset_config_impl(
    shape: usize,
    chunk: Option<usize>,
    _compress: bool,
    _shuffle: bool,
    expected_is_chunked: bool,
) {
    with_tmp_file(|file| {
        let mut builder = file.new_dataset::<i32>();
        if let Some(chunk_size) = chunk {
            builder = builder.chunk(chunk_size);
        }

        let ds = builder.shape(shape).create("test_ds").unwrap();
        assert_eq!(ds.is_chunked(), expected_is_chunked);

        // Verify data round-trips
        let data = TestData::int_1d(shape);
        ds.write(&data).unwrap();
        let read_data: Vec<i32> = ds.read_raw().unwrap();
        assert_eq!(read_data, data);
    })
}

/// Property-based testing helper for dataset operations.
///
/// Tests that an operation preserves data integrity for various shapes.
pub struct PropertyTester {
    pub shapes: Vec<Vec<usize>>,
    pub test_data: Vec<Vec<i32>>,
}

impl Default for PropertyTester {
    fn default() -> Self {
        Self {
            shapes: vec![
                vec![10],
                vec![10, 10],
                vec![5, 5, 5],
                vec![2, 3, 4, 5],
            ],
            test_data: vec![
                (0..10).collect(),
                (0..50).collect(),
                (0..125).collect(),
                (0..120).collect(),
            ],
        }
    }
}

impl PropertyTester {
    /// Test that data round-trips correctly for all configured shapes.
    pub fn test_roundtrip<F>(&self, mut dataset_builder: F) -> Result<()>
    where
        F: FnMut(&File, usize, &[usize]) -> Result<hdf5::Dataset>,
    {
        for shape in &self.shapes {
            let size: usize = shape.iter().product();
            let data: Vec<i32> = (0..size as i32).collect();

            // Create test file
            let file = hdf5::File::create(
                std::env::temp_dir().join(format!("test_roundtrip_{}.h5", size))
                    .to_str()
                    .unwrap(),
            )
            .unwrap();

            let ds = dataset_builder(&file, size, shape)?;

            ds.write(&data).unwrap();
            let read_data: Vec<i32> = ds.read_raw().unwrap();
            assert_eq!(read_data, data, "Round-trip failed for shape {:?}", shape);

            // Clean up
            std::fs::remove_file(file.filename()).unwrap();
        }
        Ok(())
    }

    /// Test that datasets can be created and queried for various shapes.
    pub fn test_shape_queries<F>(&self, mut dataset_builder: F)
    where
        F: FnMut(&File, &[usize]) -> Result<hdf5::Dataset>,
    {
        for shape in &self.shapes {
            let shape_str: Vec<String> = shape.iter().map(|n| n.to_string()).collect();
            let file = hdf5::File::create(
                std::env::temp_dir()
                    .join(format!("test_shape_{}.h5", shape_str.join("_")))
                    .to_str()
                    .unwrap(),
            )
            .unwrap();

            let ds = dataset_builder(&file, shape).unwrap();

            assert_eq!(ds.shape(), *shape);
            assert_eq!(ds.size(), shape.iter().product());
            assert_eq!(ds.ndim(), shape.len());

            std::fs::remove_file(file.filename()).unwrap();
        }
    }
}

/// Helper to test dataset property list operations.
pub struct PLTestHelper {
    file: File,
}

impl PLTestHelper {
    pub fn new() -> Result<Self> {
        Ok(Self {
            file: File::create(
                std::env::temp_dir().join("test_pl.h5").to_str().unwrap(),
            )?,
        })
    }

    /// Get or create the test group
    pub fn group(&self, name: &str) -> Result<Group> {
        if self.file.group(name).is_ok() {
            self.file.group(name)
        } else {
            self.file.create_group(name)
        }
    }

    /// Clean up the test file
    pub fn cleanup(self) {
        let _ = std::fs::remove_file(self.file.filename());
    }
}

impl Deref for PLTestHelper {
    type Target = File;

    fn deref(&self) -> &Self::Target {
        &self.file
    }
}

/// Macro to test multiple dataset builder methods in a single test.
/// The macro takes the test name, a list of method calls with their arguments,
/// and a list of assertions to run after creating the dataset.
///
/// # Example
/// ```ignore
/// test_builder_methods! {
///     test_chunked_compressed,
///     methods: {
///         chunk(10),
///         deflate(3),
///     },
///     asserts: {
///         assert!(ds.is_chunked()),
///         assert!(!ds.filters().is_empty()),
///     }
/// }
/// ```
macro_rules! test_builder_methods {
    (
        $test_name:ident,
        methods: {
            $(
                $method:ident $(($($arg:expr),*))?
            ),+
        },
        asserts: {
            $($assert:expr),*
        }
    ) => {
        #[test]
        fn $test_name() {
            with_tmp_file(|file| {
                let builder = file.new_dataset::<i32>();
                let ds = builder
                    $(
                        .$method($($($arg),*)?)
                    )+
                    .shape(100)
                    .create(stringify!($test_name))
                    .unwrap();

                // Write and verify data round-trips
                let data: Vec<i32> = (0..100).collect();
                ds.write(&data).unwrap();
                let read_data: Vec<i32> = ds.read_raw().unwrap();
                assert_eq!(read_data, data);

                // Run assertions
                $($assert);*
            })
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_test_data() {
        // Verify TestData generates correct data
        assert_eq!(TestData::int_1d(5), vec![0, 1, 2, 3, 4]);
        assert_eq!(TestData::constant_i32(42, 3), vec![42, 42, 42]);
    }

    #[test]
    fn test_dataset_config_builder() {
        let config = DatasetConfig::default()
            .chunked(10)
            .compressed();

        assert_eq!(config.chunk_size, Some(10));
        assert!(config.compress);
    }
}
