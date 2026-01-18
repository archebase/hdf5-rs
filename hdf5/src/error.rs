use std::convert::Infallible;
use std::error::Error as StdError;
use std::fmt;
use std::io;
use std::ops::Deref;
use std::panic;
use std::ptr::{self, addr_of_mut};

use ndarray::ShapeError;

#[cfg(not(feature = "1.10.0"))]
use hdf5_sys::h5::hssize_t;
use hdf5_sys::h5e::{
    H5E_auto2_t, H5E_error2_t, H5Eget_current_stack, H5Eget_msg, H5Eprint2, H5Eset_auto2, H5Ewalk2,
    H5E_DEFAULT, H5E_WALK_DOWNWARD,
};

use crate::internal_prelude::*;

// =============================================================================
// Error Category and Code System (Robocodec-style)
// =============================================================================

/// Error categories for HDF5 operations with numeric codes.
///
/// Each category has a unique prefix for error codes:
/// - File: 1000-1999
/// - Dataset: 2000-2999
/// - Attribute: 3000-3999
/// - Group: 4000-4999
/// - Datatype: 5000-5999
/// - Dataspace: 6000-6999
/// - Handle: 7000-7999
/// - Filter: 8000-8999
/// - Internal: 9000-9999
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H5ErrorCategory {
    File = 1000,
    Dataset = 2000,
    Attribute = 3000,
    Group = 4000,
    Datatype = 5000,
    Dataspace = 6000,
    Handle = 7000,
    Filter = 8000,
    Internal = 9000,
}

impl H5ErrorCategory {
    /// Returns the numeric prefix for this error category.
    #[inline]
    pub const fn code_prefix(&self) -> u16 {
        *self as u16
    }

    /// Returns the string representation of this category.
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::File => "FILE",
            Self::Dataset => "DATASET",
            Self::Attribute => "ATTRIBUTE",
            Self::Group => "GROUP",
            Self::Datatype => "DATATYPE",
            Self::Dataspace => "DATASPACE",
            Self::Handle => "HANDLE",
            Self::Filter => "FILTER",
            Self::Internal => "INTERNAL",
        }
    }

    /// Creates a category from a numeric code.
    #[inline]
    pub const fn from_code(code: u16) -> Option<Self> {
        match code {
            1000..=1999 => Some(Self::File),
            2000..=2999 => Some(Self::Dataset),
            3000..=3999 => Some(Self::Attribute),
            4000..=4999 => Some(Self::Group),
            5000..=5999 => Some(Self::Datatype),
            6000..=6999 => Some(Self::Dataspace),
            7000..=7999 => Some(Self::Handle),
            8000..=8999 => Some(Self::Filter),
            9000..=9999 => Some(Self::Internal),
            _ => None,
        }
    }
}

/// The main error type for HDF5 operations with structured error codes.
///
/// Each error variant has a specific numeric code in the format `[CATEGORY-NNNN]`
/// where CATEGORY is the error category and NNNN is the specific error code.
/// This format enables programmatic error handling and logging.
///
/// # Examples
///
/// ```ignore
/// use hdf5::error::{H5Error, H5ErrorCategory};
///
/// let err = H5Error::file_not_found("/path/to/file.h5");
/// assert_eq!(err.category(), H5ErrorCategory::File);
/// assert_eq!(err.code(), 1001);
/// ```
#[derive(Clone)]
pub enum H5Error {
    // ========================================================================
    // File errors (1000-1999)
    // ========================================================================

    /// File not found at the specified path.
    FileNotFound { path: String }, // 1001

    /// Error opening file.
    FileOpenError { path: String, reason: String }, // 1002

    /// Error creating file.
    FileCreateError { path: String, reason: String }, // 1003

    /// I/O error during file operations.
    FileIOError { path: String, operation: String }, // 1004

    /// Invalid file access mode.
    InvalidAccessMode { mode: String }, // 1005

    // ========================================================================
    // Dataset errors (2000-2999)
    // ========================================================================

    /// Dataset not found.
    DatasetNotFound { name: String }, // 2001

    /// Dataset shape mismatch.
    DatasetShapeMismatch { expected: Vec<usize>, found: Vec<usize> }, // 2002

    /// Dataset space allocation failed.
    DatasetSpaceError { reason: String }, // 2003

    /// Chunk size configuration error.
    ChunkSizeError { reason: String }, // 2004

    /// Dataset read error.
    DatasetReadError { name: String, reason: String }, // 2005

    /// Dataset write error.
    DatasetWriteError { name: String, reason: String }, // 2006

    // ========================================================================
    // Attribute errors (3000-3999)
    // ========================================================================

    /// Attribute not found.
    AttributeNotFound { name: String }, // 3001

    /// Attribute read error.
    AttributeReadError { name: String, reason: String }, // 3002

    /// Attribute write error.
    AttributeWriteError { name: String, reason: String }, // 3003

    /// Attribute delete error.
    AttributeDeleteError { name: String, reason: String }, // 3004

    // ========================================================================
    // Group errors (4000-4999)
    // ========================================================================

    /// Group not found.
    GroupNotFound { path: String }, // 4001

    /// Group creation failed.
    GroupCreateError { path: String, reason: String }, // 4002

    /// Invalid group path.
    InvalidGroupPath { path: String, reason: String }, // 4003

    /// Group iteration error.
    GroupIterationError { reason: String }, // 4004

    // ========================================================================
    // Datatype errors (5000-5999)
    // ========================================================================

    /// Type conversion failed.
    TypeConversionError { from: String, to: String }, // 5001

    /// Invalid datatype for operation.
    InvalidDatatype { datatype: String, operation: String }, // 5002

    /// Datatype creation failed.
    DatatypeCreateError { reason: String }, // 5003

    // ========================================================================
    // Dataspace errors (6000-6999)
    // ========================================================================

    /// Dataspace selection error.
    SelectionError { reason: String }, // 6001

    /// Invalid hyperslab selection.
    HyperslabError { reason: String }, // 6002

    /// Point selection error.
    PointSelectionError { reason: String }, // 6003

    /// Dimension out of bounds.
    DimensionBoundsError { index: usize, max: usize, dim_name: Option<String> }, // 6004

    /// Dimension size overflow.
    DimensionOverflow { dims: Vec<usize> }, // 6005

    // ========================================================================
    // Handle errors (7000-7999)
    // ========================================================================

    /// Invalid handle identifier.
    InvalidHandle { handle_id: hid_t, description: String }, // 7001

    /// Handle already closed.
    HandleClosed { handle_type: String }, // 7002

    /// Handle creation failed.
    HandleCreateError { object_type: String, reason: String }, // 7003

    // ========================================================================
    // Filter errors (8000-8999)
    // ========================================================================

    /// Filter not available.
    FilterNotAvailable { filter_name: String }, // 8001

    /// Filter registration failed.
    FilterRegistrationError { filter_name: String, reason: String }, // 8002

    /// Filter configuration error.
    FilterConfigError { filter_name: String, reason: String }, // 8003

    /// Compression/decompression error.
    FilterOperationError { filter_name: String, operation: String }, // 8004

    // ========================================================================
    // Internal errors (9000-9999)
    // ========================================================================

    /// HDF5 C library error with error stack.
    HDF5Error(ErrorStack), // 9001

    /// Generic internal error.
    Internal { message: String }, // 9002

    /// Feature not implemented.
    NotImplemented { feature: String }, // 9003

    /// Invalid argument provided.
    InvalidArgument { arg_name: String, reason: String }, // 9004

    /// Out of memory.
    OutOfMemory, // 9005
}

impl H5Error {
    // ========================================================================
    // Builder methods for each error category
    // ========================================================================

    // File errors
    pub fn file_not_found(path: impl Into<String>) -> Self {
        Self::FileNotFound { path: path.into() }
    }

    pub fn file_open_error(path: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::FileOpenError { path: path.into(), reason: reason.into() }
    }

    pub fn file_create_error(path: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::FileCreateError { path: path.into(), reason: reason.into() }
    }

    pub fn file_io_error(path: impl Into<String>, operation: impl Into<String>) -> Self {
        Self::FileIOError { path: path.into(), operation: operation.into() }
    }

    pub fn invalid_access_mode(mode: impl Into<String>) -> Self {
        Self::InvalidAccessMode { mode: mode.into() }
    }

    // Dataset errors
    pub fn dataset_not_found(name: impl Into<String>) -> Self {
        Self::DatasetNotFound { name: name.into() }
    }

    pub fn dataset_shape_mismatch(expected: Vec<usize>, found: Vec<usize>) -> Self {
        Self::DatasetShapeMismatch { expected, found }
    }

    pub fn dataset_space_error(reason: impl Into<String>) -> Self {
        Self::DatasetSpaceError { reason: reason.into() }
    }

    pub fn chunk_size_error(reason: impl Into<String>) -> Self {
        Self::ChunkSizeError { reason: reason.into() }
    }

    pub fn dataset_read_error(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::DatasetReadError { name: name.into(), reason: reason.into() }
    }

    pub fn dataset_write_error(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::DatasetWriteError { name: name.into(), reason: reason.into() }
    }

    // Attribute errors
    pub fn attribute_not_found(name: impl Into<String>) -> Self {
        Self::AttributeNotFound { name: name.into() }
    }

    pub fn attribute_read_error(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::AttributeReadError { name: name.into(), reason: reason.into() }
    }

    pub fn attribute_write_error(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::AttributeWriteError { name: name.into(), reason: reason.into() }
    }

    pub fn attribute_delete_error(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::AttributeDeleteError { name: name.into(), reason: reason.into() }
    }

    // Group errors
    pub fn group_not_found(path: impl Into<String>) -> Self {
        Self::GroupNotFound { path: path.into() }
    }

    pub fn group_create_error(path: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::GroupCreateError { path: path.into(), reason: reason.into() }
    }

    pub fn invalid_group_path(path: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidGroupPath { path: path.into(), reason: reason.into() }
    }

    pub fn group_iteration_error(reason: impl Into<String>) -> Self {
        Self::GroupIterationError { reason: reason.into() }
    }

    // Datatype errors
    pub fn type_conversion_error(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self::TypeConversionError { from: from.into(), to: to.into() }
    }

    pub fn invalid_datatype(datatype: impl Into<String>, operation: impl Into<String>) -> Self {
        Self::InvalidDatatype { datatype: datatype.into(), operation: operation.into() }
    }

    pub fn datatype_create_error(reason: impl Into<String>) -> Self {
        Self::DatatypeCreateError { reason: reason.into() }
    }

    // Dataspace errors
    pub fn selection_error(reason: impl Into<String>) -> Self {
        Self::SelectionError { reason: reason.into() }
    }

    pub fn hyperslab_error(reason: impl Into<String>) -> Self {
        Self::HyperslabError { reason: reason.into() }
    }

    pub fn point_selection_error(reason: impl Into<String>) -> Self {
        Self::PointSelectionError { reason: reason.into() }
    }

    pub fn dimension_bounds_error(index: usize, max: usize, dim_name: Option<String>) -> Self {
        Self::DimensionBoundsError { index, max, dim_name }
    }

    pub fn dimension_overflow(dims: Vec<usize>) -> Self {
        Self::DimensionOverflow { dims }
    }

    // Handle errors
    pub fn invalid_handle(handle_id: hid_t, description: impl Into<String>) -> Self {
        Self::InvalidHandle { handle_id, description: description.into() }
    }

    pub fn handle_closed(handle_type: impl Into<String>) -> Self {
        Self::HandleClosed { handle_type: handle_type.into() }
    }

    pub fn handle_create_error(object_type: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::HandleCreateError { object_type: object_type.into(), reason: reason.into() }
    }

    // Filter errors
    pub fn filter_not_available(filter_name: impl Into<String>) -> Self {
        Self::FilterNotAvailable { filter_name: filter_name.into() }
    }

    pub fn filter_registration_error(filter_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::FilterRegistrationError { filter_name: filter_name.into(), reason: reason.into() }
    }

    pub fn filter_config_error(filter_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::FilterConfigError { filter_name: filter_name.into(), reason: reason.into() }
    }

    pub fn filter_operation_error(filter_name: impl Into<String>, operation: impl Into<String>) -> Self {
        Self::FilterOperationError { filter_name: filter_name.into(), operation: operation.into() }
    }

    // Internal errors
    pub fn hdf5_error(stack: ErrorStack) -> Self {
        Self::HDF5Error(stack)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal { message: message.into() }
    }

    pub fn not_implemented(feature: impl Into<String>) -> Self {
        Self::NotImplemented { feature: feature.into() }
    }

    pub fn invalid_argument(arg_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidArgument { arg_name: arg_name.into(), reason: reason.into() }
    }

    pub const fn out_of_memory() -> Self {
        Self::OutOfMemory
    }

    // ========================================================================
    // Query methods
    // ========================================================================

    /// Returns the error category for this error.
    pub fn category(&self) -> H5ErrorCategory {
        match self {
            Self::FileNotFound { .. }
            | Self::FileOpenError { .. }
            | Self::FileCreateError { .. }
            | Self::FileIOError { .. }
            | Self::InvalidAccessMode { .. } => H5ErrorCategory::File,

            Self::DatasetNotFound { .. }
            | Self::DatasetShapeMismatch { .. }
            | Self::DatasetSpaceError { .. }
            | Self::ChunkSizeError { .. }
            | Self::DatasetReadError { .. }
            | Self::DatasetWriteError { .. } => H5ErrorCategory::Dataset,

            Self::AttributeNotFound { .. }
            | Self::AttributeReadError { .. }
            | Self::AttributeWriteError { .. }
            | Self::AttributeDeleteError { .. } => H5ErrorCategory::Attribute,

            Self::GroupNotFound { .. }
            | Self::GroupCreateError { .. }
            | Self::InvalidGroupPath { .. }
            | Self::GroupIterationError { .. } => H5ErrorCategory::Group,

            Self::TypeConversionError { .. }
            | Self::InvalidDatatype { .. }
            | Self::DatatypeCreateError { .. } => H5ErrorCategory::Datatype,

            Self::SelectionError { .. }
            | Self::HyperslabError { .. }
            | Self::PointSelectionError { .. }
            | Self::DimensionBoundsError { .. }
            | Self::DimensionOverflow { .. } => H5ErrorCategory::Dataspace,

            Self::InvalidHandle { .. }
            | Self::HandleClosed { .. }
            | Self::HandleCreateError { .. } => H5ErrorCategory::Handle,

            Self::FilterNotAvailable { .. }
            | Self::FilterRegistrationError { .. }
            | Self::FilterConfigError { .. }
            | Self::FilterOperationError { .. } => H5ErrorCategory::Filter,

            Self::HDF5Error(_) | Self::Internal { .. } | Self::NotImplemented { .. }
            | Self::InvalidArgument { .. } | Self::OutOfMemory => H5ErrorCategory::Internal,
        }
    }

    /// Returns the specific error code within its category.
    pub fn code(&self) -> u16 {
        let base = self.category().code_prefix();
        let offset = match self {
            Self::FileNotFound { .. } => 1,
            Self::FileOpenError { .. } => 2,
            Self::FileCreateError { .. } => 3,
            Self::FileIOError { .. } => 4,
            Self::InvalidAccessMode { .. } => 5,

            Self::DatasetNotFound { .. } => 1,
            Self::DatasetShapeMismatch { .. } => 2,
            Self::DatasetSpaceError { .. } => 3,
            Self::ChunkSizeError { .. } => 4,
            Self::DatasetReadError { .. } => 5,
            Self::DatasetWriteError { .. } => 6,

            Self::AttributeNotFound { .. } => 1,
            Self::AttributeReadError { .. } => 2,
            Self::AttributeWriteError { .. } => 3,
            Self::AttributeDeleteError { .. } => 4,

            Self::GroupNotFound { .. } => 1,
            Self::GroupCreateError { .. } => 2,
            Self::InvalidGroupPath { .. } => 3,
            Self::GroupIterationError { .. } => 4,

            Self::TypeConversionError { .. } => 1,
            Self::InvalidDatatype { .. } => 2,
            Self::DatatypeCreateError { .. } => 3,

            Self::SelectionError { .. } => 1,
            Self::HyperslabError { .. } => 2,
            Self::PointSelectionError { .. } => 3,
            Self::DimensionBoundsError { .. } => 4,
            Self::DimensionOverflow { .. } => 5,

            Self::InvalidHandle { .. } => 1,
            Self::HandleClosed { .. } => 2,
            Self::HandleCreateError { .. } => 3,

            Self::FilterNotAvailable { .. } => 1,
            Self::FilterRegistrationError { .. } => 2,
            Self::FilterConfigError { .. } => 3,
            Self::FilterOperationError { .. } => 4,

            Self::HDF5Error(_) => 1,
            Self::Internal { .. } => 2,
            Self::NotImplemented { .. } => 3,
            Self::InvalidArgument { .. } => 4,
            Self::OutOfMemory => 5,
        };
        base + offset
    }

    /// Returns structured fields for logging.
    pub fn log_fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::FileNotFound { path } => vec![("path", path.clone())],
            Self::FileOpenError { path, reason } => vec![("path", path.clone()), ("reason", reason.clone())],
            Self::FileCreateError { path, reason } => vec![("path", path.clone()), ("reason", reason.clone())],
            Self::FileIOError { path, operation } => vec![("path", path.clone()), ("operation", operation.clone())],
            Self::InvalidAccessMode { mode } => vec![("mode", mode.clone())],

            Self::DatasetNotFound { name } => vec![("name", name.clone())],
            Self::DatasetShapeMismatch { expected, found } => {
                vec![("expected", format!("{expected:?}")), ("found", format!("{found:?}"))]
            }
            Self::DatasetSpaceError { reason } => vec![("reason", reason.clone())],
            Self::ChunkSizeError { reason } => vec![("reason", reason.clone())],
            Self::DatasetReadError { name, reason } => vec![("name", name.clone()), ("reason", reason.clone())],
            Self::DatasetWriteError { name, reason } => vec![("name", name.clone()), ("reason", reason.clone())],

            Self::AttributeNotFound { name } => vec![("name", name.clone())],
            Self::AttributeReadError { name, reason } => vec![("name", name.clone()), ("reason", reason.clone())],
            Self::AttributeWriteError { name, reason } => vec![("name", name.clone()), ("reason", reason.clone())],
            Self::AttributeDeleteError { name, reason } => vec![("name", name.clone()), ("reason", reason.clone())],

            Self::GroupNotFound { path } => vec![("path", path.clone())],
            Self::GroupCreateError { path, reason } => vec![("path", path.clone()), ("reason", reason.clone())],
            Self::InvalidGroupPath { path, reason } => vec![("path", path.clone()), ("reason", reason.clone())],
            Self::GroupIterationError { reason } => vec![("reason", reason.clone())],

            Self::TypeConversionError { from, to } => vec![("from", from.clone()), ("to", to.clone())],
            Self::InvalidDatatype { datatype, operation } => {
                vec![("datatype", datatype.clone()), ("operation", operation.clone())]
            }
            Self::DatatypeCreateError { reason } => vec![("reason", reason.clone())],

            Self::SelectionError { reason } => vec![("reason", reason.clone())],
            Self::HyperslabError { reason } => vec![("reason", reason.clone())],
            Self::PointSelectionError { reason } => vec![("reason", reason.clone())],
            Self::DimensionBoundsError { index, max, dim_name } => {
                let mut fields = vec![("index", index.to_string()), ("max", max.to_string())];
                if let Some(name) = dim_name {
                    fields.push(("dim_name", name.clone()));
                }
                fields
            }
            Self::DimensionOverflow { dims } => vec![("dims", format!("{dims:?}"))],

            Self::InvalidHandle { handle_id, description } => {
                vec![("handle_id", handle_id.to_string()), ("description", description.clone())]
            }
            Self::HandleClosed { handle_type } => vec![("handle_type", handle_type.clone())],
            Self::HandleCreateError { object_type, reason } => {
                vec![("object_type", object_type.clone()), ("reason", reason.clone())]
            }

            Self::FilterNotAvailable { filter_name } => vec![("filter_name", filter_name.clone())],
            Self::FilterRegistrationError { filter_name, reason } => {
                vec![("filter_name", filter_name.clone()), ("reason", reason.clone())]
            }
            Self::FilterConfigError { filter_name, reason } => {
                vec![("filter_name", filter_name.clone()), ("reason", reason.clone())]
            }
            Self::FilterOperationError { filter_name, operation } => {
                vec![("filter_name", filter_name.clone()), ("operation", operation.clone())]
            }

            Self::HDF5Error(_) => vec![("source", "hdf5_c_library".to_string())],
            Self::Internal { message } => vec![("message", message.clone())],
            Self::NotImplemented { feature } => vec![("feature", feature.clone())],
            Self::InvalidArgument { arg_name, reason } => {
                vec![("arg_name", arg_name.clone()), ("reason", reason.clone())]
            }
            Self::OutOfMemory => vec![],
        }
    }

    /// Obtain the current HDF5 error stack and wrap it in an H5Error.
    pub fn query() -> Self {
        if let Ok(stack) = ErrorStack::from_current() {
            Self::hdf5_error(stack)
        } else {
            Self::internal("Could not get error stack")
        }
    }
}

impl fmt::Debug for H5Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let category = self.category();
        let code = self.code();
        write!(f, "[{}-{:04}] ", category.as_str(), code)?;
        match self {
            Self::FileNotFound { path } => write!(f, "File not found: {}", path),
            Self::FileOpenError { path, reason } => write!(f, "Failed to open file '{}': {}", path, reason),
            Self::FileCreateError { path, reason } => {
                write!(f, "Failed to create file '{}': {}", path, reason)
            }
            Self::FileIOError { path, operation } => {
                write!(f, "I/O error during {} on file '{}'", operation, path)
            }
            Self::InvalidAccessMode { mode } => write!(f, "Invalid access mode: {}", mode),

            Self::DatasetNotFound { name } => write!(f, "Dataset not found: {}", name),
            Self::DatasetShapeMismatch { expected, found } => {
                write!(f, "Dataset shape mismatch: expected {:?}, found {:?}", expected, found)
            }
            Self::DatasetSpaceError { reason } => write!(f, "Dataset space error: {}", reason),
            Self::ChunkSizeError { reason } => write!(f, "Chunk size error: {}", reason),
            Self::DatasetReadError { name, reason } => {
                write!(f, "Failed to read dataset '{}': {}", name, reason)
            }
            Self::DatasetWriteError { name, reason } => {
                write!(f, "Failed to write dataset '{}': {}", name, reason)
            }

            Self::AttributeNotFound { name } => write!(f, "Attribute not found: {}", name),
            Self::AttributeReadError { name, reason } => {
                write!(f, "Failed to read attribute '{}': {}", name, reason)
            }
            Self::AttributeWriteError { name, reason } => {
                write!(f, "Failed to write attribute '{}': {}", name, reason)
            }
            Self::AttributeDeleteError { name, reason } => {
                write!(f, "Failed to delete attribute '{}': {}", name, reason)
            }

            Self::GroupNotFound { path } => write!(f, "Group not found: {}", path),
            Self::GroupCreateError { path, reason } => {
                write!(f, "Failed to create group '{}': {}", path, reason)
            }
            Self::InvalidGroupPath { path, reason } => {
                write!(f, "Invalid group path '{}': {}", path, reason)
            }
            Self::GroupIterationError { reason } => write!(f, "Group iteration error: {}", reason),

            Self::TypeConversionError { from, to } => {
                write!(f, "Type conversion error: cannot convert {} to {}", from, to)
            }
            Self::InvalidDatatype { datatype, operation } => {
                write!(f, "Invalid datatype '{}' for operation {}", datatype, operation)
            }
            Self::DatatypeCreateError { reason } => write!(f, "Datatype creation error: {}", reason),

            Self::SelectionError { reason } => write!(f, "Selection error: {}", reason),
            Self::HyperslabError { reason } => write!(f, "Hyperslab error: {}", reason),
            Self::PointSelectionError { reason } => write!(f, "Point selection error: {}", reason),
            Self::DimensionBoundsError { index, max, dim_name } => {
                if let Some(name) = dim_name {
                    write!(f, "Dimension '{}' bounds error: index {} out of bounds for max {}", name, index, max)
                } else {
                    write!(f, "Dimension bounds error: index {} out of bounds for max {}", index, max)
                }
            }
            Self::DimensionOverflow { dims } => {
                write!(f, "Dimension overflow: size calculation overflowed for dims {:?}", dims)
            }

            Self::InvalidHandle { handle_id, description } => {
                write!(f, "Invalid handle {}: {}", handle_id, description)
            }
            Self::HandleClosed { handle_type } => {
                write!(f, "Handle already closed: {}", handle_type)
            }
            Self::HandleCreateError { object_type, reason } => {
                write!(f, "Failed to create {} handle: {}", object_type, reason)
            }

            Self::FilterNotAvailable { filter_name } => {
                write!(f, "Filter not available: {}", filter_name)
            }
            Self::FilterRegistrationError { filter_name, reason } => {
                write!(f, "Failed to register filter '{}': {}", filter_name, reason)
            }
            Self::FilterConfigError { filter_name, reason } => {
                write!(f, "Filter configuration error for '{}': {}", filter_name, reason)
            }
            Self::FilterOperationError { filter_name, operation } => {
                write!(f, "Filter operation '{}' failed for {}", operation, filter_name)
            }

            Self::HDF5Error(stack) => match stack.clone().expand() {
                Ok(stack) => f.write_str(stack.description()),
                Err(_) => f.write_str("HDF5 C library error (unable to expand stack)"),
            },
            Self::Internal { message } => write!(f, "Internal error: {}", message),
            Self::NotImplemented { feature } => write!(f, "Feature not implemented: {}", feature),
            Self::InvalidArgument { arg_name, reason } => {
                write!(f, "Invalid argument '{}': {}", arg_name, reason)
            }
            Self::OutOfMemory => f.write_str("Out of memory"),
        }
    }
}

impl fmt::Display for H5Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl StdError for H5Error {}

// =============================================================================
// Logging macro for structured error logging
// =============================================================================

/// Macro for structured error logging with tracing.
///
/// # Examples
///
/// ```ignore
/// use hdf5::error::{H5Error, log_error};
///
/// let err = H5Error::file_not_found("/path/to/file.h5");
/// log_error!(&err);
/// ```
#[macro_export]
macro_rules! log_error {
    ($error:expr) => {
        ::tracing::error!(
            error_code = $error.code(),
            error_category = $error.category().as_str(),
            { ::tracing::field::debug($error.log_fields()) },
            "{}", $error
        )
    };
}

/// Macro for structured warning logging with tracing.
#[macro_export]
macro_rules! log_warn {
    ($error:expr) => {
        ::tracing::warn!(
            error_code = $error.code(),
            error_category = $error.category().as_str(),
            { ::tracing::field::debug($error.log_fields()) },
            "{}", $error
        )
    };
}

// =============================================================================
// Legacy compatibility type alias
// =============================================================================

/// Legacy type alias for backward compatibility.
/// Use `H5Error` for new code.
pub type Error = H5Error;

/// A type for results generated by HDF5-related functions where the `Err` type is
/// set to `hdf5::H5Error`.
pub type Result<T, E = H5Error> = ::std::result::Result<T, E>;

// =============================================================================
// Conversions from common types
// =============================================================================

impl From<&str> for H5Error {
    fn from(desc: &str) -> Self {
        Self::internal(desc)
    }
}

impl From<String> for H5Error {
    fn from(desc: String) -> Self {
        Self::internal(desc)
    }
}

impl From<Infallible> for H5Error {
    fn from(_: Infallible) -> Self {
        unreachable!("Infallible error can never be constructed")
    }
}

impl From<ShapeError> for H5Error {
    fn from(err: ShapeError) -> Self {
        Self::internal(format!("shape error: {err}"))
    }
}

impl From<H5Error> for io::Error {
    fn from(err: H5Error) -> Self {
        Self::new(io::ErrorKind::Other, err)
    }
}

// =============================================================================
// Error stack and HDF5 error handling (legacy support)
// =============================================================================

/// Silence errors emitted by `hdf5`
///
/// Safety: This version is not thread-safe and must be synchronized
/// with other calls to `hdf5`
pub(crate) unsafe fn silence_errors_no_sync(silence: bool) {
    // Cast function with different argument types. This is safe because H5Eprint2 is
    // documented to support this interface
    let h5eprint: Option<unsafe extern "C" fn(hid_t, *mut libc::FILE) -> herr_t> =
        Some(H5Eprint2 as _);
    let h5eprint: H5E_auto2_t = std::mem::transmute(h5eprint);
    H5Eset_auto2(H5E_DEFAULT, if silence { None } else { h5eprint }, ptr::null_mut());
}

/// Silence errors emitted by `hdf5`
pub fn silence_errors(silence: bool) {
    h5lock!(silence_errors_no_sync(silence));
}

#[repr(transparent)]
#[derive(Clone)]
pub struct ErrorStack(Handle);

impl ObjectClass for ErrorStack {
    const NAME: &'static str = "errorstack";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_ERROR_STACK];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    fn short_repr(&self) -> Option<String> {
        Some(format!("<error stack id={}>", self.0.id()))
    }
}

impl ErrorStack {
    pub(crate) fn from_current() -> Result<Self> {
        let stack_id = h5lock!(H5Eget_current_stack());
        Handle::try_new(stack_id).map(Self)
    }

    /// Expands the error stack to a format which is easier to handle
    // known HDF5 bug: H5Eget_msg() used in this function may corrupt
    // the current stack, so we use self over &self
    pub fn expand(self) -> Result<ExpandedErrorStack> {
        struct CallbackData {
            stack: ExpandedErrorStack,
            err: Option<H5Error>,
        }
        unsafe extern "C" fn callback(
            _: c_uint, err_desc: *const H5E_error2_t, data: *mut c_void,
        ) -> herr_t {
            panic::catch_unwind(|| unsafe {
                let data = &mut *(data.cast::<CallbackData>());
                if data.err.is_some() {
                    return 0;
                }
                let closure = |e: H5E_error2_t| -> Result<ErrorFrame> {
                    let (desc, func) = (string_from_cstr(e.desc), string_from_cstr(e.func_name));
                    let major = get_h5_str(|m, s| H5Eget_msg(e.maj_num, ptr::null_mut(), m, s))?;
                    let minor = get_h5_str(|m, s| H5Eget_msg(e.min_num, ptr::null_mut(), m, s))?;
                    Ok(ErrorFrame::new(&desc, &func, &major, &minor))
                };
                match closure(*err_desc) {
                    Ok(frame) => {
                        data.stack.push(frame);
                    }
                    Err(err) => {
                        data.err = Some(err);
                    }
                }
                0
            })
            .unwrap_or_else(|_| {
                // Log the panic for debugging purposes before returning error code
                ::tracing::error!("Panic in HDF5 error stack expansion callback");
                -1
            })
        }

        let mut data = CallbackData { stack: ExpandedErrorStack::new(), err: None };
        let data_ptr: *mut c_void = addr_of_mut!(data).cast::<c_void>();

        let stack_id = self.handle().id();
        h5lock!({
            H5Ewalk2(stack_id, H5E_WALK_DOWNWARD, Some(callback), data_ptr);
        });

        data.err.map_or(Ok(data.stack), Err)
    }
}

#[derive(Clone, Debug)]
pub struct ErrorFrame {
    desc: String,
    func: String,
    major: String,
    minor: String,
    description: String,
}

impl ErrorFrame {
    pub(crate) fn new(desc: &str, func: &str, major: &str, minor: &str) -> Self {
        Self {
            desc: desc.into(),
            func: func.into(),
            major: major.into(),
            minor: minor.into(),
            description: format!("{func}(): {desc}"),
        }
    }

    pub fn desc(&self) -> &str {
        self.desc.as_ref()
    }

    pub fn description(&self) -> &str {
        self.description.as_ref()
    }

    pub fn detail(&self) -> Option<String> {
        Some(format!("Error in {}(): {} [{}: {}]", self.func, self.desc, self.major, self.minor))
    }
}

#[derive(Clone, Debug)]
pub struct ExpandedErrorStack {
    frames: Vec<ErrorFrame>,
    description: Option<String>,
}

impl Deref for ExpandedErrorStack {
    type Target = [ErrorFrame];

    fn deref(&self) -> &Self::Target {
        &self.frames
    }
}

impl Default for ExpandedErrorStack {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpandedErrorStack {
    pub(crate) fn new() -> Self {
        Self { frames: Vec::new(), description: None }
    }

    pub(crate) fn push(&mut self, frame: ErrorFrame) {
        self.frames.push(frame);
        if !self.is_empty() {
            let top_desc = self.frames[0].description().to_owned();
            if self.len() == 1 {
                self.description = Some(top_desc);
            } else {
                self.description =
                    Some(format!("{}: {}", top_desc, self.frames[self.len() - 1].desc()));
            }
        }
    }

    pub fn top(&self) -> Option<&ErrorFrame> {
        self.first()
    }

    pub fn description(&self) -> &str {
        match self.description {
            None => "unknown library error",
            Some(ref desc) => desc.as_ref(),
        }
    }

    pub fn detail(&self) -> Option<String> {
        self.top().and_then(ErrorFrame::detail)
    }
}

// =============================================================================
// Error code checking trait
// =============================================================================

pub fn h5check<T: H5ErrorCode>(value: T) -> Result<T> {
    H5ErrorCode::h5check(value)
}

#[allow(unused)]
pub fn is_err_code<T: H5ErrorCode>(value: T) -> bool {
    H5ErrorCode::is_err_code(value)
}

pub trait H5ErrorCode: Copy {
    fn is_err_code(value: Self) -> bool;

    fn h5check(value: Self) -> Result<Self> {
        if Self::is_err_code(value) {
            Err(H5Error::query())
        } else {
            Ok(value)
        }
    }
}

impl H5ErrorCode for hsize_t {
    fn is_err_code(value: Self) -> bool {
        value == 0
    }
}

impl H5ErrorCode for herr_t {
    fn is_err_code(value: Self) -> bool {
        value < 0
    }
}

#[cfg(feature = "1.10.0")]
impl H5ErrorCode for hid_t {
    fn is_err_code(value: Self) -> bool {
        value < 0
    }
}

#[cfg(not(feature = "1.10.0"))]
impl H5ErrorCode for hssize_t {
    fn is_err_code(value: Self) -> bool {
        value < 0
    }
}

impl H5ErrorCode for libc::ssize_t {
    fn is_err_code(value: Self) -> bool {
        value < 0
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
pub mod tests {
    use hdf5_sys::h5p::{H5Pclose, H5Pcreate};

    use crate::globals::H5P_ROOT;
    use crate::internal_prelude::*;

    use super::{ExpandedErrorStack, H5Error, H5ErrorCategory};

    #[test]
    pub fn test_error_codes() {
        let err = H5Error::file_not_found("/test.h5");
        assert_eq!(err.category(), H5ErrorCategory::File);
        assert_eq!(err.code(), 1001);
        assert_eq!(err.category().as_str(), "FILE");

        let err = H5Error::dataset_not_found("dataset1");
        assert_eq!(err.category(), H5ErrorCategory::Dataset);
        assert_eq!(err.code(), 2001);
        assert_eq!(err.category().as_str(), "DATASET");

        let err = H5Error::attribute_not_found("attr1");
        assert_eq!(err.category(), H5ErrorCategory::Attribute);
        assert_eq!(err.code(), 3001);

        let err = H5Error::group_not_found("/group");
        assert_eq!(err.category(), H5ErrorCategory::Group);
        assert_eq!(err.code(), 4001);

        let err = H5Error::type_conversion_error("int", "float");
        assert_eq!(err.category(), H5ErrorCategory::Datatype);
        assert_eq!(err.code(), 5001);

        let err = H5Error::selection_error("invalid selection");
        assert_eq!(err.category(), H5ErrorCategory::Dataspace);
        assert_eq!(err.code(), 6001);

        let err = H5Error::invalid_handle(123, "test");
        assert_eq!(err.category(), H5ErrorCategory::Handle);
        assert_eq!(err.code(), 7001);

        let err = H5Error::filter_not_available("blosc");
        assert_eq!(err.category(), H5ErrorCategory::Filter);
        assert_eq!(err.code(), 8001);

        let err = H5Error::internal("test error");
        assert_eq!(err.category(), H5ErrorCategory::Internal);
        assert_eq!(err.code(), 9002);
    }

    #[test]
    pub fn test_error_display_format() {
        let err = H5Error::file_not_found("/test.h5");
        let display = format!("{}", err);
        assert!(display.contains("[FILE-1001]"));
        assert!(display.contains("File not found: /test.h5"));

        let err = H5Error::dataset_shape_mismatch(vec![10, 20], vec![5, 20]);
        let display = format!("{}", err);
        assert!(display.contains("[DATASET-2002]"));
        assert!(display.contains("Dataset shape mismatch"));

        let err = H5Error::type_conversion_error("i32", "f32");
        let display = format!("{}", err);
        assert!(display.contains("[DATATYPE-5001]"));
        assert!(display.contains("Type conversion error"));
    }

    #[test]
    pub fn test_log_fields() {
        let err = H5Error::file_not_found("/test.h5");
        let fields = err.log_fields();
        assert_eq!(fields, vec![("path", "/test.h5".to_string())]);

        let err = H5Error::dataset_shape_mismatch(vec![10, 20], vec![5, 20]);
        let fields = err.log_fields();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].0, "expected");

        let err = H5Error::dimension_bounds_error(5, 10, Some("x".to_string()));
        let fields = err.log_fields();
        assert_eq!(fields.len(), 3);
    }

    #[test]
    pub fn test_error_stack() {
        let stack = h5lock!({
            let plist_id = H5Pcreate(*H5P_ROOT);
            H5Pclose(plist_id);
            H5Error::query()
        });

        let stack = match stack {
            H5Error::HDF5Error(stack) => stack,
            other => panic!("Expected HDF5 error, got {:?}", other),
        }
        .expand()
        .unwrap();
        assert!(stack.is_empty());

        let stack = h5lock!({
            let plist_id = H5Pcreate(*H5P_ROOT);
            H5Pclose(plist_id);
            H5Pclose(plist_id);
            H5Error::query()
        });

        let stack = match stack {
            H5Error::HDF5Error(stack) => stack,
            other => panic!("Expected HDF5 error, got {:?}", other),
        }
        .expand()
        .unwrap();
        assert_eq!(stack.description(), "H5Pclose(): can't close: can't locate ID");
        assert_eq!(
            &stack.detail().unwrap(),
            "Error in H5Pclose(): can't close [Property lists: Unable to free object]"
        );

        assert!(stack.len() >= 2 && stack.len() <= 4);
        assert!(!stack.is_empty());

        assert_eq!(stack[0].description(), "H5Pclose(): can't close");
        assert_eq!(
            &stack[0].detail().unwrap(),
            "Error in H5Pclose(): can't close \
             [Property lists: Unable to free object]"
        );

        #[cfg(not(feature = "1.14.0"))]
        {
            assert_eq!(stack[stack.len() - 1].description(), "H5I_dec_ref(): can't locate ID");
            assert_eq!(
                &stack[stack.len() - 1].detail().unwrap(),
                "Error in H5I_dec_ref(): can't locate ID \
             [Object atom: Unable to find atom information (already closed?)]"
            );
        }
        #[cfg(feature = "1.14.0")]
        {
            assert_eq!(stack[stack.len() - 1].description(), "H5I__dec_ref(): can't locate ID");
            assert_eq!(
                &stack[stack.len() - 1].detail().unwrap(),
                "Error in H5I__dec_ref(): can't locate ID \
             [Object ID: Unable to find ID information (already closed?)]"
            );
        }

        let empty_stack = ExpandedErrorStack::new();
        assert!(empty_stack.is_empty());
        assert_eq!(empty_stack.len(), 0);
    }

    #[test]
    pub fn test_h5call() {
        let result_no_error = h5call!({
            let plist_id = H5Pcreate(*H5P_ROOT);
            H5Pclose(plist_id)
        });
        assert!(result_no_error.is_ok());

        let result_error = h5call!({
            let plist_id = H5Pcreate(*H5P_ROOT);
            H5Pclose(plist_id);
            H5Pclose(plist_id)
        });
        assert!(result_error.is_err());
    }

    #[test]
    pub fn test_h5try() {
        fn f1() -> Result<herr_t> {
            h5try!(H5Pcreate(*H5P_ROOT));
            Ok(100)
        }

        assert_eq!(f1().unwrap(), 100);

        fn f2() -> Result<herr_t> {
            h5try!(H5Pcreate(123456));
            Ok(100)
        }

        assert!(f2().is_err());
    }

    #[test]
    pub fn test_category_from_code() {
        assert_eq!(
            H5ErrorCategory::from_code(1001),
            Some(H5ErrorCategory::File)
        );
        assert_eq!(
            H5ErrorCategory::from_code(2001),
            Some(H5ErrorCategory::Dataset)
        );
        assert_eq!(
            H5ErrorCategory::from_code(9999),
            Some(H5ErrorCategory::Internal)
        );
        assert_eq!(H5ErrorCategory::from_code(10000), None);
        assert_eq!(H5ErrorCategory::from_code(0), None);
    }

    #[test]
    pub fn test_all_error_variants_have_codes() {
        // Test that all error variants produce valid codes
        let err = H5Error::file_not_found("/test.h5");
        assert!(err.code() >= 1000 && err.code() < 2000);

        let err = H5Error::dataset_not_found("ds");
        assert!(err.code() >= 2000 && err.code() < 3000);

        let err = H5Error::attribute_not_found("attr");
        assert!(err.code() >= 3000 && err.code() < 4000);

        let err = H5Error::group_not_found("/g");
        assert!(err.code() >= 4000 && err.code() < 5000);

        let err = H5Error::type_conversion_error("i32", "f32");
        assert!(err.code() >= 5000 && err.code() < 6000);

        let err = H5Error::selection_error("test");
        assert!(err.code() >= 6000 && err.code() < 7000);

        let err = H5Error::invalid_handle(123, "test");
        assert!(err.code() >= 7000 && err.code() < 8000);

        let err = H5Error::filter_not_available("blosc");
        assert!(err.code() >= 8000 && err.code() < 9000);

        let err = H5Error::internal("test");
        assert!(err.code() >= 9000 && err.code() < 10000);
    }

    #[test]
    pub fn test_error_builder_methods() {
        // Test all builder methods work correctly
        let _ = H5Error::file_not_found("/test");
        let _ = H5Error::file_open_error("/test", "reason");
        let _ = H5Error::file_create_error("/test", "reason");
        let _ = H5Error::file_io_error("/test", "read");
        let _ = H5Error::invalid_access_mode("r");

        let _ = H5Error::dataset_not_found("ds");
        let _ = H5Error::dataset_shape_mismatch(vec![1], vec![2]);
        let _ = H5Error::dataset_space_error("reason");
        let _ = H5Error::chunk_size_error("reason");
        let _ = H5Error::dataset_read_error("ds", "reason");
        let _ = H5Error::dataset_write_error("ds", "reason");

        let _ = H5Error::attribute_not_found("attr");
        let _ = H5Error::attribute_read_error("attr", "reason");
        let _ = H5Error::attribute_write_error("attr", "reason");
        let _ = H5Error::attribute_delete_error("attr", "reason");

        let _ = H5Error::group_not_found("/g");
        let _ = H5Error::group_create_error("/g", "reason");
        let _ = H5Error::invalid_group_path("/g", "reason");
        let _ = H5Error::group_iteration_error("reason");

        let _ = H5Error::type_conversion_error("i32", "f32");
        let _ = H5Error::invalid_datatype("int", "convert");
        let _ = H5Error::datatype_create_error("reason");

        let _ = H5Error::selection_error("reason");
        let _ = H5Error::hyperslab_error("reason");
        let _ = H5Error::point_selection_error("reason");
        let _ = H5Error::dimension_bounds_error(1, 10, Some("x".to_string()));
        let _ = H5Error::dimension_overflow(vec![1, 2]);

        let _ = H5Error::invalid_handle(123, "test");
        let _ = H5Error::handle_closed("dataset");
        let _ = H5Error::handle_create_error("dataset", "reason");

        let _ = H5Error::filter_not_available("blosc");
        let _ = H5Error::filter_registration_error("blosc", "reason");
        let _ = H5Error::filter_config_error("blosc", "reason");
        let _ = H5Error::filter_operation_error("blosc", "compress");

        let _ = H5Error::internal("test");
        let _ = H5Error::not_implemented("feature");
        let _ = H5Error::invalid_argument("arg", "reason");
        let _ = H5Error::out_of_memory();
    }

    #[test]
    pub fn test_error_clone() {
        let err1 = H5Error::file_not_found("/test.h5");
        let err2 = err1.clone();
        assert_eq!(err1.code(), err2.code());
        assert_eq!(format!("{:?}", err1), format!("{:?}", err2));
    }

    #[test]
    pub fn test_error_from_string_conversions() {
        let err: H5Error = "test error".into();
        assert!(matches!(err, H5Error::Internal { .. }));

        let err: H5Error = String::from("test error").into();
        assert!(matches!(err, H5Error::Internal { .. }));
    }
}
