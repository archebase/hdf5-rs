# HDF5-Rust

Rust bindings for the HDF5 library, providing safe idiomatic Rust wrappers around low-level FFI bindings.

## Project Structure

This is a Cargo workspace with multiple interdependent crates:

```
hdf5-rust/
├── hdf5/                    # Main library - high-level HDF5 API
├── hdf5-types/              # Native Rust equivalents of HDF5 types
├── hdf5-derive/             # Procedural macro for deriving H5Type trait
├── hdf5-sys/                # Low-level FFI bindings to HDF5 C library
├── hdf5-src/                # Build scripts for compiling HDF5 from source
├── tests/                   # Integration tests
└── .github/workflows/       # CI configuration
```

## Key Technologies

- **HDF5 C Library**: Hierarchical Data Format for scientific data
- **Rust FFI**: Low-level bindings to libhdf5
- **Procedural Macros**: `#[derive(H5Type)]` for automatic type mappings
- **CMake**: For building bundled HDF5 from source
- **pkg-config**: For finding system HDF5 installations

## Build System

### Features

Main features in the `hdf5` crate:
- `static`: Compile and statically link bundled HDF5
- `zlib`: Enable zlib compression filter
- `lzf`: Enable LZF compression filter
- `blosc`: Enable blosc compression filters
- `mpio`: Enable MPI parallel support
- `complex`: Enable complex number types
- `f16`: Enable float16 type support

### Version-Based Features

Features are auto-enabled based on detected HDF5 version:
- `1_8_4`, `1_8_5`, ..., `1_14_4`: Specific HDF5 version support
- `hdf5_1_10_0`, `hdf5_1_12_0`, etc.: Major version milestones

### Building

```bash
cargo build                   # Build with system HDF5
cargo build --features static  # Build with bundled HDF5
cargo test                     # Run all tests
cargo fmt                      # Format code
cargo clippy                   # Run linter
```

## Code Organization

### High-Level API (hdf5/src/)

Core modules:
- `error.rs`: Error handling with `H5Error` and `Result` types
- `globals.rs`: Global library initialization and state
- `sync.rs`: Thread safety (reentrant mutexes for non-threadsafe libhdf5)
- `handle.rs`: Handle management for HDF5 objects
- `dim.rs`: Dimension utilities

High-level API (`hl/` directory):
- `file.rs`: File operations
- `group.rs`: Group (directory-like) operations
- `dataset.rs`: Dataset operations
- `datatype.rs`: Type definitions
- `dataspace.rs`: Data space and dimensions
- `attribute.rs`: Attribute operations
- `object.rs`: Generic object operations
- `location.rs`: Location/namespace management

Property lists (`plist/` directory):
- File creation/access properties
- Dataset creation/access properties
- Link creation properties

### Low-Level FFI (hdf5-sys/src/)

Organized by HDF5 C modules:
- `h5.rs`: Core HDF5 functions
- `h5a.rs`: Attributes (H5A_* functions)
- `h5d.rs`: Datasets (H5D_* functions)
- `h5f.rs`: Files (H5F_* functions)
- `h5g.rs`: Groups (H5G_* functions)
- `h5t.rs`: Datatypes (H5T_* functions)
- `h5s.rs`: Dataspaces (H5S_* functions)
- `h5p.rs`: Property lists (H5P_* functions)
- And others (h5i, h5l, h5o, h5z, etc.)

Each FFI file uses:
- `pub unsafe fn` for raw C function bindings
- `extern "C"` for C ABI
- Type-safe wrappers where applicable

## Development Guidelines

### When Adding New HDF5 Functions

1. **FFI Layer (hdf5-sys)**: Add raw bindings to appropriate `h5*.rs` file
2. **High-Level Wrapper (hdf5/hl/)**: Create safe Rust wrappers
3. **Feature Gates**: Use `cfg(feature = "1_XX_X")` for version-specific APIs
4. **Tests**: Add tests in `hdf5/tests/`

### Error Handling

All HDF5 functions can fail. Use the `Result` type:

```rust
use hdf5::Result;

fn do_something() -> Result<()> {
    let file = hdf5::File::open("data.h5")?;
    // ...
    Ok(())
}
```

### Thread Safety

The HDF5 C library is not thread-safe by default. The library uses reentrant mutexes:
- Global initialization in `globals.rs`
- Handle locking via `sync.rs`
- Do not call HDF5 functions from multiple threads without proper synchronization

### Type Derivation

Use the derive macro for custom types:

```rust
#[derive(H5Type)]
struct MyData {
    x: i32,
    y: f64,
}
```

## Testing

### Test Organization

- `hdf5/tests/`: Integration tests
- `dataset_test.rs`: Dataset operations
- `test_plist.rs`: Property lists
- `test_real_file.rs`: Real file I/O tests
- `common/dataset_test_utils.rs`: Shared utilities

### Running Tests

```bash
cargo test                      # Run all tests
cargo test --features static    # Test with bundled HDF5
cargo test --no-fail-fast       # Don't stop on first failure
```

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) tests:
- Linux (Ubuntu with system HDF5)
- macOS (Homebrew HDF5)
- Windows (vcpkg HDF5)
- Static builds (bundled HDF5)
- Feature matrix
- MSRV compliance

## Current Work

The project is undergoing updates for HDF5 1.14.4 support:
- MSRV bumped to 1.92
- Rust edition 2024
- Modernized dependency syntax
- Enhanced feature gating

## Code Style

- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Follow Rust naming conventions
- Document all public APIs
