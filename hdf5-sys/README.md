# `hdf5-sys`

[![Build](https://github.com/archebase/hdf5-rs/workflows/CI/badge.svg)](https://github.com/archebase/hdf5-rs/actions?query=branch%3Amain)
[![Latest Version](https://img.shields.io/crates/v/hdf5-sys.svg)](https://crates.io/crates/hdf5-sys)
[![Documentation](https://docs.rs/hdf5-sys/badge.svg)](https://docs.rs/hdf5-sys)
[![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This crate provides direct bindings to the HDF5 C library and allows to build
the library from C sources if need be, so it can be linked in statically.

### Static linking

This crate supports linking to a static build of HDF5. The HDF5 C library is built
via the `hdf5-src` crate which is then linked in when the `static` feature is set.
See below for a list of supported options for static builds.

As of the time of writing, the version of the HDF5 library that is built is 2.0.0.
Dynamic linking supports HDF5 versions 1.8.4 through 2.0.x.

## Crate features

Features propagated upwards will be detected based on features available for the chosen library. The library selected can always be overridden by specifying the environment variable `HDF5_DIR`.

General features:

- `mpio`: enable MPI support (not supported in static mode)
- `static`: build HDF5 C library from source (see below for a list of options)

These options are mutually exclusive. This combination could be supported in the future, see issue #101.

The following features affect the HDF5 build options when compiling it from source:

- `hl`: enable high-level features (which we don't provide bindings for)
- `threadsafe`: build a thread-safe version of the library
- `zlib`: enable `zlib` filter support
- `deprecated`:  include deprecated symbols (which we don't provide bindings for)

Note: HDF5 library has a [separate BSD-style license](https://support.hdfgroup.org/products/licenses.html).
