use std::env;

/// Register all custom cfg values to avoid "unexpected cfg" warnings.
fn register_check_cfg() {
    // Register all version-based feature flags
    // 1.8.x versions
    for v in 5..=21 {
        println!("cargo::rustc-check-cfg=cfg(feature, values(\"1.8.{}\"))", v);
    }
    // 1.10.x versions
    for v in 0..=8 {
        println!("cargo::rustc-check-cfg=cfg(feature, values(\"1.10.{}\"))", v);
    }
    // 1.12.x versions
    for v in 0..=2 {
        println!("cargo::rustc-check-cfg=cfg(feature, values(\"1.12.{}\"))", v);
    }
    // 1.14.x versions
    for v in 0..=1 {
        println!("cargo::rustc-check-cfg=cfg(feature, values(\"1.14.{}\"))", v);
    }

    // Register special feature flags
    println!("cargo::rustc-check-cfg=cfg(feature, values(\"have-parallel\"))");
    println!("cargo::rustc-check-cfg=cfg(feature, values(\"have-direct\"))");
    println!("cargo::rustc-check-cfg=cfg(feature, values(\"have-threadsafe\"))");
    println!("cargo::rustc-check-cfg=cfg(feature, values(\"have-filter-deflate\"))");

    // Register internal config flags
    println!("cargo::rustc-check-cfg=cfg(msvc_dll_indirection)");

    // Register docsrs cfg for documentation builds
    println!("cargo::rustc-check-cfg=cfg(docsrs)");
}

fn main() {
    // Register all custom cfg values first
    register_check_cfg();

    let print_feature = |key: &str| println!("cargo::rustc-cfg=feature=\"{}\"", key);
    let print_cfg = |key: &str| println!("cargo::rustc-cfg={}", key);
    for (key, _) in env::vars() {
        match key.as_str() {
            // public features
            "DEP_HDF5_HAVE_DIRECT" => print_feature("have-direct"),
            "DEP_HDF5_HAVE_PARALLEL" => print_feature("have-parallel"),
            "DEP_HDF5_HAVE_THREADSAFE" => print_feature("have-threadsafe"),
            "DEP_HDF5_HAVE_FILTER_DEFLATE" => print_feature("have-filter-deflate"),
            // internal config flags
            "DEP_HDF5_MSVC_DLL_INDIRECTION" => print_cfg("msvc_dll_indirection"),
            // public version features
            key if key.starts_with("DEP_HDF5_VERSION_") => {
                print_feature(&key.trim_start_matches("DEP_HDF5_VERSION_").replace('_', "."));
            }
            _ => continue,
        }
    }
}
