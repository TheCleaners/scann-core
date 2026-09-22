//! Compiles only the cxx-bridge shim (`src/bridge.rs` + `src/shim.cc`) and
//! links it against the CMake-built scann_core archives. scann_core's own
//! sources are never recompiled here -- see ../rust/CMakeLists.txt, which
//! writes the build description this reads and then invokes cargo.
//!
//! For standalone `cargo build` / `cargo check` / rust-analyzer, point
//! SCANN_CORE_BUILD_ENV at <cmake build dir>/rust/scann_core_rust_build.env.

use std::env;
use std::fs;
use std::path::Path;

#[derive(Default)]
struct BuildEnv {
    cxx: Option<String>,
    lib_dir: String,
    deps_archive: String,
    includes: Vec<String>,
    flags: Vec<String>,
    defines: Vec<String>,
    links: Vec<String>,
}

fn read_build_env(path: &str) -> BuildEnv {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("reading SCANN_CORE_BUILD_ENV={path}: {e}"));
    let mut env = BuildEnv::default();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else { continue };
        if value.is_empty() {
            continue;
        }
        let value = value.to_string();
        match key {
            "cxx" => env.cxx = Some(value),
            "lib_dir" => env.lib_dir = value,
            "deps_archive" => env.deps_archive = value,
            "include" => env.includes.push(value),
            "flag" => env.flags.push(value),
            "define" => env.defines.push(value),
            "link" => env.links.push(value),
            _ => {}
        }
    }
    assert!(!env.lib_dir.is_empty(), "{path}: missing lib_dir");
    assert!(!env.deps_archive.is_empty(), "{path}: missing deps_archive");
    env
}

fn main() {
    let env_path = env::var("SCANN_CORE_BUILD_ENV").expect(
        "SCANN_CORE_BUILD_ENV not set -- build through CMake (SCANN_BUILD_RUST_BINDINGS=ON), \
         or point it at <build dir>/rust/scann_core_rust_build.env",
    );
    println!("cargo:rerun-if-env-changed=SCANN_CORE_BUILD_ENV");
    println!("cargo:rerun-if-changed={env_path}");
    for f in ["src/bridge.rs", "src/shim.h", "src/shim.cc"] {
        println!("cargo:rerun-if-changed={f}");
    }
    let benv = read_build_env(&env_path);

    // 1. The shim, compiled with the core's own include paths, defines and
    //    ISA flags. Compiled (and therefore emitted on the link line) first.
    let mut build = cxx_build::bridge("src/bridge.rs");
    build.file("src/shim.cc").include("src").std("c++17");
    if let Some(cxx) = &benv.cxx {
        build.compiler(cxx);
    }
    for dir in &benv.includes {
        build.include(dir);
    }
    for flag in &benv.flags {
        build.flag(flag);
    }
    for def in &benv.defines {
        match def.split_once('=') {
            Some((k, v)) => build.define(k, Some(v)),
            None => build.define(def, None),
        };
    }
    build.compile("scann_core_rust_bridge");

    // 2. scann_core's objects, whole-archive: several distance measures
    //    register themselves from static initializers and nothing references
    //    those objects directly (Bazel's alwayslink = 1).
    println!("cargo:rustc-link-search=native={}", benv.lib_dir);
    println!("cargo:rustc-link-lib=static:+whole-archive,-bundle=scann_core");

    // 3. All transitive static dependencies, merged into one archive.
    let deps = Path::new(&benv.deps_archive);
    let deps_dir = deps.parent().expect("deps_archive has no parent dir");
    let deps_name = deps
        .file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.strip_prefix("lib"))
        .expect("deps_archive should be named lib<name>.a");
    println!("cargo:rustc-link-search=native={}", deps_dir.display());
    println!("cargo:rustc-link-lib=static:-bundle={deps_name}");

    // 4. System libraries (-pthread, -lm, ...) and the C++ runtime, after
    //    everything that needs them.
    for link in &benv.links {
        if let Some(lib) = link.strip_prefix("-l") {
            println!("cargo:rustc-link-lib=dylib={lib}");
        } else {
            println!("cargo:rustc-link-arg={link}");
        }
    }
    let cxx_runtime = match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") | Ok("ios") => "c++",
        _ => "stdc++",
    };
    println!("cargo:rustc-link-lib=dylib={cxx_runtime}");
}
