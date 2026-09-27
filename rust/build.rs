// Copyright 2026 Elias Benali (@ebenali) and TheCleaners.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Build script for the scann-core crate.
//!
//! The Rust side is a cxx bridge (rust/src/bridge.rs + rust/src/shim.cc) over
//! the C++ library, which is built by CMake. Three modes:
//!
//! * Inside scann-core's CMake build (SCANN_BUILD_RUST_BINDINGS=ON), CMake
//!   invokes cargo with SCANN_CORE_BUILD_ENV pointing at a build description
//!   (include paths, flags, archives) it wrote. For standalone `cargo build`
//!   against an existing CMake build dir, export
//!   SCANN_CORE_BUILD_ENV=<build dir>/rust/scann_core_rust_build.env.
//! * Otherwise (e.g. a crates.io dependency), this script configures and
//!   builds the C++ library itself with CMake (the `cmake` crate). That
//!   downloads the C++ dependencies unless SCANN_CORE_CMAKE_ARGS points
//!   FetchContent at local copies, e.g.
//!   SCANN_CORE_CMAKE_ARGS="-DFETCHCONTENT_SOURCE_DIR_ABSL=/src/absl ...".
//!   Needs CMake >= 3.27 and clang (picked automatically when CXX isn't set).
//! * On docs.rs (DOCS_RS set) nothing native is built: rustdoc doesn't link.

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

/// Builds the C++ library with CMake and returns its build description.
fn build_with_cmake() -> String {
    let root = env::var("CARGO_MANIFEST_DIR").unwrap();
    let mut cfg = cmake::Config::new(&root);
    cfg.profile("Release")
        .define("SCANN_BUILD_PYTHON", "OFF")
        .define("SCANN_BUILD_RUST_BINDINGS", "OFF")
        .define("SCANN_RUST_BUILD_ENV_ONLY", "ON")
        .define("SCANN_BUILD_TESTS", "OFF")
        .define("SCANN_BUILD_EXAMPLES", "OFF")
        .define("SCANN_BUILD_SHARED", "OFF")
        .build_target("scann_core_rust_inputs");
    // Prefer clang (upstream's compiler, which results are verified
    // against) when CXX isn't set; the cmake crate would otherwise pass the
    // platform default compiler explicitly. GCC >= 13 works too.
    if env::var_os("CXX").is_none() {
        if let (Some(cc), Some(cxx)) = (find_on_path("clang"), find_on_path("clang++")) {
            cfg.define("CMAKE_C_COMPILER", cc).define("CMAKE_CXX_COMPILER", cxx);
        }
    }
    println!("cargo:rerun-if-env-changed=SCANN_CORE_CMAKE_ARGS");
    if let Ok(args) = env::var("SCANN_CORE_CMAKE_ARGS") {
        for arg in args.split_whitespace() {
            let def = arg.strip_prefix("-D").unwrap_or_else(|| {
                panic!("SCANN_CORE_CMAKE_ARGS: expected -DNAME=VALUE entries, got {arg:?}")
            });
            let (name, value) = def.split_once('=').unwrap_or((def, "ON"));
            let name = name.split(':').next().unwrap(); // drop a :TYPE suffix
            cfg.define(name, value);
        }
    }
    for dir in ["CMakeLists.txt", "cmake", "core", "src", "third_party", "VERSION"] {
        println!("cargo:rerun-if-changed={root}/{dir}");
    }
    let out = cfg.build();
    out.join("build/rust/scann_core_rust_build.env").display().to_string()
}

fn find_on_path(name: &str) -> Option<std::path::PathBuf> {
    env::split_paths(&env::var_os("PATH")?)
        .map(|dir| dir.join(name))
        .find(|p| p.is_file())
}

fn main() {
    println!("cargo:rerun-if-env-changed=SCANN_CORE_BUILD_ENV");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    for f in ["rust/src/bridge.rs", "rust/src/shim.h", "rust/src/shim.cc"] {
        println!("cargo:rerun-if-changed={f}");
    }
    if env::var_os("DOCS_RS").is_some() {
        return;
    }
    let env_path = match env::var("SCANN_CORE_BUILD_ENV") {
        Ok(path) => path,
        Err(_) => build_with_cmake(),
    };
    println!("cargo:rerun-if-changed={env_path}");
    let benv = read_build_env(&env_path);

    // 1. The shim, compiled with the core's own include paths, defines and
    //    ISA flags. Compiled (and therefore emitted on the link line) first.
    let mut build = cxx_build::bridge("rust/src/bridge.rs");
    build.file("rust/src/shim.cc").include("rust/src").std("c++17");
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
    // Link the runtime of the compiler that built the C++ code. Rust links
    // through the default `cc`, which may belong to another GCC version that
    // doesn't search this compiler's library directory (e.g. cc = gcc-13
    // while the code was built with g++-14).
    if let Some(cxx) = &benv.cxx {
        if let Ok(out) = std::process::Command::new(cxx)
            .arg(format!("-print-file-name=lib{cxx_runtime}.so"))
            .output()
        {
            let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let path = Path::new(&path);
            if path.is_absolute() && path.exists() {
                if let Some(dir) = path.parent() {
                    println!("cargo:rustc-link-search=native={}", dir.display());
                }
            }
        }
    }
    println!("cargo:rustc-link-lib=dylib={cxx_runtime}");
}
