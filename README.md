# scann-core

A standalone, TensorFlow- and Python-independent extraction of ScaNN's
search core, plus first-class Rust bindings. Built with CMake instead of
the parent `scann/` tree's Bazel workspace, and with all dependencies
vendored via `FetchContent` rather than assumed to be system-installed.

## Why this exists

The parent `scann/` package couples two things that don't need to be
coupled: the actual nearest-neighbor search implementation, and a
TensorFlow custom-op wrapper around it. The Python bindings
(`scann_ops_pybind`) never touch TensorFlow at all — confirmed by tracing
the Bazel dependency graph, not by inspection alone (`_scann_ops.so`, the
TF custom op, and `scann_pybind.so`, the pybind11 module, are separate
Bazel targets with no shared TF dependency). `scann-core` is that
TF-free/Python-optional slice, pulled out into its own buildable tree:
`src/` mirrors the relevant parts of `scann/`'s package layout (`base`,
`data_format`, `distance_measures`, `hashes`, `partitioning`,
`projection`, `proto`, `tree_x_hybrid`, `trees`, `utils`, `oss_wrappers`
minus its TF glue, and just the pybind-free half of `scann_ops/cc`:
`scann.h`/`scann.cc`, i.e. `ScannInterface`).

## Layout

```
scann-core/
├── CMakeLists.txt      # options, the core library targets
├── cmake/
│   ├── Dependencies.cmake  # FetchContent-vendored deps, pinned versions
│   └── Proto.cmake         # protoc codegen helper
├── src/                 # the core C++ source tree (see above)
├── python/               # pybind11 module (ScannNumpy + scann_pybind.cc)
└── rust/                 # Rust crate: cxx bridge over ScannInterface
    ├── Cargo.toml
    ├── build.rs
    └── src/
        ├── bridge.rs      # #[cxx::bridge] FFI declarations
        ├── shim.h/.cc     # C++ adapter: absl types -> cxx-bridgeable types
        └── lib.rs         # safe Rust wrapper (ScannIndex, ScannError)
```

## Building

```sh
cmake -S scann-core -B build
cmake --build build
```

CMake options (all default `ON`):

| Option | Effect |
|---|---|
| `SCANN_BUILD_STATIC` | Build `libscann_core.a` |
| `SCANN_BUILD_SHARED` | Build `libscann_core.so`/`.dylib` |
| `SCANN_BUILD_PYTHON` | Build the pybind11 module (needs a Python + pybind11) |
| `SCANN_BUILD_RUST_BINDINGS` | Build the Rust crate (needs `cargo` on `PATH`) |

Both static and shared are built from the same compiled objects (a shared
`OBJECT` library target), so enabling both isn't 2x compile time.
`SCANN_BUILD_RUST_BINDINGS` requires `SCANN_BUILD_STATIC` — the Rust crate
links against the static archive, not the shared library.

All C++ dependencies (abseil-cpp, protobuf, highway, eigen, cnpy, zlib,
and pybind11 when `SCANN_BUILD_PYTHON=ON`) are fetched and built from
source by CMake's `FetchContent` — nothing needs to be pre-installed on
the build machine beyond a C++17 compiler, CMake ≥3.24, and (for the Rust
crate) a Rust toolchain. Versions are pinned in `cmake/Dependencies.cmake`
to match what the parent `scann/` Bazel build already validated.

## Status

Builds end to end (static + shared library, Python package, Rust crate)
and is verified equivalent to the Bazel-built upstream wheel by
`tests/equivalence/run.py`: bit-identical neighbour lists and distances on
every deterministic config, for Python and for Rust (`cargo test`). The
Rust binding currently covers construction and single-query search.

For what the underlying config/search API actually means (distance
measures, partitioning, quantization, the config string these bindings
take), see [`docs/api_reference.md`](docs/api_reference.md) and
[`docs/algorithms.md`](docs/algorithms.md) — the
core library here implements exactly that API, just without the Python
wrapper layer.
