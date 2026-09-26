# scann-core

[ScaNN](https://github.com/google-research/google-research/tree/master/scann)'s
nearest-neighbour search core without TensorFlow, as a CMake project with
C++, Python and Rust APIs:

* **C++**: `libscann_core` (static and/or shared), with ScaNN's pybind-free
  facade `research_scann::ScannInterface` and a C++ port of Python's
  `ScannBuilder` (`scann_core::ConfigBuilder`).
* **Python**: the same `scann_pybind` module and `scann.scann_ops_pybind` API
  as the upstream wheel, from upstream's Python sources (with one bug fix, see
  NOTICE); `scann/__init__.py` doesn't import the TensorFlow op.
* **Rust**: the `scann-core` crate, a safe API over the C++ library via
  [cxx](https://cxx.rs).

Extracted from google-research commit `758b894e` (`scann/` subdirectory);
see [NOTICE](NOTICE) for provenance and the list of upstream files that were
changed.

> **scann-core is a derived work of ScaNN. It is not an official Google
> product and is not affiliated with or endorsed by Google.**

## Layout

```
scann-core/
├── CMakeLists.txt        options and library targets
├── cmake/                Flags, Dependencies (pinned FetchContent), Proto,
│                         SourceFlags (per-file copts), BundleStatic
├── src/scann/            upstream C++ sources (see NOTICE for the fixes)
├── core/scann_core/      scann-core additions: ConfigBuilder
├── python/               pybind11 module + upstream Python package
├── rust/                 the Rust crate (cxx bridge, safe API, tests, examples)
├── examples/cpp/         C++ example
├── tests/                C++ tests, Python/Rust equivalence harness
└── docs/                 tutorial, API reference, algorithms, AVQ explainer
```

## Building

Requirements: CMake ≥ 3.27 and a C++17 compiler (clang or GCC); for the
optional parts, Python with numpy and a Rust toolchain.

```sh
cmake -S . -B build -G Ninja
cmake --build build
```

| Option | Default | Effect |
|---|---|---|
| `SCANN_BUILD_STATIC` | ON | `libscann_core.a`, CMake target `scann::core_static` (linked whole-archive for you) |
| `SCANN_BUILD_SHARED` | ON | `libscann_core.so`, target `scann::core_shared` |
| `SCANN_BUILD_PYTHON` | ON | the Python package in `build/python/` |
| `SCANN_BUILD_RUST_BINDINGS` | ON | the Rust crate, via cargo (needs `SCANN_BUILD_STATIC`) |
| `SCANN_BUILD_TESTS` | ON | C++ tests |
| `SCANN_BUILD_EXAMPLES` | ON | C++ example |
| `SCANN_ARCH_FLAGS` | `-mavx;-mfma` (x86-64), `-march=armv8-a+simd` (arm64) | ISA flags for scann-core **and** all dependencies |
| `SCANN_SANITIZE` | empty | e.g. `address,undefined` or `thread`; instruments dependencies too |
| `SCANN_USE_SYSTEM_DEPS` | OFF | try `find_package` first (versions must match the pins exactly) |
| `SCANN_ENABLE_LTO` | OFF | IPO for scann-core's own objects |
| `SCANN_HWY_DISABLED_TARGETS` | empty | `HWY_DISABLED_TARGETS`, applied globally |

The static and shared libraries are linked from the same object files, so
building both costs no extra compilation. `scann::core` is the static library
if it's built, otherwise the shared one.

**Static linking needs whole-archive.** Distance measures register
themselves from static initializers (Bazel's `alwayslink`), so without it a
linker drops them and configs fail at runtime with an unknown distance
measure. `scann::core_static` does this for CMake consumers. Outside CMake,
link `libscann_core.a` whole-archive plus `libscann_core_deps.a` (every
transitive static dependency merged into one archive):

```sh
c++ app.o -Wl,--whole-archive build/libscann_core.a -Wl,--no-whole-archive \
    build/libscann_core_deps.a -lpthread -lrt -lm
```

### Dependencies

All fetched with `FetchContent`, pinned by URL and SHA-256 in
[`cmake/Dependencies.cmake`](cmake/Dependencies.cmake) (latest releases as
of 2026-09-22):

| | version |
|---|---|
| abseil-cpp | 20260817.0 |
| protobuf | 36.2 (Python runtime ≥ 7.36.2 for the generated `_pb2` modules) |
| highway | 1.4.0 |
| Eigen | 5.0.1 |
| zlib | 1.3.2 (static, only for cnpy) |
| cnpy | commit `57184ee0`, vendored in [`third_party/cnpy`](third_party/cnpy) (not downloaded) |
| pybind11 | 3.1.0 (Python only) |
| cxx | 1.x (Rust only, from crates.io) |

**Offline / reproducible builds.** Point FetchContent at local sources and
forbid network access:

```sh
cmake -S . -B build -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DFETCHCONTENT_SOURCE_DIR_ABSL=/src/abseil-cpp-20260817.0 \
  -DFETCHCONTENT_SOURCE_DIR_PROTOBUF=/src/protobuf-36.2 \
  -DFETCHCONTENT_SOURCE_DIR_HIGHWAY=/src/highway-1.4.0 \
  -DFETCHCONTENT_SOURCE_DIR_EIGEN=/src/eigen-5.0.1 \
  -DFETCHCONTENT_SOURCE_DIR_ZLIB=/src/zlib-1.3.2 \
  -DFETCHCONTENT_SOURCE_DIR_PYBIND11=/src/pybind11-3.1.0
```

(an existing build's `_deps/*-src` directories work), or use
`-DSCANN_USE_SYSTEM_DEPS=ON` to take installed packages when their versions
match.

### Compile flags

Carried over from the Bazel build:

* **Global** (reach every dependency too, like `--copt`): the ISA flags
  (`SCANN_ARCH_FLAGS`), `-fsized-deallocation`, `-w`, `-std=c++17` (not
  `gnu++17`: under GCC that also changes `-ffp-contract`, i.e. distances),
  and `-O2` as the release baseline (not CMake's `-O3`).
* **Per file** ([`cmake/SourceFlags.cmake`](cmake/SourceFlags.cmake)), as in
  the Bazel `copts`: `-O3` on the LUT16 kernels and on many-to-many
  distances; `-mtune=generic` on the many-to-many fixed8/sfp8/orthogonality
  files (upstream's workaround for an AMX codegen problem);
  `-fno-tree-vectorize` on `limited_inner_product`; `-fomit-frame-pointer`
  on `asymmetric_hashing_impl_omit_frame_pointer`.
* The LUT16 template sharding (`{BATCH_SIZE}` = 1..9) of Bazel's
  `batch_size_sharder`.

Not carried over: `HWY_DISABLED_TARGETS=(HWY_AVX3_SPR|HWY_AVX10_2)`. It
worked around highway 1.3.0 failing to compile vqsort with clang 23, and
isn't needed with highway 1.4.0. Thin LTO isn't on by default
(`SCANN_ENABLE_LTO`).

## Using it

### C++

```cpp
#include "scann/scann_ops/cc/scann.h"
#include "scann_core/config_builder.h"

scann_core::TreeOptions tree;
tree.num_leaves = 200;
tree.num_leaves_to_search = 20;
scann_core::AhOptions ah;
ah.anisotropic_quantization_threshold = 0.2;
auto config = scann_core::ConfigBuilder(10, scann_core::DistanceMeasure::kDotProduct, dim)
                  .Tree(tree).ScoreAh(ah).Reorder({100})
                  .BuildText(n);

research_scann::ScannInterface index;
absl::Status s = index.Initialize(dataset /* n*dim floats */, n, *config, 0);
research_scann::NNResultsVector res;
s = index.Search(query_ptr, &res, /*final_nn=*/-1, /*pre_reorder_nn=*/-1, /*leaves=*/-1);
```

Full program: [`examples/cpp/quickstart.cc`](examples/cpp/quickstart.cc)
(`cmake --build build --target scann_core_example_quickstart`).
`ConfigBuilder` produces the same configs as the Python builder, except that
it returns an error where Python silently drops or ignores an option. See
the header for the full list.

### Python

`build/python/` is an importable package with upstream's API:

```sh
PYTHONPATH=build/python python -c "import scann; print(scann.scann_ops_pybind.builder)"
```

`scann.scann_ops` (the TensorFlow op) is not included.

### Rust

```rust
use scann_core::{AhOptions, ConfigBuilder, DistanceMeasure, ReorderOptions,
                 SearchOptions, TreeOptions};

let index = ConfigBuilder::new(10, DistanceMeasure::DotProduct, dim)
    .tree(TreeOptions::new(200, 20))
    .score_ah(AhOptions::new(2).anisotropic_quantization_threshold(0.2))
    .reorder(ReorderOptions::new(100))
    .build_index(&dataset)?;
let nn = index.search(query, SearchOptions::default())?;   // nn.indices, nn.distances
```

`ScannIndex` covers single, batched and parallel batched search,
`add`/`upsert`/`delete`/`reserve`/`rebalance`, `serialize`/`load` (the
on-disk format is shared with Python), `set_num_threads`, `config` and health
stats. It is `Send + Sync`: search takes `&self` and can run from many
threads, and mutation takes `&mut self`. Shapes are validated before
anything reaches C++, and every C++ error becomes a `ScannError`.

Full program: [`rust/examples/quickstart.rs`](rust/examples/quickstart.rs).
CMake builds the crate and its examples with the `scann_core_rust` target.
For plain `cargo` (and rust-analyzer), export
`SCANN_CORE_BUILD_ENV=<build>/rust/scann_core_rust_build.env`. build.rs
compiles only the small bridge shim and links the CMake-built archives.

## Testing

```sh
ctest --test-dir build --output-on-failure
```

runs everything that needs nothing beyond the build:

| test | what |
|---|---|
| `api_exercise`, `api_exercise_threaded` | the C++ API end to end on synthetic data, for 12 configs (brute force, AH, autopilot, tree + AH + reorder for both distances, SOAR with bfloat16 reordering): search modes agree, serialize/reload, mutation, retraining, bad input |
| `config_builder` | `ConfigBuilder` against the Python builder's output for 75 option sets (the expected configs are generated from this build's Python package first) |
| `python_docid_bookkeeping` | a failed `upsert`/`delete` leaves docids in sync with the index |
| `rust` | `cargo test`: exactness against naive search, mode agreement, round trip, mutation, concurrency, errors |

The Python tests need numpy and protobuf ≥ 7.36.2 in the interpreter the
module is built for; CMake says so at configure time if they're missing.

The comparison against the upstream wheel is separate, since it needs that
wheel installed: `tests/equivalence/run.py --build-dir build --python <wheel
venv python> --core-python <python with numpy/protobuf>` writes fixtures, and
configuring with `-DSCANN_TEST_FIXTURES=build/equivalence/fixtures` adds them
to `ctest`. The Rust test picks them up automatically.

### Equivalence with upstream

On a fixed seed with two datasets (5000×128 and 4000×768), every
**deterministic** config gives bit-identical neighbour lists and distances
to the upstream wheel, for single and batched search from Python and from
Rust. The configs are brute force, AH + int8 reorder, autopilot, and
tree + AH + reorder with k-means++ initialization, for dot product and
squared L2. Indexes serialized by either build load in the other.

Upstream's default `tree(random_init=True)` is **not reproducible even
against itself**. The initial centers go into an `absl::flat_hash_set`,
whose iteration order is randomized per process. For those configs the
harness compares recall distributions over 8 trainings per build, and they
agree within noise.

### Sanitizers and static analysis

The C++ API test (every fixture config, all search modes, serialization,
mutation, retraining, bad input) runs clean under ASan + UBSan and under
TSan (threaded training and parallel search). Valgrind memcheck on the
portable (`-mavx -mfma`) build reports no leaks and no errors in ScaNN
code. Its only reports are uninitialised-value warnings inside protobuf's
descriptor/reflection code; these look like the known false positive with
clang's combined bitfield loads, but that hasn't been confirmed.
clang-tidy (`bugprone-*`, `clang-analyzer-*` and a few others) was run
over all sources and its findings triaged.

Bugs found and fixed this way are listed in [NOTICE](NOTICE). Among them:
configs with parse errors were silently accepted; `n_points == 0` divided by
zero; `RetrainAndReindex` destroyed a locked mutex; parallel batched search
crashed with `batch_size = 0` (SIGFPE) or without a thread pool (null
dereference; the default pool is `GetNumCPUs() - 1` threads, so this hit
every single-CPU machine); a failed Python `upsert`/`delete` left docids
pointing at the wrong vectors.

## Intentional differences from upstream

* No TensorFlow op (`scann.scann_ops`); `scann/__init__.py` doesn't import
  TensorFlow.
* CMake instead of Bazel; dependencies are upgraded to current releases.
* The bug fixes above: some inputs upstream accepted (bad configs,
  inconsistent shapes, `batch_size = 0`) are now errors.
* Rust batched search returns exactly the neighbours found per query. The
  Python API pads short rows with index 0 and NaN distance.
* `ConfigBuilder` (C++/Rust) returns errors where Python's builder silently
  ignores options, and keeps `upper_tree(soar_lambda=0)` (Python turns it
  into 1.5).

## Documentation

* [`docs/tutorial/`](docs/tutorial/README.md): a seven-part, hands-on
  tutorial on a real million-vector dataset. It covers measuring recall
  and speed, the partition/score/reorder pipeline, tuning, serving,
  updating, and C++ and Rust. Every number in it comes from running the
  scripts included with it.
* [`docs/api_reference.md`](docs/api_reference.md): the config options and
  search parameters, and what they mean.
* [`docs/algorithms.md`](docs/algorithms.md): partitioning, asymmetric
  hashing, anisotropic quantization, reordering.
* [`docs/anisotropic_quantization_explained.md`](docs/anisotropic_quantization_explained.md):
  a plain-language walkthrough of the paper.

## License

Apache 2.0 (see [LICENSE](LICENSE)).

* ScaNN: Copyright The Google Research Authors.
* scann-core's additions and modifications: Copyright 2026 ebenali and
  TheCleaners.

Dependencies carry their own licenses (see [NOTICE](NOTICE)). scann-core is
not an official Google product and is not affiliated with or endorsed by
Google.
