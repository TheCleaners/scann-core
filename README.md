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

It runs on x86-64 and aarch64 Linux. On x86-64 it's about 50% faster than the
upstream wheel at the same recall: the wheel never detects the CPU, so its
AVX2/AVX-512 kernels never run. On aarch64 it includes Arm's Neon/SVE
kernels. See [docs/benchmarks.md](docs/benchmarks.md).

Extracted from google-research commit `758b894e` (`scann/` subdirectory);
see [NOTICE](NOTICE) for provenance and the list of upstream files that were
changed.

> **scann-core is a derived work of ScaNN. It is not an official Google
> product and is not affiliated with or endorsed by Google.**

## Contents

* [Install](#install)
* [Layout](#layout)
* [Building](#building): [dependencies](#dependencies),
  [compile flags](#compile-flags),
  [cross-compiling for aarch64](#cross-compiling-for-aarch64)
* [Using it](#using-it): [C++](#c), [Python](#python), [Rust](#rust),
  [examples](#examples)
* [Testing](#testing): [equivalence with upstream](#equivalence-with-upstream),
  [sanitizers and static analysis](#sanitizers-and-static-analysis)
* [Intentional differences from upstream](#intentional-differences-from-upstream)
* [Documentation](#documentation)
* [License](#license)

## Install

| | |
|---|---|
| Python | `pip install scann-core` (release candidates: `pip install --pre scann-core`), or `pip install .` from a checkout |
| Rust | `cargo add scann-core` (release candidates: `cargo add scann-core@<version>`), or a git/path dependency on this repository |
| C++ | CMake `FetchContent`/`add_subdirectory`, linking `scann::core`; see [`examples/fetchcontent`](examples/fetchcontent) |

All three build the C++ library from source, which needs:

* **Linux on x86-64 or aarch64.**
  * Both are built and tested in CI (GitHub Actions, Ubuntu 26.04), aarch64
    on native Arm runners. aarch64 was also tested on AWS Graviton4
    (Neoverse V2).
  * The C++ tests also run under QEMU on six emulated Arm CPUs, from Neon-only
    Cortex-A57 to SVE2 (see [Cross-compiling for aarch64](#cross-compiling-for-aarch64)).
  * The SIMD kernels (AVX2/AVX-512 on x86-64, Neon/SVE on aarch64) are
    chosen at run time from the CPU's features.
  * The macOS code paths exist, inherited from upstream, but are untested.
* **clang ≥ 19 or GCC ≥ 13.** Tested with clang 19–24 and GCC 13–16. CI
  runs the oldest and newest that Ubuntu 26.04 packages: clang 19 and 22,
  and GCC 13 and 15. clang is
  upstream's compiler, and the one the
  [equivalence checks](#equivalence-with-upstream) use. GCC builds give the same recall (checked on GloVe-100),
  with last-bit differences in distances. They are slower: the partitioned
  pipeline by about 5%, and batched brute-force search at about half of
  clang's throughput. Use clang for speed. When no compiler is chosen, clang
  is picked if it's on PATH. The AMX kernels (Sapphire Rapids and later)
  need clang ≥ 20.
* **CMake ≥ 3.27**, and network access to download the C++ dependencies
  (or local copies; see [Dependencies](#dependencies)).
* For Python: Python ≥ 3.10 with numpy and protobuf ≥ 7.36.2 (pip installs
  them). Free-threaded Python (3.14t, 3.15t) is supported: the module runs
  without the GIL (see [Threads](#threads)). For Rust: Rust ≥ 1.88.

The version is in [`VERSION`](VERSION); see [CHANGELOG.md](CHANGELOG.md).

## Layout

```
scann-core/
├── CMakeLists.txt        options and library targets
├── cmake/                Flags, Dependencies (pinned FetchContent), Proto,
│                         SourceFlags (per-file copts), BundleStatic
├── src/scann/            upstream C++ sources (see NOTICE for the fixes)
├── core/scann_core/      scann-core additions: ConfigBuilder
├── python/               pybind11 module + upstream Python package
├── rust/                 the Rust crate (cxx bridge, safe API, tests)
├── examples/             Python, C++ and Rust examples, FetchContent consumer template
├── third_party/          vendored: cnpy, googletest's gtest_prod.h
├── tests/                C++/Python tests, upstream-equivalence harness
├── benchmarks/           ann-benchmarks runner, GloVe by default (docs/benchmarks.md)
├── scripts/ci.sh         what CI runs (also runnable locally)
├── scripts/python-versions.sh  the Python tests on CPython 3.10-3.15t
├── scripts/cross-aarch64.sh  aarch64 cross-build + tests under QEMU
├── Cargo.toml            the Rust crate (sources in rust/)
├── pyproject.toml        the Python package (scikit-build-core)
└── docs/                 tutorial, benchmarks, API reference, algorithms, AVQ explainer,
                          TensorFlow
```

## Building

Requirements as under [Install](#install).

```sh
cmake -S . -B build -G Ninja
cmake --build build
```

| Option | Default | Effect |
|---|---|---|
| `SCANN_BUILD_STATIC` | ON | `libscann_core.a`, CMake target `scann::core_static` (linked whole-archive for you) |
| `SCANN_BUILD_SHARED` | ON\* | `libscann_core.so`, target `scann::core_shared` |
| `SCANN_BUILD_PYTHON` | ON\* | the Python package in `build/python/` (for `python3` on PATH, or `-DPython_EXECUTABLE=`) |
| `SCANN_BUILD_RUST_BINDINGS` | ON\* | the Rust crate, via cargo (needs `SCANN_BUILD_STATIC`) |
| `SCANN_BUILD_TESTS` | ON\* | tests, run with `ctest` |
| `SCANN_BUILD_EXAMPLES` | ON\* | C++ examples |
| `SCANN_ARCH_FLAGS` | `-mavx;-mfma` (x86-64), `-march=armv8-a+simd` (arm64) | ISA flags for scann-core **and** all dependencies |
| `SCANN_SANITIZE` | empty | e.g. `address,undefined` or `thread`; instruments dependencies too |
| `SCANN_USE_SYSTEM_DEPS` | OFF | try `find_package` first (versions must match the pins exactly) |
| `SCANN_ENABLE_LTO` | OFF | IPO for scann-core's own objects |
| `SCANN_HWY_DISABLED_TARGETS` | empty | `HWY_DISABLED_TARGETS`, applied globally |
| `SCANN_ALLOW_UNSUPPORTED_COMPILER` | OFF | configure with a compiler other than clang or GCC anyway (expect errors) |

\* ON when scann-core is the top-level project, OFF when it's pulled into
another one with FetchContent or `add_subdirectory`.

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
| googletest | `gtest_prod.h` only, v1.18.0, vendored in [`third_party/googletest`](third_party/googletest) |
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
  `gnu++17`, as in the Bazel build), and `-O2` as the release baseline (not CMake's `-O3`).
* **Per file** ([`cmake/SourceFlags.cmake`](cmake/SourceFlags.cmake)), as in
  the Bazel `copts`: `-O3` on the LUT16 kernels and on many-to-many
  distances; `-mtune=generic` on the many-to-many fixed8/sfp8/orthogonality
  files (upstream's workaround for an AMX codegen problem);
  `-fno-tree-vectorize` on `limited_inner_product`; `-fomit-frame-pointer`
  on `asymmetric_hashing_impl_omit_frame_pointer`.
* The LUT16 template sharding (`{BATCH_SIZE}` = 1..9) of Bazel's
  `batch_size_sharder`.

On aarch64 the default `-march=armv8-a+simd` covers every Armv8 CPU. The
SVE kernels are compiled with a `target("+sve")` attribute and used only
when the CPU has SVE, so `-march=native` isn't needed for them.

Not carried over: `HWY_DISABLED_TARGETS=(HWY_AVX3_SPR|HWY_AVX10_2)`. It
worked around highway 1.3.0 failing to compile vqsort with clang 23, and
isn't needed with highway 1.4.0. Thin LTO isn't on by default
(`SCANN_ENABLE_LTO`).

### Cross-compiling for aarch64

[`cmake/toolchains/aarch64-linux-gnu.cmake`](cmake/toolchains/aarch64-linux-gnu.cmake)
cross-compiles with clang and lld, against the Debian/Ubuntu
`aarch64-linux-gnu` sysroot. It uses `qemu-aarch64` to run the tools the
build runs (protoc) and the tests:

```sh
cmake -S . -B build-aarch64 -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-linux-gnu.cmake \
  -DSCANN_BUILD_PYTHON=OFF -DSCANN_BUILD_RUST_BINDINGS=OFF
cmake --build build-aarch64
QEMU_CPU=neoverse-n2 ctest --test-dir build-aarch64
```

`SCANN_CLANG_SUFFIX=-19` picks `clang-19`. The build needs `clang`, `lld`,
`g++-aarch64-linux-gnu` and `qemu-user`.
[`scripts/cross-aarch64.sh`](scripts/cross-aarch64.sh) does all of it in a
throwaway container. It runs the tests on six emulated CPUs with different
feature sets, covering both the Neon and the SVE kernels:

```sh
docker run --rm --platform linux/amd64 -v $PWD:/src -w /src ubuntu:26.04 scripts/cross-aarch64.sh
```

Emulated timings mean nothing. For speed, see the Graviton4 numbers in
[docs/benchmarks.md](docs/benchmarks.md#aarch64-aws-graviton4). Python and
Rust build natively on aarch64 the same way as on x86-64.

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

Full programs: [`examples/cpp/quickstart.cc`](examples/cpp/quickstart.cc)
(`cmake --build build --target scann_core_example_quickstart`) and
[`examples/cpp/updating.cc`](examples/cpp/updating.cc) (adding, updating and
removing points).
`ConfigBuilder` produces the same configs as the Python builder, except that
it returns an error where Python silently drops or ignores an option. See
the header for the full list.

### Python

`pip install .` builds and installs the `scann` package (same import name
and API as upstream's wheel, so don't install both in one environment). In a
CMake build, `build/python/` is the same package, importable directly:

```sh
PYTHONPATH=build/python python -c "import scann; print(scann.__version__)"
```

Full programs: [`examples/python/`](#examples).

`scann.scann_ops` (the TensorFlow op) is not included. `scann.tf` wraps the
searcher for TensorFlow code instead (eager mode, `tf.function`, `tf.data`;
not SavedModels): `pip install 'scann-core[tf]'`, and see
[docs/tensorflow.md](docs/tensorflow.md), which also covers serving
alongside a TensorFlow model.

#### Threads

A searcher can be shared between Python threads:
* Searches run in parallel with each other.
* `upsert`, `delete`, `rebalance`, `reserve`, `set_num_threads` and
  `serialize` each run on their own. A search sees the index before or after
  an update, never during one.

With the GIL, searches release it while they run, but the Python-side
work around each call is serialized. On a 64-thread machine, concurrent
`search()` calls plateau around 100k QPS.

On free-threaded Python (3.14t and later), the module declares that it
doesn't need the GIL, so importing it doesn't turn the GIL back on. The
same threads then reach about 330k QPS, level with batched search. See
[tutorial part 5](docs/tutorial/05-saving-and-serving.md#serving-batch-size-latency-and-throughput).

Upstream relied on the GIL for thread safety, but its searches release it.
A search could therefore run during an `upsert` or `delete`, and map its
results to the wrong docids. scann-core adds a reader/writer lock in C++
and one around the docid bookkeeping. Plain searches cost the same; a
search on a searcher with docids costs about 1 µs more.

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

Full programs: [`examples/rust/quickstart.rs`](examples/rust/quickstart.rs)
and [`examples/rust/updating.rs`](examples/rust/updating.rs) (upsert, delete,
rebalance, re-saving).

The crate's manifest is the repository root's `Cargo.toml`. Built on its own
(`cargo build`, or as a dependency), its build script builds the C++ library
with CMake; `SCANN_CORE_CMAKE_ARGS` passes extra `-D` options, such as local
dependency sources for offline builds. Inside a CMake build, the
`scann_core_rust` target builds the crate against the CMake-built libraries
instead; for plain `cargo` or rust-analyzer to do the same, export
`SCANN_CORE_BUILD_ENV=<build>/rust/scann_core_rust_build.env`.

### Examples

Short programs on synthetic data, meant to be copied from. Each runs in
about a second and checks its own results, so they also run as tests
(`example_*` below).

| | |
|---|---|
| [`python/quickstart.py`](examples/python/quickstart.py) | build tree + AH + reorder, search one query and batches, recall against exact search, save with docids and load from a moved directory |
| [`python/updating.py`](examples/python/updating.py) | upsert new and existing docids, delete, health stats, `rebalance()`, saving over an existing index |
| [`python/serving_threads.py`](examples/python/serving_threads.py) | `search_batched_parallel` vs. Python threads calling `search()`, with or without the GIL |
| [`python/tensorflow_wrapper.py`](examples/python/tensorflow_wrapper.py) | `scann.tf` eagerly and in `tf.function`, docids with `tf.gather` (needs TensorFlow) |
| [`python/tensorflow_serving.py`](examples/python/tensorflow_serving.py) | a Keras query tower exported as a SavedModel, the index saved beside it, and a service that loads both (needs TensorFlow) |
| [`cpp/quickstart.cc`](examples/cpp/quickstart.cc), [`rust/quickstart.rs`](examples/rust/quickstart.rs) | build with the config builder, search, add a point, save and reload |
| [`cpp/updating.cc`](examples/cpp/updating.cc), [`rust/updating.rs`](examples/rust/updating.rs) | add, update and delete points by index, retrain, save over an existing index and reload |
| [`fetchcontent/`](examples/fetchcontent) | a CMake project that pulls in scann-core with `FetchContent` (built by `scripts/ci.sh`, not a ctest) |

Run a Python one with `PYTHONPATH=build/python python examples/python/quickstart.py`,
the C++ ones from `build/examples/`, the Rust ones with
`cargo run --release --example updating` (see [Rust](#rust)).

## Testing

```sh
ctest --test-dir build --output-on-failure
```

runs everything that needs nothing beyond the build:

| test | what |
|---|---|
| `api_exercise`, `api_exercise_threaded` | the C++ API end to end on synthetic data, for 12 configs (brute force, AH, autopilot, tree + AH + reorder for both distances, SOAR with bfloat16 reordering): search modes agree, serialize/reload, mutation, retraining, bad input (including NaN/infinity) |
| `api_exercise_avx2` | the same, with the AVX2 kernels forced on an AVX-512 machine (`SCANN_TEST_FORCE_AVX2=1`), so both kernel sets get tested (and sanitized) |
| `mutation_regressions` | a failed `rebalance()` leaves a working index; tree + bfloat16 add/update/delete; a failed update in a SOAR tree (injected leaf failure) changes nothing; every stored vector keeps finding itself |
| `artifact_loading` | about 60 damaged or mixed index directories, generated at run time (bad `.npy` headers, dtypes and shapes, out-of-range tokens, files from another index, SOAR mismatches, manifest errors) fail to load with an error; all-deleted and bfloat16-leaf trees round-trip; `SerializeToDirectory` replaces a previous index, and one that fails midway leaves a directory that fails to load |
| `artifact_loading_avx2` | the same, with the AVX2 kernels forced (`SCANN_TEST_FORCE_AVX2=1`); includes searching an index whose leaves are all empty |
| `config_regressions` | raw configs that crashed upstream (zero block sizes, LUT16 with other than 16 clusters, binary or unsupported distances, bad quantiles) are errors; tree + PCA/TRUNCATE + AH without residuals builds, searches well and reloads |
| `config_builder` | `ConfigBuilder` against the Python builder's output for 75 option sets (the expected configs are generated from this build's Python package first) |
| `python_docid_bookkeeping` | a failed `upsert`/`delete` leaves docids in sync with the index |
| `python_input_validation` | NaN/infinity in builds, upserts and batched queries, and `leaves_to_search` on indexes without a tree, are clean errors (not crashes); a failed batch upsert changes nothing; padded results map to `None` |
| `python_config_validation` | the same raw configs through `create_searcher`; the builder's `pca()`/`truncate()` with a squared-L2 tree builds; `incremental_threshold` with `pca()`, `truncate()` or `upper_tree()` raises `ValueError` |
| `python_wrapper_edge_cases` | `upsert` with a repeated docid is rejected before anything changes; zero-query batches return empty `(0, k)` results; more than 2^32 - 1 rows is a clear error; spherical trees store unit vectors at build, upsert and `rebalance()`; `rebalance()` without float data fails cleanly |
| `python_projection_mutation` | trees with PCA/TRUNCATE projections through inserts, updates, deletes and `rebalance()`: points stay findable, health stats stay consistent |
| `python_rebalance_flow` | an index grown from empty with batched upserts, then retrained with `rebalance(config)` into a SOAR tree (the big-ann-benchmarks flow); the builder's SOAR options; a clear error for more leaves than points |
| `python_serialization` | `serialize()`/`load_searcher()` round trips for 10 configs, also with every point deleted; re-serializing over another index leaves no stale files or docids; a re-serialize that fails or is killed (`SIGKILL`) midway leaves the old index, the new one, or a directory that fails to load, never a mix |
| `python_concurrency` | 3 s of concurrent searches, upserts, deletes and rebalances from Python threads; every point keeps finding itself by docid. On free-threaded Python, also checks that importing scann keeps the GIL disabled |
| `python_tf` | `scann.tf` returns exactly the pybind searcher's results as int32/float32 tensors, eagerly and in `tf.function` (unknown batch size, static shapes), from `tf.data` maps and concurrent threads; docids, padding, `serialize_to_module()` raising; `import scann` doesn't import TensorFlow. Skipped without TensorFlow |
| `python_langchain` | LangChain's ScaNN vector store on scann-core: results equal an exact search for both distance strategies, `normalize_L2` and a tree + AH config; filters; save/load and re-saving into the same folder. Skipped without langchain-community |
| `python_torch_input` | PyTorch tensors at every entry point give the same results as numpy arrays: float32/64/16 and bfloat16, non-contiguous views, tensors that require grad, zero rows, CUDA tensors when a GPU is present; no copy for float32 CPU tensors. Skipped without PyTorch |
| `rust` | `cargo test`: exactness against naive search, mode agreement, round trip, mutation, concurrency, errors |
| `example_py_quickstart`, `example_py_updating`, `example_py_serving_threads` | the Python [examples](#examples): recall above 0.9, identical results after reloading, every inserted or updated point found under its docid, a repeated upsert docid rejected, concurrent `search()` calls agreeing with a batched search |
| `example_py_tensorflow_wrapper`, `example_py_tensorflow_serving` | the TensorFlow examples: `scann.tf` results equal the pybind searcher's; the SavedModel + index service returns docids with recall above 0.9. Skipped without TensorFlow |
| `example_cpp_quickstart`, `example_cpp_updating` | the C++ examples (built with `SCANN_BUILD_EXAMPLES`) |
| `example_rust_quickstart`, `example_rust_updating` | the Rust examples, with `cargo run --example` |

The Python tests need numpy and protobuf ≥ 7.36.2 in the interpreter the
module is built for; CMake says so at configure time if they're missing.
`python_tf` and the two TensorFlow examples also need TensorFlow, and
`python_langchain` needs langchain-community and `python_torch_input`
PyTorch; ctest reports them as skipped without those.
[`scripts/python-versions.sh`](scripts/python-versions.sh) runs them (and the
Python examples) on
every supported CPython, 3.10 to 3.15 and free-threaded 3.14t and 3.15t,
with interpreters from [uv](https://docs.astral.sh/uv/), and installs
`tensorflow-cpu`, langchain-community and PyTorch (CPU) for 3.12 so that
those tests run there. It compiles the
C++ library once and rebuilds only the Python module for each version.

The comparison against the upstream wheel is separate, since it needs that
wheel installed: `tests/equivalence/run.py --build-dir build --python <wheel
venv python> --core-python <python with numpy/protobuf>` writes fixtures, and
configuring with `-DSCANN_TEST_FIXTURES=build/equivalence/fixtures` adds them
to `ctest`. The Rust test picks them up automatically.

### Equivalence with upstream

The harness uses a fixed seed and two datasets (5000×128 and 4000×768). The
**deterministic** configs are brute force, AH + int8 reorder, autopilot, and
tree + AH + reorder with k-means++ initialization, for dot product and
squared L2. Indexes serialized by either build load in the other.

* **aarch64:** every deterministic config gives bit-identical neighbour
  lists and distances to the upstream wheel, for single and batched search.
  Checked on Graviton4.
* **x86-64:** bit-identical as well in scann-core 0.1.0. Since 0.2.0, the
  CPU-detection fix makes scann-core run the AVX2/AVX-512 kernels that the
  wheel never does. Distances now differ in the last bits (≤ 2×10⁻⁷).
  * Neighbour lists are identical in 13 of the 14 config/search-mode pairs.
  * The exception is k-means++ training on the 768-dimensional data. It
    trains a slightly different partitioner: 183 of 200 queries give
    identical neighbours, and recall is 0.962 against 0.982.
  * The wheel gives the same 0.962 on aarch64, so this is ordinary training
    variation.
  * The harness, which demands exact equality, therefore reports a mismatch
    for that config on x86-64. On GloVe the recall matches to the fourth
    decimal place.
  * Details in [docs/benchmarks.md](docs/benchmarks.md#correctness).

Upstream's default `tree(random_init=True)` is **not reproducible even
against itself**. The initial centers go into an `absl::flat_hash_set`,
whose iteration order is randomized per process. For those configs the
harness compares recall distributions over 8 trainings per build, and
checks that the means agree within 3 standard errors.

On one config (tree + AH + reorder, dot product, 5000×128), scann-core's
mean recall has come out 0.2–0.9 points lower than the wheel's in every run
so far. On x86-64 that stays within the bound. On aarch64 it doesn't: the
wheel's random init there is nearly deterministic (1–2 distinct indexes in
8), which narrows the bound. That gives 0.991 ± 0.006 against 0.998 ± 0.002.
It happens with and without Arm's kernels, and the k-means++ configs aren't
affected. Why is still open.

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
  TensorFlow. `scann.tf` offers the op's Python API without the op, and
  can't be saved in a SavedModel; see [docs/tensorflow.md](docs/tensorflow.md)
  for that and for serving next to a TensorFlow model.
* CMake instead of Bazel; dependencies are upgraded to current releases.
* The bug fixes above: some inputs upstream accepted (bad configs,
  inconsistent shapes, `batch_size = 0`) are now errors.
* The Python module is safe to share between threads, and runs without the
  GIL on free-threaded Python (see [Threads](#threads)).
* Rust batched search returns exactly the neighbours found per query. The
  Python API pads short rows with index 0 and NaN distance (docid `None`
  when the searcher has docids; upstream returned `docids[0]`).
* `ConfigBuilder` (C++/Rust) returns errors where Python's builder silently
  ignores options, and keeps `upper_tree(soar_lambda=0)` (Python turns it
  into 1.5).

## Documentation

* [`docs/tutorial/`](docs/tutorial/README.md): a seven-part, hands-on
  tutorial on a real million-vector dataset. It covers measuring recall
  and speed, the partition/score/reorder pipeline, tuning, serving,
  updating, and C++ and Rust. Every number in it comes from running the
  scripts included with it.
* [`docs/benchmarks.md`](docs/benchmarks.md): speed and recall against the
  upstream wheel on x86-64 and aarch64 (Graviton4), and how to reproduce
  them with [`benchmarks/ann_benchmarks.py`](benchmarks/ann_benchmarks.py)
  (GloVe by default, or any ann-benchmarks dataset).
* [`docs/api_reference.md`](docs/api_reference.md): the config options and
  search parameters, and what they mean.
* [`docs/tensorflow.md`](docs/tensorflow.md): using scann-core from
  TensorFlow code (`scann.tf`), serving retrieval next to a TensorFlow
  model, and why there is no TensorFlow op.
* [`docs/integrations.md`](docs/integrations.md): installing scann-core in
  place of the `scann` wheel, PyTorch tensors as inputs, and libraries
  that use it (LangChain).
* [`docs/algorithms.md`](docs/algorithms.md): partitioning, asymmetric
  hashing, anisotropic quantization, reordering.
* [`docs/anisotropic_quantization_explained.md`](docs/anisotropic_quantization_explained.md):
  a plain-language walkthrough of the paper.

## License

Apache 2.0 (see [LICENSE](LICENSE)).

* ScaNN: Copyright The Google Research Authors.
* scann-core's additions and modifications: Copyright 2026 Elias Benali
  ([@ebenali](https://github.com/ebenali)) and TheCleaners.

Dependencies carry their own licenses (see [NOTICE](NOTICE)). scann-core is
not an official Google product and is not affiliated with or endorsed by
Google.
