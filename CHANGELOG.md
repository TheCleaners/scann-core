# Changelog

All notable changes to scann-core. Versions follow
[semantic versioning](https://semver.org); the version is in `VERSION`.

## 0.1.0 (2026-09-26)

First release. scann-core is ScaNN's search core, extracted from
google-research commit `758b894e` (`scann/`), without TensorFlow.

### Build and packaging
- CMake build replacing Bazel; C++ dependencies pinned by URL and SHA-256
  and fetched with FetchContent (abseil 20260817.0, protobuf 36.2, highway
  1.4.0, Eigen 5.0.1, zlib 1.3.2, pybind11 3.1.0); cnpy and googletest's
  `gtest_prod.h` vendored in `third_party/`. Offline builds via
  `FETCHCONTENT_SOURCE_DIR_*`.
- Static and shared libraries (`scann::core_static`, `scann::core_shared`),
  usable from other CMake projects with FetchContent/`add_subdirectory`.
- Python package (`pip install .`, scikit-build-core) with upstream's
  `scann.scann_ops_pybind` API.
- Rust crate `scann-core` (cxx bindings) that builds the C++ library itself.
- Builds with clang ≥ 19 (tested 19, 21, 23, 24) and GCC ≥ 13 (tested
  13, 14, 16). Upstream only ever built with clang; making GCC work fixed
  type errors, missing declarations and C++17 violations that clang let
  through (see NOTICE).
- CI (clang 19 and 20, GCC 13 and 14, ASan/UBSan, TSan, clippy, MSRV
  1.88) and a release
  workflow for manylinux wheels, PyPI and crates.io.

### Added
- `scann_core::ConfigBuilder`: C++ port of Python's `ScannBuilder`, checked
  against the Python builder for 75 option sets. It returns errors where the
  Python builder silently ignores options.
- Rust API: single/batched/parallel search, add/upsert/delete/reserve/
  rebalance, serialize/load (compatible with Python), health stats,
  `ConfigBuilder`. `ScannIndex` is `Send + Sync`.
- Tutorial (`docs/tutorial/`), API reference, algorithms guide, anisotropic
  quantization explainer; C++, Rust and FetchContent examples.

### Fixed (upstream bugs; details in NOTICE)
- Heap corruption when updating or deleting points in a SOAR (spilled)
  index.
- Malformed or misspelled configs were silently accepted.
- Division by zero for `n_points == 0`; silently wrong dimensionality when
  the dataset size isn't a multiple of `n_points`.
- `RetrainAndReindex` destroyed a locked mutex.
- Parallel batched search crashed with `batch_size = 0`, or without a
  thread pool (single-CPU machines, `set_num_threads(0)`).
- Python `upsert(batch_size=0)` divided by zero; a failed `upsert`/`delete`
  left docids out of sync with the index; `delete` didn't re-attach the
  mutation thread pool after a retrain.
- Undefined behavior in bfloat16 decoding; empty optionals dereferenced in
  memory logging; `MemoryUsage` measured a pointer.
- Clearer errors for wrong dimensionalities.

### Equivalence with upstream
- Deterministic configurations give bit-identical results to the upstream
  Bazel-built wheel, from Python and from Rust; saved indexes load in both
  directions.
