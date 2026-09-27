# Changelog

All notable changes to scann-core. Versions follow
[semantic versioning](https://semver.org); the version is in `VERSION`.

## 0.2.0-rc.1 (2026-09-27)

Release candidate for 0.2.0 (on PyPI as `0.2.0rc1`; `pip install --pre`).

### Performance
- x86-64: ScaNN's AVX2/AVX-512 kernels are now used. Upstream's open-source
  builds compile CPU detection out (`PLATFORM_IS_X86` is never defined), so
  the upstream wheel and scann-core 0.1.0 always ran the fallback kernels.
  The fix comes from Arm's google-research PR #3374. On GloVe-100 this gives
  about 50% more throughput than the upstream wheel at the same recall for
  tree + AH + reorder, with a third less latency. int8 brute force is also
  about 50% faster. See `docs/benchmarks.md`.

### Added
- aarch64 Linux support, with Arm's work from google-research PR #3374 and
  lizhang-arm/google-research PRs #1–#3 (authors preserved; see NOTICE):
  - run-time CPU feature detection (`getauxval`);
  - Neon many-to-many distances;
  - Neon residual statistics and `IndexDatapointNoiseShaped`, for AH with
    anisotropic quantization;
  - Neon and SVE int8 × float dot products.

  On Graviton4, index builds are 7–10% faster than without these kernels,
  and batched float32 brute force gets 22% more throughput.
- Tested natively on AWS Graviton4 (C++, Python and Rust), and under QEMU
  on six emulated CPUs from Cortex-A57 (Neon only) to SVE2 and 2048-bit
  SVE.
- Cross-compiling: `cmake/toolchains/aarch64-linux-gnu.cmake` (clang +
  lld, with QEMU running protoc and the tests), and
  `scripts/cross-aarch64.sh`, which builds and tests in a container.
- Free-threaded Python (3.14t, 3.15t). The module declares that it doesn't
  need the GIL, and release wheels are built for cp314t and cp315t
  (cibuildwheel 4.2.1). On the tutorial's index, concurrent `search()` calls
  from 64 plain Python threads reach 327k QPS, against about 100k with the
  GIL.
- Release wheels for Linux aarch64 (built on native Arm runners) and for
  CPython 3.15, alongside x86-64 and 3.10-3.14.
- CI on Ubuntu 26.04 (GitHub Actions):
  - x86-64 with clang 19 and 22, and GCC 13 and 15;
  - aarch64 natively;
  - the aarch64 cross build under QEMU;
  - the Python tests on CPython 3.10-3.15t (`scripts/python-versions.sh`).

  Builds are cached with ccache.
- `benchmarks/glove.py` and `docs/benchmarks.md`: build time, recall,
  throughput and latency against the upstream wheel, on x86-64 and
  Graviton4.
- The C++ API test also checks recall against an exact search it computes
  itself, and covers a tree with int8 (fixed-point) centers.

### Changed
- x86-64 results are no longer bit-identical to the upstream wheel, because
  the two now run different kernels. Distances differ in the last bits, and
  k-means++ training can end up with a slightly different partitioner.
  Recall on GloVe-100 matches to the fourth decimal place. aarch64 results
  are bit-identical to the upstream wheel.
- The tutorial's figures were re-measured with this version. Part 5 now
  also measures plain threads, and shows free-threaded Python.
- Copyright and package metadata name Elias Benali (@ebenali) and
  TheCleaners.

### Fixed
- Python searches could run concurrently with `upsert`/`delete` (searches
  release the GIL), and could map their results to the wrong docids. Without
  the GIL, concurrent upserts corrupted the index. `ScannNumpy` now holds a
  reader/writer lock (searches shared, everything else exclusive), and the
  docid bookkeeping has one too. Plain searches cost the same; searches with
  docids cost about 1 µs more. New test: `python_concurrency`.
- Upstream's aarch64 build:
  - `int8_tile.cc` included its per-target header before
    `hwy/foreach_target.h`, which broke Highway's Neon pass;
  - `hwy-compact.cc` required AES for its static Neon target.
- Builds without `NDEBUG` (Debug, or no build type) didn't compile: in
  Highway's debug mode, ScaNN's Highway one-to-many kernels compile to
  nothing, but are still called. Such builds now turn Highway's debug mode
  off, which only disables Highway's internal assertions. Before, scann-core
  hid this by forcing a Release build even as a subproject.
- Used as a subproject (FetchContent/`add_subdirectory`), scann-core no
  longer overrides the parent project's `BUILD_TESTING` or build type.
- `SCANN_USE_SYSTEM_DEPS=ON` rejected a matching system zlib when CMake
  reports both `ZLIB_VERSION` and `ZLIB_VERSION_STRING` (CMake >= 3.26).
- The aarch64 toolchain file never found `qemu-aarch64-static`.
- CMake builds run cargo with `--locked`, so they never rewrite `Cargo.lock`.
- ctest timeouts (5 minutes for the Python tests, 15 for the C++ ones), so a
  hang fails the test instead of stalling for ctest's default 25 minutes.
- Rust: `ScannIndex::health_stats(&self)` could race with itself when
  called from several threads (the C++ call updates cached figures in a
  `mutable` member), which safe Rust must never allow; calls are now
  serialized. Searches are unaffected.
- Rust: `ScannIndex::delete` deleted the given indices one at a time, each
  against the index as the previous deletion left it, so several indices
  could fail halfway, or delete the wrong points (a duplicate deleted two).
  Indices now refer to the index before the call, duplicates are rejected
  up front, and the result lists every datapoint that moved as
  `(old index, new index)`.
- Rust: when cross-compiling (`cargo build --target ...`), the build script
  no longer picks the host's `clang`.
- The Rust equivalence test (opt-in, with fixtures) uses the C++ fixture
  check's criterion (neighbour overlap >= 0.9), since on x86-64 results are
  no longer bit-identical to the upstream wheel.

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
  through (see NOTICE). GCC builds have the same recall; batched brute-force
  search runs at about half of clang's throughput.
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
