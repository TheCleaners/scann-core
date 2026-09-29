# Changelog

All notable changes to scann-core. Versions follow
[semantic versioning](https://semver.org); the version is in `VERSION`.

## Unreleased

### Added
- Autopilot recall target: `autopilot(target_recall=0.95)` (Python; C++
  and Rust `AutopilotOptions::target_recall`) calibrates the index's
  default `leaves_to_search` and `pre_reorder_num_neighbors` when it is
  built: the cheapest setting (by a per-query cost model) that reaches the
  target recall@k on sample queries, against brute force. The index
  structure is the rules' as before. The choice and the target are
  recorded in the config (`autopilot { tree_ah { target_recall ...
  calibration { ... } } }`), so saving, loading and rebalancing keep them.
  - Sample queries: `calibration_queries=` (C++: the last argument of
    `ScannInterface::Initialize`; Rust:
    `ConfigBuilder::build_index_with_calibration_queries`), or else 1000
    datapoints (`calibration_sample_size=`) picked with a fixed seed, each
    one's own datapoint left out of its neighbors. C++
    `ScannInterface::CalibrateSearchDefaults()` recalibrates an index.
  - Measured on the datasets' real queries with datapoints as calibration
    queries (targets 0.9 / 0.95 / 0.99): GloVe-100 k=10 0.8986 / 0.9486 /
    0.9887, arxiv-768 (300k rows) k=100 0.8984 / 0.9504 / 0.9903, SIFT-128
    k=10 0.9086 / 0.9393 / 0.9885; calibrated on 1,000 real SIFT queries,
    the other 9,000 got 0.9166 / 0.9557 / 0.9902. Single-query QPS at the
    calibrated settings: 0.98–1.36× that of the fastest setting of a
    hand-sweep that reaches the target; against the defaults, e.g. 2.0× on
    SIFT at 0.99 (defaults: 0.999). The calibration added 0.2–0.9 s to 2–3 s
    builds of the 1M-point sets and about 1.5 s to a 4.8 s build at k=100
    (16 threads). Details: docs/tuning.md, "A recall target".
  - Without `target_recall`, configs and default settings are unchanged
    (`rules="upstream"` still gives 0.2.0's configs), and saved indexes
    keep their saved defaults. The default settings were left as they are:
    they are cautious on SIFT (half the leaves: 0.9991 → 0.9937 at 1.55×
    QPS) but not on GloVe-100 (0.9625 → 0.9284) or at k=100 on GloVe-100
    (0.941 at the defaults); docs/tuning.md, "Why the default settings
    stay where they are".
  - Rust: `AutopilotOptions` no longer implements `Eq` (it has an `f64`
    field now); `PartialEq` stays.

### Fixed
- A search asking for more neighbors than the index's default
  pre-reordering count (e.g. `search(q, final_num_neighbors=400)` on an
  autopilot index built for k=10, whose default is 317) returned only that
  many; the default now grows to the number of neighbors asked for.

### Performance
- The AH (`lut16`) scan uses ScaNN's AVX-512 kernel on CPUs with AVX-512
  (F, BW, DQ). Upstream compiled it but never ran it: nothing produced the
  code layout it reads. Searchers now keep their LUT16 codes in that layout
  in memory, converted in place when an index is built or loaded (no second
  copy), and adds, updates and deletes keep it.
  - Results are bit-identical to the AVX2 kernel's (ids and distances;
    checked over 30 index kinds, mutated indexes, single, batched and
    parallel search).
  - Saved indexes are unchanged: 0.2.x indexes load and give the same
    results, and indexes saved now load in 0.2.1.
  - Single-query latency on Zen 4 (Threadripper PRO 7975WX), in-process
    A/B against the AVX2 kernel: GloVe-100 tuned −3 to −5 % at recall 0.80,
    −7 to −9 % at 0.90 and 0.95; SIFT-128 (`l2_as_dot_product` tuned) −4 to
    −6 %. The scan alone (microbenchmark): +15–20 % with the codes in L1/L2,
    0 to +11 % from L3. Zen 4 splits 512-bit operations in two; CPUs with
    full-width AVX-512 units may gain more (not measured yet).
  - The kernel is picked when an index is built or loaded; `ignore_avx512`
    (or `SCANN_TEST_FORCE_AVX2` in the tests) set before that keeps AVX2.
  - Upstream kernel fixes on the way: it read up to 48 bytes past the
    codes and the lookup table, required 64-byte-aligned codes, and its
    top-N functions could return different neighbors from the AVX2 ones
    (threshold clamping, int16 accumulators beyond 256 blocks).

## 0.2.1 (2026-09-29)

### Performance
- Thread pools follow the CPUs the process may use: the CPU affinity mask
  (`taskset`, cpusets, containers), capped by a cgroup CPU quota
  (`docker --cpus`, Kubernetes limits), counted at each build or load,
  instead of every CPU of the machine. Under `taskset -c` with 4 of 64
  CPUs, a searcher now runs 4 threads instead of 64 (63 were started per
  index, even for single-query serving): build context switches −88 %,
  `search_batched_parallel` +2.6 % (GloVe-100, 10k queries). The query
  pool starts on the first call that needs it, so searchers that are only
  queried with `search()` start no threads. `search_batched_parallel`
  also counts the calling thread as a worker and hands out one chunk at a
  time: with a small pool, one worker could be left with three chunks
  while the others idled (4 threads: 52k → 61k QPS).
- Batched search does less serial work around its parallel part: the
  queries are searched in place (they were copied with the GIL held, then
  each chunk copied its share again), NaN/infinity is checked per chunk in
  the parallel region (it was checked serially over all queries first, then
  again per chunk), and each chunk writes its results straight into the
  returned arrays (they were reshaped serially, then copied by pybind11
  with the GIL held). The Rust `search_batched*` no longer copy the queries
  either.
- `search_batched_parallel` on tree indexes searches chunks of 8 queries or
  fewer as single queries (with `search()`'s code), which is cheaper for
  one query and spreads small batches over more threads: batch of 1
  −15 % latency (now the same as `search()`), batches of 8 and 64 +16 % and
  +11 % throughput with 8 threads (GloVe-100, tutorial config); batches of
  512 and more are unchanged. Distances of queries searched this way are
  `search()`'s, which can differ from a batched search's in the last bits.

- Less fixed cost per `search()` call: without docids, the Python wrapper
  calls the C++ searcher directly (no docid lock or context manager); the
  results are written straight into the returned arrays (they were copied
  twice); the tree parameters for a `leaves_to_search` value are made once
  per thread instead of per query; the query dimensionality check no longer
  copies a shared pointer that every searching thread shares; the SOAR
  duplicate merge no longer allocates. On a 1,000-point index (1 leaf,
  reorder 10): 3.54 → 3.00 µs per `search()` call (−0.53 µs; −0.26 µs of
  it in C++). On GloVe-100 (53 µs per query) that is within the noise.
- Loading an index (`load_searcher`, and building one) computes the
  health statistics' quantization error in parallel (bit-identical; it
  was a serial pass over the whole dataset): `load_searcher` of GloVe-100
  with float reordering 0.62 → 0.43 s on 16 CPUs. And `load_searcher`
  releases the GIL: other Python threads kept running (the longest stall
  a 1 ms ticker thread saw went from 624 ms to 1 ms).
- `upsert` passes the rows to C++ as one float32 array, read with the GIL
  released (they became one Python array object per row), and runs the
  index's incremental maintenance once per call (it ran after every
  batch; with autopilot that recomputed the autopilot config each time).
  With the new default `batch_size=256` (see Changed), 10,000 rows into a
  200k-point GloVe tree: 128 → 32 ms; at an equal `batch_size`, −6 to
  −7 %. The Rust `upsert` also maintains once per call.
- Huge pages: after a build, load or rebalance, the reordering data and a
  tree's AH codes are `madvise`d `MADV_HUGEPAGE` (and collapsed with
  `MADV_COLLAPSE` on Linux ≥ 6.1). Single-query latency −4.0 % on GloVe-100
  (1500 leaves, bfloat16 reordering) and −2.6 % on SIFT-128, page-table
  walks 224 → 2 per query, on a host with THP `always`, where the heap-
  allocated codes weren't on huge pages; unchanged with 8 processes on
  one CCD (bound by memory bandwidth). About +25 ms of load time.
  `SCANN_HUGEPAGES=0` turns it off. See
  [docs/api_reference.md](docs/api_reference.md#memory-huge-pages).
- `search()` costs less per query, with bit-identical results (the next
  three items). Through the Python API, one pinned thread, tuned
  GloVe-100 (1,500 leaves, int8 centroids, bfloat16 reordering) and
  SIFT-128 (`l2_as_dot_product`, 1,000 leaves), 6 alternating runs,
  medians: 1 leaf / 10 candidates 11.4 → 9.6 µs (GloVe) and 10.4 →
  8.7 µs (SIFT); at recall 0.8 / 0.9 / 0.95, −6 / −2 / 0 % (GloVe, noise
  about ±1.5 %) and −9 / −7 / −6 % (SIFT). `search_batched` and
  `search_batched_parallel` (8 threads) +1 to +11 % QPS.
- The int8 kernel that scores a query against the tree's centroids
  (`quantize_centroids=True`, the tuned configs' tokenization, most of a
  query's cost when few leaves are searched) computes six centroids at a
  time instead of three, reads each 8-byte half straight into the
  conversion, and prefetches each cache line once (results bit-identical:
  each centroid's sum is computed with the same operations in the same
  order). Its bfloat16 twin, used for reordering, gets the same. C++
  `Search()` with 1 leaf and 10 candidates on GloVe-100 (1,500 leaves):
  10.1 → 9.1 µs.
- A tree + AH `search()` makes 2 heap allocations instead of 14 (with the
  next item's per-thread lookup-table buffer): the distances to the
  centroids, the int8-adjusted query, the list of leaves and the top-N
  buffers of tokenization and of the leaf scan are per-thread buffers that
  are reused (the top-N ones initialized exactly as new ones, so results
  are bit-identical). The time saved was within the noise (glibc's
  per-thread cache made those allocations cheap), but a search no longer
  allocates and zero-fills 4–8 KB per query.
- The lookup table of a dot-product AH index (LUT16, the default with
  `score_ah(1–4)` on `dot_product` and `l2_as_dot_product` indexes) is
  computed in one pass over all blocks (it was one distance call per
  block, 50 for GloVe-100) and its conversion to int8 vectorizes, with the
  same values bit for bit: the new code evaluates each entry with the
  operations the generic code uses on this build, and checks that against
  the generic code when the index is first searched (falling back to it
  otherwise). C++ `Search()` with 1 leaf and 10 candidates: GloVe-100
  9.1 → 8.4 µs, SIFT-128 (`l2_as_dot_product`) 8.2 → 7.7 µs.

### Changed
- `autopilot()` chooses the configuration with scann-core's tuned rules
  (`rules="tuned"`, the new default; C++ `AutopilotRules::kTuned`, Rust
  `AutopilotRules::Tuned`; `autopilot { tree_ah { rules: TUNED_V1 } }` in
  the config) instead of upstream ScaNN's. They keep upstream's
  brute-force cutoff, candidate count and reordering precision, and change
  the index: at most about √n leaves, with `leaves_to_search` scaled to
  them; for dot product, tree AVQ 2.5, 2 dimensions per AH block up to 384
  dimensions and about 192 blocks above, and an anisotropic threshold of
  0.2 × the norms' 5th percentile up to 128 dimensions, falling as
  d^-0.75 above (upstream: 0.2 whatever the dimension and norm); rounded
  AH lookup tables; and squared L2 through `l2_as_dot_product` when the
  squared norms vary by at most 5% and the data isn't far from the origin.
  On eight datasets (GloVe-100, SIFT-128, arxiv-nomic-768 and
  imagenet-clip-512 at k=100, 100k subsamples of GloVe and SIFT, and two
  synthetic sets), against upstream's rules: recall at the default search
  settings the same or higher (GloVe-100 0.961 → 0.971); QPS at equal
  recall 1.15–1.28× on SIFT, 1.2–1.9× at 512–768 dimensions (builds
  4.6–4.9× faster there), 1.0–1.9× elsewhere; QPS at the default settings
  0.89–1.39×. Previews (`create_config()`, `ConfigBuilder::Build(n)`)
  assume unit norms and plain squared L2, since the rules measure the data
  when the index is built; the searcher's config records the threshold.
  Indexes built with autopilot by upstream ScaNN or scann-core 0.2.0 keep
  upstream's rules when loaded, retrained or updated, and
  `autopilot(rules="upstream")` builds exactly the previous configs. See
  [docs/tuning.md](docs/tuning.md#defaults-and-autopilot).
- The tuning guide no longer recommends bfloat16 reordering unconditionally:
  on the same index at autopilot's default settings it cost 0.0019
  recall@100 on imagenet-clip-512 and 0.0009–0.0044 exact-id recall on
  three more datasets (none on GloVe-100, SIFT-128 or arxiv-768). The
  builders' defaults (float32) are unchanged.
- The x86 builds use `-mpopcnt` by default (`SCANN_ARCH_FLAGS`), and the
  AVX2/AVX-512 kernels' target attributes include POPCNT (every AVX CPU has
  it; GCC compiled `popcount` to a bit-twiddling sequence without it; with
  clang, ScaNN's hot code already used the instruction: same instruction
  count on GloVe-100).
- ScaNN's x86 `ignore_avx2`, `ignore_avx512`, `ignore_avx512_vnni` and
  `ignore_amx` flags are honored (upstream read none of them). `ignore_amx`
  now defaults to false, so the AMX kernels stay on on Sapphire Rapids and
  later, as they were in practice (upstream documented a default of true
  but never read the flag).
- `upsert(docids, database, batch_size=256)`: the default `batch_size`
  was 1 (upstream's), which prepared and inserted the rows one at a
  time; 256 matches the C++ and Rust APIs. Rows prepared in one batch are
  assigned to leaves from the index as it was before the batch, so
  results can differ slightly from one-row-at-a-time upserts; pass
  `batch_size=1` for the previous behavior.
- `set_num_threads(n)` (Python, `ScannInterface::SetNumThreads`, Rust
  `set_num_threads`) now means n threads in all, the calling one
  included: n − 1 pool threads, and 0 or 1 runs everything on the calling
  thread. It started n pool threads before (n + 1 threads searched),
  while the default was one less than the CPU count. The defaults, for
  the query pool and for `training_threads=0`, are the CPUs the process
  may use; the new `SCANN_NUM_THREADS` environment variable overrides
  them. `rebalance()` trains with the index's `training_threads` (the
  query pool's threads when it was built with the default, or loaded),
  not always the query pool. See
  [docs/api_reference.md](docs/api_reference.md#thread-configuration).
- Tutorial part 5: the reason a batch of one was slower than `search()`
  was wrong (it never reached the thread pool; the batched code path has a
  fixed cost per call), and it is no longer slower.
- `scann.tf` and `scann_tf_ops` are one API with two backends. `import
  scann.tf` uses scann-core's TensorFlow op when `scann_tf_ops` is
  importable (built from source with `-DSCANN_BUILD_TF_OP=ON`), so its
  searchers save in SavedModels (`serialize_to_module()`,
  `searcher_from_module()`), and the `tf.numpy_function` wrapper otherwise
  (the wheel), whose `serialize_to_module()` still raises, now saying how
  to get the op. Results are identical between the backends (dtypes,
  shapes, padding, static shapes; tested call for call). `scann_tf_ops`
  still imports directly: it is the op backend's module. See
  [docs/tensorflow.md](docs/tensorflow.md#backends).
- Search errors from `scann.tf` are `tf.errors.InvalidArgumentError` with
  both backends, eagerly and in graphs, as with upstream's op. The Python
  backend raised the pybind searcher's `ValueError` or `RuntimeError`
  eagerly, and `InvalidArgumentError` or `UnknownError` in graphs.
- The two backends' signatures are aligned on upstream's:
  `create_searcher(db, config, training_threads=0, container="",
  shared_name=None, docids=None)` (the Python backend took `docids` as the
  fourth positional argument) and `load_searcher(dir,
  assets_backcompat_shim=True, shared_name=None)` (the op's second
  positional argument was `shared_name`; a string there still is).
  Eagerly, the Python backend calls the pybind searcher directly instead
  of through `tf.numpy_function`.

### Added
- `autopilot(rules=..., allow_l2_as_dot_product=...)` (Python); C++
  `ConfigBuilder::Autopilot(AutopilotOptions)` with `AutopilotRules`; Rust
  `ConfigBuilder::autopilot_with(AutopilotOptions)` with `AutopilotRules`;
  config fields `AutopilotTreeAH.rules`, `noise_shaping_threshold` and
  `allow_l2_as_dot_product`. Tests: `autopilot` (C++), `python_autopilot`,
  the Rust `autopilot_rules`, and mutation-fuzzer configs with tree AVQ.
- Squared L2 search through an inner-product index, the exact L2 → MIPS
  reduction: `builder(db, k, "squared_l2").....l2_as_dot_product()`
  (C++ `ConfigBuilder::L2AsDotProduct()`, Rust
  `ConfigBuilder::l2_as_dot_product()`; config field
  `l2_as_dot_product { scale center }`). The index stores
  `[x, (center − |x|²) / (2·scale)]` and searches `[q, scale]` by dot
  product, so euclidean data gets ScaNN's dot-product-only techniques
  (residual and anisotropic AH, tree AVQ, SOAR, AH blocks over d + 1
  dimensions); searches still return squared L2 distances. The searcher
  (`ScannInterface`) applies it to the dataset, every query (single,
  batched, parallel; Python, C++, Rust, `scann.torch`, `scann.tf` and the
  TensorFlow and PyTorch ops), every upsert and rebalance, and saves
  `scale` and `center` with the index; loaders without it (upstream ScaNN,
  scann-core 0.2.0) refuse such an index at load time. Defaults: `center`
  = the mean |x|², `scale` = 0.4 × the RMS norm. On SIFT-128 it returns the
  same ids as the manual recipe of the tuning guide (20–55 % faster than
  the best plain `squared_l2` configs at recall 0.9–0.995), 2–7 % faster
  than that recipe (the query's coordinate is appended in C++); see
  [tuning.md](docs/tuning.md#euclidean-data-the-exact-l2--inner-product-reduction).
  C++ callers that mutate through `GetMutator()` convert vectors with the
  new `ScannInterface::ToStoredDatapoints()`.
- A mutation fuzzer (`tests/fuzz/mutfuzz.cc`): random adds, updates,
  deletes, retrains and save/reload on 46 index configs, checked against a
  shadow copy with exhaustive searches that must repeat bit for bit, with
  optional injected leaf failures. Every config runs as a ctest (label
  `mutfuzz`, about 1 s in all; also in the CI sanitizer job), and
  `scripts/fuzz.sh` runs whole campaigns over seeds and modes.
- `scann.tf.backend()` (`"op"` or `"python"`), `get_backend(name)` (either
  backend's module), `available_backends()`, and the `SCANN_TF_BACKEND`
  environment variable (`auto`, `op`, `python`) to override the choice. An
  op that is installed but fails to import falls back to the Python
  backend with a `RuntimeWarning`.
- Both backends have `from_pybind()`, `to_pybind()` and
  `ScannSearcher(pybind_searcher)`; the op backend's `load_searcher()`
  loads directories without a manifest (the backcompat shim) and reports
  a missing directory as the pybind loader does.
- Tests: `python_tf` runs against both backends (the op one when built)
  and checks that they agree, backend selection, and a SavedModel saved
  through `scann.tf` reloaded in a fresh process; with the op built,
  `python_tf_without_op` (the op hidden) and
  `example_py_tensorflow_wrapper_python`.
- `scann.torch` native backend, in a new optional wheel,
  `scann-core-torch` (`pip install scann-core-torch`; package
  `scann_torch_ops`, built from `torch_op/`, or in a CMake build with
  `-DSCANN_BUILD_TORCH_OP=ON`). C++ ops (`torch.ops.scann.search`,
  `search_batched`) built against LibTorch's stable ABI only, so one
  `py3-none-manylinux_2_34` wheel per architecture works with every torch
  >= 2.10 (CPU, CUDA, ROCm) and Python >= 3.10. With it installed,
  `scann.torch.Searcher` keeps the index in its buffers and searches with
  these ops; same API and results as the Python backend (bit for bit).
  What it adds: `torch.export` and AOTInductor programs of models that
  search carry the index and run in other processes; `state_dict()`
  contains the index (`load_state_dict()` of a 0.2.0-era state_dict, which
  has none, keeps the current one); models holding a `Searcher` can be
  pickled (`torch.save(model)`). `scann.torch.backend()`,
  `SCANN_TORCH_BACKEND` and a `backend` argument choose the backend; without
  the package, scann.torch is 0.2.0's Python backend, unchanged. See
  docs/integrations.md.
- New tests: `python_torch_native` (parity for 10 index types, compile,
  export and AOTInductor in fresh processes, state_dict and pickling round
  trips, stale state, errors, concurrency) and `torch_op_symbols` (the op
  library exports nothing and imports only LibTorch's stable C functions
  and the C/C++ runtime). `python_torch` runs on both backends.
- AOTInductor packages of models that search, on NVIDIA GPUs: verified
  with torch 2.14.0+cu132 and 2.11.0+cu128 (RTX 3060 Ti), with the CUDA
  headers from NVIDIA's pip packages (`cuda-toolkit[crt,cccl]` or
  `[nvcc,cccl]` of torch's CUDA version; no system toolkit). The recipe is
  in docs/integrations.md ("AOTInductor on NVIDIA GPUs").
  `python_torch_native` skips that case only when no CUDA toolkit is found,
  not on any compile error.
- `scann_ops_pybind.ScannSearcher` counts the calls that may change the
  index (`upsert`, `delete`, `rebalance`), so that the native backend
  refreshes its copy.
- Docs: [docs/frameworks.md](docs/frameworks.md), scann-core in batch
  jobs and services. Batch retrieval on Ray Data, Spark (`mapInArrow`,
  `mapInPandas`) and Dask: one searcher per worker process, loaded from a
  shipped directory and cached in a module (a cache in `__main__` reloads
  per task: measured), the index's thread pool (sized from the machine's
  CPUs, not the process's) against the framework's parallelism, memory
  per worker, docids. Serving with FastAPI (a shared searcher, `def`
  endpoints, micro-batching, free-threaded Python), Ray Serve, and
  patterns for BentoML and Triton's Python backend. Zero-copy inputs from
  Arrow, pandas and Polars (checked in code, including which conversions
  copy). Dense + sparse hybrid retrieval with an external sparse engine and
  score fusion (RRF, weighted); scann-core's API has no sparse search.
- Examples: [`fastapi_service.py`](examples/python/fastapi_service.py) (a
  FastAPI service run in-process with `TestClient`) and
  [`batch_retrieval.py`](examples/python/batch_retrieval.py) (a saved index
  searched from worker processes, queries from Arrow without a copy, and
  Ray Data when installed), run as the ctests
  `example_py_fastapi_service` and `example_py_batch_retrieval` (skipped
  without FastAPI and httpx2, or pyarrow).
- Tutorial [part 8](docs/tutorial/08-pytorch-and-tensorflow.md), PyTorch
  and TensorFlow, on GloVe-100 with the part 3 index: `scann.torch`
  (tensors in and out, a compiled model that encodes and searches, GPU
  queries, the native backend's `state_dict` and `torch.export`), `scann.tf`
  (`tf.function` on both backends, a SavedModel with the index inside),
  per-call overheads against plain `scann_ops_pybind`, and when to use
  which. Scripts: `docs/tutorial/code/part8_torch.py` and
  `part8_tensorflow.py`.
- Docs: [docs/tuning.md](docs/tuning.md), a tuning guide from single-query
  measurements on GloVe-100, SIFT-128 (k=10) and 768-dimensional
  embeddings (k=100). It covers `num_leaves` and `leaves_to_search`,
  `dimensions_per_block`, how the anisotropic threshold scales with
  dimension and norm, tree AVQ, reorder count and precision (bfloat16),
  SOAR, an exact L2 → inner-product reduction for euclidean data (as a
  manual recipe), k=10 against k=100, a per-query cost model, how to
  measure (single queries, batches, concurrency), and starting
  configurations with their measured recall and QPS. Tutorial part 4, the
  API reference and algorithms.md link to it; part 4 no longer suggests
  2000 leaves around a million points, and it and the API reference say
  that a threshold of 0.2 and 2 dimensions per block suit ~100-dimensional
  data, not every dimension.

### Fixed
- A tree with AVQ (`tree(avq=...)`) and incremental training
  (`incremental_threshold`, or autopilot's `ONLINE` modes) failed on the
  first upsert with "Dimensionality mismatch (d vs. <garbage>)": upstream's
  `KMeansTreeNode::ApplyAvq` left the tree's centres with a cached mutator
  of a destroyed dataset, which incremental training then wrote through
  (undefined behavior). The centres now get a mutator of their own.
- The wheels no longer contain Eigen's headers: 0.2.0's installed 681
  files under `include/eigen3` into site-packages (from Eigen's install
  rules). The wheel now installs only the `scann` package.
- The `tensorflow_serving` example's recall check failed now and then:
  its query tower's weights were unseeded.
- Health stats: trees with a PCA or TRUNCATE projection report their
  `avg_quantization_error` (the distance between the projected datapoints
  and their centroids, as the partitioner sees them), kept up to date by
  upserts and deletes. It was 0 (not computed, as upstream).
- Health stats: `avg_quantization_error` is NaN when it can't be known,
  instead of a wrong number. Trees without float reordering (brute-force,
  int8 or bfloat16 leaves, AH without reordering) release their float data
  after the build, so after an upsert or delete the error can't be updated:
  upstream kept the build's sum divided by the new number of points (it
  went *down* when far-away points were added), and reported 0 after
  `initialize_health_stats()`. `rebalance()` computes it again: it used to
  release the data before computing the stats, which gave 0. (AH without
  reordering saves no float data either: NaN after loading, 0 before.)
- Searches of a SOAR (spilled) tree are deterministic. Candidates found in
  two leaves were merged through a hash map and passed to reordering in its
  iteration order, which abseil varies per table; reordering rounds
  differently depending on a candidate's position, so repeating a search on
  an unchanged index could return distances differing in the last bits.
  Duplicates are now merged by a stable sort on the datapoint index (same
  merged values). The PyPI wheel isn't affected (its abseil doesn't vary
  the order per table).

### Build and packaging
- CI: a `torch-op` job (not on pull requests) builds the op against torch
  2.10.0's headers and tests it with the newest CPU torch; the release
  builds the scann-core-torch wheels (`.github/workflows/torch-wheels.yml`),
  tests them with the scann-core wheel before publishing, and publishes
  them to PyPI after scann-core.
- CI: ccache works for every build job. The FetchContent example and
  `pip install .` builds use fixed directories (they used random ones, so
  about half of each job's compiles missed); the cross-aarch64 job, which
  builds in a container, uses ccache at all.
- CI: the scann-core-torch wheels build in `/project/build/torch-op`, at
  the depth of the scann-core wheels' build directories (it was one level
  deeper, under `torch_op/`), so the relative paths in their command lines
  match and a build without its own cache reuses the scann-core wheels'
  for every compile but the op's own source (about a quarter missed
  before).
- The wheel build is a reusable workflow (`.github/workflows/wheels.yml`).
  CI builds one wheel per architecture on pushes to main, which checks the
  wheel build between releases and saves the ccache that release tags
  restore (a tag's run can't read another tag's caches).
- `api_exercise_avx2` and `artifact_loading_avx2` are registered only on
  x86-64 (elsewhere they repeated the plain tests), and cross-aarch64 runs
  each emulated CPU's tests in parallel.
- CI's `tf-op` job runs every TensorFlow test and example, with
  `SCANN_TF_BACKEND=op` so that an op that fails to load is an error
  rather than a fallback to the Python backend.

## 0.2.0 (2026-09-27)

Changes since 0.2.0-rc.1: fixes for the rest of the audit findings
(loading and saving indexes, config values, mutation), TensorFlow
(`scann.tf`, and an optional op built from source), PyTorch (`scann.torch`,
and tensors as inputs everywhere), examples, and the first release on
crates.io (rc.1 was published to PyPI only). See 0.2.0-rc.1 below for the
rest of 0.2.0. There was no 0.2.0-rc.2 release.

### Changed
- Spherical partitioning (`tree(spherical=True)`) stores unit vectors: the
  dataset's rows are L2-normalized at build time, and so are upserted
  vectors. Upstream only tagged the dataset unit-norm: points added later
  were normalized in some configurations (float brute force, float
  reordering) and stored as given in others, while the original points were
  stored as given, so the same vector scored differently depending on when
  it was added. Data that is already normalized, as spherical partitioning
  expects, builds the same index as before.
- Python `upsert()` rejects a docid listed more than once (ValueError),
  before changing anything. Upstream added a new docid twice to the index
  but mapped it once, leaving a duplicate in `docids`. The Rust `upsert`
  rejects a repeated index the same way.

### Fixed
- Loading a damaged or inconsistent index directory crashed the process
  (segfault, SIGFPE, abort, uncaught C++ exceptions) or loaded silently
  wrong data. Of 57 damaged variants of small indexes, the upstream wheel
  crashed on 9 and loaded 13 inconsistent ones; all 22 are now errors (7
  others are self-consistent and still load, e.g. a missing tokenization
  is recomputed). The loader checks `.npy` headers, dtypes and shapes, row
  counts and dimensionalities across files, token ranges, the partitioner,
  the AH codes and SOAR assets. Files swapped between two indexes of
  identical shape still can't be detected. New test: `artifact_loading`.
- `serialize()` wrote into the directory in place, config first, so an
  interrupted re-save left a new config next to the old files. It now
  stages the files and commits them behind a marker that makes loading
  fail until the save completes. It also removes the previous index's
  files that the new one lacks; before, a stale `scann_docids.pkl` was
  attached to an index saved without docids. New test:
  `python_serialization`.
- `serialize(dir)` with a relative `dir` and the default
  `relative_path=False` wrote an index that didn't load (upstream too): the
  manifest recorded `dir/name`, which loading resolved to `dir/dir/name`.
  The recorded paths are now absolute.
- Searching a tree + AH index where every searched leaf is empty (e.g.
  after deleting all points) crashed with SIGFPE in the AVX2 LUT16 kernel,
  which divided by the leaves' block count of 0 (upstream too; AVX-512 CPUs
  use other kernels). Empty leaves are now skipped. New test:
  `artifact_loading_avx2`; the CI sanitizer job also runs
  `mutation_regressions`.
- A tree with every point deleted serialized to a directory that couldn't
  be loaded; so did any tree with bfloat16 brute-force leaves.
- A failed AH lookup table in a tree search threw from `.value()` instead
  of returning an error.
- Raw config values that crashed the process are now errors when the index
  is built (a config string passed to Python `create_searcher`, Rust
  `ScannIndex::new` or the C API, or a loaded `scann_config.pb`; also
  affects upstream ScaNN). From a sweep of about 1,100 single-field changes
  to builder-made configs, under ASan+UBSan:
  - AH `num_dims_per_block` or `num_blocks` of 0 (SIGFPE or CHECK abort);
  - `INT8_LUT16` with other than 16 clusters per block (heap overflow);
  - binary distances (Hamming, ...) on float data, in any distance field
    (LOG(FATAL));
  - bfloat16 brute force, or an upper tree with int8 or bfloat16 scoring,
    with a distance other than dot product or squared L2 (LOG(FATAL));
  - `fixed_point_multiplier_quantile` outside (0, 1] or NaN for int8 tree
    leaves or int8 reordering (undefined behavior);
  - reloading an AH index with `LimitedInnerProductDistance` and int8
    reordering (null reference; the int8 data isn't serialized).
- A tree with `pca()` or `truncate()` scored with AH without residual
  quantization (e.g. every `squared_l2` tree) failed to build with a bare
  "SCANN_RET_CHECK failure" (upstream too). It now builds and searches with
  the same recall as other trees.
- Incremental training (`incremental_threshold`) together with `pca()`,
  `truncate()` or `upper_tree()` failed at build time with an unclear
  error. The Python builder, `ConfigBuilder` and Rust now reject it when
  the config is built; raw configs get a clear error.
- Undefined behavior (null pointer arithmetic) in distances between an
  empty sparse datapoint and a dense one (UBSan).
- A failed update of a datapoint in a tree index (e.g. its new SOAR spill
  leaf rejecting it) left the datapoint half-updated: the base held the new
  vector while some leaves held the old one, and its old spill assignment
  could be gone. Updates are now all-or-nothing. (Since rc.1's input
  validation, no such failure is known to be reachable from Python or
  Rust; tree + bfloat16 leaves used to fail this way.)
- `rebalance()` of a spherical tree failed with "Input vectors must be unit
  L2-norm" for most scoring configurations; it now works wherever the index
  can be retrained. Retraining an index that keeps only quantized data
  (int8 or bfloat16 brute force, including tree + bfloat16, or AH without
  reordering) still fails, now with an error that says why.
- Python `search_batched`/`search_batched_parallel` with zero queries
  return empty `(0, k)` results instead of failing with a misleading
  dimensionality error. The parallel path reports a query dimensionality
  mismatch clearly instead of with a bare `SCANN_RET_CHECK_EQ failure`.
- The Python searcher keeps its own copy of the `docids` passed to
  `build()`. Upstream kept the caller's list, which `upsert()` then
  appended to and `delete()` reordered.
- A Python dataset with more than 2^32 - 1 rows is rejected instead of
  having its row count truncated to 32 bits; upserts that would grow an
  index past that are rejected too (Python and Rust).

### Added
- An optional TensorFlow op, built from source against the installed
  TensorFlow with `-DSCANN_BUILD_TF_OP=ON` (Linux; not part of the wheel,
  unsupported), and the `scann_tf_ops` package with upstream's
  `scann_ops` API: `builder()`, `create_searcher()`, `search*()`, and
  working `serialize_to_module()` / `searcher_from_module()`, so a model
  that searches saves as a SavedModel, index included. Unlike upstream's
  op it uses only TensorFlow's C API (no TensorFlow abseil/protobuf; the
  library exports no symbols), keeps every index file in `tf.Variable`s
  (so SOAR, int8, bfloat16 and all-deleted trees work), and caches the
  searcher per index id and variable fingerprint, rebuilding it when the
  variables change. Verified with TensorFlow 2.21.0 (`tensorflow-cpu`, and
  the same library in the CUDA-built `tensorflow` wheel, also with a GPU
  in use); it can't be
  loaded by TensorFlow Serving's stock model server. Tests
  `python_tf_ops`, `tf_op_symbols`, example
  [`examples/python/tensorflow_op.py`](examples/python/tensorflow_op.py),
  a CI job, and [docs/tensorflow.md](docs/tensorflow.md#the-op-backend-scann_tf_ops-build-from-source).
- `scann.torch`: the searcher as a `torch.nn.Module` (`builder()`,
  `create_searcher()`, `load_searcher()`, `Searcher.from_pybind()`), whose
  `search`/`search_batched`/`search_batched_parallel` take query tensors on
  any device and return int64/float32 tensors on the same device, always k
  wide, missing results as index -1 and distance NaN. The searches are
  PyTorch custom ops with exact fake shapes, so models that search compile
  with `torch.compile(fullgraph=True)`, also with dynamic batch sizes;
  `torch.export` raises (planned with a native op in 0.2.1). Optional
  dependency: `pip install 'scann-core[torch]'` (torch ≥ 2.10); `import
  scann` still doesn't import PyTorch. Checked with torch 2.10 (CPU), 2.11
  and 2.14 (CUDA) and 2.14 (ROCm). Test `python_torch`, example
  [`examples/python/torch_retrieval.py`](examples/python/torch_retrieval.py),
  [docs/integrations.md](docs/integrations.md#scanntorch-searching-from-pytorch-models).
- C++: `ScannInterface::LoadArtifactsFromMemory()` loads an index from its
  files held in memory, with the same validation as loading a directory.
- Python: vectors can be PyTorch tensors (and other arrays numpy can read)
  everywhere: a float32 CPU tensor without a copy; a GPU tensor copied to
  host memory; a tensor that requires grad detached; bfloat16 converted.
  Upstream rejected the last three with a pybind signature error. New
  test: `python_torch_input`.
- `docs/integrations.md`: scann-core as a drop-in for the `scann` wheel.
  LangChain's ScaNN vector store works unchanged (checked against the
  upstream wheel; the new `python_langchain` test runs it in CI).
- Release: the crate is built and verified in parallel with the wheels,
  with ccache; `cargo publish` only uploads. CI builds the packaged crate
  (`cargo package`) on pushes to main, which catches files missing from the
  crate before a release and fills that ccache.
- C++: `ScannInterface::SerializeToDirectory(dir)` writes the whole index,
  manifest included, so that an interrupted save can't leave a mixed
  directory (see Fixed). The Python and Rust `serialize` use it.
- `config_regressions` (C++) and `python_config_validation` tests; the CI
  sanitizer job runs `config_regressions` too.
- Tests: `python_wrapper_edge_cases`; a failed-update case in
  `mutation_regressions`; Rust tests for repeated upsert ids, empty
  batches and spherical upserts.
- `scann.tf`: upstream's TensorFlow `scann_ops` API (`builder`,
  `create_searcher`, `search`/`search_batched`/`search_batched_parallel`
  returning int32/float32 tensors) over the pybind searcher, through
  `tf.numpy_function`. Works eagerly, in `tf.function` and in `tf.data`;
  `serialize_to_module()`/`searcher_from_module()` raise
  `NotImplementedError`. `import scann` still doesn't import TensorFlow.
  Optional dependency: `pip install 'scann-core[tf]'` (TensorFlow ≥ 2.21,
  the first whose protobuf range includes scann-core's).
- `docs/tensorflow.md`: `scann.tf`, serving retrieval next to a TensorFlow
  query model, and why there is no TensorFlow op.
- Test `python_tf`, skipped when TensorFlow isn't installed; CI runs it on
  Python 3.12 with `tensorflow-cpu`.
- Examples: Python (`examples/python/`: quickstart, updating, serving from
  threads, `scann.tf`, serving next to a TensorFlow SavedModel), and
  updating in C++ and Rust. They check their own results and run as ctests
  (`example_py_*`, `example_cpp_*`, `example_rust_*`), with the C++ and
  Rust quickstarts.

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
- `benchmarks/ann_benchmarks.py` and `docs/benchmarks.md`: build time, recall,
  throughput and latency against the upstream wheel, on x86-64 and
  Graviton4. It runs GloVe-100 by default, or any ann-benchmarks dataset
  (angular ones by dot product, euclidean ones by squared L2), and records
  the environment with the results.
- The C++ API test also checks recall against an exact search it computes
  itself, and covers a tree with int8 (fixed-point) centers.
- `python_rebalance_flow` test: an index grown from empty by batched
  upserts, then retrained with `rebalance(config)` into a SOAR tree (the
  flow of big-ann-benchmarks' ScaNN entry; see google-research#2712).
  Also covers the builder's SOAR options, and the error for a tree with
  more leaves than points.

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
- Configs ScaNN mishandled now fail with a clear error when the searcher is
  built:
  - PCA or TRUNCATE to 0 dimensions (PCA aborted the process; negative
    TRUNCATE failed with "vector::_M_range_insert");
  - a tree searching 0 leaves (the searcher built, then every search failed
    with a bare "SCANN_RET_CHECK failure").

  Both are reachable from upstream's Python builder (`pca(0, None)`,
  `tree(n, 0)`).
- `ConfigBuilder` (C++ and Rust) rejects values that don't make sense,
  where the Python builder passes them through:
  - counts below 1;
  - `reordering_num_neighbors` below `num_neighbors` (searches silently
    returned fewer neighbors);
  - `dimensions_per_block` above the dimensionality;
  - residual quantization without a tree;
  - out-of-range `Pca`/`Truncate` dimensions.
- Crashes and index corruption found by an audit of the C++ core (each
  with a regression test; also affects upstream ScaNN):
  - upserting a vector containing NaN or infinity into a tree index
    crashed the process (segfault or heap overflow). Python and Rust
    upserts now check every row before changing anything, so a batch that
    fails on a later row no longer half-applies;
  - building an index on data containing NaN or infinity aborted the
    process; it is now an error naming the row. This includes loading an
    index that contains such values;
  - batched search accepted NaN/infinity queries and returned garbage; it
    now rejects them, as single search already did;
  - `leaves_to_search` on an int8 brute-force index segfaulted; it is now
    ignored on indexes without partitioning;
  - a failed `rebalance()` (e.g. fewer points than leaves after deletions,
    or more children than points) left the index using freed memory, so the
    next upsert or delete crashed or corrupted it. A failed retrain now
    leaves the index unchanged;
  - tree + bfloat16 brute-force indexes couldn't be mutated: every upsert or
    delete failed after half-applying (size shrank, searches returned
    out-of-range indices). More generally, a failed leaf mutation no longer
    leaves a tree's size out of step with its leaves;
  - mutating a tree with a PCA or TRUNCATE projection read out of bounds in
    the health-stats collector (`avg_quantization_error` became inf). For
    projected trees the quantization error isn't tracked (reported as 0);
    partition sizes and imbalance stay correct.
- Python `search_batched` with docids mapped short-result padding to
  `docids[0]`; padded entries are now `None`.
- Undefined behavior in the AVX2 LUT16 search kernel: its prefetch did
  pointer arithmetic on a null pointer for the last partition (UBSan). It
  only runs on CPUs without AVX-512, and only since scann-core uses the AVX2
  kernels. `SCANN_TEST_FORCE_AVX2=1` makes the C++ API test use the AVX2
  kernels on AVX-512 machines too (ctest `api_exercise_avx2`, and in the
  CI sanitizer job).
- Python: `delete()` on a searcher built without docids raised
  `AttributeError`; it now raises the same `ValueError` as `upsert()`.
- Rust: option structs and `ConfigBuilder` are `#[must_use]`, so a dropped
  builder call is a compiler warning.
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
