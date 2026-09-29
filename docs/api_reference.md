# ScaNN Python API Reference

This page documents the `scann.scann_ops_pybind` Python API in detail: every
builder method, every `ScannSearcher` runtime method, their real defaults, and
the gotchas you'll hit that aren't obvious from the one-line docstrings in
`scann_builder.py` / `scann_ops_pybind.py`. For conceptual background on *why*
partitioning/quantization/rescoring exist, see
[docs/algorithms.md](algorithms.md); this page is the "what exactly do I pass"
companion to that page's "why." For measured recommendations on which
values to choose, see the [tuning guide](tuning.md).

Everything here reflects the code in this repository. scann-core's Python
package exposes the same `scann.scann_ops_pybind` API as the upstream
wheel; upstream's TensorFlow-op variant (`scann.scann_ops`) is not part of
the wheel. `scann.tf` provides upstream's `scann_ops` API for TensorFlow
code instead (the same builder, searches returning tensors): from the
wheel through `tf.numpy_function`, and with scann-core's TensorFlow op,
whose searchers can be saved in SavedModels, when that is built from
source with `-DSCANN_BUILD_TF_OP=ON`; see
[docs/tensorflow.md](tensorflow.md#backends).
`scann.torch` wraps it as a `torch.nn.Module` whose searches return
tensors and compile with `torch.compile`; see
[integrations.md](integrations.md#scanntorch-searching-from-pytorch-models).

## Quickstart

```python
import numpy as np
import scann

dataset = np.random.rand(10000, 128).astype(np.float32)
queries = np.random.rand(5, 128).astype(np.float32)

searcher = (
    scann.scann_ops_pybind.builder(dataset, 10, "dot_product")
    .score_brute_force()  # fine for datasets this small; see "Choosing a scoring method" below
    .build()
)

neighbors, distances = searcher.search_batched(queries)
```

For anything past a few tens of thousands of points, you want the full
tree + AH + reorder pipeline. The [tutorial](tutorial/README.md) builds it up
step by step on a million-vector dataset, measures each stage, tunes it, and
then serves it from Python, C++ and Rust.

## The builder entry point

```python
scann.scann_ops_pybind.builder(db, num_neighbors, distance_measure)
```

- `db`: a 2D row-major `np.ndarray`, one datapoint per row.
- `num_neighbors`: the default `k` baked into the searcher (can be overridden
  per-query via `final_num_neighbors` at search time).
- `distance_measure`: **only `"dot_product"` or `"squared_l2"` are valid.**
  Anything else raises `ValueError` — but only when you call `.build()`, not
  when you call `builder()`. The C++ distance-measure registry
  (`distance_measure_factory.cc`) supports many more names (`CosineDistance`,
  `L1Distance`, Jaccard/Hamming variants, etc.), but none of those are
  reachable from this top-level Python parameter. **For cosine similarity,
  L2-normalize your vectors yourself and use `"dot_product"`** — dot product
  on unit vectors is cosine similarity; pair it with `.tree(spherical=True)`
  if you're also partitioning (see below).

## Builder chain grammar

The builder is a fluent config accumulator: each config method just stashes
its kwargs and returns `self`; nothing is validated or assembled into the
actual proto config until `.build()` runs. This has real consequences:

- **Calling the same method twice raises** `Exception("{key} has already been
  configured")`.
- **Exactly one of `.score_ah(...)` / `.score_brute_force(...)` is required.**
  Omitting both, or setting both, raises `ValueError`. This is the one
  non-optional step — every other builder call is optional.
- **`.pca(...)` and `.truncate(...)` are mutually exclusive** — setting both
  raises `ValueError("Exactly 1 of pca or truncate must be set")`. Note this
  message is also raised if you accidentally set *neither* in a code path
  that expects one — neither is required by default, so this only fires if
  something in your code explicitly touches both or your logic assumes one is
  always set.
- **`.autopilot(...)` silently discards every other builder call.** If you
  call `.autopilot()` alongside `.tree()`/`.score_ah()`/`.reorder()`/etc., the
  autopilot path in `create_config()` returns immediately after generating
  its own config — no error, no warning, your manual config is just never
  used. Don't mix `.autopilot()` with manual tuning calls.
- **`.upper_tree(...)` silently does nothing if `.tree(...)` wasn't also
  called.** The docstring says "REQUIRES tree() call in builder," but nothing
  in the code enforces it — the `upper_tree` config is simply never read.

Putting it together, the full grammar is:

```
builder(db, k, distance_measure)
  [.pca(...) | .truncate(...)]
  [.tree(...) [.upper_tree(...)]]
  (.score_ah(...) | .score_brute_force(...))   # exactly one, required
  [.reorder(...)]
  [.l2_as_dot_product(...)]                     # squared_l2 only (scann-core)
  [.set_n_training_threads(...)]
  .build(docids=None)
```

or, as a completely separate mode:

```
builder(db, k, distance_measure).autopilot(...).build(...)
```

## Projection: `.pca(...)` / `.truncate(...)`

Both are optional dimensionality-reduction steps applied before
partitioning/scoring.

- **`.pca(reduction_dim=None, pca_significance_threshold=0.80,
  pca_truncation_threshold=0.6)`** — PCA projection. `input_dim` is read
  automatically from `db.shape[1]`. `pca_significance_threshold` has a
  non-`None` default, so if you want to specify a fixed `reduction_dim`
  instead of a significance-based cutoff, you must explicitly pass
  `pca_significance_threshold=None`.
- **`.truncate(reduction_dim)`** — simple dimensionality truncation (keep the
  first `reduction_dim` dims, drop the rest). Intended for
  Matryoshka/MRL-style embeddings where a prefix of dimensions is meaningful
  on its own. Raises `ValueError` if `reduction_dim >= db.shape[1]`.

## Partitioning: `.tree(...)`

```python
.tree(
    num_leaves,
    num_leaves_to_search,
    training_sample_size=100000,
    min_partition_size=50,
    training_iterations=12,
    spherical=False,
    quantize_centroids=False,
    random_init=True,
    incremental_threshold=None,
    avq=None,
    soar_lambda=None,
    overretrieve_factor=None,
)
```

Partitioning is IVF-style coarse quantization: k-means clusters the dataset
into `num_leaves` "leaves"; every datapoint is assigned to its nearest leaf;
at query time, only the `num_leaves_to_search` leaves nearest the query are
scored, instead of the whole dataset. **Without `.tree(...)`, every query
scores 100% of the dataset.**

**When to use it** (from `docs/algorithms.md`, matches the code's own
`autopilot` cutoffs): brute force under ~20k points, AH without a tree
between ~20k–100k, tree + AH + reorder above ~100k.

- **`num_leaves` / `num_leaves_to_search`** — `num_leaves` should generally be
  on the order of `sqrt(n)` for `n` datapoints (this is the documented rule of
  thumb; ScaNN's own `autopilot()` uses a more elaborate cache-aware formula
  instead — see below). `num_leaves_to_search / num_leaves` is roughly the
  fraction of the dataset scanned per query, so it's your main recall/latency
  knob; tune it against a recall target rather than a fixed formula. Around
  a million points, 1000–1500 leaves did best in single-query tests; see
  [tuning.md](tuning.md#partitioning-num_leaves-and-leaves_to_search).
- **`training_sample_size` (default 100,000)** — how many datapoints are
  sampled to *train* the k-means centroids (every datapoint still gets
  assigned to a leaf afterward regardless of this number; it only affects
  training quality/speed). **This default does not scale with `num_leaves`.**
  ScaNN's own `autopilot()` path sets it to roughly `num_leaves * 200`
  instead. If you're using a large `num_leaves` (tens of thousands or more),
  consider raising this explicitly — the flat 100k default can mean well
  under one training point per leaf on average, producing poor clusters.
- **`spherical`** — `False` (default) uses ordinary Euclidean k-means
  (centroid = mean of cluster). `True` uses spherical k-means (centroid =
  L2-normalized mean direction), the right objective for cosine/angular
  similarity. **A spherical index stores unit vectors:** scann-core
  L2-normalizes the dataset's rows at build time and every upserted vector,
  so scores are computed against the normalized vectors whatever
  `distance_measure` you chose (for `dot_product`, that makes them cosine
  similarities, up to the query's norm). Normalize your data yourself if you
  want to know exactly what is stored; already-normalized rows are kept
  bit for bit. Upstream ScaNN only *tagged* the dataset as unit-norm without
  normalizing it, and then normalized upserted vectors in some
  configurations but not others, so the same vector could score differently
  depending on when it was added.
- **Partitioning training always uses squared L2, regardless of
  `distance_measure`.** Even if you built with `distance_measure="dot_product"`,
  cluster training and database-point-to-leaf assignment both run on
  `SquaredL2Distance`; only *query*-to-leaf routing at search time uses your
  chosen distance measure. (This is intentional — L2 gives geometrically
  balanced clusters — but it's non-obvious from the API.)
- **`random_init` (default `True`)** — uses plain random k-means
  initialization, not k-means++, despite the underlying proto's default being
  k-means++. This is a deliberate builder-level override (likely for training
  speed at scale) but means you get less reproducible/potentially lower-quality
  initial clusters than you might expect. Set `random_init=False` for
  k-means++ if cluster quality matters more than training speed.
- **`min_partition_size` (default 50)** — clusters that end up smaller than
  this during training get reseeded/split. The builder's default (50) is much
  stricter than the raw proto default (1).
- **`avq` (anisotropic vector quantization for centroids) and `soar_lambda`
  (SOAR spilling)** — both **require `distance_measure="dot_product"`**;
  using either with `"squared_l2"` raises `ValueError`. `avq` re-centers
  partition centroids using the anisotropic loss (same idea as
  `anisotropic_quantization_threshold` below, applied to centroids instead of
  AH codes) instead of the plain mean, to better preserve dot-product ranking
  within a cluster. `soar_lambda` spills each point into a second, ideally
  orthogonal, nearby leaf so points near cluster boundaries aren't lost to
  the wrong side. Measured values: `avq=2.5` (best of 1, 2.5 and 5), and
  SOAR only for high recall targets; see [tuning.md](tuning.md#tree-avq)
  and [SOAR](tuning.md#soar).
- **`incremental_threshold`** — an `int` becomes a datapoint-count threshold,
  a `float` becomes a fraction threshold, for when incremental
  re-partitioning should trigger as data is upserted (see `upsert` below).
- **`quantize_centroids`** — if `True`, stores centroids as int8 instead of
  float32 for cheaper query-to-leaf routing, at a small accuracy cost.

### `.upper_tree(...)`

A second, coarser partitioning level over the first tree's own centroids —
useful when `num_leaves` is large enough that scanning every leaf centroid at
query time becomes a bottleneck. Same `num_leaves`/`num_leaves_to_search`/
`avq`/`soar_lambda` knobs, plus `scoring_mode` (how the upper tree's own
centroid lookup is quantized: `FIXED8` / `BFLOAT16` / `FLOAT32`). As noted
above, this is silently inert unless `.tree(...)` is also configured.

## Choosing a scoring method: `.score_ah(...)` vs `.score_brute_force(...)`

Exactly one of these is required.

### `.score_ah(...)` — Asymmetric Hashing (product quantization)

```python
.score_ah(
    dimensions_per_block,
    anisotropic_quantization_threshold=float("nan"),
    training_sample_size=100000,
    min_cluster_size=100,   # accepted but ignored, see below
    hash_type="lut16",
    training_iterations=10,
)
```

**What it does, conceptually:** each vector is split into consecutive chunks
("blocks") of `dimensions_per_block` dimensions. For each block position, a
k-means codebook is trained (16 centroids per block for `hash_type="lut16"`,
256 for `"lut256"`), and every database vector's sub-vector at that block
position is replaced by the ID of its nearest centroid — one byte per block
instead of 4 bytes per float dimension. At query time, the query's distance
to every centroid in every block is precomputed once into small lookup
tables; a database point's approximate distance to the query is then a sum of
table lookups, which is why this is fast and memory-cheap compared to full
float distance computation. "Asymmetric" refers to the query staying
full-precision while only database points are quantized (more accurate than
quantizing both sides, at the same storage cost). Codebooks are trained via
k-means, controlled by `training_sample_size` (default 100,000, points
sampled for training) and `training_iterations` (default 10).

- `dimensions_per_block` — no default, required. `docs/algorithms.md`
  recommends **2** as a starting point; higher-dimensional data can use
  more (4 did best on 768-d embeddings, see
  [tuning.md](tuning.md#ah-dimensions_per_block-and-hash_type)). If it
  doesn't evenly divide the dataset's dimensionality, the last block is automatically a smaller
  "remainder" block — this fallback is silent, no warning.
- `hash_type` — `"lut16"` (default, 16 centroids/block, the modern
  SIMD-optimized `INT8_LUT16` lookup path) or `"lut256"` (256 centroids/block,
  plain `INT8` lookup — finer-grained/more accurate but slower per lookup,
  closer to textbook PQ). Anything else raises `ValueError`.
- `min_cluster_size` — **accepted but explicitly ignored** (`del
  min_cluster_size  # Deprecated field.` in the source). Don't rely on it;
  it's a no-op left in the signature for backward compatibility.
- `anisotropic_quantization_threshold` — see the dedicated section below.
  This is the actual named contribution of the ScaNN paper and the single
  most impactful accuracy knob for AH.
- Whether *residual* quantization is used, and whether the "global top-N"
  optimization kicks in, are **not exposed as parameters** — they're derived
  automatically inside `create_config()` from whether `.tree()` was
  configured and whether `distance_measure == "dot_product"`. You can't
  toggle these directly from Python.

#### `anisotropic_quantization_threshold`: what it actually does

Standard product quantization minimizes total reconstruction error (squared
distance between a vector and its quantized approximation) treating error in
every direction equally. But for nearest-neighbor / inner-product search,
what matters isn't total reconstruction error — it's how much the
quantization error corrupts the *ranking*. Error in the direction *parallel*
to a vector biases its estimated norm/inner product directly; error
*perpendicular* to it mostly cancels out in aggregate and barely affects
ranking. The ScaNN paper ("Accelerating Large-Scale Inference with
Anisotropic Vector Quantization," Guo et al. 2020, linked in the README)
introduces a quantization loss that's deliberately anisotropic: it spends its
error budget mostly on the perpendicular directions and protects the parallel
direction, instead of treating both equally.

(For a layman's walkthrough of the paper itself, see
[docs/anisotropic_quantization_explained.md](anisotropic_quantization_explained.md).)

In code (`scann/utils/noise_shaping_utils.h`), the `threshold` value is
compared against each vector's own squared norm to compute a
parallel-to-perpendicular cost ratio, which then greedily biases how
quantization rounds each dimension. A `NaN` (the default) disables this
entirely — plain isotropic quantization. There's no single "correct" value
asserted anywhere in this codebase's comments; the paper's own guidance (and
common practice) is to tune empirically starting somewhere around 0.2, but
treat that as a starting point to sweep, not a hardcoded recommendation from
this repo. Because the ratio depends on the dimension and the norm, the same
value means different things on different data: 0.2 suits unit vectors of
~100 dimensions, but caps recall at 0.795 on 768-dimensional ones, where
0.05 works. [tuning.md](tuning.md#the-anisotropic-threshold) gives the
formula and how to scale the threshold. The same knob (same underlying
idea) is also exposed on `.reorder(...)` and `.tree(..., avq=...)` — three
different pipeline stages, same anisotropic-loss concept applied to AH
codes, rescoring, and partition centroids respectively.

### `.score_brute_force(quantize=ReorderType.FLOAT32)`

Skips AH; every database point is scored exactly against the query. The only
knob is `quantize`: `FLOAT32` (no compression, exact), `INT8` (quantize
*stored* vectors to int8 for a 4x memory reduction — the distance computation
itself still runs exactly against the quantized values), or `BFLOAT16`.
Quantized brute force can be faster in memory-bandwidth-bound scenarios but
is *slower* than plain brute force when compute-bound (large batches, or a
dataset small enough to fit in cache) — it's a bandwidth/compute tradeoff,
not a strict win. `quantize` also accepts legacy `True`/`False` (`True` →
`INT8`, `False` → `FLOAT32`) for backward compatibility with an older
signature.

## Rescoring: `.reorder(...)`

```python
.reorder(
    reordering_num_neighbors,
    quantize=ReorderType.FLOAT32,
    anisotropic_quantization_threshold=float("nan"),
)
```

AH scoring is fast but lossy. Reordering takes the top
`reordering_num_neighbors` candidates from the (approximate) scoring pass and
re-scores *just those* against the query using exact float32 vectors (or a
less-aggressive `INT8`/`BFLOAT16` re-quantization via `quantize`), then
returns the final `k` from this re-ranked set. **Highly recommended whenever
you use `.score_ah(...)`.**

`reordering_num_neighbors` should be greater than `k` (the final
`num_neighbors`). There's no single hardcoded multiplier asserted for this
specific parameter in the codebase; as a starting point for tuning, try
something in the 2–10x range and adjust against your recall target — raising
it trades speed for accuracy (the fastest settings measured used 7–50× k at
k=10 and 2–10× k at k=100, rising with the recall target). Same
`True`/`False` backward-compatibility shim on `quantize` as
`.score_brute_force()`.

For `quantize`, `BFLOAT16` is the measured recommendation: recall within
0.0001 of `FLOAT32` on the same index, half the memory, and faster single
queries. `INT8`
lost 0.006–0.009 recall on GloVe and 0.046 on SIFT (euclidean). See
[tuning.md](tuning.md#reordering-how-many-candidates-and-at-what-precision).

## `.autopilot(mode=IncrementalMode.NONE, quantize=ReorderType.FLOAT32)`

Instead of manually tuning `.tree()`/`.score_ah()`/`.reorder()`, delegates to
ScaNN's own autotuner, which picks parameters from dataset size and
dimensionality using real cache-aware formulas (see
`single_machine_autopilot.cc`) — e.g. leaf size is sized to fit L1 cache
during AH scoring, `num_leaves_to_search` grows sub-linearly (roughly a
log-scaled curve) as `num_leaves` grows, and below a size threshold it falls
back to pure brute force automatically (the code-level version of the "under
20k points, use brute force" rule of thumb). **As noted above, this discards
any other builder configuration you've set** — don't combine it with manual
`.tree()`/`.score_ah()`/`.reorder()` calls. `mode` controls whether the
resulting config supports incremental updates (`NONE` / `ONLINE` /
`ONLINE_INCREMENTAL`); see `upsert`/`delete` below for what that enables.

## `.l2_as_dot_product(scale=None, center=None)` (scann-core)

For `"squared_l2"` builders: searches by squared L2 distance through an
inner-product index, the exact L2 → MIPS reduction, so that the
dot-product-only techniques apply to euclidean data (residual and
anisotropic AH, `tree(avq=...)`, `tree(soar_lambda=...)`). The index stores
each datapoint x as `[x, (center − |x|²) / (2·scale)]` and searches each
query q as `[q, scale]`; `q'·x' = q·x − |x|²/2 + center/2` is largest
exactly where `|q − x|²` is smallest. Everything else in the chain is then
configured as for `"dot_product"` over `d + 1` dimensions (AH blocks
included: 129 dimensions are 43 blocks of 3).

- **Results are squared L2 distances** (`|q|² + center − 2·q'·x'`, clamped
  at 0), and the searcher applies the reduction to every query, batch and
  upsert, in every binding. `rebalance()` keeps `scale` and `center`,
  `serialize()` saves them, and `config()` shows them.
- **`scale`** (> 0): the query's extra coordinate. Default: 0.4 × the
  dataset's RMS norm, `0.4·sqrt(mean |x|²)`. It weighs the norm term in
  partitioning and quantization, never exactness; see
  [tuning.md](tuning.md#euclidean-data-the-exact-l2--inner-product-reduction)
  for the measurements.
- **`center`**: any constant (it centres the extra coordinate). Default:
  the dataset's mean `|x|²`. Pass it (and `scale`) to reproduce an index
  built with the manual recipe.
- Not combinable with `tree(spherical=True)`, `truncate()` or
  `autopilot()`; an empty dataset needs an explicit `scale`.
- A saved index records the reduction in `scann_config.pb`, with a
  `distance_measure` that loaders without it (upstream ScaNN, scann-core
  0.2.0) reject at load time.

C++: `ConfigBuilder::L2AsDotProduct({.scale, .center})`; Rust:
`ConfigBuilder::l2_as_dot_product(L2AsDotProductOptions)`.

## Thread configuration

- **`.set_n_training_threads(threads)`** — threads used only during index
  *training* (and by `rebalance()` on that searcher). Default 0: the CPUs
  the process may use (see below).
- **`searcher.set_num_threads(num_threads)`** — a separate, *runtime* setting:
  how many threads `search_batched_parallel`, `upsert` (with `batch_size` >
  1), `delete` and, for an index built with the default training threads
  or loaded, `rebalance` use, **counting the calling thread**: the searcher
  starts `num_threads - 1` pool threads, and 0 or 1 means everything runs
  on the calling thread. (Before 0.2.1 it started `num_threads` pool
  threads, so `set_num_threads(n)` used n + 1 threads.) `search()` and
  `search_batched()` always run on the calling thread.
- **Defaults: the CPUs the process may use**, counted when the searcher is
  built or loaded: the CPUs in the process's affinity mask (`taskset`,
  `numactl`, cpusets, a container's cpuset), capped by a cgroup CPU quota
  (`docker --cpus`, Kubernetes CPU limits; rounded up). Before 0.2.1 it was
  every CPU of the machine, whatever the affinity or quota. The **`SCANN_NUM_THREADS`**
  environment variable (a positive integer) overrides that count for both
  defaults. The pool is started on the first call that needs it, not when
  the searcher is created, so a searcher that is only ever queried with
  `search()` starts no threads.

## Memory: huge pages

After an index is built, loaded or rebalanced, scann-core asks Linux to back
its large, hot buffers with 2 MiB pages (`madvise(MADV_HUGEPAGE)` and, on
Linux ≥ 6.1, `MADV_COLLAPSE`): the reordering data (float32, bfloat16 or
int8 rows) and a tree's AH codes. With 4 KiB pages nearly every reordering
row and every searched leaf costs a TLB miss and a page-table walk. On a
GloVe-100 index (1500 leaves, bfloat16 reordering) this cut page walks from
224 to 2 per query and single-query latency by 4 % (SIFT-128: 2.6 %), on a
host whose THP mode is `always` (the codes, on the malloc heap, weren't
backed by huge pages even then). On hosts in `madvise` mode (Ubuntu's and
Debian's default) the reordering data benefits too. It costs about 25 ms of
load time for GloVe-100. Set **`SCANN_HUGEPAGES=0`** to turn it off. Without
it, `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` makes glibc's malloc request
huge pages for everything (the reordering data included, not the heap-
allocated codes). On multi-socket (or NPS2/NPS4) machines, `numactl
--interleave=all` spreads an index over the memory nodes.

## `ScannSearcher` runtime API

Vectors (queries here, the dataset for `builder`/`create_searcher`, new
points for `upsert`) can be numpy arrays or anything numpy can read,
including PyTorch tensors on the CPU or a GPU; see
[integrations.md](integrations.md#arrays-from-pytorch-and-other-libraries).

### Searching

- **`search(q, final_num_neighbors=-1, pre_reorder_num_neighbors=-1,
  leaves_to_search=-1)`** — single query. `-1` on any parameter means "use
  the value baked in at build time." If you didn't configure `.reorder(...)`,
  `pre_reorder_num_neighbors` is silently forced to equal
  `final_num_neighbors` (a reorder-candidate count is meaningless without a
  reorder stage). Returns `(ids_or_docids, distances)`.
- **`search_batched(queries, final_num_neighbors=None,
  pre_reorder_num_neighbors=None, leaves_to_search=None)`** — same semantics,
  but the "use the build-time default" sentinel is `None` here instead of
  `-1` (internally converted). Runs single-threaded over the batch. Returns
  2D arrays shaped `[num_queries, final_num_neighbors]`. With docids, the
  indices come back as a list of lists of docids instead. Zero queries (a
  `(0, dim)` array) return empty results shaped `[0, k]` (`k` being
  `final_num_neighbors` or the build-time default), or an empty list with
  docids; upstream failed with a misleading dimensionality error.
- **`search_batched_parallel(queries, ..., batch_size=256)`** — same as
  `search_batched` but parallelized: the queries are split into one chunk
  per thread (see `set_num_threads`), of at most `batch_size` queries. On a
  tree index, chunks that would hold 8 queries or fewer are searched one
  query at a time, as `search()` does (for small batches that is faster:
  more threads get work, for about the same total cost), so their
  distances are `search()`'s, which can differ from a batched search's in
  the last bits.

### Persistence: `serialize` / `load_searcher`

```python
searcher.serialize(artifacts_dir, relative_path=False)
searcher2 = scann.scann_ops_pybind.load_searcher(artifacts_dir)
```

- `artifacts_dir` must already exist as a directory — `serialize()` does not
  create it.
- Writes a `scann_assets.pbtxt` manifest plus whichever binary asset files
  are relevant to your config (e.g. `ah_codebook.pb`,
  `serialized_partitioner.pb`, `hashed_dataset.npy`, `int8_dataset.npy`,
  `dp_norms.npy`, `dataset.npy` — not all of these are written for every
  config, only the ones your build actually produced). If you passed
  `docids` to `builder(...).build(docids=...)`, they're additionally pickled
  to `scann_docids.pkl`.
- `relative_path=True` records asset paths in the manifest relative to
  `artifacts_dir`, so the whole directory can be moved/copied intact.
  `relative_path=False` (default) records absolute paths, also when
  `artifacts_dir` is given as a relative path (upstream recorded
  `artifacts_dir/name` as given, which didn't load for a relative
  `artifacts_dir`).
- **Re-serializing into a directory that holds an index replaces it.**
  scann-core writes and fsyncs every file in a staging subdirectory
  (`.scann_staging_*`) first, then replaces `scann_assets.pbtxt` with a
  marker that makes loading fail, renames the new files into place
  (removing index files the new index doesn't have, including a stale
  `scann_docids.pkl`), and renames the new `scann_assets.pbtxt` in last.
  Each rename is atomic; the whole sequence is not. If the process dies or
  the disk fills during a `serialize()`, `load_searcher()` afterwards finds
  either the complete old index, the complete new one, or a directory it
  refuses to load ("This index directory is incomplete: serialize() was
  interrupted ... Serialize the index again."), never a mix of the two.
  Loading a directory *while* another process serializes into it isn't
  covered (it may see the marker, or with bad timing a mix), and neither are
  two concurrent `serialize()` calls into one directory. Upstream ScaNN
  writes in place, config first, so an interrupted re-save there leaves a
  new config next to old files.
- `load_searcher` checks the files against each other and the config (dtypes,
  shapes, row counts, token ranges, SOAR assets, codebook size) and raises
  on any mismatch instead of crashing or loading garbage. This catches most
  mixed directories left by other writers (e.g. an interrupted upstream
  save), but not files swapped between two indexes of identical shape (same
  config, dimensionality and size).
- `load_searcher(artifacts_dir, assets_backcompat_shim=True)` does **not**
  need the original dataset or config to reload — everything required is
  reconstructed from the artifacts directory. If `scann_assets.pbtxt` is
  missing (a directory serialized by an older ScaNN version, before the
  assets-proto convention) and `assets_backcompat_shim=True` (the default),
  it prints a notice and **writes a new `scann_assets.pbtxt` into that
  directory** by inferring it from known filenames — a real, silent
  side-effecting migration of your directory, not a read-only operation.
- **Only load indexes you trust.** `load_searcher` unpickles
  `scann_docids.pkl` when it's present, and unpickling can run arbitrary
  code. Treat an index directory like code, not like data. The C++ and Rust
  loaders don't read that file.

Round-trip pattern:

```python
searcher.serialize(artifacts_dir, relative_path=False)
searcher2 = scann.scann_ops_pybind.load_searcher(artifacts_dir)
# searcher2.search(...) reproduces searcher.search(...)
```

### Dynamic updates: `upsert` / `delete` / `reserve` / `rebalance`

These all require `docids` to have been set at build time
(`builder(...).build(docids=[...])`) — a list of strings, one per datapoint,
duplicates rejected at construction. The searcher keeps its own copy
(`searcher.docids`); upstream used your list, and upserts and deletes then
changed it.

- **`upsert(docids, database, batch_size=256)`** — insert-or-update by docid.
  Accepts a single docid/vector or lists/arrays (1D vectors are
  auto-promoted to a single row). Each docid may appear once per call: a
  repeated one raises `ValueError` before anything changes (upstream added
  a repeated new docid to the index twice but mapped it once, leaving a
  duplicate in `searcher.docids`). The rows are assigned to leaves
  `batch_size` at a time, on the searcher's thread pool when `batch_size` >
  1 (the default was 1 before 0.2.1: one row at a time, about 4× slower
  for 10k rows). After all rows are in, the call runs the index's
  incremental maintenance once (before 0.2.1: after every `batch_size`
  rows), and if that determines the index needs rebuilding, it
  transparently calls a full `rebalance()` for you — **an upsert call can
  silently become an expensive full retrain**, not a cheap incremental
  insert, depending on how much has changed.
- **`delete(docids)`** — removes by docid; raises `KeyError` for unknown
  docids. Uses swap-with-last-element removal internally (mirroring the C++
  side's index recycling) to keep the docid mapping dense.
- **`reserve(num_datapoints)`** — pre-allocates mutator capacity ahead of a
  batch of upserts.
- **`rebalance(config=None)`** — despite the name, this is currently always a
  **full retrain** (there's a `TODO` in the source acknowledging this isn't
  yet a lightweight incremental rebalance). `config`, if given, is a
  text-proto string to retrain with a different configuration than the one
  the searcher currently has; leave it as `None`/empty to keep the same
  config. Retraining needs the datapoints' float values, so it fails with
  `RuntimeError` for indexes that keep only quantized data (int8 or
  bfloat16 brute force, tree or not, and AH without reordering). A failed
  `rebalance()` leaves the searcher unchanged.

### Diagnostics

- **`size()`** — number of indexed datapoints.
- **`config()`** — returns the searcher's live text-proto config. Useful for
  inspecting what `.autopilot()` (or any defaults) actually resolved to,
  since the builder doesn't otherwise expose the assembled config before
  `.build()`.
- **`get_health_stats()`** / **`initialize_health_stats()`** — quantization
  error and partition-imbalance diagnostics. Call
  `initialize_health_stats()` before `get_health_stats()`, or the latter's
  numbers won't be meaningful (this ordering isn't enforced or checked by the
  wrapper — it's on you to call them in the right order).
  `avg_quantization_error` is NaN when it can't be known. A tree without
  float reordering drops its float data once the searcher is created (built,
  loaded, rebalanced), so the error is computed then, and is NaN after an
  `upsert()`/`delete()` or `initialize_health_stats()` until the next
  `rebalance()`. AH without reordering doesn't save float data either: NaN
  after loading. For a tree with a PCA/TRUNCATE projection it is measured
  in the projected space.

## Distance measures: the full picture

The Python builder's `distance_measure` argument accepts **exactly two
strings**: `"dot_product"` and `"squared_l2"`. Nothing else. The C++
distance-measure registry (`distance_measure_factory.cc`) knows about many
more (`CosineDistance`, `L1Distance`, `GeneralJaccardDistance`,
`GeneralHammingDistance`, and others), but none of those are reachable from
this top-level parameter — they exist for internal/proto-level use only. If
you want cosine similarity, L2-normalize your vectors and use
`"dot_product"`. For euclidean data, `"squared_l2"` with
[`.l2_as_dot_product()`](#l2_as_dot_productscalenone-centernone-scann-core)
searches through an inner-product index and still returns squared L2
distances.

## Common errors and what they mean

| Error | Cause |
|---|---|
| `ValueError: distance_measure must be one of ['dot_product', 'squared_l2']` | Passed something other than those two strings to `builder(...)`. |
| `ValueError: hash_type must be one of ['lut16', 'lut256']` | Bad `hash_type` in `.score_ah(...)`. |
| `ValueError: Exactly 1 of pca or truncate must be set` | Both (or an invalid combination of) `.pca(...)`/`.truncate(...)` configured. |
| `ValueError: Exactly 1 of score_ah or score_brute_force must be set` | Neither (or both) `.score_ah(...)`/`.score_brute_force(...)` called — one is mandatory. |
| `ValueError: AVQ only applies to dot product distance (or squared_l2 with l2_as_dot_product()).` | `.tree(avq=...)` used with `distance_measure="squared_l2"`. |
| `ValueError: SOAR requires dot product distance (or squared_l2 with l2_as_dot_product()).` | `.tree(soar_lambda=...)` used with `distance_measure="squared_l2"`. (With `.l2_as_dot_product()`, both are allowed.) |
| `ValueError: l2_as_dot_product() requires the squared_l2 distance measure.` / `... can't be combined with ...` | `.l2_as_dot_product()` on a `"dot_product"` builder, or with `tree(spherical=True)`, `truncate()` or `autopilot()`. |
| `RuntimeError: ... Invalid distance_measure: 'SquaredL2Distance [l2_as_dot_product: needs scann-core >= 0.2.1]'` | Loading an index built with `.l2_as_dot_product()` with a ScaNN that doesn't have it (upstream ScaNN, scann-core 0.2.0). |
| `RuntimeError: Failed to retrain searcher: l2_as_dot_product's scale and center can't change when retraining ...` | `rebalance(config)` with another `scale`/`center` than the index's (or, similarly, a config without `l2_as_dot_product` for an index that has it). |
| `Exception: {key} has already been configured` | Called the same builder method (e.g. `.tree(...)`) twice. |
| `Exception: build() called but no builder lambda was set.` | Internal invariant — shouldn't happen via the public `builder()` entrypoint; indicates a custom/incomplete builder setup. |
| `ValueError: docid and database size mismatch` | `len(docids) != db.shape[0]` passed to `.build(docids=...)`. |
| `ValueError: Duplicates found in docids.` | Non-unique docids passed to `.build(docids=...)`. |
| `ValueError: reduction_dim must be less than {dim}` | `.truncate(reduction_dim=...)` with `reduction_dim >= db.shape[1]`. |
| `KeyError` from `searcher.delete(docids)` | Deleting a docid that doesn't exist in the index. |
| `ValueError` from `searcher.upsert(...)` | Called without `docids` having been set at build time — mutation requires docids. |
| `RuntimeError: ... Failed to parse research_scann.ScannConfig text proto: <line>:<col>: ...` | The config text (from `create_config()` edits, or a hand-written config) doesn't parse, e.g. a misspelled field name. scann-core reports this; upstream silently ran with the part of the config that parsed before the error. |
| `RuntimeError: ... Query has dimensionality X, but the dataset has Y` | A query passed to `search`/`search_batched*` has the wrong number of dimensions. |
| `ValueError: Upsert vector has dimensionality X, but the dataset has Y` | A vector passed to `upsert` has the wrong number of dimensions. Nothing is changed. |
| `ValueError: Upsert batch_size must be >= 1.` / `RuntimeError: ... batch_size must be >= 1` | `batch_size=0` (or negative) passed to `upsert` / `search_batched_parallel`. Upstream crashed the process with a division by zero. |
| `KeyError: Docids to delete are not unique: [...]` | The same docid appears twice in one `delete` call. Nothing is deleted. |
| `ValueError: Docids to upsert are not unique: [...]` | The same docid appears twice in one `upsert` call. Nothing is changed. |
| `ValueError: Dataset has N rows; ScaNN supports at most 4294967295` | Datapoint indices are 32-bit. Upstream silently truncated the row count. `upsert` rejects growing an index past that too. |
| `RuntimeError: Failed to retrain searcher: Retraining (rebalance) needs the datapoints' float values, ...` | `rebalance()` on an index that keeps only quantized data. The searcher is unchanged. |

An `upsert` or `delete` rejected with one of the errors above leaves the
searcher unchanged, docids included: they are checked before anything is
modified. (In upstream ScaNN either could leave the docid mapping out of sync
with the index.)
