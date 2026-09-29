# Benchmarks

Speed and recall of scann-core against the upstream `scann` wheel from PyPI,
on x86-64 and on aarch64 (AWS Graviton4). Everything here comes from
[`benchmarks/ann_benchmarks.py`](../benchmarks/ann_benchmarks.py). See
[Reproducing](#reproducing) to run it yourself, on GloVe or on any other
[ann-benchmarks](http://ann-benchmarks.com) dataset.

## Summary

* **x86-64: about 50% more throughput than the upstream wheel for the same
  recall.** On a partitioned index (tree + AH + reorder), latency drops by
  about a third. Quantized (int8) brute force is also about 50% faster.
  Upstream's open-source builds never detect the CPU, so their AVX2 and
  AVX-512 kernels never run. Neither did scann-core 0.1.0's. The fix comes
  from Arm's google-research PR #3374; see [why](#why-x86-got-faster).
* **aarch64: supported and tested, including natively on Graviton4.**
  Arm's Neon/SVE kernels build indexes 7–10% faster and give 22% more
  throughput on float32 brute force. Partitioned search improves by about
  1%, and quantized (int8) brute force is unchanged. Against the upstream
  wheel, that's 4% more throughput and 7% faster index builds.
* **Recall is unchanged** on both platforms: identical on aarch64, and
  within the fourth decimal place on x86 (see
  [Correctness](#correctness)).

## What's measured

The dataset is **GloVe-100** from [ann-benchmarks](http://ann-benchmarks.com):
1,183,514 vectors of dimension 100 and 10,000 queries, with the exact 100
nearest neighbours of each query. Vectors are L2-normalized and searched by
dot product (cosine similarity). It's the tutorial's dataset; see
[the tutorial](tutorial/README.md#the-dataset).

| config | what it is |
|---|---|
| `brute_force_f32` | exact search, float32 |
| `brute_force_int8` | brute force on int8-quantized vectors |
| `tree_ah_reorder` | 2000-leaf tree searching 100 leaves, AH with 2 dimensions per block and anisotropic quantization (threshold 0.2), float32 reordering of the top 100: the tutorial's main index |
| `tree_ah_reorder_int8` | the same with int8 reordering |
| `tree_ah_reorder_noaq` | the same without anisotropic quantization |

Trees are trained with k-means++ (`random_init=False`) on 250,000 samples,
so repeated builds of one implementation train the same index. (Other
datasets scale these settings; see [Other datasets](#other-datasets).)

For each config the script reports:

* **build**: wall time of `builder.build()`, which trains the tree and
  quantizer and encodes the dataset;
* **recall@10** against ann-benchmarks' exact neighbours;
* **QPS**: `search_batched_parallel` over all 10,000 queries using every
  hardware thread, best of at least 5 passes and 2 seconds;
* **latency**: `search()` one query at a time on one thread, mean over
  1,000 queries.

Every figure below is the mean of two full runs. The two runs agreed within
3% for QPS and index build time. Single-query latency agreed within 1% on
Graviton4 and within 7% on x86-64.

## x86-64

AMD Ryzen Threadripper PRO 7975WX (Zen 4: AVX-512 with VNNI), 32 cores /
64 threads; Linux, clang 23.1.1, Python 3.12. Both scann-core builds use
the default ISA flags (`-mavx -mfma`), the same as the upstream wheel's
Bazel build.

| config | | upstream `scann` 1.4.2 | scann-core 0.1.0 | **scann-core 0.2.0** |
|---|---|---|---|---|
| `tree_ah_reorder` | QPS | 323,600 | 329,600 | **487,100** (+51%) |
| | latency | 0.109 ms | 0.108 ms | **0.074 ms** |
| | recall@10 | 0.9006 | 0.9006 | 0.9002 |
| | build | 4.20 s | 4.17 s | 4.25 s |
| `tree_ah_reorder_int8` | QPS | 337,100 | 340,200 | **529,200** (+57%) |
| | latency | 0.107 ms | 0.104 ms | **0.073 ms** |
| | recall@10 | 0.8866 | 0.8866 | 0.8860 |
| `tree_ah_reorder_noaq` | QPS | 326,200 | 327,700 | **490,500** (+50%) |
| | recall@10 | 0.8857 | 0.8857 | 0.8851 |
| `brute_force_f32` | QPS | 11,870 | 12,310 | **14,910** (+26%) |
| | latency | 9.91 ms | 9.69 ms | 9.67 ms |
| `brute_force_int8` | QPS | 2,435 | 2,427 | **3,698** (+52%) |
| | latency | 8.86 ms | 8.68 ms | **4.78 ms** |
| | recall@10 | 0.9745 | 0.9745 | 0.9745 |

Percentages compare with the upstream wheel. Index build times are the same
for all three builds, within 5%.

### Why x86 got faster

ScaNN picks its AVX2 / AVX-512 kernels at run time with
`port::TestCPUFeature()`. The CPUID code behind it sits under
`#ifdef PLATFORM_IS_X86`, a macro defined by a TensorFlow header that
open-source builds never include. It therefore compiles to
`return false`. The released wheel's `scann_pybind.so` has exactly that:
`xor eax,eax; ret`. Every x86 CPU runs the fallback kernels. (Code
dispatched through Highway detects the CPU itself and wasn't affected.)

The first commit of Arm's PR #3374 defines the platform macros from the
compiler's when TensorFlow's header is absent, which turns detection on.
scann-core carries that commit (see [NOTICE](../NOTICE)). Upstream will
get the same speedup once that PR is merged.

`-march=native` doesn't change this: the fallback kernels were chosen at
run time, not compile time. (The [tutorial](tutorial/README.md) builds with
`-march=native`, and its figures are from scann-core 0.2.0.)

## aarch64 (AWS Graviton4)

AWS `c8g.4xlarge`: Graviton4 (Neoverse V2: Neon, dot product, i8mm, SVE2
at 128 bits, bf16), 16 vCPUs; Ubuntu 24.04, clang 20.1.2. scann-core ran
under Python 3.12 and the upstream wheel under Python 3.9 (its aarch64
wheels only work there; see
[the note below](#upstreams-aarch64-wheels-need-python-39)). Both
scann-core builds use the default ISA flags (`-march=armv8-a+simd`). The
Neon/SVE kernels are chosen at run time from `getauxval(AT_HWCAP)`.

The middle column is scann-core without Arm's kernel work: the Neon/SVE
commits from lizhang-arm/google-research PRs #1–#3 reverted. It keeps
PR #3374's run-time feature detection and scann-core's aarch64 build
fixes, so it shows what those kernels add.

| config | | upstream `scann` 1.4.2 | scann-core, no Arm kernels | **scann-core 0.2.0** |
|---|---|---|---|---|
| `tree_ah_reorder` | build | 9.83 s | 10.14 s | **9.13 s** (−10%) |
| | QPS | 68,900 | 71,070 | 71,750 (+1%) |
| | latency | 0.206 ms | 0.200 ms | 0.202 ms |
| | recall@10 | 0.9002 | 0.9002 | 0.9002 |
| `tree_ah_reorder_int8` | build | 9.95 s | 10.23 s | **9.21 s** (−10%) |
| | QPS | 70,000 | 72,350 | 72,990 (+1%) |
| | recall@10 | 0.8860 | 0.8860 | 0.8860 |
| `tree_ah_reorder_noaq` | build | 9.07 s | 9.33 s | **8.71 s** (−7%) |
| | QPS | 69,380 | 71,550 | 72,150 (+1%) |
| `brute_force_f32` | QPS | 2,513 | 2,475 | **3,021** (+22%) |
| | latency | 17.9 ms | 17.8 ms | 17.8 ms |
| `brute_force_int8` | QPS | 573 | 610 | 607 (0%) |
| | latency | 19.7 ms | 20.8 ms | 20.9 ms |
| | recall@10 | 0.9745 | 0.9745 | 0.9745 |

Percentages compare with the middle column.

What the Arm kernels change:

* **Index build is 7–10% faster.** That's the partitioning pass (Neon
  many-to-many distances, PR #1) and AH encoding with anisotropic
  quantization (Neon `IndexDatapointNoiseShaped`, PR #2). AQ's share of
  the build time halves, from 0.8 s to 0.4 s. Arm's PR descriptions
  report about 16% (PR #1) and up to 5% (PR #2) on their own
  measurements.
* **Batched float32 brute force gets 22% more throughput.** Single-query
  latency doesn't change, since that path doesn't use the many-to-many
  kernels.
* **Partitioned search improves by 1%, and int8 brute force doesn't
  change.** PR #3's int8 dot-product kernels (Neon and SVE, about 13%
  faster in the PR's own measurements) don't show up in these
  end-to-end numbers. Int8 brute force is slower than float32 on this CPU
  (20.9 ms against 17.8 ms per query; on x86 it's twice as fast). This is
  the obvious place for future aarch64 work.

Against the upstream wheel, scann-core 0.2.0 is 4% faster at partitioned
search, 7% faster at building, and 20% faster at float32 brute force. The
wheel's single-query int8 brute force is 6% faster (19.7 against
20.9 ms); in batch throughput, scann-core is 6% ahead.

## Correctness

The equivalence harness ([`tests/equivalence/run.py`](../tests/equivalence/run.py))
runs 7 deterministic configs on two synthetic datasets (5000×128 and
4000×768) through scann-core and the upstream wheel, and compares
neighbour lists and distances. See
[README: Equivalence](../README.md#equivalence-with-upstream).

* **aarch64 (Graviton4):** bit-identical to the upstream wheel, for single
  and batched search, with and without Arm's kernels. The C++, Python and
  Rust tests all pass natively. Under QEMU they also pass on six emulated
  CPUs: Cortex-A57 (Neon only), Neoverse N1 (+ dot product), V1 (+ SVE
  256-bit, i8mm), N2 (+ SVE2), and SVE at 512 and 2048 bits. Instruction
  traces confirmed that Cortex-A57 and N1 run the Neon int8 kernels and V1
  and N2 the SVE ones. Recall agrees on all of them and with x86, to
  within one neighbour in 2000.
  See [`scripts/cross-aarch64.sh`](../scripts/cross-aarch64.sh).
* **x86-64:** no longer bit-identical to the wheel, because the two now run
  different kernels, and different kernels round differently.
  * Distances differ by up to 2×10⁻⁷.
  * Neighbour lists are identical in 13 of the 14 config/search-mode
    pairs.
  * The exception is k-means++ tree training on the 768-dimensional
    dataset. There the rounding differences make training pick a slightly
    different partitioner: 183 of 200 queries give identical neighbours,
    and recall@10 is 0.962 against the wheel's 0.982.
  * The wheel itself gives 0.962 on aarch64, where its kernels round
    differently too, so this is the normal spread between trained
    partitioners, not a loss of quality.
  * On GloVe (above) recall differs in the fourth decimal place.
  * Indexes with int8 centroids (`quantize_centroids=True`, not among the
    harness's configs) also pick their leaves with scann-core's
    fixed-point kernel since 0.3: 99.6–100 % of queries search the same
    leaves as with ScaNN's kernel, recall@10 within ±0.0001 on GloVe-100,
    SIFT-128 and 768-d embeddings. `SCANN_EXACT_TOKENIZATION=1` restores
    ScaNN's kernel (see the [changelog](../CHANGELOG.md)).

## Upstream's aarch64 wheels need Python 3.9

Every Linux aarch64 wheel of upstream `scann` 1.4.2 (cp39 to cp313), and
of 1.4.0, contains the same `scann_pybind.so`, built for Python 3.9. On any
other Python version, `pip install scann` succeeds, but `import scann`
fails:

```
ImportError: Python version mismatch: module was compiled for Python 3.9,
but the interpreter version is incompatible: 3.12.3 ...
```

scann-core builds from source for the interpreter you use, so it isn't
affected. To run the upstream wheel on aarch64 for comparison, use a
Python 3.9 environment. For example, with uv:
`uv venv -p 3.9 wheel-venv && VIRTUAL_ENV=wheel-venv uv pip install scann==1.4.2 h5py`.

## Reproducing

```sh
# scann-core (defaults: portable ISA flags, Release)
cmake -S . -B build -G Ninja && cmake --build build
python -m venv .venv && .venv/bin/pip install numpy "protobuf>=7.36.2" h5py
PYTHONPATH=build/python .venv/bin/python benchmarks/ann_benchmarks.py --label scann-core --json scann-core.json

# the upstream wheel, in its own environment (Python 3.9 on aarch64)
python -m venv wheel-venv && wheel-venv/bin/pip install scann==1.4.2 h5py
wheel-venv/bin/python benchmarks/ann_benchmarks.py --label upstream --json upstream.json
```

`--only brute_force_int8 tree_ah_reorder` runs a subset of the configs,
and `--threads N` sets the number of threads for the QPS pass. Datasets
are downloaded on first use into `$SCANN_TUTORIAL_DATA` (default
`~/.cache/scann-core-tutorial`); GloVe is ~485 MB. `--json` also records
the environment: CPU, thread count, scann and Python versions, dataset and
tree settings. Run on an otherwise idle machine, and run the
implementations one after another, not concurrently.

### Other datasets

`--dataset <name>` runs any ann-benchmarks dataset.
- Names ending in `-angular` are L2-normalized and searched by dot
  product, as GloVe is.
- Names ending in `-euclidean` are searched by squared L2, without
  anisotropic quantization (it applies to dot product only), so
  `tree_ah_reorder_noaq` isn't run.
- The tree scales with the dataset: 2000 leaves from a million points up,
  otherwise 2√n, searching 5% of them.

For example, `--dataset fashion-mnist-784-euclidean` (60,000 × 784,
217 MB) on the x86-64 machine above, one run, scann-core 0.2.0-rc.1 with
the default ISA flags:

```
fashion-mnist-784-euclidean: 60000 x 784, squared_l2, 10000 queries; 64 threads
brute_force_f32        build    0.02 s  recall@10 1.0000      33247 QPS    3.871 ms/query
brute_force_int8       build    0.08 s  recall@10 0.9761      14809 QPS    1.619 ms/query
tree_ah_reorder        build    6.21 s  recall@10 0.9986     248518 QPS    0.068 ms/query
tree_ah_reorder_int8   build    6.10 s  recall@10 0.9749     415693 QPS    0.064 ms/query
```

Other sizes to try: `nytimes-256-angular` (290k × 256),
`sift-128-euclidean` (1M × 128), `gist-960-euclidean` (1M × 960, 3.6 GB).
