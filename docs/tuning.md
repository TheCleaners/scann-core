# Tuning guide

A reference for choosing the parameters of a partitioned index
(`.tree()`, `.score_ah()`, `.reorder()`) and its search-time settings
(`leaves_to_search`, `pre_reorder_num_neighbors`). It is based on a tuning
study on three datasets: GloVe-100 (angular), SIFT-128 (euclidean) and a
768-dimensional text-embedding set searched at k=100. Every number here
comes from that study unless it says otherwise;
[where the numbers come from](#where-the-numbers-come-from) describes how
each kind was measured.

For a hands-on introduction to the same dials, see
[tutorial part 4](tutorial/04-tuning.md). For each parameter's exact
meaning and default, see [api_reference.md](api_reference.md).

## Contents

* [Recommendations at a glance](#recommendations-at-a-glance)
* [Where the numbers come from](#where-the-numbers-come-from)
* [A per-query cost model](#a-per-query-cost-model)
* [Partitioning: `num_leaves` and `leaves_to_search`](#partitioning-num_leaves-and-leaves_to_search)
* [AH: `dimensions_per_block` and `hash_type`](#ah-dimensions_per_block-and-hash_type)
* [The anisotropic threshold](#the-anisotropic-threshold)
* [Tree AVQ](#tree-avq)
* [Reordering: how many candidates, and at what precision](#reordering-how-many-candidates-and-at-what-precision)
* [SOAR](#soar)
* [Euclidean data: the exact L2 → inner-product reduction](#euclidean-data-the-exact-l2--inner-product-reduction)
* [k=10 and k=100](#k10-and-k100)
* [Knobs that made no measurable difference](#knobs-that-made-no-measurable-difference)
* [How to measure](#how-to-measure)
* [Starting points](#starting-points)

## Recommendations at a glance

For about a million points; the sections below give the evidence and the
exceptions.

| parameter | recommendation | measured effect |
|---|---|---|
| `num_leaves` | 1000–1500 (512–1024 for 768-d at k=100) | GloVe: 1000–1500 beat 2000–4000 at every recall |
| `leaves_to_search` | the recall knob; sweep it | 1–15% of the leaves for recall 0.8–0.995 at k=10 |
| `dimensions_per_block` | 2 at d=100; 3 for SIFT through the L2 → MIPS reduction; 4 at d=768 | 3 on SIFT: +7–17% QPS up to recall 0.995 |
| `anisotropic_quantization_threshold` | scale it with dimension and norm: 0.2 at d=100 and 0.05 at d=768 for unit vectors | 0.2 at d=768 caps recall at 0.795 |
| `tree(avq=...)` | 2.5, dot product only | GloVe +7–8% at recall 0.8–0.9; SIFT (via MIPS) +9–21% |
| `reorder(quantize=...)` | `BFLOAT16` | recall within 0.0001 of float32, half the memory, +1–7% QPS at k=10, +12–26% at k=100 on 768-d |
| `reorder(n)` | 70–500 at k=10, 200–1000 at k=100 (recall 0.8–0.995) | more candidates only fill up to the ceiling `leaves_to_search` sets |
| `tree(soar_lambda=...)` | only for high recall: 0.5 on GloVe (recall ≥ 0.99), 1.0 on 768-d (≥ 0.95) | 3–10% slower than without on GloVe at 0.8–0.95 |
| euclidean data | build on the [L2 → MIPS augmentation](#euclidean-data-the-exact-l2--inner-product-reduction) | SIFT: +12–55% QPS over the untuned plain `squared_l2` grid at recall 0.8–0.995 |
| `hash_type` | `"lut16"` | `"lut256"` was 2.5–3× slower |

Together these gave 9–65% more single-query QPS than the parameter grid
ann-benchmarks used for ScaNN on GloVe, 12–55% on SIFT, and 13–51% on the
768-d set ([table](#against-the-untuned-grids)).

## Where the numbers come from

**Machine and software.** AMD Ryzen Threadripper PRO 7975WX (Zen 4, 32
cores, 32 MB of L3 per 8-core CCD), `powersave` governor, transparent huge
pages `always`, Python 3.12, and the **scann-core 0.2.0 wheel from PyPI**
(the portable build, not `-march=native`). Other work ran on other cores
throughout. The [tutorial](tutorial/README.md)'s numbers come from a
`-march=native` build, and most of them from batched search on 64 threads,
so they aren't directly comparable with these.

**Datasets.**

| dataset | points × dims | metric | k | queries |
|---|---|---|---:|---:|
| glove-100-angular (ann-benchmarks) | 1,183,514 × 100 | angular: unit rows, `dot_product` | 10 | 10,000 |
| sift-128-euclidean (ann-benchmarks) | 1,000,000 × 128 | euclidean | 10 | 10,000 |
| arxiv-nomic-768-normalized (VIBE) | 1,344,643 × 768 | unit rows, `dot_product` | 100 | 1,000 |

The last one is called "the 768-d set" below.

**Protocol.** Each index was built once with 8 training threads,
serialized, and reloaded with `load_searcher` for its sweeps. Searches ran
with `set_num_threads(1)` in one process pinned to one core, one query per
`searcher.search(q, k, reorder, leaves)` call, each call timed alone (the
way ann-benchmarks times its `query()`). QPS is 1 / the mean latency of the
best of 3 passes. Recall is ann-benchmarks' knn recall: a returned point
counts if its true distance is at most the k-th true distance + 0.001.
Measured this way, the untuned grid matched ann-benchmarks' own
single-worker run of scann-core to within 1–7%, with identical recall.

**Measurement sets.** Each number below names the set it comes from.

| set | how | queries × passes |
|---|---|---|
| **fronts** | the final recall/QPS fronts and the untuned-grid baselines; one sweep at a time | 10,000 × 3 (GloVe, SIFT); 1,000 × 3 (768-d) |
| **screens** | earlier sweeps, one at a time on the same core | 2,000 × 3 (GloVe, SIFT); 1,000 × 3 (768-d) |
| **A/B** | several variants timed interleaved in one process, so they all see the same machine state; used for close calls and the cost breakdown | 5,000 |
| **benchmark runs** | ann-benchmarks with 12 parallel workers and a 1-worker rerun; VIBE with 6 workers | the harnesses' own |
| **audit** | a performance audit by code reading (2026-09-28): explanations and estimates only, **nothing measured** | – |

The raw measurements are in the study's result files (one JSON-lines file
per dataset plus a summary of the fronts); they aren't part of this
repository. Numbers marked *derived* are computed here from those
measurements (differences, ratios), not measured directly.

## A per-query cost model

A single query's time is roughly

```
time ≈ fixed + points scanned × cost per point + candidates reordered × cost per candidate
```

where points scanned ≈ n × `leaves_to_search` / `num_leaves` (about twice
that with SOAR, which lists each point in a second leaf). Measured by
changing one parameter at a time (**A/B**, k=10):

| index | fixed (1 leaf, 10 candidates) | AH scan | reorder, per extra candidate |
|---|---:|---|---|
| GloVe, 1500 leaves (tuned) | 12.4 µs | 38 leaves ≈ 30k points: 29 µs (≈ 1.0 ns/point; 50 blocks) | bf16: 58–72 ns |
| GloVe, 2000 leaves (untuned) | 14.2 µs | 60 leaves: 36 µs | f32: 70 ns; bf16: 59 ns |
| SIFT, 600 leaves, plain `squared_l2` | 10.3 µs | 10 leaves ≈ 16.7k points: 18 µs (≈ 1.1 ns/point; 64 blocks) | f32: 99 ns; bf16: 90–117 ns |
| SIFT through MIPS, 1000 leaves, 3 dims/block | 12.6 µs | 17 leaves ≈ 17k points: 14 µs (≈ 0.8 ns/point; 43 blocks) | bf16: 69 ns |

For the 768-d set (k=100, 1024 leaves, 4 dims per block, bf16), *derived*
from the **screens**: about 3.3 ns per scanned point (192 blocks) and
104–112 ns per extra candidate. The cheapest setting measured there, 2
leaves and 100 candidates, took 59.8 µs.

What this says:

* **The fixed cost is 10–14 µs** at d=100–129 and k=10, of which Python and pybind take about
  1.9 µs (a 20-point brute-force searcher answers in 1.9 µs). It caps a
  single thread at about 80,000 QPS on 100-d data (80,722 QPS at 1 leaf and
  10 candidates on GloVe), and is about 30% of a SIFT query at recall 0.9.
* **The AH scan costs about 0.02 ns per block per point** (*derived*: 1.0 ns
  / 50 blocks, 1.1 / 64, 0.8 / 43, 3.3 / 192). On GloVe it is 65–75% of the
  time at recall 0.9–0.95, so recall per scanned point is what counts:
  fewer, larger leaves, tree AVQ, SOAR, and the block size all act on it.
* **Each reorder candidate costs 60–120 ns.** That is one DRAM access per
  candidate: latency, not bandwidth. bf16 halves the bytes but saved only
  about 15% of the reorder time on GloVe (290 candidates: 20.2 µs float32,
  17.2 µs bf16). Rows of 768 floats (3 KB) gain more; see
  [reordering](#precision-bfloat16).

A worked example, GloVe at recall 0.9 (1500 leaves, 38 leaves searched, 100
candidates): 12.4 µs + 29 µs + 90 × 58–72 ns ≈ 47–48 µs. The **fronts**
measured 47.7 µs (20,983 QPS). SIFT through MIPS at recall 0.95 (1000
leaves, 25 searched, 150 candidates): 12.6 + 25,000 × 0.8 ns + 140 × 69 ns ≈
42 µs, against 40.1 µs (24,921 QPS) measured.

The model is for one query on one otherwise idle core. Under concurrency the
same query costs more; see [concurrency](#concurrency).

## Partitioning: `num_leaves` and `leaves_to_search`

### `num_leaves`

Fewer, larger partitions than the usual grids won (**screens**, **fronts**):

| dataset | best | notes |
|---|---|---|
| GloVe (1.18M) | 1000–1500 | beat 2000–4000 at every recall; 500–700 matched 1000 up to recall 0.9 and fell behind from 0.95; 3000 was worse everywhere |
| SIFT (1M) | 600–1000 with plain `squared_l2`; 1000–1500 through MIPS | |
| 768-d (1.34M, k=100) | 512–1024 up to recall 0.95 | above that, SOAR or 2048 leaves |

Fewer centres make routing (comparing the query with every centre) cheaper,
and larger leaves scan more efficiently, while recall per scanned point
barely changed between 1000 and 2000 leaves. For a million points,
1000–1500 is about 1–1.4 × √n, the rule of thumb in
[algorithms.md](algorithms.md). The study covered only datasets of 1–1.35
million points; how the best value scales with n wasn't measured.

The [tutorial](tutorial/04-tuning.md#num_leaves-forgiving) found 1000, 2000
and 4000 leaves nearly equivalent. It measured batched throughput on 64
threads, where routing is batched and costs relatively less; this study
measured single queries on one thread. Tune in the mode you'll serve in
([how to measure](#how-to-measure)).

### `leaves_to_search`

`leaves_to_search` sets the recall ceiling; the reorder count fills up to it.
The fastest settings found for each recall target (**fronts**):

| recall | GloVe (k=10) | SIFT through MIPS (k=10) | 768-d (k=100) |
|---:|---|---|---|
| 0.8 | 10 of 1000 (1.0%), 70 candidates | 10 of 1000 (1.0%), 70 | 8 of 1024 (0.8%), 200 |
| 0.9 | 38 of 1500 (2.5%), 100 | 19 of 1500 (1.3%), 100 | 12 of 1024 (1.2%), 300 |
| 0.95 | 90 of 1500 (6.0%), 150 | 25 of 1000 (2.5%), 150 | 8 of 1024 with SOAR, 400 |
| 0.99 | 150 of 1500 with SOAR (10%), 300 | 75 of 1500 (5.0%), 300 | 24 of 1024 with SOAR, 600 |
| 0.995 | 225 of 1500 with SOAR (15%), 300 | 70 of 1000 (7.0%), 500 | 24 of 1024 with SOAR, 1000 |

Both values can be passed to every search call without rebuilding:
`searcher.search(q, leaves_to_search=38, pre_reorder_num_neighbors=100)`.

## AH: `dimensions_per_block` and `hash_type`

`score_ah(dimensions_per_block)` splits each vector into d / dims_per_block
blocks. More dimensions per block means fewer blocks, so a cheaper scan
(about 0.02 ns per block per point, [above](#a-per-query-cost-model)), but
coarser codes. What won (**screens**, **A/B**):

| data | best | alternatives |
|---|---|---|
| GloVe (d=100, T=0.2) | **2** | 1: 30% more time per point for +0.012 recall. 3: recall caps at 0.981. 4 with `lut256`: 3× slower |
| SIFT through MIPS (d=129) | **3** (43 blocks) | at equal settings +23–28% QPS and −0.009 to −0.029 recall; on the front +7–17% up to recall 0.995. 2 only for recall above 0.999 |
| SIFT, plain `squared_l2` | **2** | 3 and 4 were worse; 4 caps recall at 0.995 |
| 768-d (T=0.05) | **4** (192 blocks) | with bf16 reordering, so the larger candidate counts it needs stay cheap |

The block size and the anisotropic threshold interact: at d=768 with 4 dims
per block, T=0.05 worked, while T=0.1 capped recall at 0.97 and T=0.2 at
0.795 ([next section](#the-anisotropic-threshold)). At d=768, 2 dims per block
makes 384 blocks; the builder turns off its global top-N scan above 256
blocks. The study didn't separate that effect from the others.

`hash_type="lut256"` was 2.5–3× slower than `"lut16"` wherever it was
tried.

## The anisotropic threshold

`score_ah(..., anisotropic_quantization_threshold=T)` makes AH penalise
quantization error parallel to each datapoint more than perpendicular error
(the [explainer](anisotropic_quantization_explained.md) covers why). The
penalty ratio, which ScaNN computes per datapoint
(`ComputeParallelCostMultiplier` in the source), is

```
ratio = (d − 1) · (T² / |x|²) / (1 − T² / |x|²)
```

with d the dimensionality and |x| the norm of the indexed vector (after any
projection or augmentation; the residual's norm isn't used). The same T
therefore means very different things at different dimensions and norms:
the ratio grows with d and shrinks with |x|. Solved for T:

```
T = |x| · sqrt(ratio / (d − 1 + ratio))
```

What was measured (**screens**; ratios *derived* with the formula):

| data | d | \|x\| | T | T / \|x\| | ratio | result |
|---|---:|---:|---:|---:|---:|---|
| GloVe | 100 | 1 | 0.15 | 0.15 | 2.3 | same as 0.2 |
| GloVe | 100 | 1 | **0.2** | 0.2 | 4.1 | used; turning it off costs a median 0.015 recall, up to 0.048 at small reorder counts |
| GloVe | 100 | 1 | 0.3 | 0.3 | 9.8 | same as 0.2 |
| 768-d | 768 | 1 | **0.05** | 0.05 | 1.9 | used; vs. no threshold +9% QPS at recall 0.9, +11% at 0.95, +31% at 0.995 |
| 768-d | 768 | 1 | 0.1 | 0.1 | 7.7 | recall caps at 0.97 |
| 768-d | 768 | 1 | 0.2 | 0.2 | 32 | recall caps at 0.795 |
| SIFT through MIPS | 129 | ≈ 508 | 100 | 0.20 | 5.2 | works |
| SIFT through MIPS | 129 | ≈ 508 | **150** | 0.30 | 12.2 | used |
| SIFT through MIPS | 129 | ≈ 508 | 200 | 0.39 | 23.5 | works |

**Choosing T.** Start from a ratio of about 2–4, then sweep:

* **d=100, unit vectors:** ratio 4 gives T = sqrt(4 / 103) ≈ 0.20. GloVe
  tolerated ratios 2.3–9.8 (T = 0.15–0.3) equally.
* **d=768, unit vectors:** ratio 2 gives T = sqrt(2 / 769) ≈ 0.05. The
  tolerance is narrow here: ratio 7.7 (T=0.1) already capped recall at 0.97.
  Err low at high dimension. (Ratio 2–4 at d=512 is T ≈ 0.06–0.09, and at
  d=1024 T ≈ 0.04–0.06; *derived*, not measured.)
* **Unnormalized data:** T scales with the norm, so pick T / |x| as for
  unit vectors and multiply by a typical norm. On SIFT through MIPS (d=129,
  |x| ≈ 508), 100–200 all worked and 150 ≈ 0.3 × |x| was used. That is a
  ratio of about 12, above the 2–4 that suited the unit-norm sets; ratio 2–4
  there would be T ≈ 63–88, which wasn't tried.

Notes:

* The formula breaks down once T reaches |x| (the perpendicular weight
  becomes zero or negative). With varying norms, keep T well below the
  smallest norms in the data.
* With plain `squared_l2`, a threshold slightly hurt on SIFT (−0.0065
  recall). Through the L2 → MIPS reduction it was needed: without it, recall
  topped out at 0.96 (cause not investigated).
* The VIBE benchmark's grid tries only "off" and 0.1–0.55, all too large at
  d=768 by the table above, which is why its best scann-core configs all
  had the threshold off.

## Tree AVQ

`tree(avq=eta)` places the partition centres with the anisotropic loss
instead of at the plain mean. It requires `distance_measure="dot_product"`.
Of 1, 2.5 and 5, **2.5** was best.

* **GloVe** (**A/B**, 2000 leaves, bf16): recall +0.0067, +0.0025 and
  +0.0011 at the same leaves and candidates (55/95, 130/150, 400/300), at
  2–3% lower QPS. On the fronts that is +7–8% QPS at recall 0.8–0.9 and
  0–5% above.
* **SIFT through MIPS:** +9% at recall 0.8, +11% at 0.9–0.95, +21% at 0.99,
  the largest single build-time gain on SIFT. (Plain `squared_l2` can't use
  it.)
* **768-d:** no gain with 2 dims per block. It wasn't tried with the final
  4-dims, T=0.05 configuration.

## Reordering: how many candidates, and at what precision

### How many candidates

`reorder(n)` (or `pre_reorder_num_neighbors` per search) rescores the n best
AH candidates exactly. Past the ceiling set by `leaves_to_search`, more
candidates cost time and gain nothing. GloVe, 38 of 1500 leaves (**A/B**):

| candidates | recall@10 | latency |
|---:|---:|---:|
| 10 | 0.615 | 41.7 µs |
| 100 | 0.902 | 48.3 µs |
| 300 | 0.906 | 58.8 µs |

On the 768-d set with 4 leaves searched, recall stays at 0.79–0.81 from 300
to 1000 candidates (**screens**).

The candidate counts on the fronts rise with the recall target: 70 → 500 at
k=10 (7–50 × k), and 200 → 1000 at k=100 (2–10 × k). So pick the count
together with `leaves_to_search` rather than as a fixed multiple of k:
sweep both.

### Precision: bfloat16

`reorder(n, quantize=scann.ReorderType.BFLOAT16)` stores the reordering
vectors as bfloat16 instead of float32. Same index otherwise, **A/B**,
GloVe, 2000 leaves:

| leaves / candidates | float32 QPS | bf16 QPS | bf16 gain | int8 QPS | int8 recall loss |
|---|---:|---:|---:|---:|---:|
| 55 / 95 (recall 0.907) | 18,020 | 18,316 | +1.6% | 18,686 | −0.0064 |
| 130 / 150 (0.956) | 9,787 | 10,050 | +2.7% | 10,173 | −0.0076 |
| 400 / 300 (0.991) | 3,844 | 4,024 | +4.7% | 4,114 | −0.0094 |
| 800 / 1000 (0.998) | 1,811 | 1,930 | +6.6% | – | – |

* **bf16:** recall changed by at most 0.0001 on every setting swept. QPS
  +1.0 to +4.0% on SIFT (600 leaves, plain `squared_l2`), and on the 768-d
  set (k=100, 3 KB float32 rows) +12% at recall 0.8, +18% at 0.9, +17% at
  0.95 and +26% at 0.99. It halves the reordering data: the GloVe index went
  from 539 to 301 MB, SIFT's from 581 to 326 MB.
* **int8** is 1–2% faster than bf16 on GloVe but loses 0.006–0.009 recall
  there. On SIFT (plain `squared_l2`, 2000 leaves, 40 / 200) it lost
  **0.046** recall (0.9696 → 0.9239) and was 4% *slower* than bf16 (13,034
  against 13,544 QPS). Don't use it for L2 data; on dot-product data, only
  if memory matters more than that recall.
* The `.reorder()` `anisotropic_quantization_threshold` wasn't studied.

## SOAR

`tree(soar_lambda=λ)` also lists each point in a second leaf, chosen so its
quantization error is unlikely to coincide with the first's. Leaves hold
more points, so it costs scan time and build time, and pays only where
recall would otherwise need many more leaves. Dot product only.

| data | λ | effect (**fronts**, **screens**) |
|---|---:|---|
| GloVe | 0.5 | 3–10% *slower* than no SOAR at recall 0.8–0.95. Best at ≥ 0.99: 4,222 vs 4,203 QPS at 0.99, 3,040 vs 2,948 at 0.995; it reached 0.9996, against 0.9981 at the largest setting swept without it (600 leaves, 500 candidates). Build 9.6 s instead of 7.9 s, index 366 MB instead of 302 |
| 768-d (k=100) | 1.0 | +16–17% at recall 0.95–0.995. Build 61 s instead of 49 s (8 threads) |
| SIFT through MIPS | – | no gain |

On GloVe the gain at 0.99–0.995 is small (+0.5% and +3%); SOAR's main
benefit there is reaching recall the plain index doesn't. Between 0.95 and
0.99 the SOAR and plain fronts interleave. λ 0.5 ≥ 1.0 > 1.5, and
`overretrieve_factor` 1.5 vs 2.0 made no difference. Build SOAR as a
second index for the high-recall end rather than as the default.

## Euclidean data: the exact L2 → inner-product reduction

ScaNN keeps residual AH, the anisotropic loss, tree AVQ and SOAR for
`dot_product`. Euclidean data can use all of them through an exact
reduction: append one coordinate to each database vector and one to each
query, and search by dot product.

With c = the mean of |x|² over the database and a scale s:

```
x' = [x, (c − |x|²) / (2s)]        q' = [q, s]
q'·x' = q·x − |x|²/2 + c/2
```

Since |q − x|² = |q|² − 2(q·x − |x|²/2), the largest q'·x' is exactly the
smallest |q − x|², and reordering rescores the same quantity. (c only
centres the extra coordinate; any constant would do.)

```python
import numpy as np
import scann

S = 1000.0  # the query's extra coordinate; 1,000 and 10,000 measured the same

def augment_database(x, c=None, s=S):                     # x: (n, d) float32
    sq = np.einsum("ij,ij->i", x, x, dtype=np.float64)   # |x|², in float64
    if c is None:
        c = sq.mean()                                     # once, at build time
    extra = (c - sq) / (2.0 * s)
    return np.hstack([x, extra[:, None]]).astype(np.float32), c

def augment_queries(q, s=S):                              # q: (n, d) float32
    return np.hstack([q, np.full((len(q), 1), s, np.float32)])

xa, c = augment_database(x)                               # store c with the index
searcher = (
    scann.scann_ops_pybind.builder(xa, 10, "dot_product")
    .tree(num_leaves=1000, num_leaves_to_search=25,
          training_sample_size=len(xa), spherical=False,
          quantize_centroids=True, avq=2.5)
    .score_ah(3, anisotropic_quantization_threshold=150)  # ≈ 0.3 × |x| for SIFT; see above
    .reorder(150, quantize=scann.ReorderType.BFLOAT16)
    .build()
)

qa = augment_queries(queries)
ids, scores = searcher.search_batched(qa)
sq_l2 = (queries * queries).sum(1)[:, None] + c - 2.0 * scores   # squared L2, if needed
```

* **Keep c and s.** Upserted vectors need the same c and s
  (`augment_database(new_rows, c)`, not a new mean), and so does any code
  that converts scores back to distances. The augmented index returns dot
  products, largest first.
* **Don't use `spherical=True`:** it would normalize the augmented rows.
* **s:** 1,000 and 10,000 were within noise; 100 was slightly worse.
* The threshold is in the augmented data's units: set it from the norms
  ([above](#the-anisotropic-threshold)). The study's queries were augmented
  inside the timed call, so the QPS numbers include that cost.

What it gave on SIFT (**screens**, **A/B**, **fronts**):

* With 2 dims per block, float32 reordering and no tree AVQ, it roughly
  broke even against plain `squared_l2`: +11–17% QPS and −0.007 to −0.013
  recall at the same leaves and candidates.
* It made the anisotropic threshold, tree AVQ and 3 dims per block
  available. With them (and bf16 reordering), the tuned index was 21–55%
  faster than the untuned plain `squared_l2` grid at recall 0.9–0.995
  ([table](#against-the-untuned-grids)).
* SOAR through it didn't help, and PCA to 64 or 96 dimensions had no
  measurable effect.

The builder doesn't offer this reduction itself; the recipe above is manual.

## k=10 and k=100

The k=100 results come from a different dataset (768-d text embeddings)
than the k=10 ones (100-d and 128-d), so the study can't separate the effect
of k from that of dimension. With that caveat:

| | k=10 (GloVe, SIFT) | k=100 (768-d) |
|---|---|---|
| reorder candidates at recall 0.8–0.995 | 70–500 | 200–1000 |
| leaves searched at recall 0.8–0.995 | 1–15% | 0.8–2.3% |
| bf16 reordering vs float32 | +1–7% | +12–26% |
| SOAR pays from recall | 0.99 (GloVe); no gain (SIFT) | 0.95 |
| cost per reorder candidate (bf16) | 58–117 ns | 104–112 ns (*derived*) |
| single-thread QPS at recall 0.9 | 20,983 (GloVe), 31,503 (SIFT) | 7,853 |

At k=100 every query rescores at least 100 candidates. Consistent with
that, the changes that make candidates cheaper (bf16) or better (a suitable
threshold, SOAR) paid more there.

## Knobs that made no measurable difference

On these datasets (**screens**):

* `tree(quantize_centroids=False)` against `True`.
* k-means++ initialization (`random_init=False`) with 20 iterations.
* `score_ah(training_sample_size=len(X))` against the default.
* PCA to 64 or 96 dimensions on SIFT.
* SOAR's `overretrieve_factor` 1.5 against 2.0 (GloVe).

The tuned builds all used `tree(training_sample_size=len(X))`,
`quantize_centroids=True` and the default random initialization.

## How to measure

### Single queries or batches

Tune in the mode you'll serve in. The two modes rank settings differently:
the [tutorial](tutorial/04-tuning.md)'s batched sweep (64 threads) found
`num_leaves` from 1000 to 4000 nearly equivalent, while single queries on
one thread favoured 1000–1500. For single queries, time each
`searcher.search()` call alone:

```python
import os, time
os.sched_setaffinity(0, {8})      # one core
searcher.set_num_threads(1)
leaves, reorder = 38, 100         # the setting under test
best = float("inf")
for _ in range(3):                # best of 3 passes
    total = 0.0
    for q in queries:
        t0 = time.perf_counter()
        searcher.search(q, 10, reorder, leaves)
        total += time.perf_counter() - t0
    best = min(best, total / len(queries))
qps = 1.0 / best
```

For batched serving, time `search_batched_parallel` on the batch sizes and
thread counts you'll use ([tutorial part 5](tutorial/05-saving-and-serving.md#serving-batch-size-latency-and-throughput)).
Use plain `search()` for single queries: a batch of one through
`search_batched_parallel` took 0.089 ms against 0.075 ms (tutorial part 5).

### Concurrency

The same index and settings cost more when other processes run on the
machine (**benchmark runs**, compared with one process alone on its
core):

| setup | slowdown |
|---|---|
| GloVe, 12 ann-benchmarks workers | 1.2× at recall 0.8, 1.7× at 0.9, 1.9× at 0.95, 2.2× at 0.99–0.995 |
| SIFT, 12 ann-benchmarks workers | 1.3–1.4× |
| 768-d, 6 VIBE workers | median 1.83× (10th percentile 1.46×, 90th 2.14×) |
| 4 sweeps on one 8-core CCD (this study's own first screens) | about 40% slower, ±20% noise; one setting gave 4,300–5,400 QPS concurrently and 8,500 alone |

The slowdown grows with recall on GloVe, where queries scan more leaves and
rescore more candidates. The **audit** reads the code as dominated by
memory traffic (AH codes, reorder rows, centres), which fits these
slowdowns but wasn't profiled. Because the slowdown differs between
settings (1.2× to 2.2× on GloVe), the fastest setting for a recall target
can differ too. Check the front under the concurrency you'll run with, and
compare published numbers only when they were measured with the same
parallelism.

### Noise and close calls

* **Recall is deterministic** for a given index and settings, so recall can
  be screened with many sweeps in parallel. QPS can't: time sequentially.
* For differences of a few percent, time the variants interleaved in one
  process (each round runs every variant once), so they all see the same
  machine state.
* Record the machine state with the numbers: CPU governor, transparent huge
  pages mode (`/sys/kernel/mm/transparent_hugepage/enabled`), and the build
  flags. This study ran with THP `always`; the **audit** expects hosts with
  the common `madvise` setting to be slower on reorder-heavy settings
  (an estimate, not measured).
* Use enough queries: 1,000–10,000 here, with 3 passes.

## Starting points

All numbers in this section are from the **fronts**: one query at a time
on one core of a Threadripper PRO 7975WX, scann-core 0.2.0 wheel. Builds
used 8 training threads. Expect different absolute numbers on other
machines, and lower ones [under concurrency](#concurrency); the settings
are starting points for your own sweep, on your own queries.

### About 1M angular vectors, k=10 (GloVe-100)

```python
x = x / np.linalg.norm(x, axis=1, keepdims=True)    # cosine: unit rows
searcher = (
    scann.scann_ops_pybind.builder(x, 10, "dot_product")
    .tree(num_leaves=1500, num_leaves_to_search=38,
          training_sample_size=len(x), spherical=True,
          quantize_centroids=True, avq=2.5)          # add soar_lambda=0.5 for recall >= 0.99
    .score_ah(2, anisotropic_quantization_threshold=0.2)
    .reorder(100, quantize=scann.ReorderType.BFLOAT16)
    .build()
)
```

Build 7.9 s with 8 threads, 302 MB on disk (with SOAR: 9.6 s, 366 MB).

| recall@10 | QPS | leaves | candidates | index |
|---:|---:|---:|---:|---|
| 0.8158 | 37,668 | 15 | 50 | 1500 leaves |
| 0.9017 | 20,983 | 38 | 100 | 1500 leaves |
| 0.9522 | 11,359 | 90 | 150 | 1500 leaves |
| 0.9908 | 4,203 | 300 | 300 | 1500 leaves |
| 0.9905 | 4,222 | 150 | 300 | 1500 leaves + SOAR 0.5 |
| 0.9954 | 3,040 | 225 | 300 | 1500 leaves + SOAR 0.5 |
| 0.9996 | 1,147 | 600 | 500 | 1500 leaves + SOAR 0.5 |

At the low end, 1000 leaves was slightly faster: recall 0.8056 at 39,041
QPS (10 leaves, 70 candidates).

### About 1M euclidean vectors, k=10 (SIFT-128)

The [L2 → MIPS recipe](#euclidean-data-the-exact-l2--inner-product-reduction)
above: 1000 leaves, `avq=2.5`, `score_ah(3, anisotropic_quantization_threshold=150)`
(≈ 0.3 × SIFT's typical norm of 508), bf16 reordering. Build 6.4 s, 306 MB.
For recall above 0.999, the same with `score_ah(2, ...)`: 7.0 s, 328 MB.

| recall@10 | QPS | leaves | candidates | index |
|---:|---:|---:|---:|---|
| 0.8276 | 41,962 | 10 | 70 | 3 dims/block |
| 0.9114 | 31,118 | 18 | 100 | 3 dims/block |
| 0.9521 | 24,921 | 25 | 150 | 3 dims/block |
| 0.9921 | 13,723 | 60 | 300 | 3 dims/block |
| 0.9974 | 10,410 | 80 | 500 | 3 dims/block |
| 0.9990 | 7,175 | 100 | 300 | 2 dims/block |
| 0.9999 | 3,945 | 200 | 500 | 2 dims/block |

### 768-dimensional unit-norm embeddings, k=100 (arxiv-nomic-768)

```python
searcher = (
    scann.scann_ops_pybind.builder(x, 100, "dot_product")
    .tree(num_leaves=1024, num_leaves_to_search=12,
          training_sample_size=len(x), spherical=True,
          quantize_centroids=True)                    # add soar_lambda=1.0 for recall >= 0.95
    .score_ah(4, anisotropic_quantization_threshold=0.05)
    .reorder(300, quantize=scann.ReorderType.BFLOAT16)
    .build()
)
```

Build 49 s (61 s with SOAR), 8 threads. (bf16 reordering data for 1.34M ×
768 is about 2.1 GB, against 4.1 GB in float32; *derived*.) The same
settings weren't tuned on other 512–1024-d sets; start the threshold from
the [formula](#the-anisotropic-threshold) for your dimension.

| recall@100 | QPS | leaves | candidates | index |
|---:|---:|---:|---:|---|
| 0.8327 | 10,164 | 8 | 200 | 1024 leaves |
| 0.9034 | 7,853 | 12 | 300 | 1024 leaves |
| 0.9533 | 6,124 | 8 | 400 | 1024 leaves + SOAR 1.0 |
| 0.9927 | 3,141 | 24 | 600 | 1024 leaves + SOAR 1.0 |
| 0.9954 | 2,655 | 24 | 1000 | 1024 leaves + SOAR 1.0 |
| 0.9999 | 1,078 | 96 | 1000 | 1024 leaves + SOAR 1.0 |

### Against the untuned grids

QPS at a recall target, both measured the same way (**fronts**). "Untuned"
is the best point of the parameter grid ann-benchmarks used for ScaNN (on
GloVe mostly 2000 leaves, T=0.2, 2 dims per block, float32 reordering; on
SIFT 600 leaves, plain `squared_l2`, no threshold), and for the 768-d set
the best of 7 of the VIBE grid's 30 builds (float32 reordering).

| dataset | | 0.8 | 0.9 | 0.95 | 0.99 | 0.995 | max recall |
|---|---|---:|---:|---:|---:|---:|---:|
| GloVe | tuned | 39,041 | 20,983 | 11,359 | 4,222 | 3,040 | 0.9996 |
| | untuned | 28,079 | 17,181 | 9,909 | 3,867 | 1,838 | 0.9985 |
| | ratio | 1.39× | 1.22× | 1.15× | 1.09× | 1.65× | |
| SIFT | tuned | 41,962 | 31,503 | 24,921 | 13,767 | 11,160 | 0.9999 |
| | untuned | 37,308 | 25,954 | 18,525 | 9,860 | 7,222 | 0.9969 |
| | ratio | 1.12× | 1.21× | 1.35× | 1.40× | 1.55× | |
| 768-d | tuned | 10,164 | 7,853 | 6,124 | 3,141 | 2,655 | 0.9999 |
| | untuned | 8,989 | 6,124 | 4,057 | 2,217 | 2,065 | 0.9999 |
| | ratio | 1.13× | 1.28× | 1.51× | 1.42× | 1.29× | |
