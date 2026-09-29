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
* [Defaults and autopilot](#defaults-and-autopilot)

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
| `reorder(quantize=...)` | `BFLOAT16` if the recall holds on your data | half the memory, +1–7% QPS at k=10, +12–26% at k=100 on 768-d; recall within 0.0001 of float32 on the three study datasets, but 0.001–0.004 lower on four of the five others of the [autopilot validation](#defaults-and-autopilot) |
| `reorder(n)` | 70–500 at k=10, 200–1000 at k=100 (recall 0.8–0.995) | more candidates only fill up to the ceiling `leaves_to_search` sets |
| `tree(soar_lambda=...)` | only for high recall: 0.5 on GloVe (recall ≥ 0.99), 1.0 on 768-d (≥ 0.95) | 3–10% slower than without on GloVe at 0.8–0.95 |
| euclidean data | `.l2_as_dot_product()`, the [L2 → MIPS reduction](#euclidean-data-the-exact-l2--inner-product-reduction) | SIFT: +12–55% QPS over the untuned plain `squared_l2` grid at recall 0.8–0.995 |
| `hash_type` | `"lut16"` | `"lut256"` was 2.5–3× slower |
| no time to tune | `autopilot()` | its rules come from this study: on eight datasets, the same or higher recall at its default settings than upstream's rules, and 1.15–1.9× the QPS at equal recall on SIFT and at 512–768 dimensions ([defaults and autopilot](#defaults-and-autopilot)) |

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
  recall). Through the L2 → MIPS reduction it was needed: without it,
  recall was 0.76 instead of 0.95 at 25 leaves / 150 candidates, and 0.92
  at 80 / 500 (0.997 with it). It isn't a hard cap (the study saw recall
  top out at 0.96 in its grid): with 400 leaves and 2,000 candidates it
  reached 0.99, at a quarter of the QPS (`l2_as_dot_product()`, 2,000
  queries; the manual recipe measured the same). Why the anisotropic loss
  matters this much here wasn't investigated.
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

* **bf16:** recall changed by at most 0.0001 on every setting swept on
  these three datasets. It isn't always so: in the
  [autopilot validation](#defaults-and-autopilot) (the same index with
  float32 and bfloat16 reordering, at autopilot's default settings) it
  cost 0.0019 recall@100 on imagenet-clip-512 (CLIP embeddings, whose
  energy sits in a few large coordinates) and, under exact-id recall
  (without ann-benchmarks' 0.001 distance tolerance), 0.0044 on a 100k
  subsample of GloVe, 0.0037 and 0.0009 on two synthetic sets; SIFT's
  integer coordinates are exact in bfloat16. QPS
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

With a constant c and a scale s:

```
x' = [x, (c − |x|²) / (2s)]        q' = [q, s]
q'·x' = q·x − |x|²/2 + c/2
|q − x|² = |q|² + c − 2 q'·x'
```

Since |q − x|² = |q|² − 2(q·x − |x|²/2), the largest q'·x' is exactly the
smallest |q − x|², and reordering rescores the same quantity. (c only
centres the extra coordinate; any constant would do.)

`l2_as_dot_product()` (scann-core 0.2.1) does this inside the index:

```python
import scann

searcher = (
    scann.scann_ops_pybind.builder(x, 10, "squared_l2")      # x: (n, d) float32
    .tree(num_leaves=1000, num_leaves_to_search=25,
          training_sample_size=len(x), quantize_centroids=True, avq=2.5)
    .score_ah(3, anisotropic_quantization_threshold=150)     # ≈ 0.3 × |x| for SIFT; see above
    .reorder(150, quantize=scann.ReorderType.BFLOAT16)
    .l2_as_dot_product()
    .build()
)
ids, sq_l2 = searcher.search_batched(queries)                # squared L2 distances
```

C++: `ConfigBuilder(10, DistanceMeasure::kSquaredL2, d)....L2AsDotProduct()`;
Rust: `ConfigBuilder::new(10, DistanceMeasure::SquaredL2, d)....l2_as_dot_product(L2AsDotProductOptions::new())`.

* **Everything else is configured as for `dot_product` over d + 1
  dimensions.** `tree(avq=..., soar_lambda=...)` are allowed, AH is
  residual by default, and the AH blocks cover d + 1 dimensions (SIFT:
  129 = 43 blocks of 3). The top-level distance stays `squared_l2`.
* **The index keeps c and s.** The searcher appends the extra coordinate to
  the dataset at build time, to every query (single, batched, parallel; the
  Python, C++, Rust, `scann.torch` and `scann.tf` APIs) and to every
  upserted vector; `rebalance()` keeps them, and `serialize()` saves them.
  `config()` shows them: `l2_as_dot_product { scale: … center: … }`.
* **Results are squared L2 distances**, |q|² + c − 2 q'·x' computed in
  double, clamped at 0; the ids are those of the manual recipe below with
  the same c and s.
* **Defaults:** c = the dataset's mean |x|²; s = 0.4 × its RMS norm,
  0.4 · sqrt(mean |x|²) (SIFT: s ≈ 203). Pass `scale=` and `center=` to
  override them (an empty dataset needs an explicit `scale`).
* **Not with** `tree(spherical=True)` (it would normalize the stored
  vectors, extra coordinate included), `truncate()` (it would drop the
  extra coordinate) or `autopilot()`, whose tuned rules use it by
  themselves where the norms are nearly constant
  ([defaults and autopilot](#defaults-and-autopilot)). The threshold
  applies to the stored vectors x' (|x'|² = |x|² plus the extra coordinate squared; on SIFT, whose
  norms vary little, |x'| ≈ |x|): set it from their norms
  ([above](#the-anisotropic-threshold)).

**The scale** sets how much the extra coordinate weighs in partitioning and
quantization; it never affects exactness (reordering scores the exact
q'·x'). On SIFT-128 (1000 leaves, 3 dims per block, threshold 150, bf16
reordering; 2,000 queries), s = 25 to 10,000 (0.05× to 20× the RMS norm of
508) gave the same recall within ±0.004 at every point of the grid, e.g.
0.906–0.909 at 19 leaves / 100 candidates and 0.990–0.991 at 60 / 300. On
synthetic data it mattered: Gaussian and clustered sets (d = 16 and 64,
20,000 points, 2 or 3 dims per block, threshold 0.3 × RMS norm, 8 of 40
leaves, 100 candidates), recall@10 was best at 0.25–0.5 × the RMS norm,
up to 0.28 lower at 0.125×, and mostly lower above 1×, most with 3 dims per
block (d = 64: 0.52 at 0.25×, 0.34 at 2×, 0.19 at 16×; d = 16: 0.65 at
0.5×, 0.32 at 16×). The default 0.4× sits in the best range, and scales
with the data.

**Numerics.** The extra coordinate is computed in double
((c − |x|²) / (2s), with |x|² around 2.6·10⁵ on SIFT) and stored as
float32; its rounding moves q'·x' by about 10⁻⁷ × |c − |x|²|, below the
float32 dot product's own error. The distance |q|² + c − 2 q'·x' cancels
large terms, so its error is relative to |q|² + |x|² rather than to the
distance itself. Measured on SIFT (1,000 queries, 10 results each, typical
distance 4.8·10⁴): with float32 reordering at most 0.016 (3·10⁻⁸ of
|q|² + |x|²); with bfloat16 reordering at most 6.3, median 0.55
(1.2·10⁻⁵ of |q|² + |x|²; SIFT's integer coordinates are exact in
bfloat16, only the extra coordinate and the products round). On data whose
coordinates don't fit bfloat16 exactly it is larger: the mutation fuzzer's
8-dimensional Gaussian data needed a tolerance of 10⁻² × (|q|² + |x|²),
where plain `squared_l2` with bfloat16 needs one relative to the distance.
Recall is unaffected,
but with bfloat16 or int8 reordering distances near 0 (duplicates) aren't
exact. Distances are clamped at 0.

**Points outside the build data.** An upserted point with |x|² > c just
gets a negative extra coordinate; results stay exact. But the coordinate
grows with |x|², so a point whose norm is far outside the data is quantized
coarsely (AH finds it only with many candidates), and retraining
(`rebalance()`) on such outliers degrades the quantization of every point —
as with plain `squared_l2`, where one point at 20× the typical norm took
recall from 0.99 to 0.09 after a rebalance on synthetic data.

**Saved indexes.** `scann_config.pb` holds `l2_as_dot_product` and the
distance measure `"SquaredL2Distance [l2_as_dot_product: needs scann-core
>= 0.2.1]"`, which loaders that don't know the reduction (upstream ScaNN,
scann-core 0.2.0) reject when they load the index ("Invalid
distance_measure"), rather than serving the stored d + 1-dimensional
vectors as a dot-product index. In C++, load such an index with
`ScannInterface` (`LoadArtifacts` or `LoadArtifactsFromMemory`, then
`Initialize`); `CreateSearcher()` alone refuses it. Mutations through
`GetMutator()` take vectors as the index stores them: convert them with
`ToStoredDatapoints()`.

**The manual recipe** (what the tuning study ran) builds the same index
from augmented data; indexes built this way keep working as before:

```python
import numpy as np

S = 1000.0

def augment_database(x, c=None, s=S):                     # x: (n, d) float32
    sq = np.einsum("ij,ij->i", x, x, dtype=np.float64)   # |x|², in float64
    if c is None:
        c = sq.mean()                                     # once, at build time
    extra = (c - sq) / (2.0 * s)
    return np.hstack([x, extra[:, None]]).astype(np.float32), c

xa, c = augment_database(x)                               # keep c for upserts
searcher = scann.scann_ops_pybind.builder(xa, 10, "dot_product")...build()
qa = np.hstack([queries, np.full((len(queries), 1), S, np.float32)])
ids, scores = searcher.search_batched(qa)                 # dot products
sq_l2 = (queries * queries).sum(1)[:, None] + c - 2.0 * scores
```

With the same config, `l2_as_dot_product(scale=S, center=c)` builds the
same index and returns the same ids: on SIFT with
`tree(random_init=False)`, all 10,000 queries at three settings matched,
and the distances matched |q|² + c − 2·score to 0.003. (Pass `center=c`:
the option sums the norms in another order than `np.mean`. With the
default random initialization, training isn't reproducible run to run, so
two builds of either kind can differ.)

What it gave on SIFT (**screens**, **A/B**, **fronts**, with the manual
recipe and s = 1000):

* With 2 dims per block, float32 reordering and no tree AVQ, it roughly
  broke even against plain `squared_l2`: +11–17% QPS and −0.007 to −0.013
  recall at the same leaves and candidates.
* It made the anisotropic threshold, tree AVQ and 3 dims per block
  available. With them (and bf16 reordering), the tuned index was 21–55%
  faster than the untuned plain `squared_l2` grid at recall 0.9–0.995
  ([table](#against-the-untuned-grids)).
* SOAR through it didn't help, and PCA to 64 or 96 dimensions had no
  measurable effect.

`l2_as_dot_product()` against the manual recipe on SIFT, with the tuned
settings above (1000 leaves, `avq=2.5`, 3 dims per block, threshold 150,
bf16 reordering): single queries (`searcher.search()`, the manual recipe
appending s with `np.append`), one thread pinned to one core, 10,000
queries, best of 3 passes, each variant built and measured twice
(scann-core built with `-march=native`, so the QPS are higher than in the
tables below):

| leaves | candidates | manual (s = 1000) recall / QPS | option (default s ≈ 203) recall / QPS | option, s = 1000 |
|---:|---:|---:|---:|---:|
| 10 | 70 | 0.8263 / 44,500 | 0.8264 / 47,500 | 0.8263 / 47,300 |
| 18 | 100 | 0.9102 / 32,800 | 0.9107 / 34,400 | 0.9102 / 34,600 |
| 25 | 150 | 0.9508 / 26,000 | 0.9518 / 27,200 | 0.9508 / 27,200 |
| 30 | 300 | 0.9746 / 20,100 | 0.9748 / 20,800 | 0.9746 / 20,600 |
| 60 | 300 | 0.9915 / 14,300 | 0.9918 / 14,700 | 0.9915 / 14,700 |
| 80 | 500 | 0.9974 / 10,800 | 0.9975 / 11,000 | 0.9974 / 11,000 |

The option reproduces the recipe's recall (to the default scale's
±0.001) and is 2–7% faster: the query's extra coordinate is appended in
C++, not by numpy in Python.


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

The [L2 → MIPS reduction](#euclidean-data-the-exact-l2--inner-product-reduction)
above (`l2_as_dot_product()`; the study used the manual recipe): 1000 leaves, `avq=2.5`, `score_ah(3, anisotropic_quantization_threshold=150)`
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

## Defaults and autopilot

`builder(db, k, distance).autopilot()` chooses the whole configuration from
the data: brute force below a size that depends on the dimensionality,
otherwise a tree with AH and reordering, plus the default search settings
(`leaves_to_search`, `pre_reorder_num_neighbors`, both overridable per
search). Since scann-core 0.2.1 it uses rules derived from this study,
`autopilot(rules="tuned")`, the default. `autopilot(rules="upstream")` uses
upstream ScaNN's rules, as scann-core 0.2.0 did (the same configs, value for
value). An index keeps the rules it was built with: indexes built with
autopilot by upstream ScaNN or scann-core 0.2.0 reload, retrain and update
with upstream's.

### What the tuned rules change

| | upstream's rules | tuned rules | why |
|---|---|---|---|
| brute force below | 42 leaves of 4 × 32 KiB / d points (55,020 at d=100, 8,400 from d=656) | the same | exact results for small data |
| leaves | n / (4 × 32 KiB / d), capped for L3 and training time: 903 for GloVe, 5,000 at 512–768 dimensions | upstream's, at most round(√n): 1,160 for the 768-d set | 1024 leaves beat 2048 on the 768-d set up to recall 0.99; upstream's 5,000 took 5× longer to build |
| `leaves_to_search` | 42 × 2^log₁₀(leaves / 42): 106 of 903, 178 of 5,000 | upstream's × √(leaves / upstream's) with fewer leaves: 86 of 1,160 | coarser leaves need a larger share for the same recall: on the 768-d set, searching upstream's share of 1,160 leaves (42) gave recall 0.990 where upstream's config gave 0.998; twice as many gave 0.998 |
| candidates (`pre_reorder_num_neighbors`) | max(2k, 100√k): 317 at k=10, 1000 at k=100 | the same | |
| reordering precision | `quantize` (float32 by default) | the same | [bfloat16](#precision-bfloat16) lost recall on four of the eight datasets below |
| dims per AH block | 2 | dot product: 2 up to 384 dimensions, then ⌈d / 192⌉ (3 at 512, 4 at 768: at most 192 blocks) | [block size](#ah-dimensions_per_block-and-hash_type); at 512 dimensions 3 balanced speed and recall best (2 was slower, 4 lost 0.035 recall at 300 candidates) |
| anisotropic threshold | 0.2 (dot product) | dot product: 0.2 × the norms' 5th percentile up to 128 dimensions, × (128 / d)^0.75 above (0.071 at 512, 0.052 at 768 on unit vectors) | [the threshold](#the-anisotropic-threshold): 0.2 caps recall at 768 dimensions; it scales with the norm |
| tree AVQ | none | 2.5 for dot product | [tree AVQ](#tree-avq) |
| AH lookup tables | truncated to int8 | rounded (as `score_ah()` builds them) | at 512 dimensions (171 blocks), truncation cost 0.001 recall at the default settings |
| squared L2 | plain | through [`l2_as_dot_product`](#euclidean-data-the-exact-l2--inner-product-reduction) when the squared norms vary by at most 5% (coefficient of variation) and the data isn't far from the origin; `autopilot(allow_l2_as_dot_product=False)` turns it off | SIFT (0.3%): 1.15–1.28× QPS; synthetic clusters whose squared norms vary by 20%: recall at the defaults 0.924 → 0.881 through it |

The threshold, and the choice of `l2_as_dot_product`, come from the data,
so only the built index shows them: `searcher.config()` records the
threshold (`autopilot { tree_ah { noise_shaping_threshold: ... } }`), which
reloading, retraining and incremental maintenance then reuse.
`create_config()` (and C++/Rust `ConfigBuilder` with a point count) shows
the configuration without the data: unit norms for the threshold, and a
squared L2 index without `l2_as_dot_product`.

### Validation

The tuned rules were fitted on the three study datasets and on
imagenet-clip-512. A first validation run on all eight datasets below then
added the limit on `l2_as_dot_product` (the synthetic clusters lost recall
through it) and the rounded lookup tables (imagenet-clip-512); GloVe 100k,
SIFT 100k and the synthetic MIPS set were never used to choose a rule. The
final rules against upstream's:

| dataset | points × dims | distance | k | queries | recall |
|---|---|---|---:|---:|---|
| glove-100-angular | 1,183,514 × 100 | dot product, unit rows | 10 | 10,000 | ann-benchmarks' |
| sift-128-euclidean | 1,000,000 × 128 | squared L2 | 10 | 10,000 | ann-benchmarks' |
| arxiv-nomic-768-normalized | 1,344,643 × 768 | dot product, unit rows | 100 | 1,000 | ann-benchmarks' |
| imagenet-clip-512-normalized (VIBE) | 1,281,167 × 512 | dot product, unit rows | 100 | 1,000 | ann-benchmarks' |
| GloVe 100k: 100,000 random rows of GloVe | 100,000 × 100 | dot product, unit rows | 10 | 2,000 | exact ids |
| SIFT 100k: 100,000 random rows of SIFT | 100,000 × 128 | squared L2 | 10 | 2,000 | exact ids |
| synthetic clusters: 4,096 Gaussian clusters | 5,000,000 × 64 | squared L2 | 10 | 1,000 | exact ids |
| synthetic MIPS: 2,048 clusters, lognormal norms (σ = 0.5) | 1,000,000 × 96 | dot product | 10 | 1,000 | exact ids |

"ann-benchmarks'" recall counts a result whose true distance is within
0.001 of the k-th true distance; "exact ids" counts only the true k
nearest.

**Protocol.** A `-march=native` build of scann-core with the tuned rules.
Every index was built with `autopilot()` (float32 reordering, the default)
and 8 training threads, serialized and reloaded. Single queries, one call
to `searcher.search(q, k, candidates, leaves)` timed at a time, in one
process pinned to one core, one process per dataset timing all its indexes'
settings in turns of 50 queries (the order rotating), 3 rounds; QPS is 1 /
the mean latency. The settings: each index's defaults, and
`leaves_to_search` at 1/32 to 4 times its default with its default
candidate count and with 100 (k=10) or 300 (k=100). QPS at a recall target
is interpolated along each index's front. An ann-benchmarks run shared the
cores' L3 cache and other builds ran on the machine throughout, so the
absolute QPS are about half of those in the sections above: read the
ratios, which are the tuned rules' QPS over upstream's.

| dataset | leaves (searched by default), upstream's → tuned | recall at the defaults | QPS at the defaults | QPS at recall 0.9 / 0.95 / 0.99 | build (8 threads) | index |
|---|---|---|---:|---|---|---|
| GloVe-100 | 903 (106) → 903 (106) | 0.9612 → 0.9713 | 0.96× | 1.16× / 1.11× / 1.87× | 3.0 → 3.1 s | 538 → 538 MB |
| SIFT-128 | 976 (109) → 984 (109), through `l2_as_dot_product` | 0.9990 → 0.9991 | 1.18× | 1.15× / 1.16× / 1.22× | 3.6 → 3.7 s | 581 → 586 MB |
| arxiv-768 (k=100) | 5,000 (178) → 1,160 (86) | 0.9977 → 0.9983 | 1.39× | 1.73× / 1.21× / 1.45× | 116 → 24 s | 4,683 → 4,402 MB |
| imagenet-512 (k=100) | 5,004 (178) → 1,132 (85) | 0.9996 → 0.9998 | 1.18× | 1.85× / 1.73× / 1.59× | 72 → 16 s | 2,978 → 2,853 MB |
| GloVe 100k | 76 (51) → 76 (51) | 0.9961 → 0.9971 | 0.96× | 1.15× / 1.10× / 1.02× | 0.4 → 0.4 s | 46 → 46 MB |
| SIFT 100k | 97 (55) → 98 (55), through `l2_as_dot_product` | 0.9998 → 0.9998 | 1.21× | 1.21× / 1.24× / 1.28× | 0.4 → 0.5 s | 58 → 59 MB |
| synthetic clusters 5M × 64 | 2,441 (143) → 2,236 (137), plain L2 | 0.9237 → 0.9237 | 0.98× | 1.00× / – / – | 10.0 → 10.0 s | 1,461 → 1,461 MB |
| synthetic MIPS 1M × 96 | 732 (100) → 732 (100) | 0.9916 → 0.9925 | 0.89× | 1.01× / 1.50× / 1.20× | 2.3 → 2.3 s | 437 → 437 MB |

* **Recall at the default settings** is the same or higher everywhere
  (GloVe-100 +0.010).
* **At equal recall** the tuned rules are faster everywhere but on the
  synthetic clusters, where both configs are nearly the same: 1.2–1.9× at
  512–768 dimensions, with builds 4.6–4.9× faster, and 1.15–1.28× on SIFT.
  (Near the top of the recall range the fronts have few points; SIFT at
  0.995 interpolates to 0.86×.)
* **At the default settings** QPS is 0.89–1.39×: the defaults stay at
  upstream's high recall. On the synthetic MIPS set, tree AVQ makes the
  default 100 leaves cost 11% more, for +0.0009 recall; about 80 leaves
  there would match upstream's recall.
* Two earlier runs timed each setting as a block of 3 passes (best of 3)
  instead of in turns. They gave the same recall, but their QPS ratios
  scattered with the load between blocks (imagenet-512 at the defaults:
  1.26× in one, 0.80× in the other), hence the turns.

**Against hand-tuned configurations.** The study's tuned configurations
(the [starting points](#starting-points) above) were built and measured in
the same processes. QPS as a share of theirs:

| dataset | recall | upstream's rules | tuned rules | tuned rules, bfloat16 |
|---|---:|---:|---:|---:|
| GloVe-100 | 0.8 / 0.9 / 0.95 | 54% / 63% / 78% | 85% / 72% / 87% | 75% / 74% / 87% |
| SIFT-128 | 0.8 / 0.9 / 0.95 | 74% / 72% / 70% | 83% / 83% / 81% | 90% / 83% / 84% |
| arxiv-768 | 0.8 / 0.9 | 41% / 50% | 81% / 86% | 98% / 99% |

The rest of the gap is what the rules leave out, below: bfloat16, and on
GloVe 1500 leaves, training on every point and quantized centroids; on
SIFT 3 dimensions per block.

### What the tuned rules leave out

* **bfloat16 reordering** by default: half the memory and −1% to +13% QPS
  at the defaults here (+9–13% at 512–768 dimensions), but at the default
  settings (the same index, float32 against bfloat16):

  | | GloVe-100 | SIFT-128 | arxiv-768 | imagenet-512 | GloVe 100k | SIFT 100k | clusters 5M | MIPS 1M |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | recall change | 0.0000 | −0.0001 | 0.0000 | −0.0019 | −0.0044 | 0.0000 | −0.0037 | −0.0009 |

  Ask for it with `autopilot(quantize=scann.ReorderType.BFLOAT16)` after
  checking recall on your data.
* **SOAR**: it pays only at recall ≥ 0.99 on GloVe and ≥ 0.95 on the 768-d
  set, and costs index size and build time ([SOAR](#soar)).
* **`l2_as_dot_product` with varying norms**: on the synthetic clusters
  (squared norms varying by 20%) it lost recall at the defaults (0.924 →
  0.881; the anisotropic threshold recovered part of it, tree AVQ none).
* **3 dimensions per block below 384 dimensions**: better on SIFT through
  MIPS, but GloVe's recall capped at 0.981 with it.
* **Quantized centroids, training on every point, spherical
  partitioning**: on GloVe, training k-means on every point instead of 200
  per leaf raised recall by 0.005–0.01 at the same speed, for a 1.7×
  longer build; spherical partitioning and rounding changed nothing.
  Quantized centroids were 2–8% faster at 768 dimensions in one A/B, with
  the same recall; not checked further.
* Nothing for **incremental training** (`mode=ONLINE`): the same rules
  apply. (Upstream failed on the first upsert into a tree with both AVQ
  and incremental training, through a dangling pointer; scann-core 0.2.1
  fixes that.)
