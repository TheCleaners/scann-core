# Part 4: Tuning

Script: [`code/part4_tuning.py`](code/part4_tuning.py)

Part 3 used one set of parameters. There are two kinds of dial:

* **query-time** parameters, which can change per search without
  rebuilding;
* **build-time** parameters, which shape the index itself.

Start with the first kind. It's free to explore.

This part tunes by hand on one dataset. [tuning.md](../tuning.md) is the
reference that goes with it: per-parameter recommendations measured on
GloVe, SIFT and 768-dimensional embeddings, a cost model, and starting
configurations with their measured recall and speed.

## Query time: `leaves_to_search` and `pre_reorder_num_neighbors`

Both can be passed to every search method. They override the values the index
was built with (`num_leaves_to_search`, and the `reorder()` count):

```python
searcher.search_batched_parallel(queries, leaves_to_search=200,
                                 pre_reorder_num_neighbors=200)
```

The script sweeps both over the part 3 index (2000 leaves):

```
leaves  25 pre_reorder  50         recall@10 0.7711    991058 QPS   0.035 ms/query
leaves  25 pre_reorder 100         recall@10 0.7836    861723 QPS   0.037 ms/query
leaves  25 pre_reorder 200         recall@10 0.7857    693907 QPS   0.047 ms/query
leaves  25 pre_reorder 400         recall@10 0.7859    497715 QPS   0.052 ms/query
leaves  50 pre_reorder  50         recall@10 0.8315    822996 QPS   0.052 ms/query
leaves  50 pre_reorder 100         recall@10 0.8490    710172 QPS   0.055 ms/query
leaves  50 pre_reorder 200         recall@10 0.8529    567769 QPS   0.056 ms/query
leaves  50 pre_reorder 400         recall@10 0.8534    412865 QPS   0.066 ms/query
leaves 100 pre_reorder  50         recall@10 0.8774    540551 QPS   0.081 ms/query
leaves 100 pre_reorder 100         recall@10 0.9007    483354 QPS   0.077 ms/query
leaves 100 pre_reorder 200         recall@10 0.9064    422794 QPS   0.083 ms/query
leaves 100 pre_reorder 400         recall@10 0.9074    328432 QPS   0.095 ms/query
leaves 200 pre_reorder  50         recall@10 0.9102    330538 QPS   0.131 ms/query
leaves 200 pre_reorder 100         recall@10 0.9387    312748 QPS   0.139 ms/query
leaves 200 pre_reorder 200         recall@10 0.9468    279899 QPS   0.141 ms/query
leaves 200 pre_reorder 400         recall@10 0.9485    232795 QPS   0.152 ms/query
leaves 400 pre_reorder  50         recall@10 0.9318    184380 QPS   0.224 ms/query
leaves 400 pre_reorder 100         recall@10 0.9649    177110 QPS   0.230 ms/query
leaves 400 pre_reorder 200         recall@10 0.9753    168082 QPS   0.242 ms/query
leaves 400 pre_reorder 400         recall@10 0.9779    149961 QPS   0.263 ms/query
```

Reading it:

* **`leaves_to_search` sets the ceiling.** Each doubling adds 3–7 points of
  recall, less as recall rises, and costs 20–45% of the throughput. Doubling
  the leaves doubles the vectors scored, but some per-query costs stay
  fixed.
* **`pre_reorder_num_neighbors` fills up to the ceiling.** Past a point,
  more candidates don't help. At 100 leaves, going from 200 to 400
  candidates gains 0.001 recall and costs 22% of the throughput. The more
  leaves you search, the more candidates it takes to reach the ceiling. At
  400 leaves, 50 candidates leave nearly 5 points on the table.
* **A good rule of thumb:** keep `pre_reorder_num_neighbors` around 10–20×
  *k*, and use `leaves_to_search` to pick the recall you need. Higher
  recall targets need more candidates: the fastest settings in
  [tuning.md](../tuning.md#how-many-candidates) used 7–50× *k* at k=10 and
  2–10× *k* at k=100.

For a target like "95% recall", read the fastest row that meets it:
here, 400 leaves with 100 candidates, at about 177,000 QPS. For 90%, it's
100 leaves with 100 candidates, at about 483,000 QPS, though only just
(0.9007). 200 candidates give more margin (0.9064) at 423,000 QPS.

**Always choose parameters on held-out queries from your real workload.** The
curve depends on your data.

## Build time

Each variant below is built fresh and then swept over three
`leaves_to_search` values, 2.5%, 5% and 10% of its leaves, with 200 reorder
candidates. Variants are only comparable at equal speed, so compare curves,
not single rows.

```
1000 leaves: built in 2.9 s
  leaves   25                      recall@10 0.8387    572713 QPS   0.054 ms/query
  leaves   50                      recall@10 0.8968    422288 QPS   0.082 ms/query
  leaves  100                      recall@10 0.9401    279247 QPS   0.125 ms/query
2000 leaves: built in 3.3 s
  leaves   50                      recall@10 0.8522    531180 QPS   0.059 ms/query
  leaves  100                      recall@10 0.9050    402683 QPS   0.094 ms/query
  leaves  200                      recall@10 0.9460    271306 QPS   0.140 ms/query
4000 leaves: built in 3.7 s
  leaves  100                      recall@10 0.8424    475050 QPS   0.075 ms/query
  leaves  200                      recall@10 0.9002    383874 QPS   0.116 ms/query
  leaves  400                      recall@10 0.9433    249248 QPS   0.174 ms/query
2000 leaves, plain PQ (no AQ): built in 3.0 s
  leaves   50                      recall@10 0.8477    532702 QPS   0.058 ms/query
  leaves  100                      recall@10 0.8995    416526 QPS   0.085 ms/query
  leaves  200                      recall@10 0.9393    279824 QPS   0.138 ms/query
2000 leaves, 4 dims/block: built in 2.7 s
  leaves   50                      recall@10 0.7783    606956 QPS   0.045 ms/query
  leaves  100                      recall@10 0.8060    487431 QPS   0.063 ms/query
  leaves  200                      recall@10 0.8233    348511 QPS   0.095 ms/query
2000 leaves, SOAR lambda 1.5: built in 4.2 s
  leaves   50                      recall@10 0.9093    313784 QPS   0.097 ms/query
  leaves  100                      recall@10 0.9503    228780 QPS   0.153 ms/query
  leaves  200                      recall@10 0.9775    146916 QPS   0.262 ms/query
```

### `num_leaves`: forgiving

The 1000, 2000 and 4000-leaf curves nearly coincide: 0.897 at 422,000 QPS,
0.905 at 403,000, and 0.900 at 384,000. With fewer, bigger
partitions you search fewer of them. With more, smaller ones you search
more of them. The work evens out.

More leaves cost somewhat more training time. With many thousands of leaves,
comparing the query against every centre starts to matter too; ScaNN's
`.upper_tree()` adds a partitioning level above the leaves for that case.
The rule of thumb in [algorithms.md](../algorithms.md) is `num_leaves` ≈
√n, here 1,088, with anything in that neighbourhood fine.

That holds for this sweep, which measures batched throughput on 64 threads.
Measured one query at a time on one thread, fewer leaves won: on this
dataset 1000–1500 leaves beat 2000–4000 at every recall, and 3000 was worse
everywhere ([tuning.md](../tuning.md#num_leaves)). Tune in the mode you'll
serve in.

### Anisotropic quantization: free, and big when reordering is short

With 200 reorder candidates, AQ adds only 0.5–0.7 points of recall over plain
product quantization (0.9050 vs 0.8995; 0.9460 vs 0.9393). Reordering 200
candidates already repairs most of AH's ranking errors, and AQ improves
exactly the AH stage. The last section of the script takes that safety net
away:

```
AQ vs plain PQ with less reordering (2000 leaves, 100 searched)
AQ, no reordering                  recall@10 0.6070    505822 QPS   0.071 ms/query
AQ, reorder 20                     recall@10 0.7747    503050 QPS   0.071 ms/query
AQ, reorder 50                     recall@10 0.8759    472790 QPS   0.080 ms/query
plain PQ, no reordering            recall@10 0.5489    500805 QPS   0.071 ms/query
plain PQ, reorder 20               recall@10 0.7141    488277 QPS   0.071 ms/query
plain PQ, reorder 50               recall@10 0.8427    469424 QPS   0.087 ms/query
```

When AH does most of the ranking, AQ is worth 3.3–6.1 points of recall at
the same speed. That's the effect the
[explainer](../anisotropic_quantization_explained.md) describes: the
scores of the vectors that matter are distorted less. AQ costs nothing at
query time, so use it for dot-product search. 0.2 suits normalized
100-dimensional data like this. The right value shrinks as the
dimension grows: at 768 dimensions 0.2 caps recall at 0.795 and 0.05 works
(see [tuning.md](../tuning.md#the-anisotropic-threshold) for the formula,
and the
[explainer's section on the threshold](../anisotropic_quantization_explained.md#the-threshold-how-much-do-you-protect)).

### `dimensions_per_block`: 2 for this data

With 4 dimensions per block, the codes are half as long and scoring is
faster, but recall plateaus around 0.82 however many leaves you search. The
approximate scores are so coarse that 200 candidates no longer contain the
true neighbours. With 2 dimensions per block, the fast AH scoring still
keeps the true neighbours among its top candidates, so reordering can
recover them.

That is for 100 dimensions with a threshold of 0.2. Higher-dimensional data
can afford larger blocks: on 768-dimensional embeddings, 4 dimensions per
block with a threshold of 0.05 was best, and 128-dimensional SIFT searched
through an inner-product reduction did best with 3
([tuning.md](../tuning.md#ah-dimensions_per_block-and-hash_type)).

### SOAR: better partitions, for dot product

[SOAR](https://arxiv.org/abs/2404.00774) assigns each vector to a *second*
partition as well as its nearest one. That rescues neighbours that sit near
a partition boundary, which were the reason for part 3's recall ceiling. The
second partition is chosen with an "orthogonality-amplified residual" loss,
so that the error it makes is unlikely to coincide with the first one's.
The cost: each vector is listed in two partitions, so every leaf searched
holds more vectors to score, and building takes longer (4.2 s vs 3.3 s).

That is why SOAR's rows are slower at the same `leaves_to_search`, and why
it has to be compared at equal speed. Take the query-time sweep's rows at the
same throughput:

| QPS | plain 2000 leaves | SOAR |
|---|---|---|
| ~313,000 | 0.9387 (200 leaves, 100 candidates; 312,748 QPS) | 0.9093 (50 leaves; 313,784 QPS) |
| ~229,000–233,000 | 0.9485 (200 leaves, 400 candidates; 232,795 QPS) | 0.9503 (100 leaves; 228,780 QPS) |
| ~147,000–150,000 | 0.9779 (400 leaves, 400 candidates; 149,961 QPS) | 0.9775 (200 leaves; 146,916 QPS) |

At equal speed on GloVe, SOAR is level with plain partitioning at high recall
(within 0.2 points either way). At lower recall it is behind: 0.909 against
0.939 at about 313,000 QPS. The paper reports larger gains on other datasets
and at larger scales, so treat SOAR as something to try on your own data,
not a default. SOAR requires dot product.

This sweep used lambda 1.5. Measured one query at a time on one thread,
lambda 0.5 did better on GloVe (0.5 ≥ 1.0 > 1.5). SOAR was 3–10% slower
than plain partitioning at recall 0.8–0.95 and slightly faster from 0.99; on
768-dimensional embeddings at k=100 it paid from 0.95
([tuning.md](../tuning.md#soar)).

## A tuning recipe

1. Compute exact ground truth for a few thousand real queries (part 2).
2. Build with part 3's pipeline: `num_leaves` ≈ √n (1000–1500 around a
   million points), `score_ah(2, anisotropic_quantization_threshold=0.2)`
   for normalized data of about 100 dimensions (smaller thresholds at
   higher dimensions), and `reorder(100)` to `reorder(200)`. Consider
   storing the reordering data as bfloat16
   (`reorder(..., quantize=scann.ReorderType.BFLOAT16)`): half the memory
   (part 5) and a few percent faster in single-query tests, usually at
   nearly the same recall, but check: on some datasets it cost 0.002–0.004
   ([tuning.md](../tuning.md#precision-bfloat16)). Or start from
   `.autopilot()`, whose rules come from the same study
   ([tuning.md](../tuning.md#defaults-and-autopilot)), and sweep its
   search settings as in step 3 — or let it do that sweep:
   `.autopilot(target_recall=0.95, calibration_queries=queries[:1000])`
   picks the cheapest `leaves_to_search` and `pre_reorder_num_neighbors`
   that reach recall 0.95 on those queries when it builds, and saves them
   as the index's defaults. Check the recall on other queries: calibrated
   on 1,000 real queries, SIFT's other 9,000 met the targets; with
   datapoints as queries (no `calibration_queries`) it landed within 0.002
   of the target on GloVe and a 768-d set, but 0.011 short on SIFT
   ([tuning.md](../tuning.md#a-recall-target-autopilottarget_recall)).
3. Sweep `leaves_to_search`, and `pre_reorder_num_neighbors` at a few
   values, at query time. Pick the fastest setting that meets your recall
   target.
4. Only if that isn't good enough, try build-time changes (`tree(avq=2.5)`
   and SOAR for dot product, a different `num_leaves`), and compare them at
   equal speed. [tuning.md](../tuning.md) has measured starting points and
   what each change did on three datasets.

**Next:** [Part 5: Saving and serving](05-saving-and-serving.md).
