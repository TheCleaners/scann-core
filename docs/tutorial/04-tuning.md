# Part 4: Tuning

Script: [`code/part4_tuning.py`](code/part4_tuning.py)

Part 3 used one set of parameters. There are two kinds of dial:

* **query-time** parameters, which can change per search without
  rebuilding;
* **build-time** parameters, which shape the index itself.

Start with the first kind. It's free to explore.

## Query time: `leaves_to_search` and `pre_reorder_num_neighbors`

Both can be passed to every search method. They override the values the index
was built with (`num_leaves_to_search`, and the `reorder()` count):

```python
searcher.search_batched_parallel(queries, leaves_to_search=200,
                                 pre_reorder_num_neighbors=200)
```

The script sweeps both over the part 3 index (2000 leaves):

```
leaves  25 pre_reorder  50         recall@10 0.7701    722308 QPS   0.046 ms/query
leaves  25 pre_reorder 100         recall@10 0.7830    650244 QPS   0.048 ms/query
leaves  25 pre_reorder 200         recall@10 0.7849    529338 QPS   0.054 ms/query
leaves  25 pre_reorder 400         recall@10 0.7851    386502 QPS   0.065 ms/query
leaves  50 pre_reorder  50         recall@10 0.8298    539653 QPS   0.067 ms/query
leaves  50 pre_reorder 100         recall@10 0.8482    462043 QPS   0.074 ms/query
leaves  50 pre_reorder 200         recall@10 0.8516    393281 QPS   0.079 ms/query
leaves  50 pre_reorder 400         recall@10 0.8520    303486 QPS   0.093 ms/query
leaves 100 pre_reorder  50         recall@10 0.8761    333257 QPS   0.115 ms/query
leaves 100 pre_reorder 100         recall@10 0.8995    307823 QPS   0.113 ms/query
leaves 100 pre_reorder 200         recall@10 0.9053    285163 QPS   0.124 ms/query
leaves 100 pre_reorder 400         recall@10 0.9062    225813 QPS   0.136 ms/query
leaves 200 pre_reorder  50         recall@10 0.9090    184744 QPS   0.201 ms/query
leaves 200 pre_reorder 100         recall@10 0.9379    177083 QPS   0.213 ms/query
leaves 200 pre_reorder 200         recall@10 0.9464    169489 QPS   0.232 ms/query
leaves 200 pre_reorder 400         recall@10 0.9479    150783 QPS   0.239 ms/query
leaves 400 pre_reorder  50         recall@10 0.9301    103396 QPS   0.378 ms/query
leaves 400 pre_reorder 100         recall@10 0.9637    101386 QPS   0.389 ms/query
leaves 400 pre_reorder 200         recall@10 0.9747     99862 QPS   0.378 ms/query
leaves 400 pre_reorder 400         recall@10 0.9771     88020 QPS   0.427 ms/query
```

Reading it:

* **`leaves_to_search` sets the ceiling.** Each doubling adds 3–7 points of
  recall, less as recall rises, and costs 25–40% of the throughput. Doubling
  the leaves doubles the vectors scored, but some per-query costs stay
  fixed.
* **`pre_reorder_num_neighbors` fills up to the ceiling.** Past a point,
  more candidates don't help. At 100 leaves, going from 200 to 400
  candidates gains 0.0009 recall and costs 20% of the throughput. The more
  leaves you search, the more candidates it takes to reach the ceiling. At
  400 leaves, 50 candidates leave nearly 5 points on the table.
* **A good rule of thumb:** keep `pre_reorder_num_neighbors` around 10–20×
  *k*, and use `leaves_to_search` to pick the recall you need.

For a target like "95% recall", read the fastest row that meets it:
here, 400 leaves with 100 candidates, at about 101,000 QPS. For 90%, it's
100 leaves with 200 candidates, at about 285,000 QPS. With 100 candidates
the same setting reaches 0.8995, just short.

**Always choose parameters on held-out queries from your real workload.** The
curve depends on your data.

## Build time

Each variant below is built fresh and then swept over three
`leaves_to_search` values, 2.5%, 5% and 10% of its leaves, with 200 reorder
candidates. Variants are only comparable at equal speed, so compare curves,
not single rows.

```
1000 leaves: built in 2.8 s
  leaves   25                      recall@10 0.8389    397740 QPS   0.079 ms/query
  leaves   50                      recall@10 0.8965    261645 QPS   0.118 ms/query
  leaves  100                      recall@10 0.9399    167230 QPS   0.203 ms/query
2000 leaves: built in 3.4 s
  leaves   50                      recall@10 0.8525    378564 QPS   0.074 ms/query
  leaves  100                      recall@10 0.9057    262916 QPS   0.126 ms/query
  leaves  200                      recall@10 0.9466    161339 QPS   0.220 ms/query
4000 leaves: built in 4.3 s
  leaves  100                      recall@10 0.8450    346620 QPS   0.098 ms/query
  leaves  200                      recall@10 0.9012    241394 QPS   0.159 ms/query
  leaves  400                      recall@10 0.9438    148647 QPS   0.273 ms/query
2000 leaves, plain PQ (no AQ): built in 3.2 s
  leaves   50                      recall@10 0.8495    385766 QPS   0.076 ms/query
  leaves  100                      recall@10 0.9007    268054 QPS   0.123 ms/query
  leaves  200                      recall@10 0.9401    161038 QPS   0.219 ms/query
2000 leaves, 4 dims/block: built in 2.8 s
  leaves   50                      recall@10 0.7795    493981 QPS   0.056 ms/query
  leaves  100                      recall@10 0.8075    378410 QPS   0.084 ms/query
  leaves  200                      recall@10 0.8247    258444 QPS   0.141 ms/query
2000 leaves, SOAR lambda 1.5: built in 4.1 s
  leaves   50                      recall@10 0.9109    240341 QPS   0.123 ms/query
  leaves  100                      recall@10 0.9512    158009 QPS   0.216 ms/query
  leaves  200                      recall@10 0.9783     92815 QPS   0.367 ms/query
```

### `num_leaves`: forgiving

The 1000, 2000 and 4000-leaf curves nearly coincide: 0.897 at 262,000 QPS,
0.906 at 263,000, and 0.901 at 241,000. With fewer, bigger
partitions you search fewer of them. With more, smaller ones you search
more of them. The work evens out.

More leaves cost somewhat more training time. With many thousands of leaves,
comparing the query against every centre starts to matter too; ScaNN's
`.upper_tree()` adds a partitioning level above the leaves for that case.
The rule of thumb in [algorithms.md](../algorithms.md) is `num_leaves` ≈
√n, here 1,088, with anything in that neighbourhood fine. Don't agonize over
it.

### Anisotropic quantization: free, and big when reordering is short

With 200 reorder candidates, AQ adds only 0.3–0.7 points of recall over plain
product quantization (0.9057 vs 0.9007; 0.9466 vs 0.9401). Reordering 200
candidates already repairs most of AH's ranking errors, and AQ improves
exactly the AH stage. The last section of the script takes that safety net
away:

```
AQ vs plain PQ with less reordering (2000 leaves, 100 searched)
AQ, no reordering                  recall@10 0.6060    328589 QPS   0.109 ms/query
AQ, reorder 20                     recall@10 0.7729    318404 QPS   0.107 ms/query
AQ, reorder 50                     recall@10 0.8761    316129 QPS   0.112 ms/query
plain PQ, no reordering            recall@10 0.5479    325418 QPS   0.107 ms/query
plain PQ, reorder 20               recall@10 0.7142    313504 QPS   0.109 ms/query
plain PQ, reorder 50               recall@10 0.8439    305471 QPS   0.115 ms/query
```

When AH does most of the ranking, AQ is worth 3.2–5.9 points of recall at
the same speed. That's the effect the
[explainer](../anisotropic_quantization_explained.md) describes: the
scores of the vectors that matter are distorted less. AQ costs nothing at
query time, so use it for dot-product search. 0.2 is the usual threshold
for normalized data; see the
[explainer's section on the threshold](../anisotropic_quantization_explained.md#the-threshold-how-much-do-you-protect).

### `dimensions_per_block`: keep it at 2

With 4 dimensions per block, the codes are half as long and scoring is
faster, but recall plateaus around 0.82 however many leaves you search. The
approximate scores are so coarse that 200 candidates no longer contain the
true neighbours. With 2 dimensions per block, the fast AH scoring still
keeps the true neighbours among its top candidates, so reordering can
recover them.

### SOAR: better partitions, for dot product

[SOAR](https://arxiv.org/abs/2404.00774) assigns each vector to a *second*
partition as well as its nearest one. That rescues neighbours that sit near
a partition boundary, which were the reason for part 3's recall ceiling. The
second partition is chosen with an "orthogonality-amplified residual" loss,
so that the error it makes is unlikely to coincide with the first one's.
The cost: each vector is listed in two partitions, so every leaf searched
holds more vectors to score, and building takes longer (4.1 s vs 3.4 s).

That is why SOAR's rows are slower at the same `leaves_to_search`, and why
it has to be compared at equal speed. Take the query-time sweep's rows at the
same throughput:

| QPS | plain 2000 leaves | SOAR |
|---|---|---|
| ~150,000–158,000 | 0.9479 (200 leaves, 400 candidates; 150,783 QPS) | 0.9512 (100 leaves; 158,009 QPS) |
| ~88,000–100,000 | 0.9747 (400 leaves, 200 candidates; 99,862 QPS)<br>0.9771 (400 leaves, 400 candidates; 88,020 QPS) | 0.9783 (200 leaves; 92,815 QPS) |

On GloVe, SOAR is ahead by 0.1–0.4 points of recall at equal speed: real,
but small. The paper reports larger gains on other datasets and at larger
scales. Treat it as something to try when you need the last bit of recall,
and measure it on your data. SOAR requires dot product.

## A tuning recipe

1. Compute exact ground truth for a few thousand real queries (part 2).
2. Build with the defaults from part 3: `num_leaves` ≈ √n (or 2000 around
   a million points), `score_ah(2, anisotropic_quantization_threshold=0.2)`,
   `reorder(100)` to `reorder(200)`.
3. Sweep `leaves_to_search`, and `pre_reorder_num_neighbors` at a few
   values, at query time. Pick the fastest setting that meets your recall
   target.
4. Only if that isn't good enough, try build-time changes (SOAR for dot
   product, a different `num_leaves`), and compare them at equal speed.

**Next:** [Part 5: Saving and serving](05-saving-and-serving.md).
