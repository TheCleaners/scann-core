# Part 3: The pipeline

Script: [`code/part3_pipeline.py`](code/part3_pipeline.py)

Brute force compares every query against every vector, exactly. ScaNN avoids
that with a three-stage pipeline:

1. **Partitioning.** The dataset is split into clusters ahead of time, and a
   query only looks inside the few clusters nearest to it.
2. **Scoring with asymmetric hashing (AH).** Inside those clusters, vectors
   are stored compressed and scored approximately, which is much cheaper
   than an exact dot product.
3. **Reordering.** The best few hundred approximate candidates are rescored
   exactly, and the top *k* of those are returned.

Each stage can be used on its own. Adding them one at a time shows what
each contributes. Everything below uses the same 10,000 queries as part 2.
For reference, exact brute force there gave **recall 1.0 at 12,097 QPS and
10 ms per query**.

## The configurations

```python
def tree(b):
  return b.tree(num_leaves=2000, num_leaves_to_search=100,
                training_sample_size=250000)

def ah(b):
  return b.score_ah(2, anisotropic_quantization_threshold=0.2)

configs = [
    ("1. tree + brute force", lambda: tree(builder()).score_brute_force()),
    ("2. AH only", lambda: ah(builder())),
    ("3. AH + reorder 100", lambda: ah(builder()).reorder(100)),
    ("4. tree + AH", lambda: ah(tree(builder()))),
    ("5. tree + AH + reorder 100", lambda: ah(tree(builder())).reorder(100)),
]
```

```
1. tree + brute force: built in 3.1 s
1. tree + brute force              recall@10 0.9066    104279 QPS   0.700 ms/query
2. AH only: built in 1.5 s
2. AH only                         recall@10 0.5613     24013 QPS   1.337 ms/query
3. AH + reorder 100: built in 1.5 s
3. AH + reorder 100                recall@10 0.9583     24130 QPS   1.318 ms/query
4. tree + AH: built in 3.6 s
4. tree + AH                       recall@10 0.6052    334980 QPS   0.099 ms/query
5. tree + AH + reorder 100: built in 3.3 s
5. tree + AH + reorder 100         recall@10 0.9000    294492 QPS   0.111 ms/query
```

The last line is the headline: **90% recall at 24× the throughput and 1/90th
the latency of brute force**, trained in 3.3 seconds. Here's where each
piece of that comes from.

## Stage 1: partitioning (`.tree`)

```python
.tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000)
```

At build time, k-means clusters the dataset into `num_leaves` = 2000
partitions ("leaves"), about 590 vectors each. To save time it trains on a
random `training_sample_size` of 250,000 vectors, then assigns every vector
to its nearest cluster centre.

At query time, the query is compared against the 2000 centres, and only the
`num_leaves_to_search` = 100 nearest partitions are searched: 5% of the data.

Configuration 1 searches those partitions exactly. It finds **90.7%** of the
true neighbours while touching 5% of the vectors, which is 8.6× faster than
brute force. The other 9.3% are neighbours that sit in a partition whose
centre wasn't among the query's 100 nearest. This usually happens near a
cluster boundary.

That 0.9066 is a **ceiling**. No scoring method can find a neighbour in a
partition that was never searched, so with 100 of 2000 leaves, recall can't
go above it. The only way to raise the ceiling is to search more leaves, which
is the main dial in [part 4](04-tuning.md).

## Stage 2: asymmetric hashing (`.score_ah`)

```python
.score_ah(2, anisotropic_quantization_threshold=0.2)
```

AH is product quantization:

* Each 100-dimensional vector is cut into 50 blocks of
  `dimensions_per_block` = 2 dimensions.
* Each block is replaced by the nearest of 16 learned "codewords" for that
  block, a 4-bit code, so a vector becomes 50 codes.
* To score a query, ScaNN first computes the query's dot product with every
  codeword of every block: 50 small tables of 16 entries.
* A vector's approximate score is then the sum of 50 table lookups instead
  of 100 multiply-adds.

With 16 entries, a whole table fits in one SIMD register, so a single
shuffle instruction does the lookups for many vectors at once. That's the
default "LUT16" format. The hashing is *asymmetric* because the
query is never compressed, only the dataset is.
[algorithms.md](../algorithms.md) has the details.

`anisotropic_quantization_threshold` is ScaNN's own contribution. Ordinary
product quantization picks codewords that minimize *all* compression error.
Anisotropic quantization penalizes error *along* each vector more than
error perpendicular to it, because only the parallel error changes dot
products with the vectors that score high. The
[explainer](../anisotropic_quantization_explained.md) builds this up from
scratch. Part 4 measures what it buys on GloVe.

Configuration 2, AH over the whole dataset, gets only **56% recall**.
The approximate scores are too coarse to put the top 10 in the right
order. Adding a reorder stage (configuration 3) shows that AH still did its
job: **95.8%** of the true neighbours were somewhere in AH's top 100. **AH is
a good filter and a poor ranker**, and the pipeline uses it as a filter.

## Stage 3: reordering (`.reorder`)

```python
.reorder(100)
```

This keeps the top 100 candidates from the scoring stage, rescores them with
exact dot products, and returns the best 10. 100 exact dot products per
query cost almost nothing. Compare configurations 4 and 5: throughput drops
by 12%, and recall goes from 0.605 to 0.900.

That brings recall up to 0.9000, against the partitioning ceiling of 0.9066.
Once the right partitions are searched, AH plus reordering lose almost
nothing, so **the partitioning decides recall, and AH decides speed**.

To rescore exactly, reordering needs the original float32 vectors. They
are 451 MiB of the index's 514 MiB total. [Part 5](05-saving-and-serving.md)
shows how to shrink that.

## Why the stages multiply

* The tree cuts the work to 5% of the vectors.
* AH makes scoring each remaining vector several times cheaper, and more
  cache-friendly: 25 bytes of codes instead of 400 bytes of floats.
* Reordering then fixes AH's ranking errors at a fixed cost of 100 exact
  scores per query.

Configuration 1 (tree + exact) and configuration 3 (AH + reorder, no tree)
are each only 2–9× faster than brute force. Together they are 24× faster.

**Next:** these were one set of parameters. [Part 4](04-tuning.md) explores
the others.
