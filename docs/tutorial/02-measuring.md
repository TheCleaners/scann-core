# Part 2: Measuring

Script: [`code/part2_measuring.py`](code/part2_measuring.py)

Approximate search trades accuracy for speed. To make that trade
deliberately, you need to measure both sides.

## Recall

The standard accuracy measure is **recall@k**: of the true *k* nearest
neighbours, what fraction did the search return?

```python
def recall(found, true_neighbors):
  k = found.shape[1]
  hits = sum(np.intersect1d(f, t[:k]).size for f, t in zip(found, true_neighbors))
  return hits / (found.shape[0] * k)
```

A recall@10 of 0.9 means that, on average, 9 of the 10 results are among
the true top 10. The 10th is usually the 11th or 12th true neighbour, a very
near miss rather than garbage, since it has to score well to be returned at
all.

Recall needs ground truth, meaning exact results for a set of queries.
ann-benchmarks ships it for GloVe. For your own data, compute it once with
a brute-force searcher like the one from part 1, on a few thousand queries
drawn from your real query distribution. Recall on random vectors tells you
nothing about recall on real queries.

## Speed: throughput and latency

There are two numbers, answering two different questions:

* **Throughput** (queries per second, QPS): how many queries per second
  the machine can answer when it has plenty of work. This is what matters for
  offline jobs and busy services. We measure it with
  `search_batched_parallel`, which spreads a batch over a thread pool, using
  one thread per core.
* **Latency** (ms per query): how long one query takes on its own. This is
  what matters for a user waiting on a response. We measure it with
  `search()` on one query at a time, on one thread.

`tutorial_data.evaluate()` reports both, plus recall:

```python
def evaluate(name, searcher, queries, true_neighbors, latency_queries=1000,
             **search_args):
  searcher.set_num_threads(os.cpu_count())
  best, total, passes = float("inf"), 0.0, 0
  while passes < 3 or total < 1.0:
    with Timer() as t:
      neighbors, _ = searcher.search_batched_parallel(queries, **search_args)
    best, total, passes = min(best, t.seconds), total + t.seconds, passes + 1
  qps = len(queries) / best
  ...
```

The timed pass is repeated until at least three passes and one second have
elapsed, and the best pass is reported. A fast index gets through 10,000
queries in 30 ms, too short to time reliably in a single run. Taking the best
of several passes, as ann-benchmarks does, filters out noise from whatever
else the machine is doing.

## Brute force, three ways

`score_brute_force` can also store the dataset at lower precision:

```python
for name, quantize in [("float32", scann.ReorderType.FLOAT32),
                       ("bfloat16", scann.ReorderType.BFLOAT16),
                       ("int8", scann.ReorderType.INT8)]:
  searcher = builder().score_brute_force(quantize=quantize).build()
  evaluate(f"brute force, {name}", searcher, queries, true_neighbors)
```

```
brute force, float32               recall@10 1.0000     15054 QPS   9.541 ms/query
brute force, bfloat16              recall@10 0.9948      1876 QPS   5.057 ms/query
brute force, int8                  recall@10 0.9745      3417 QPS   4.703 ms/query
```

* **bfloat16** keeps the top 16 bits of each float (the full exponent and 7
  mantissa bits). **int8** scales each dimension into −127…127. That's half
  and a quarter of the memory respectively, at a small cost in recall.
* **Latency** follows memory size: one query has to read the entire dataset,
  so the half-size bfloat16 copy is nearly twice as fast as float32.
* **Throughput** goes the other way: float32 is 4–8× *faster*. Batched
  float32 search uses a matrix-multiplication kernel that loads a block of
  data once and scores it against many queries. The quantized formats score
  one query at a time, so every query streams the whole dataset through
  memory again. bfloat16 goes from about 200 QPS on one thread to only 1,876
  on 64, and that is about 440 GB/s of vectors read. When 64 threads give
  a 10× speedup, the bottleneck is almost certainly the memory system, not
  the cores.

So "quantized = faster" isn't a rule. It depends on whether you are limited
by memory traffic or by computation, and batching changes which one it is.
You have to measure your own workload.

Either way, exact search tops out around 15,000 QPS here. Every query
touches all 1.18 million vectors. To go faster, a query has to touch fewer
of them, or do less work per vector. Those are the two ideas behind ScaNN.

**Next:** [Part 3: The pipeline](03-the-pipeline.md).
