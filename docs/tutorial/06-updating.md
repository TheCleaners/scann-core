# Part 6: Updating

Script: [`code/part6_updating.py`](code/part6_updating.py)

Collections change: documents are added, edited and removed. Rebuilding the
index for every change works, but it's wasteful. A ScaNN index can be
modified in place.

## Docids

To update an index, you need a way to name its points. Pass **docids**,
one string per row, at build time:

```python
docids = [str(i) for i in range(n)]
searcher = builder(dataset[:first]).build(docids=docids[:first])
```

Searches then return docids instead of row numbers. `upsert` and `delete`
take docids, and the searcher keeps track of which docid is at which
internal position. Positions move when points are deleted: the last point
is moved into the freed slot. So always refer to points by docid. Docids are
saved with the index as `scann_docids.pkl`.

Here the docid is simply the row number in the full dataset, so the results
can still be checked against the ground truth.

## Streaming data in

The script builds an index on the first 70% of GloVe only, then adds the
remaining 30% with `upsert`, 10,000 at a time:

```python
for start in range(first, n, 10000):
  end = min(start + 10000, n)
  searcher.upsert(docids[start:end], dataset[start:end], batch_size=10000)
```

```
built on 828459 points in 2.7 s
70% indexed                              size   828459  recall@10 0.6351  imbalance 0.278  quantization error 0.7931
upserted 355055 points in 0.9 s (394582 points/s)
after upserting the other 30%            size  1183514  recall@10 0.9011  imbalance 0.279  quantization error 0.7940
rebalanced in 3.1 s
after rebalance()                        size  1183514  recall@10 0.8996  imbalance 0.276  quantization error 0.7942
(fresh build on everything: 3.3 s)
built from scratch on 100%               size  1183514  recall@10 0.8990  imbalance 0.276  quantization error 0.7942
```

* **With 70% indexed, recall is 0.63.** About 30% of each query's true
  neighbours aren't in the index yet, and 0.7 × 0.90 ≈ 0.63. The index is
  working fine; it just can't find what it doesn't have.
* **`upsert` added 355,055 points in 0.9 s**, about 395,000 per second.
  Each new point is assigned to its nearest existing partition and encoded
  with the existing AH codebook. Nothing is retrained.
* **Afterwards, recall matches a fresh build** (0.9011 vs 0.8990; the
  difference is run-to-run noise). The partitions and codebook learned from
  the first 70% describe the rest just as well.

`upsert` with a docid that is already in the index replaces that point's
vector instead of adding a new one. `batch_size` controls how many points are
encoded at a time, in parallel on the index's thread pool. Larger is faster
for bulk loads.

## When to retrain: health stats and `rebalance()`

That last result depends on the new data looking like the old. When it
doesn't, updates slowly degrade the index. That happens, for example, when a
new topic appears, or when embeddings come from a new model version:

* Partitions become unbalanced. New points pile into whichever centres are
  nearest, and a query that lands in an overfull partition does more work.
* Quantization error grows, because the codebook wasn't trained on vectors
  like these.

The searcher tracks both:

```python
searcher.initialize_health_stats()  # compute from scratch
stats = searcher.get_health_stats()
stats["partition_avg_relative_positive_imbalance"]  # 0 = perfectly even
stats["avg_quantization_error"]
```

Call `initialize_health_stats()` once. Updates then keep the numbers
current. Here neither moved (imbalance 0.278 → 0.279, quantization error
0.7931 → 0.7940), which says what the recall already showed: no drift.

When they do grow, `searcher.rebalance()` retrains the partitioning and the
codebook on the index's current contents, with the same configuration. It
took 3.1 s here, about the same as building from scratch (3.3 s), because
it *is* a rebuild. The difference is that the searcher, its docids and your
serving code stay in place. With nothing to fix, it changed nothing measurable.

A rule of thumb: record the health stats after building, and rebalance when
the imbalance or the quantization error has grown by a noticeable fraction,
or when recall on a held-out query set drops. Keep in mind that `upsert`
calls may retrain on their own. When incremental maintenance decides the
index needs it, an `upsert` silently includes a full rebalance (see
[api_reference.md](../api_reference.md#dynamic-updates-upsert--delete--reserve--rebalance)).

## Updating and deleting single points

```python
top = searcher.search(queries[0])[0]
searcher.delete(top[0])
searcher.upsert(top[1], -queries[0])  # same docid, new vector
```

```
query 0 top-3 before: ['97478', '846101', '671078']
query 0 top-3 after deleting 97478 and moving 846101 : ['671078', '727732', '544474']
```

The deleted point is gone. The moved point now points *away* from the query
and drops out. Both take effect immediately, with no rebuild.

Query 0's exact top 3 are 97478, 262700 and 846101 (part 1). This index
missed 262700, a reminder that a 90%-recall index misses one result in ten.

Two behaviours to know about:

* `delete` raises `KeyError` for an unknown docid, and changes nothing if it
  does. A failed `upsert` (a vector of the wrong dimension, say) also leaves
  the index and its docids unchanged. In upstream ScaNN both could leave the
  docids pointing at the wrong vectors; scann-core fixed this.
* Saving an updated index (`serialize`) saves its current contents and
  docids. Loading it gives you the updated index, not the original build.

**Next:** [Part 7: C++ and Rust](07-cpp-and-rust.md).
