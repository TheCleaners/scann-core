# Part 5: Saving and serving

Script: [`code/part5_serving.py`](code/part5_serving.py)

Building the part 3 index takes 3.3 s here. At larger scale, training takes
minutes or hours, so you build once, save, and load wherever you serve.

## Saving and loading

```python
searcher.serialize(index_dir)
loaded = scann.scann_ops_pybind.load_searcher(index_dir)
```

```
serialized in 0.3 s to /tmp/glove-index-oasp2b9q:
  ah_codebook.pb                        0.0 MiB
  datapoint_to_token.npy                4.5 MiB
  dataset.npy                         451.5 MiB
  hashed_dataset.npy                   56.4 MiB
  scann_assets.pbtxt                    0.0 MiB
  scann_config.pb                       0.0 MiB
  serialized_partitioner.pb             1.6 MiB
loaded in 0.5 s
loaded index gives identical results: True
```

`index_dir` must exist. A saved index is a directory:

| File | Holds |
|---|---|
| `scann_config.pb` | the full configuration (what `searcher.config()` shows) |
| `serialized_partitioner.pb` | the 2000 partition centres |
| `datapoint_to_token.npy` | which partition each vector is in |
| `ah_codebook.pb` | the AH codewords |
| `hashed_dataset.npy` | every vector's AH codes |
| `dataset.npy` | the float32 vectors, used for reordering |
| `scann_assets.pbtxt` | the list of the files above |
| `scann_docids.pkl` | docids, if the index has them (Python only; see [part 6](06-updating.md)) |

Loading involves no training: it's 0.5 s, mostly reading `dataset.npy`, and
the loaded index returns exactly the same results.

By default `scann_assets.pbtxt` records absolute paths. Pass
`serialize(index_dir, relative_path=True)` to make the directory movable,
for example to copy it to other machines. The format is the same across
Python, C++ and Rust: [part 7](07-cpp-and-rust.md) loads this index from
both.

## Making the index smaller

`dataset.npy` is 451 MiB of the 514 MiB total. That's the float32 copy
reordering uses. It can be stored at lower precision, just like brute force
in part 2:

```python
.reorder(100, quantize=scann.ReorderType.BFLOAT16)   # or INT8
```

```
reorder precision vs. index size
  reorder float32 :  514.0 MiB on disk, recall@10 0.8980
  reorder bfloat16:  288.3 MiB on disk, recall@10 0.8974
  reorder int8    :  175.4 MiB on disk, recall@10 0.8866
```

bfloat16 nearly halves the index for less than 0.1 points of recall, and int8
cuts it by two-thirds for 1.1 points. If size is the constraint, bfloat16 is the
easy win. Any recall loss can be won back with a few more
`leaves_to_search`, as in part 4.

## Serving: batch size, latency and throughput

A serving process answers queries as they arrive. There are two ways to use
the machine's cores:

1. **Batch the queries.** Collect requests for a moment, then search them
   together with `search_batched_parallel`, which spreads a batch over the
   index's thread pool (`searcher.set_num_threads(n)`).
2. **Search concurrently.** Each request thread calls `search()` on the
   shared index. ScaNN searches are thread-safe.

Batching first:

```
batch size vs. throughput / latency (search_batched_parallel, 64 threads)
  batch     1:    11707 QPS,   0.085 ms per batch
  batch     8:    64617 QPS,   0.124 ms per batch
  batch    64:   182964 QPS,   0.350 ms per batch
  batch   512:   236128 QPS,   2.168 ms per batch
  batch  4096:   324594 QPS,  12.619 ms per batch
  search() one query at a time:    13571 QPS, 0.074 ms per query
```

This is the classic trade-off: bigger batches give more throughput, and
every query in the batch waits for the whole batch. With batches of 64, the
machine does 183k QPS and a query waits at most about a third of a millisecond,
plus however long it waited for the batch to fill.

For a single query, plain `search()` is slightly faster than a batch of one
(0.074 vs 0.085 ms), because it doesn't hand off to the thread pool.

Then concurrent `search()` calls from Python threads:

```
concurrent search() calls from a Python thread pool
   1 threads:    12666 QPS
   8 threads:    62683 QPS
  32 threads:    60715 QPS
  64 threads:    55700 QPS
```

ScaNN releases Python's global interpreter lock while it searches, so
threads run in parallel: 8 threads give 4.9× the throughput. Beyond that
the curve flattens around 60k QPS. The rest of each call, converting
arguments and building result arrays, still holds the lock. At 0.07 ms per
search, that serial part is the likely bottleneck.

**In Python**, either keep about 8 request threads, or batch. Batching goes
further: 325k QPS, against about 60k for threads. **From C++ or Rust**
there is no interpreter lock. [Part 7](07-cpp-and-rust.md) measures
concurrent `search()` calls from Rust threads.

**Next:** real collections change. [Part 6: Updating](06-updating.md).
