# Part 5: Saving and serving

Script: [`code/part5_serving.py`](code/part5_serving.py)

Building the part 3 index takes 3.2 s here. At larger scale, training takes
minutes or hours, so you build once, save, and load wherever you serve.

## Saving and loading

```python
searcher.serialize(index_dir)
loaded = scann.scann_ops_pybind.load_searcher(index_dir)
```

```
serialized in 0.3 s to /tmp/glove-index-q0wxlsid:
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

Only load index directories you trust: if the index has docids,
`load_searcher` unpickles `scann_docids.pkl`, and unpickling can run
arbitrary code.

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
  batch     1:    11222 QPS,   0.089 ms per batch
  batch     8:    61310 QPS,   0.130 ms per batch
  batch    64:   176586 QPS,   0.362 ms per batch
  batch   512:   249586 QPS,   2.051 ms per batch
  batch  4096:   293822 QPS,  13.940 ms per batch
  search() one query at a time:    13331 QPS, 0.075 ms per query
```

This is the classic trade-off: bigger batches give more throughput, and
every query in the batch waits for the whole batch. With batches of 64, the
machine does 177k QPS and a query waits at most about a third of a millisecond,
plus however long it waited for the batch to fill.

For a single query, plain `search()` is slightly faster than a batch of one
(0.075 vs 0.089 ms), because it doesn't hand off to the thread pool.

Then concurrent `search()` calls from Python threads, first through a
`ThreadPoolExecutor`, then with plain threads that each search their own
share of the queries:

```
Python 3.12.14, GIL enabled
concurrent search() calls from a Python thread pool
   1 threads:    12722 QPS
   8 threads:    64877 QPS
  32 threads:    55133 QPS
  64 threads:    54306 QPS
concurrent search() calls, plain threads
   1 threads:    13608 QPS
   8 threads:    96921 QPS
  32 threads:   115389 QPS
  64 threads:   103993 QPS
```

ScaNN releases Python's global interpreter lock while it searches, so
threads run in parallel: 8 plain threads give 7.1× the throughput of one.
Beyond that the curve flattens around 100–115k QPS. The rest of each call,
converting arguments and building result arrays, still holds the lock. At
0.07 ms per search, that serial part is the likely bottleneck. The thread
pool flattens earlier, around 55–65k, because handing each query to it
takes the lock too.

**On free-threaded Python** (3.14t and later) there is no global lock:
scann-core's module declares that it doesn't need one, and it does its own
locking. The same script, on the same machine:

```
Python 3.14.7, GIL disabled
concurrent search() calls from a Python thread pool
   1 threads:    12476 QPS
   8 threads:    95911 QPS
  32 threads:   115706 QPS
  64 threads:   124659 QPS
concurrent search() calls, plain threads
   1 threads:    13671 QPS
   8 threads:    99995 QPS
  32 threads:   314045 QPS
  64 threads:   327429 QPS
```

Plain threads now scale to 327k QPS on 64 threads, 24× one thread, on a
par with batching and with Rust ([part 7](07-cpp-and-rust.md)). The thread
pool still flattens, at about 125k: with one query per task, its shared
work queue is now the bottleneck, not ScaNN. Give each thread a share of
the work rather than one query at a time.

**In Python with the GIL**, either keep about 8 request threads, or batch.
Batching goes further: 294k QPS, against about 100k for threads. **On
free-threaded Python** threads get as far as batching. **From C++ or Rust**
there is no interpreter lock. [Part 7](07-cpp-and-rust.md) measures
concurrent `search()` calls from Rust threads.

**Next:** real collections change. [Part 6: Updating](06-updating.md).
