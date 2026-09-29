# Using scann-core with data and serving frameworks

scann-core is a library: a searcher lives in one process, searches with
that process's CPUs, and is saved to and loaded from a directory. Batch
frameworks (Ray, Spark, Dask) and servers (FastAPI, Ray Serve, BentoML,
Triton) run many processes and threads of their own, so using scann-core
with them comes down to three decisions: where each process gets its
searcher, how many threads it searches with, and how much memory that
takes. This page covers those, then batch retrieval, serving, inputs from
Arrow, pandas and Polars, and combining scann-core with a sparse
(keyword) engine.

For PyTorch and TensorFlow models, see
[integrations.md](integrations.md#scanntorch-searching-from-pytorch-models),
[tensorflow.md](tensorflow.md) and
[tutorial part 8](tutorial/08-pytorch-and-tensorflow.md). For libraries
built on the `scann` API (LangChain), see [integrations.md](integrations.md).

Two runnable examples go with this page:

* [`examples/python/batch_retrieval.py`](../examples/python/batch_retrieval.py):
  an index saved once and loaded by each worker process, queries read
  from an Arrow table without a copy, searched with `multiprocessing` and,
  when Ray is installed, with Ray Data.
* [`examples/python/fastapi_service.py`](../examples/python/fastapi_service.py):
  a FastAPI service with one shared searcher, a batch endpoint and
  server-side micro-batching, run in-process with FastAPI's `TestClient`.

The versions the code on this page was run with, on Linux x86-64 with
Python 3.14: Ray 2.58.0 (Ray Data and Ray Serve), PySpark 4.2.0 (local
mode, OpenJDK 25), Dask and distributed 2026.8.0, FastAPI 0.141.1
(Starlette 1.7.0, uvicorn 0.54.0, httpx2 2.13.1), pyarrow 25.0.1, pandas
3.0.6 and Polars 1.44.2. The BentoML and Triton sections are patterns
checked against those projects' source and documentation (versions given
there), not run.

## The common pattern

### Save once, load once per process

Build the index once, and save it with `relative_path=True`, which makes
the directory movable:

```python
searcher = (scann.scann_ops_pybind.builder(embeddings, 10, "dot_product")
            .tree(num_leaves=2000, num_leaves_to_search=100)
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(100)
            .build(docids=ids))
searcher.serialize(index_dir, relative_path=True)
```

Then copy the directory to wherever the workers can read it (a shared
file system, the image, a download from object storage at start-up), and
have each worker process call `scann.scann_ops_pybind.load_searcher(dir)`
once. Loading reads the files and does no training (0.5 s for the
1.2-million-vector index of the [tutorial](tutorial/05-saving-and-serving.md)).

**Ship the directory, not the searcher.** A searcher can't be pickled
(`TypeError: cannot pickle 'scann_pybind.ScannNumpy' object`), and every
framework here pickles what it sends to workers. The same goes for a
closure or a global that holds one: Dask, for example, fails to submit a
task whose function refers to a global searcher in `__main__`.

**Keep the cache in a module.** A searcher cached in a global of the
driver script (`__main__`, or a notebook) doesn't survive between tasks:
Ray, Spark and Dask pickle functions from `__main__` by value, with the
globals they use, so each task gets a fresh copy of the global and loads
the index again. Measured with 16 partitions: Spark (`local[4]`) loaded it
16 times with a global in `__main__` and 4 times (once per Python worker)
with the cache in an imported module; Dask (2 worker processes) 16 times
against once per worker. So put the loader in a module the workers
import:

```python
# retrieval_worker.py: importable on every worker (installed, or shipped
# with the job: Spark's addPyFile, Dask's upload_file, Ray's runtime_env).
import threading

import numpy as np
import pyarrow as pa
import scann

_lock = threading.Lock()
_searchers = {}


def get_searcher(index_dir, threads):
  """The searcher saved in index_dir, loaded once per process."""
  with _lock:
    searcher = _searchers.get(index_dir)
    if searcher is None:
      searcher = scann.scann_ops_pybind.load_searcher(index_dir)
      searcher.set_num_threads(threads)
      _searchers[index_dir] = searcher
    return searcher


def embedding_matrix(column):
  """An Arrow list or fixed_size_list column as a [rows, dim] numpy view."""
  if isinstance(column, pa.ChunkedArray):
    # combine_chunks() copies, even a single chunk.
    column = (column.chunk(0) if column.num_chunks == 1
              else column.combine_chunks())
  if column.null_count:
    raise ValueError("null embeddings")
  if pa.types.is_fixed_size_list(column.type):
    dim = column.type.list_size
  else:  # list<float>: check that every row has the same length
    lengths = np.diff(column.offsets.to_numpy())
    dim = int(lengths[0]) if len(lengths) else 0
    if (lengths != dim).any():
      raise ValueError("embeddings of different lengths")
  # flatten(), not .values: it respects slicing.
  return (column.flatten().to_numpy(zero_copy_only=True)
          .reshape(len(column), dim))
```

The lock matters where a worker runs several tasks at once in threads
(Dask, Ray Serve, FastAPI): without it, two threads that start together
both load the index (an unlocked `functools.cache` loaded it twice per
Dask worker with 2 threads each). `embedding_matrix` is explained under
[Inputs](#inputs-from-arrow-pandas-and-polars).

### Threads: the index's pool vs the framework's

Each searcher has its own thread pool for `search_batched_parallel` (and
mutations). By default it uses **as many threads as the process has CPUs**:
the CPUs in its affinity mask (`taskset`, cpusets, the container's cpuset),
capped by a cgroup CPU quota (`docker --cpus`, Kubernetes limits), counted
when the index is built or loaded, and started on the first parallel call.
(Before 0.2.1 it was every CPU of the machine: under `taskset -c 0-3` on a
64-thread machine, a new searcher started 63 threads.) The frameworks' own
settings (Ray's `num_cpus`, Spark's task CPUs, Dask's
`threads_per_worker`) only schedule work; they set no affinity, so a
worker still sees every CPU. With several worker processes per machine,
each running a parallel search on every CPU, the machine is oversubscribed
many times over.

So set it: `searcher.set_num_threads(n)`, with n the cores that worker
may use (n threads in all, the calling one included), so that workers × n
≈ the machine's cores; or set `SCANN_NUM_THREADS=n` in the workers'
environment, which also sets the default training threads. (`search()`
and `search_batched()` don't use the pool at all; they run on the calling
thread.) The two ends of the range:

* **Many single-threaded workers** (n = 1, one worker per core): simple,
  but every worker process holds a copy of the index.
* **Few workers with many threads** (one per machine, n = cores): one
  copy of the index, and `search_batched_parallel` over big partitions
  keeps the cores busy. Better when the index is large.

In between, pick the fewest processes the framework lets you have.

### Memory

A loaded searcher takes about the size of its directory in memory: the
tutorial's 514 MiB GloVe index added 499 MiB of RSS when loaded, and 66 MiB
more after searching 10,000 queries (results, per-thread scratch). Every
process that loads the index holds its own copy; nothing is shared
between processes (Ray's object store doesn't help, as loading copies the
data into the searcher). Budget index size × processes per machine, and
shrink the index if needed: `reorder(..., quantize=BFLOAT16)` nearly halves
it for 0.06 points of recall
([tutorial part 5](tutorial/05-saving-and-serving.md#making-the-index-smaller)).

### Docids

Built with `build(docids=ids)`, the searcher saves the ids with the index
(`scann_docids.pkl`) and returns them instead of row indices. That file is
a pickle: load only directories you trust. For batch jobs over millions
of queries, a list of Python strings per query is the slowest part of the
output. The alternative is an index without docids, whose searches return
row indices (a uint32 numpy array), mapped to ids in the framework, for
example with `pyarrow.compute.take` or a join against an id table.

A row shorter than k (only when the index has fewer than k points) is
padded with docid `None` (index 0) and distance NaN. Filter those before
writing results.

## Batch retrieval

The job: many queries (a table of embeddings), the k nearest neighbours of
each, written back as a table. Each partition of the queries is one
`search_batched_parallel()` call on the worker's searcher.

### Ray Data

A callable class passed to `map_batches` runs in a pool of actors, and
its constructor runs once per actor: that is where the index is loaded.

```python
import pyarrow as pa
import ray
import scann
from retrieval_worker import embedding_matrix

INDEX_DIR = "/mnt/indexes/products"   # readable from every node
THREADS = 8                           # CPUs per actor


class Search:
  def __init__(self, index_dir, threads):
    self.searcher = scann.scann_ops_pybind.load_searcher(index_dir)
    self.searcher.set_num_threads(threads)

  def __call__(self, batch):  # a pyarrow.Table
    docids, scores = self.searcher.search_batched_parallel(
        embedding_matrix(batch.column("embedding")), final_num_neighbors=10)
    return pa.table({
        "query_id": batch.column("query_id"),
        "docids": pa.array(docids, pa.list_(pa.string())),
        "scores": pa.array(list(scores), pa.list_(pa.float32()))})


queries = ray.data.read_parquet("s3://bucket/queries/")   # query_id, embedding
results = queries.map_batches(
    Search,
    fn_constructor_args=(INDEX_DIR, THREADS),
    compute=ray.data.ActorPoolStrategy(size=4),  # 4 actors = 4 index copies
    num_cpus=THREADS,                            # CPUs reserved per actor
    batch_size=1024,
    batch_format="pyarrow")
results.write_parquet("s3://bucket/neighbors/")
```

* **`batch_format="pyarrow"`** for embeddings stored as Arrow lists (as
  Parquet files written by pyarrow, Polars or Spark hold them), which
  `embedding_matrix` reads without a copy. With `"numpy"`, Ray Data turned
  a fixed-size-list column into an object array of per-row arrays, which
  then needs an `np.stack` copy. Columns made with `ray.data.from_numpy`
  (Ray's tensor type) arrive as 2-D arrays in `"numpy"` batches.
* **`num_cpus` and `set_num_threads` together**: Ray reserves the CPUs for
  the actor, and the searcher's pool uses them. Leave some CPUs over for
  reading and writing: with 2 actors × 2 CPUs on a 4-CPU Ray, the
  pipeline above stalled, the Parquet read task waiting for a CPU that
  the actors held.
* The constructor loads the index itself rather than calling a helper
  from `__main__`: Ray pickles the class, with the globals it uses, by
  value (a lock among them fails to pickle).
* `concurrency=` still works for the pool size, but Ray 2.58 warns that it
  is deprecated (since 2.51) in favour of `compute=`.

Runnable (with a local Ray, and the same with `multiprocessing`):
[`examples/python/batch_retrieval.py`](../examples/python/batch_retrieval.py).
It gives Ray a temporary directory of its own (`ray.init(_temp_dir=...)`,
removed at the end); note that Ray's session directory must have a short
path, as Unix socket paths are limited to 107 bytes.

### Spark

`mapInArrow` passes a partition's rows as Arrow record batches; Spark's
`array<float>` becomes Arrow `list<float>`, which `embedding_matrix`
reads without a copy. `mapInPandas` works the same way with pandas frames,
where the column holds one numpy array per row and needs `np.stack` (a
copy).

```python
import numpy as np
import pandas as pd
import pyarrow as pa
from pyspark import TaskContext
from pyspark.sql import SparkSession

spark = SparkSession.builder.config("spark.task.cpus", "8").getOrCreate()
spark.sparkContext.addPyFile("retrieval_worker.py")
INDEX_DIR = "/mnt/indexes/products"   # the same path on every executor
SCHEMA = "query_id long, docids array<string>, scores array<float>"


def search_arrow(batches):
  import retrieval_worker
  searcher = retrieval_worker.get_searcher(INDEX_DIR, TaskContext.get().cpus())
  for batch in batches:
    emb = retrieval_worker.embedding_matrix(batch.column("embedding"))
    docids, scores = searcher.search_batched_parallel(emb, final_num_neighbors=10)
    yield pa.RecordBatch.from_pydict({
        "query_id": batch.column("query_id"),
        "docids": pa.array(docids, pa.list_(pa.string())),
        "scores": pa.array(list(scores), pa.list_(pa.float32()))})


def search_pandas(frames):
  import retrieval_worker
  searcher = retrieval_worker.get_searcher(INDEX_DIR, TaskContext.get().cpus())
  for df in frames:
    emb = np.stack(df["embedding"].to_numpy())   # object column: one copy
    docids, scores = searcher.search_batched_parallel(emb, final_num_neighbors=10)
    yield pd.DataFrame({"query_id": df["query_id"], "docids": docids,
                        "scores": list(scores)})


queries = spark.read.parquet("/data/queries")   # query_id, embedding array<float>
results = queries.mapInArrow(search_arrow, SCHEMA)   # or mapInPandas(search_pandas, SCHEMA)
results.write.parquet("/data/neighbors")
```

* **Threads.** An executor runs (executor cores ÷ `spark.task.cpus`)
  tasks at once, each in its own Python worker process;
  `TaskContext.get().cpus()` is `spark.task.cpus` inside the task (it was
  2 in the local run with that setting), so the searcher's pool matches
  what Spark reserved. More `spark.task.cpus` means fewer Python workers,
  so fewer copies of the index per executor.
* **Python workers are reused** between tasks (Spark's default), so the
  module cache loads the index once per worker process, not per task.
* **The workers' Python must have scann-core.** In local mode the worker
  processes ran the system `python3` until `PYSPARK_PYTHON` pointed them
  at the virtual environment's interpreter.
* PySpark 4.2.0 warns that pandas >= 3.0 isn't fully supported yet; both
  functions above gave correct results with pandas 3.0.6.

### Dask

Dask workers are processes with several threads each. One searcher per
worker process is shared by its threads (a searcher is thread-safe), and
`get_searcher`'s lock keeps them from loading it twice. With `dask.array`:

```python
import dask.array as da
import numpy as np
from dask.distributed import Client

client = Client("scheduler:8786")
client.upload_file("retrieval_worker.py")
INDEX_DIR = "/mnt/indexes/products"


def search_block(block):
  import retrieval_worker
  searcher = retrieval_worker.get_searcher(INDEX_DIR, threads=8)
  docids, _ = searcher.search_batched_parallel(block, final_num_neighbors=10)
  return np.array(docids, dtype=object)


queries = da.from_zarr("queries.zarr")                  # [n, dim] float32
queries = queries.rechunk((10_000, queries.shape[1]))   # whole rows per block
docids = queries.map_blocks(search_block, chunks=(10_000, 10), dtype=object,
                            meta=np.empty((0, 10), dtype=object))
```

Pass `meta=`: without it, Dask calls the function on the client to infer
the output type. For dataframes, `map_partitions` works the same way, with
`embedding_matrix` or `np.stack` on the embedding column.

### Without a framework

On one machine, `multiprocessing` with an initializer does the same:

```python
def load(index_dir, threads):
  global searcher
  searcher = scann.scann_ops_pybind.load_searcher(index_dir)
  searcher.set_num_threads(threads)

def search_partition(batch):   # an Arrow RecordBatch
  docids, _ = searcher.search_batched_parallel(
      embedding_matrix(batch.column("embedding")), final_num_neighbors=10)
  return batch.column("query_id").to_pylist(), docids

with multiprocessing.get_context("spawn").Pool(
    4, initializer=load, initargs=(index_dir, 16)) as pool:
  results = pool.map(search_partition, table.to_batches(max_chunksize=10_000))
```

([`batch_retrieval.py`](../examples/python/batch_retrieval.py) runs this.)

Often one process is enough: `search_batched_parallel` over all queries
already uses every core, with one copy of the index. Split into processes
when the queries don't fit in memory at once, or the job needs to write
results as it goes.

## Serving

A service loads the index once at start-up and shares the searcher
between all requests: searches can run concurrently from any number of
threads, and `upsert`, `delete` and `rebalance` wait for running searches
([README: Threads](../README.md#threads)). The choices are how requests
reach the searcher:

* **One search per request**, from the server's request threads.
  Lowest latency; with the GIL, Python threads plateau at about 100k
  searches per second on a 64-thread machine, and at about 330k on
  free-threaded Python (3.14t), where scann-core runs without the GIL
  ([tutorial part 5](tutorial/05-saving-and-serving.md#serving-batch-size-latency-and-throughput)).
* **Micro-batching**: collect the requests that arrive within a millisecond
  or two and search them with one `search_batched_parallel` call. More
  throughput with the GIL (in part 5, 177k queries per second in batches
  of 64, 250k in batches of 512), at the cost of the wait.
* **Client batches**: a request that carries many queries is one batched
  call.

### FastAPI

[`examples/python/fastapi_service.py`](../examples/python/fastapi_service.py)
has all three:

```python
@contextlib.asynccontextmanager
async def lifespan(app):
  # Load once per process. The searcher is shared by every request.
  searcher = scann.scann_ops_pybind.load_searcher(os.environ["SCANN_INDEX_DIR"])
  batcher = MicroBatcher(searcher)            # see the example
  task = asyncio.create_task(batcher.run())
  app.state.searcher, app.state.batcher = searcher, batcher
  yield
  task.cancel()

app = fastapi.FastAPI(lifespan=lifespan)

@app.post("/search")              # def, not async def: runs in the thread pool
def search(query: Query):
  docids, scores = app.state.searcher.search(
      vector_array(query.vector), final_num_neighbors=query.k)
  return response(docids, scores)

@app.post("/search_micro")        # concurrent requests, one batched search
async def search_micro(query: Query):
  docids, scores = await app.state.batcher.search(
      vector_array(query.vector), query.k)
  return response(docids, scores)
```

* **`def` endpoints for direct searches.** FastAPI runs a plain `def`
  endpoint in its thread pool, where the search releases the GIL. Calling
  the searcher from an `async def` endpoint would block the event loop for
  the whole search; there, use `await asyncio.to_thread(...)` (the
  micro-batcher does).
* **The thread pool has 40 threads** by default (AnyIO's default
  limiter, which Starlette uses for `def` endpoints; checked in AnyIO
  4.15.1). With the GIL, more request threads than that wouldn't search
  faster anyway; on free-threaded Python, raise it if the searches keep
  all threads busy, with
  `anyio.to_thread.current_default_thread_limiter().total_tokens = n` in
  the lifespan function (it needs the running event loop).
* **Micro-batching** in the example is an `asyncio.Queue` drained by one
  task: take the first request, wait up to 2 ms (or 64 requests) for
  more, search them in one `search_batched_parallel` call in a worker
  thread, and resolve each request's future. In the example's runs, 400
  concurrent requests from 16 client threads became 30 to 34 batches.
* **One server process, not many.** `uvicorn --workers N` starts N
  processes, each loading its own copy of the index. The searcher is
  thread-safe, so a single process with threads (or batching) uses the
  cores with one copy; add processes only when the Python around the
  search (parsing, serialization) is the bottleneck, and set each index's
  `set_num_threads` accordingly.
* **NaN is not JSON.** Padding (docid `None`, distance NaN) must be
  dropped or converted before returning.
* The example runs the app in-process with `TestClient`, which needs
  `httpx2` with Starlette 1.7 (`httpx` still works, with a deprecation
  warning). For a real server:
  `SCANN_INDEX_DIR=... uvicorn fastapi_service:app` (checked with uvicorn
  0.54.0).

### Ray Serve

A deployment is a class whose constructor loads the index, and
`@serve.batch` does the micro-batching (Ray 2.58.0; run with a handle
from `serve.run`):

```python
import numpy as np
import scann
from ray import serve


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 4})
class Retrieval:
  def __init__(self, index_dir: str, threads: int):
    self.searcher = scann.scann_ops_pybind.load_searcher(index_dir)
    self.searcher.set_num_threads(threads)

  @serve.batch(max_batch_size=64, batch_wait_timeout_s=0.002)
  async def search(self, vectors: list[np.ndarray]) -> list[list[str]]:
    docids, _ = self.searcher.search_batched_parallel(np.stack(vectors))
    return docids

  async def __call__(self, request) -> dict:
    body = await request.json()
    return {"docids": await self.search(np.asarray(body["vector"], np.float32))}


app = Retrieval.bind("/mnt/indexes/products", 4)
# serve.run(app), or `serve run module:app`
```

Each replica is a process with its own copy of the index; match
`set_num_threads` to the replica's `num_cpus`. The batched search above
runs on the event loop's thread; for long searches, move it to a thread
(`await asyncio.to_thread(...)`) so the replica keeps accepting requests.

### BentoML

The same shape: a service class that loads the index in its constructor,
and an API method that BentoML's adaptive batching feeds with stacked
inputs. A sketch, checked against the BentoML 1.4.39 source
(`bentoml.service`, `bentoml.api(batchable=True, batch_dim=0,
max_batch_size=..., max_latency_ms=...)`), not run:

```python
import bentoml
import numpy as np
import scann


@bentoml.service(workers=1)
class Retrieval:
  def __init__(self):
    self.searcher = scann.scann_ops_pybind.load_searcher("/models/index")

  @bentoml.api(batchable=True, max_batch_size=64, max_latency_ms=10)
  def search(self, vectors: np.ndarray) -> np.ndarray:
    indices, _ = self.searcher.search_batched_parallel(vectors)
    return np.asarray(indices)   # an index built without docids
```

Each BentoML worker is a process with its own copy of the index.

### Triton Inference Server (Python backend)

Triton's Python backend runs a `model.py` whose `TritonPythonModel` has
`initialize(args)` (called when the model is loaded) and `execute(requests)`
(a list of requests, which dynamic batching may group). Load the index in
`initialize`, and search all requests of an `execute` call in one batch.
A sketch, checked against the python_backend README (commit `26764b9`,
Triton 2.72.0 current at the time), not run:

```python
import os
import numpy as np
import scann
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
  def initialize(self, args):
    index_dir = os.path.join(args["model_repository"], args["model_version"],
                             "index")
    self.searcher = scann.scann_ops_pybind.load_searcher(index_dir)

  def execute(self, requests):
    queries = [pb_utils.get_input_tensor_by_name(r, "QUERY").as_numpy()
               for r in requests]
    sizes = [len(q) for q in queries]
    indices, scores = self.searcher.search_batched_parallel(
        np.concatenate(queries))            # an index built without docids
    responses, start = [], 0
    for n in sizes:
      out = [pb_utils.Tensor("INDICES", np.asarray(indices[start:start + n], np.int32)),
             pb_utils.Tensor("SCORES", np.asarray(scores[start:start + n], np.float32))]
      responses.append(pb_utils.InferenceResponse(output_tensors=out))
      start += n
    return responses
```

scann-core has to be installed for the Python the backend's stub runs
(Python 3.12 in NVIDIA's Triton containers; another version needs a
custom stub, per the README). Every model instance (`instance_group`)
loads its own copy of the index.

## Inputs from Arrow, pandas and Polars

scann-core reads a **C-contiguous float32 numpy array in place**; anything
else is converted to one first, which copies. Measured with `tracemalloc`
on a 4.9 MiB batch of queries: a C-contiguous float32 array allocated only
the results (1.5 MiB), while the same data in Fortran order or as float64
allocated 4.9 MiB more for the converted copy. Building an index always
stores its own copy of the data, so zero-copy input matters most for
queries, and for keeping a build's peak memory at one extra copy.

The table (all checked in code, with the versions at the top of this page):

| source | to numpy | copy? |
|---|---|---|
| Arrow `fixed_size_list<float>[d]` (pyarrow's `FixedSizeListArray`, and Parquet written from one) | `embedding_matrix(column)` | no (one chunk, no nulls) |
| Arrow `list<float>` with equal lengths (Spark `array<float>`, Parquet from lists) | `embedding_matrix(column)` | no |
| Arrow `fixed_shape_tensor` extension | `column.to_numpy_ndarray()` | no |
| pandas column with `ArrowDtype` (`pd.read_parquet(..., dtype_backend="pyarrow")`) | `embedding_matrix(pa.array(df["embedding"]))` | no |
| pandas column of numpy arrays (the default `read_parquet`, `list(arr)`) | `np.stack(df["embedding"].to_numpy())` | yes, unavoidable |
| pandas frame with one column per dimension | `df.to_numpy()` is Fortran-ordered | yes, at every search: convert once with `np.ascontiguousarray` |
| Polars `Array(Float32, d)` | `series.to_numpy(allow_copy=False)` | no |
| Polars `List(Float32)` | `cast(pl.Array(pl.Float32, d))` first; `to_numpy(allow_copy=False)` raises on a `List` | the cast |
| PyTorch CPU float32 tensor | passed as is | no ([integrations.md](integrations.md#arrays-from-pytorch-and-other-libraries)) |

Details for Arrow (pyarrow 25.0.1):

* `flatten()`, not `.values`: `.values` is the whole child array, ignoring
  the slice a batch may be of a larger array (a 50-row slice of a
  3,000-row column gave all 3,000 rows' values). `flatten()` respects it.
* `to_numpy(zero_copy_only=True)` raises instead of copying silently, so
  the view is guaranteed. The result is read-only, which scann-core
  accepts for building and searching.
* `ChunkedArray.combine_chunks()` copies, even when there is only one
  chunk; `embedding_matrix` takes the chunk directly in that case. Several
  chunks need a copy, or one search per chunk.
* Null embeddings, and `float64` values, can't be viewed as a float32
  matrix. `embedding_matrix` rejects nulls; float64 values give a float64
  view, which scann-core converts (a copy).

## Dense + sparse hybrid retrieval

**scann-core has no sparse search.** Its Python, C++ facade and Rust
APIs take dense float32 vectors only. ScaNN's C++ core does have sparse
datapoint types and a sparse path in brute-force scoring, inherited from
upstream, but no API reaches them and scann-core doesn't test them.

For hybrid retrieval (dense embeddings plus keyword or learned-sparse
matching), run the sparse side in an engine built for it: BM25 in Lucene,
Elasticsearch or OpenSearch, or a learned-sparse model (SPLADE and
similar) in an engine that supports it. Then fuse:

1. Search both with the same query, each for more than you need
   (100 to 1,000 candidates each).
2. Use the same ids on both sides: build the scann-core index with
   `docids=` set to the documents' ids in the sparse engine.
3. Fuse the two ranked lists into one.

**Reciprocal rank fusion** (Cormack, Clarke and Büttcher, SIGIR 2009)
uses only ranks, so the two engines' scores don't need to be comparable:

```python
def rrf(rankings, k=60):
  """Reciprocal rank fusion: rankings are lists of docids, best first."""
  scores = {}
  for ranking in rankings:
    for rank, docid in enumerate(ranking, start=1):
      if docid is not None:  # ScaNN pads short rows with None
        scores[docid] = scores.get(docid, 0.0) + 1.0 / (k + rank)
  return sorted(scores, key=scores.get, reverse=True)


dense_ids, dense_scores = searcher.search(query_embedding, final_num_neighbors=100)
sparse_ids = [hit["_id"] for hit in bm25_hits]   # from your sparse engine
top10 = rrf([dense_ids, sparse_ids])[:10]
```

**Weighted fusion** combines scores, so it needs them on one scale;
min-max normalization per query is the simple choice:

```python
def weighted(dense, sparse, alpha=0.5):
  """alpha * dense + (1 - alpha) * sparse, each min-max normalized.

  dense, sparse: {docid: score}, higher is better.
  """
  def norm(s):
    if not s:
      return {}
    lo, hi = min(s.values()), max(s.values())
    return {d: (v - lo) / (hi - lo) if hi > lo else 1.0 for d, v in s.items()}
  dense, sparse = norm(dense), norm(sparse)
  fused = {d: alpha * dense.get(d, 0.0) + (1 - alpha) * sparse.get(d, 0.0)
           for d in dense.keys() | sparse.keys()}
  return sorted(fused, key=fused.get, reverse=True)


top10 = weighted({d: float(s) for d, s in zip(dense_ids, dense_scores)
                  if d is not None},
                 dict(zip(sparse_ids, sparse_scores)), alpha=0.7)[:10]
```

scann-core's scores are dot products for a `"dot_product"` index (higher
is better) and squared distances for `"squared_l2"` (lower is better:
negate them before fusing). With AH scoring and no reordering they are
approximations; `reorder()` makes the returned scores exact. The weight
`alpha` is worth tuning on labelled queries; RRF's `k = 60` rarely is.

Many sparse engines also have dense vector search, and can fuse
internally. scann-core is the choice for the dense side when its speed or
recall at the same cost matters more than having one system; fusing in
the application, as above, is then a few lines.
