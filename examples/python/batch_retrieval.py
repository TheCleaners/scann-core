#!/usr/bin/env python3
# Copyright 2026 Elias Benali (@ebenali) and TheCleaners.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline batch retrieval: one saved index, many worker processes.

The pattern behind batch retrieval on Ray, Spark or Dask:

  1. build the index once and save it with relative_path=True, so the
     directory can be copied anywhere (here: to a "shipped" directory);
  2. each worker process loads it once (load_searcher), and sizes the
     index's thread pool to its share of the CPUs (set_num_threads);
  3. the queries are split into partitions; each partition is one
     search_batched_parallel() call; indices come back as docids.

The queries are an Arrow table with a fixed-size-list embedding column, as
read from Parquet; the embeddings reach the searcher without a copy.

Runs with Python's multiprocessing, and also with Ray Data when Ray is
installed (`pip install 'ray[data]'`). Ray's session files go to a
temporary directory (under $RAY_TMPDIR if set), removed at the end. Needs
pyarrow; exits with status 77 (reported as skipped by ctest) without it.
Run with scann-core installed, or from a CMake build tree:
  PYTHONPATH=<build>/python python examples/python/batch_retrieval.py
"""

import multiprocessing
import os
import shutil
import sys
import tempfile
import threading

try:
  import pyarrow as pa
except ImportError:
  print("pyarrow is not installed; skipping (pip install pyarrow)")
  sys.exit(77)

import numpy as np
import scann

DIM, K = 32, 10
WORKERS, THREADS_PER_WORKER = 2, 2


def embedding_matrix(column):
  """A fixed_size_list<float32>[DIM] column as a [rows, DIM] numpy view.

  No copy for one chunk without nulls. flatten() (not .values) respects the
  slice a batch may be of a larger array.
  """
  if isinstance(column, pa.ChunkedArray):
    # combine_chunks() copies, even a single chunk; take that chunk instead.
    column = (column.chunk(0) if column.num_chunks == 1
              else column.combine_chunks())
  width = column.type.list_size
  return column.flatten().to_numpy(zero_copy_only=True).reshape(-1, width)


# --- One searcher per worker process ------------------------------------------
# Module-level state, set by the pool's initializer: loaded once per worker,
# not once per task. (A searcher can't be pickled; ship its directory.)
_searcher = None
_lock = threading.Lock()


def load(index_dir, threads):
  global _searcher
  with _lock:
    if _searcher is None:
      _searcher = scann.scann_ops_pybind.load_searcher(index_dir)
      _searcher.set_num_threads(threads)
  return _searcher


def search_partition(batch):
  """An Arrow RecordBatch of queries -> (query ids, docids)."""
  docids, _ = _searcher.search_batched_parallel(
      embedding_matrix(batch.column("embedding")), final_num_neighbors=K)
  return batch.column("query_id").to_pylist(), docids


class RaySearch:
  """A Ray Data map_batches() class: constructed once per actor.

  Ray pickles this class by value (it is in __main__), with every global
  it refers to, so it loads the index itself instead of calling load(),
  whose lock can't be pickled.
  """

  def __init__(self, index_dir, threads):
    self.searcher = scann.scann_ops_pybind.load_searcher(index_dir)
    self.searcher.set_num_threads(threads)

  def __call__(self, batch):  # a pyarrow.Table
    docids, scores = self.searcher.search_batched_parallel(
        embedding_matrix(batch.column("embedding")), final_num_neighbors=K)
    return pa.table({"query_id": batch.column("query_id"),
                     "docids": pa.array(docids, pa.list_(pa.string())),
                     "scores": pa.array(list(scores),
                                        pa.list_(pa.float32()))})


def run_ray(index_dir, table):
  import ray  # pylint: disable=g-import-not-at-top
  import ray.data  # pylint: disable=g-import-not-at-top

  ray_tmp = tempfile.mkdtemp(prefix="ray-", dir=os.environ.get("RAY_TMPDIR"))
  try:
    ray.init(num_cpus=WORKERS * THREADS_PER_WORKER, _temp_dir=ray_tmp,
             include_dashboard=False, log_to_driver=False)
    ray.data.DataContext.get_current().enable_progress_bars = False
    out = (ray.data.from_arrow(table)
           .map_batches(RaySearch,
                        fn_constructor_args=(index_dir, THREADS_PER_WORKER),
                        compute=ray.data.ActorPoolStrategy(size=WORKERS),
                        num_cpus=THREADS_PER_WORKER,
                        batch_size=500, batch_format="pyarrow")
           .to_arrow_refs())
    result = pa.concat_tables(ray.get(out)).sort_by("query_id")
    return result.column("docids").to_pylist()
  finally:
    ray.shutdown()
    shutil.rmtree(ray_tmp, ignore_errors=True)


def main():
  rng = np.random.default_rng(0)
  centers = rng.standard_normal((50, DIM))

  def clustered(n):
    x = centers[rng.integers(0, 50, n)] + 0.3 * rng.standard_normal((n, DIM))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)

  items = clustered(20_000)
  item_ids = [f"item-{i}" for i in range(len(items))]
  queries = clustered(4_000)
  table = pa.table({
      "query_id": pa.array(np.arange(len(queries))),
      "embedding": pa.FixedSizeListArray.from_arrays(
          pa.array(queries.ravel()), DIM),
  })
  emb = embedding_matrix(table.column("embedding"))
  print("embeddings read from Arrow without a copy:",
        np.shares_memory(emb, np.frombuffer(
            table.column("embedding").chunk(0).values.buffers()[1],
            dtype=np.float32)))

  with tempfile.TemporaryDirectory() as tmp:
    # 1. Build once, save a movable directory, "ship" it.
    built = os.path.join(tmp, "built")
    os.makedirs(built)
    index = (scann.scann_ops_pybind.builder(items, K, "dot_product")
             .tree(num_leaves=140, num_leaves_to_search=20, random_init=False)
             .score_ah(2, anisotropic_quantization_threshold=0.2)
             .reorder(100)
             .build(docids=item_ids))
    index.serialize(built, relative_path=True)
    expected, _ = index.search_batched(queries)
    exact = np.argsort(-queries @ items.T, axis=1)[:, :K]
    hits = sum(len(set(e) & {item_ids[j] for j in x})
               for e, x in zip(expected, exact))
    print(f"index: {len(items)} items; recall@{K} {hits / exact.size:.3f}")
    shipped = os.path.join(tmp, "shipped")
    shutil.copytree(built, shipped)
    shutil.rmtree(built)  # the copy must not depend on the original

    # 2 + 3. multiprocessing: WORKERS processes, THREADS_PER_WORKER each.
    ctx = multiprocessing.get_context("spawn")  # the same on every platform
    partitions = table.to_batches(max_chunksize=500)
    with ctx.Pool(WORKERS, initializer=load,
                  initargs=(shipped, THREADS_PER_WORKER)) as pool:
      found = {}
      for ids, docids in pool.imap_unordered(search_partition, partitions):
        found.update(zip(ids, docids))
    got = [found[i] for i in range(len(queries))]
    assert got == expected
    print(f"multiprocessing: {len(partitions)} partitions on {WORKERS} "
          f"workers x {THREADS_PER_WORKER} threads; results equal a direct "
          "search")

    try:
      import ray  # pylint: disable=g-import-not-at-top,unused-import
    except ImportError:
      print("Ray is not installed; skipping the Ray Data run")
      return
    got = run_ray(shipped, table)
    assert got == expected
    print(f"Ray Data: map_batches on {WORKERS} actors x "
          f"{THREADS_PER_WORKER} threads; results equal a direct search")


if __name__ == "__main__":
  main()
