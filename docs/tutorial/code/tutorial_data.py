"""Shared helpers for the scann-core tutorial scripts.

The tutorial uses GloVe-100 from ann-benchmarks (http://ann-benchmarks.com):
1,183,514 word vectors of dimension 100, 10,000 held-out query vectors, and
the exact 100 nearest neighbours of every query (by cosine similarity). It is
the dataset of the ScaNN paper's benchmarks.

The file (~485 MB) is downloaded once into $SCANN_TUTORIAL_DATA (default
~/.cache/scann-core-tutorial).
"""

import os
import time
import urllib.request

import numpy as np

URL = "http://ann-benchmarks.com/glove-100-angular.hdf5"


def data_dir():
  d = os.environ.get("SCANN_TUTORIAL_DATA",
                     os.path.expanduser("~/.cache/scann-core-tutorial"))
  os.makedirs(d, exist_ok=True)
  return d


def load_glove():
  """Returns (dataset, queries, true_neighbors), vectors L2-normalized."""
  import h5py  # pip install h5py

  path = os.path.join(data_dir(), "glove-100-angular.hdf5")
  if not os.path.exists(path):
    print(f"downloading {URL} -> {path} (~485 MB)")
    urllib.request.urlretrieve(URL, path + ".part")
    os.rename(path + ".part", path)
  with h5py.File(path, "r") as f:
    dataset = f["train"][:]
    queries = f["test"][:]
    true_neighbors = f["neighbors"][:]
  # "angular" = cosine similarity. Normalized, cosine is the dot product.
  dataset /= np.linalg.norm(dataset, axis=1, keepdims=True)
  queries /= np.linalg.norm(queries, axis=1, keepdims=True)
  return dataset, queries, true_neighbors


def recall(found, true_neighbors):
  """recall@k: the fraction of the true k nearest neighbours that were found.

  found: (num_queries, k) indices; true_neighbors: (num_queries, >= k).
  """
  k = found.shape[1]
  hits = sum(
      np.intersect1d(f, t[:k]).size for f, t in zip(found, true_neighbors))
  return hits / (found.shape[0] * k)


class Timer:
  """with Timer() as t: ...; t.seconds"""

  def __enter__(self):
    self.start = time.perf_counter()
    return self

  def __exit__(self, *exc):
    self.seconds = time.perf_counter() - self.start


def evaluate(name, searcher, queries, true_neighbors, latency_queries=1000,
             **search_args):
  """Prints recall@k, throughput and single-query latency of a searcher.

  Throughput: search_batched_parallel over all queries with one thread per
  core, best of at least 3 passes and 1 second (a single fast pass is too
  short to time reliably). Latency: search() one query at a time on the
  calling thread, over the first `latency_queries` queries. search_args
  (leaves_to_search, pre_reorder_num_neighbors, ...) are passed to both.
  """
  searcher.set_num_threads(os.cpu_count())
  best, total, passes = float("inf"), 0.0, 0
  while passes < 3 or total < 1.0:
    with Timer() as t:
      neighbors, _ = searcher.search_batched_parallel(queries, **search_args)
    best, total, passes = min(best, t.seconds), total + t.seconds, passes + 1
  qps = len(queries) / best
  with Timer() as t1:
    for q in queries[:latency_queries]:
      searcher.search(q, **search_args)
  r = recall(neighbors, true_neighbors)
  print(f"{name:34} recall@10 {r:.4f}  {qps:8.0f} QPS  "
        f"{1000 * t1.seconds / latency_queries:6.3f} ms/query")
  return r, qps
