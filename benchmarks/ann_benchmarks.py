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

"""ann-benchmarks datasets: index build time, recall@10, throughput, latency.

The numbers in docs/benchmarks.md come from this script (GloVe-100, the
default). It runs against whichever `scann` package the interpreter imports,
so the same script measures scann-core and the upstream wheel:

  PYTHONPATH=build/python python benchmarks/ann_benchmarks.py --label scann-core
  /path/to/wheel-venv/bin/python benchmarks/ann_benchmarks.py --label upstream
  ... --dataset sift-128-euclidean      # any ann-benchmarks dataset

Datasets come from http://ann-benchmarks.com (HDF5, downloaded once into
$SCANN_TUTORIAL_DATA, default ~/.cache/scann-core-tutorial). Names ending in
-angular are L2-normalized and searched by dot product (cosine); names
ending in -euclidean are searched by squared L2. Examples: glove-100-angular
(1.18M x 100, 485 MB), nytimes-256-angular (290k x 256), sift-128-euclidean
(1M x 128), fashion-mnist-784-euclidean (60k x 784, 217 MB),
gist-960-euclidean (1M x 960, 3.6 GB).

For each config it reports:
  build    wall time of builder.build() (partitioner training, hashing, ...),
           with scann's default training threads (one per core)
  recall   recall@10 against the dataset's exact neighbours
  QPS      search_batched_parallel over all queries on --threads threads,
           best of at least 5 passes and 2 seconds
  latency  search() one query at a time on one thread, mean over the first
           --latency-queries queries

Trees are trained with k-means++ (random_init=False), so every build of a
given implementation trains the same partitioner. The tree scales with the
dataset: 2000 leaves from a million points up (GloVe's documented setting),
2*sqrt(n) below that, searching 5% of them.
"""

import argparse
import datetime
import json
import math
import os
import platform
import shutil
import sys
import time
import urllib.request

import numpy as np
import scann

# scann-core has scann.ReorderType; the upstream wheel takes a bool.
INT8 = scann.ReorderType.INT8 if hasattr(scann, "ReorderType") else True
CONFIGS = ("brute_force_f32", "brute_force_int8", "tree_ah_reorder",
           "tree_ah_reorder_int8", "tree_ah_reorder_noaq")


def load(name):
  """Returns (dataset, queries, true neighbours, distance measure)."""
  import h5py  # pip install h5py

  if name.endswith("-angular"):
    distance = "dot_product"
  elif name.endswith("-euclidean"):
    distance = "squared_l2"
  else:
    raise SystemExit(f"{name}: expected an ann-benchmarks name ending in "
                     "-angular or -euclidean")
  cache = os.environ.get("SCANN_TUTORIAL_DATA",
                         os.path.expanduser("~/.cache/scann-core-tutorial"))
  os.makedirs(cache, exist_ok=True)
  path = os.path.join(cache, f"{name}.hdf5")
  if not os.path.exists(path):
    url = f"http://ann-benchmarks.com/{name}.hdf5"
    print(f"downloading {url} -> {path}", flush=True)
    # ann-benchmarks.com answers Python's default User-Agent with 403.
    request = urllib.request.Request(url, headers={"User-Agent": "scann-core"})
    with urllib.request.urlopen(request, timeout=60) as response, \
         open(path + ".part", "wb") as out:
      shutil.copyfileobj(response, out, length=1 << 20)
    os.rename(path + ".part", path)
  with h5py.File(path, "r") as f:
    dataset = np.asarray(f["train"][:], dtype=np.float32)
    queries = np.asarray(f["test"][:], dtype=np.float32)
    truth = f["neighbors"][:]
  if distance == "dot_product":
    dataset /= np.linalg.norm(dataset, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
  return dataset, queries, truth, distance


def recall(found, truth):
  k = found.shape[1]
  hits = sum(np.intersect1d(f, t[:k]).size for f, t in zip(found, truth))
  return hits / (found.shape[0] * k)


def configs(dataset, distance):
  n = dataset.shape[0]
  leaves = 2000 if n >= 1_000_000 else max(16, round(2 * math.sqrt(n)))
  search = max(1, leaves // 20)
  sample = min(250_000, n)
  dot = distance == "dot_product"
  b = lambda: scann.scann_ops_pybind.builder(dataset, 10, distance)
  tree = lambda: b().tree(leaves, search, training_sample_size=sample,
                          random_init=False)
  aq = {"anisotropic_quantization_threshold": 0.2} if dot else {}
  out = {
      "brute_force_f32": lambda: b().score_brute_force(),
      "brute_force_int8": lambda: b().score_brute_force(quantize=INT8),
      "tree_ah_reorder": lambda: tree().score_ah(2, **aq).reorder(100),
      "tree_ah_reorder_int8":
          lambda: tree().score_ah(2, **aq).reorder(100, quantize=INT8),
  }
  if dot:  # without AQ; for squared L2, tree_ah_reorder already is that
    out["tree_ah_reorder_noaq"] = lambda: tree().score_ah(2).reorder(100)
  return out, {"num_leaves": leaves, "leaves_to_search": search,
               "training_sample_size": sample}


def environment(args, dataset, distance, tree):
  """What the numbers depend on, recorded with them."""
  cpu = platform.processor() or platform.machine()
  try:
    with open("/proc/cpuinfo") as f:
      for line in f:
        if line.startswith("model name"):
          cpu = line.split(":", 1)[1].strip()
          break
  except OSError:
    pass
  gil = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
  return {
      "label": args.label,
      "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
      "scann": getattr(scann, "__version__", "?"),
      "scann_file": scann.__file__,
      "python": sys.version.split()[0] + ("" if gil else " (free-threaded)"),
      "numpy": np.__version__,
      "machine": platform.machine(),
      "cpu": cpu,
      "cpu_count": os.cpu_count(),
      "threads": args.threads,
      "dataset": args.dataset,
      "points": int(dataset.shape[0]),
      "dimensionality": int(dataset.shape[1]),
      "distance": distance,
      "tree": tree,
  }


def main():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--dataset", default="glove-100-angular")
  p.add_argument("--label", default="scann")
  p.add_argument("--json", help="also write the results (and environment) here")
  p.add_argument("--threads", type=int, default=os.cpu_count())
  p.add_argument("--latency-queries", type=int, default=1000)
  p.add_argument("--only", nargs="+", choices=CONFIGS, metavar="CONFIG",
                 help=f"run only these configs ({', '.join(CONFIGS)})")
  args = p.parse_args()

  dataset, queries, truth, distance = load(args.dataset)
  builders, tree = configs(dataset, distance)
  env = environment(args, dataset, distance, tree)
  print(f"{args.dataset}: {env['points']} x {env['dimensionality']}, "
        f"{distance}, {len(queries)} queries; {env['cpu']}, "
        f"{args.threads} threads; scann {env['scann']}, "
        f"Python {env['python']}", flush=True)

  results = {}
  for name, make in builders.items():
    if args.only and name not in args.only:
      continue
    builder = make()
    t0 = time.perf_counter()
    searcher = builder.build()
    build_s = time.perf_counter() - t0

    searcher.set_num_threads(args.threads)
    best, total, passes = float("inf"), 0.0, 0
    neighbors = None
    while passes < 5 or total < 2.0:
      t0 = time.perf_counter()
      neighbors, _ = searcher.search_batched_parallel(queries)
      dt = time.perf_counter() - t0
      best, total, passes = min(best, dt), total + dt, passes + 1

    n_lat = min(args.latency_queries, len(queries))
    t0 = time.perf_counter()
    for q in queries[:n_lat]:
      searcher.search(q)
    latency_ms = 1000 * (time.perf_counter() - t0) / n_lat

    r = dict(build_s=build_s, recall=recall(np.asarray(neighbors), truth),
             qps=len(queries) / best, latency_ms=latency_ms)
    results[name] = r
    print(f"{args.label:12} {name:22} build {r['build_s']:7.2f} s  "
          f"recall@10 {r['recall']:.4f}  {r['qps']:9.0f} QPS  "
          f"{r['latency_ms']:7.3f} ms/query", flush=True)
    del searcher

  if args.json:
    with open(args.json, "w") as f:
      json.dump({"environment": env, "results": results}, f, indent=1)


if __name__ == "__main__":
  main()
