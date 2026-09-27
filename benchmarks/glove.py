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

"""GloVe-100 benchmark: index build time, recall@10, throughput, latency.

The numbers in docs/benchmarks.md come from this script. It runs against
whichever `scann` package the interpreter imports, so the same script
measures scann-core and the upstream wheel:

  PYTHONPATH=build/python python benchmarks/glove.py --label scann-core
  /path/to/wheel-venv/bin/python benchmarks/glove.py --label upstream

For each configuration it reports:
  build    wall time of builder.build() (partitioner training, hashing, ...)
  recall   recall@10 against ann-benchmarks' exact neighbours
  QPS      search_batched_parallel over all 10,000 queries on every core,
           best of at least 5 passes and 2 seconds
  latency  search() one query at a time on one thread, mean over the first
           --latency-queries queries

The dataset is downloaded on first use (~485 MB; see
docs/tutorial/code/tutorial_data.py). Tree training uses k-means++
(random_init=False), so every build of a given implementation trains the
same partitioner and the recall figures are reproducible.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import scann

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "docs", "tutorial", "code"))
from tutorial_data import load_glove, recall  # pylint: disable=g-import-not-at-top

# scann-core has scann.ReorderType; the upstream wheel takes a bool.
INT8 = scann.ReorderType.INT8 if hasattr(scann, "ReorderType") else True


def configs(dataset):
  b = lambda: scann.scann_ops_pybind.builder(dataset, 10, "dot_product")
  tree = lambda: b().tree(2000, 100, training_sample_size=250000,
                          random_init=False)
  return {
      "brute_force_f32": lambda: b().score_brute_force(),
      "brute_force_int8": lambda: b().score_brute_force(quantize=INT8),
      "tree_ah_reorder": lambda: tree().score_ah(
          2, anisotropic_quantization_threshold=0.2).reorder(100),
      "tree_ah_reorder_int8": lambda: tree().score_ah(
          2, anisotropic_quantization_threshold=0.2).reorder(
              100, quantize=INT8),
      "tree_ah_reorder_noaq": lambda: tree().score_ah(2).reorder(100),
  }


def main():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--label", default="scann")
  p.add_argument("--json", help="also write the results here")
  p.add_argument("--threads", type=int, default=os.cpu_count())
  p.add_argument("--latency-queries", type=int, default=1000)
  p.add_argument("--only", nargs="*", help="run only these configs")
  args = p.parse_args()

  dataset, queries, truth = load_glove()
  results = {}
  for name, make in configs(dataset).items():
    if args.only and name not in args.only:
      continue
    builder = make()
    t0 = time.perf_counter()
    searcher = builder.build()
    build_s = time.perf_counter() - t0

    searcher.set_num_threads(args.threads)
    best, total, passes = float("inf"), 0.0, 0
    while passes < 5 or total < 2.0:
      t0 = time.perf_counter()
      neighbors, _ = searcher.search_batched_parallel(queries)
      dt = time.perf_counter() - t0
      best, total, passes = min(best, dt), total + dt, passes + 1

    t0 = time.perf_counter()
    for q in queries[:args.latency_queries]:
      searcher.search(q)
    latency_ms = 1000 * (time.perf_counter() - t0) / args.latency_queries

    r = dict(build_s=build_s, recall=recall(np.asarray(neighbors), truth),
             qps=len(queries) / best, latency_ms=latency_ms)
    results[name] = r
    print(f"{args.label:12} {name:22} build {r['build_s']:7.2f} s  "
          f"recall@10 {r['recall']:.4f}  {r['qps']:9.0f} QPS  "
          f"{r['latency_ms']:7.3f} ms/query", flush=True)
    del searcher

  if args.json:
    with open(args.json, "w") as f:
      json.dump({"label": args.label, "threads": args.threads,
                 "results": results}, f, indent=1)


if __name__ == "__main__":
  main()
