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

"""Serving one scann-core index from many threads.

Two ways to use several cores for search:

1. Batch the queries and call search_batched_parallel(), which splits the
   batch across the searcher's own C++ thread pool.
2. Share one searcher between Python threads, each calling search() for
   its own requests (a web server's worker threads, say).

Both are safe; this script runs both and compares their throughput. On
regular CPython the searches release the GIL, but the Python code around
each call still runs one thread at a time. On free-threaded Python (3.14t
and later) scann-core runs without the GIL, and approach 2 scales further.

Run with scann-core installed (`pip install scann-core`), or from a CMake
build tree:
  PYTHONPATH=<build>/python python examples/python/serving_threads.py
"""

import os
import platform
import sys
import sysconfig
import threading
import time

import numpy as np
import scann

# Which kind of interpreter is this? sys._is_gil_enabled() exists from
# Python 3.13. On a free-threaded build, importing an extension module that
# doesn't declare itself GIL-free would turn the GIL back on; scann-core's
# module declares it, so after `import scann` the GIL stays off.
free_threaded_build = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
gil_enabled = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
print(f"Python {platform.python_version()}"
      f"{' (free-threaded build)' if free_threaded_build else ''}, "
      f"GIL {'enabled' if gil_enabled else 'disabled'}")
if free_threaded_build:
  assert not gil_enabled, "importing scann turned the GIL back on"

N, DIM, NUM_QUERIES, K = 20000, 64, 20000, 10
THREADS = min(8, os.cpu_count() or 1)
rng = np.random.default_rng(0)
centers = rng.standard_normal((100, DIM))


def clustered(n):
  x = centers[rng.integers(0, 100, n)] + 0.3 * rng.standard_normal((n, DIM))
  return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


searcher = (
    scann.scann_ops_pybind.builder(clustered(N), K, "dot_product")
    .tree(num_leaves=140, num_leaves_to_search=15, random_init=False)
    .score_ah(2, anisotropic_quantization_threshold=0.2)
    .reorder(100)
    .build())
queries = clustered(NUM_QUERIES)
print(f"index: {N} x {DIM}; {NUM_QUERIES} queries; {THREADS} threads")

# The reference answers, one query at a time on this thread.
start = time.perf_counter()
expected, _ = searcher.search_batched(queries)
single = NUM_QUERIES / (time.perf_counter() - start)
print(f"  {'search_batched, 1 thread':36}{single:9.0f} QPS")

# --- 1. One batch, split across the searcher's thread pool ---------------
# set_num_threads() sizes that pool (the default is one thread less than
# the number of CPUs). batch_size is how many queries each task takes.
searcher.set_num_threads(THREADS)
start = time.perf_counter()
found, _ = searcher.search_batched_parallel(queries, batch_size=64)
batched = NUM_QUERIES / (time.perf_counter() - start)
np.testing.assert_array_equal(found, expected)
print(f"  {'search_batched_parallel':36}{batched:9.0f} QPS")

# --- 2. Python threads sharing the searcher, one query per call ----------
# The searcher may be shared freely: searches run in parallel, and upsert,
# delete, rebalance and serialize (not used here) wait for running searches
# and block new ones while they change the index.
results = [None] * NUM_QUERIES


def serve(first):
  for i in range(first, NUM_QUERIES, THREADS):
    results[i], _ = searcher.search(queries[i])


threads = [threading.Thread(target=serve, args=(t,)) for t in range(THREADS)]
start = time.perf_counter()
for t in threads:
  t.start()
for t in threads:
  t.join()
concurrent = NUM_QUERIES / (time.perf_counter() - start)
np.testing.assert_array_equal(np.array(results), expected)
print(f"  {f'search() from {THREADS} Python threads':36}{concurrent:9.0f} QPS")

# Which to pick: batching gives the best throughput when queries arrive in
# groups (offline jobs, a request carrying many queries, or a server that
# collects requests for a millisecond or two). Per-request search() calls
# from threads give the lowest latency for single queries; with the GIL
# they level off at a few threads' worth of throughput, without it they
# keep scaling. docs/tutorial/05-saving-and-serving.md has measurements on
# a 64-thread machine.
print("mode:", "free-threaded, GIL disabled" if not gil_enabled
      else "GIL enabled (searches release it while they run)")
