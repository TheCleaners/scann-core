# Copyright 2026 ebenali and TheCleaners.
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

"""Part 5: saving, loading, and serving: threads, batching, latency."""

import os
import tempfile

import numpy as np
import scann

from tutorial_data import Timer, load_glove, recall

dataset, queries, true_neighbors = load_glove()

with Timer() as t:
  searcher = (scann.scann_ops_pybind.builder(dataset, 10, "dot_product")
              .tree(num_leaves=2000, num_leaves_to_search=100,
                    training_sample_size=250000)
              .score_ah(2, anisotropic_quantization_threshold=0.2)
              .reorder(100)
              .build())
print(f"built in {t.seconds:.1f} s")

# --- Save and load. ---------------------------------------------------------
index_dir = tempfile.mkdtemp(prefix="glove-index-")
with Timer() as t:
  searcher.serialize(index_dir)
print(f"serialized in {t.seconds:.1f} s to {index_dir}:")
for name in sorted(os.listdir(index_dir)):
  size = os.path.getsize(os.path.join(index_dir, name))
  print(f"  {name:32} {size / 2**20:8.1f} MiB")

with Timer() as t:
  loaded = scann.scann_ops_pybind.load_searcher(index_dir)
print(f"loaded in {t.seconds:.1f} s")
a, _ = searcher.search_batched(queries)
b, _ = loaded.search_batched(queries)
print("loaded index gives identical results:", np.array_equal(a, b))

# --- Smaller indexes: quantize the reordering data. -------------------------
# dataset.npy is the float32 copy of every vector that reorder() rescores
# with. Reordering with int8 or bfloat16 copies shrinks it 4x / 2x.
print("\nreorder precision vs. index size")
for name, quantize in [("float32", scann.ReorderType.FLOAT32),
                       ("bfloat16", scann.ReorderType.BFLOAT16),
                       ("int8", scann.ReorderType.INT8)]:
  s = (scann.scann_ops_pybind.builder(dataset, 10, "dot_product")
       .tree(num_leaves=2000, num_leaves_to_search=100,
             training_sample_size=250000)
       .score_ah(2, anisotropic_quantization_threshold=0.2)
       .reorder(100, quantize=quantize)
       .build())
  d = tempfile.mkdtemp(prefix=f"glove-index-{name}-")
  s.serialize(d)
  size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
  s.set_num_threads(os.cpu_count())
  found, _ = s.search_batched_parallel(queries)
  print(f"  reorder {name:8}: {size / 2**20:6.1f} MiB on disk, "
        f"recall@10 {recall(found, true_neighbors):.4f}")

# --- Serving: batch size vs. throughput and latency. ------------------------
print("\nbatch size vs. throughput / latency (search_batched_parallel, "
      f"{os.cpu_count()} threads)")
loaded.set_num_threads(os.cpu_count())
for batch in (1, 8, 64, 512, 4096):
  n = min(len(queries), max(batch * 20, 2000))  # enough batches to time
  with Timer() as t:
    for start in range(0, n, batch):
      loaded.search_batched_parallel(queries[start:start + batch])
  print(f"  batch {batch:5}: {n / t.seconds:8.0f} QPS, "
        f"{1000 * t.seconds / (n / batch):7.3f} ms per batch")

with Timer() as t:
  for q in queries[:2000]:
    loaded.search(q)
print(f"  search() one query at a time: {2000 / t.seconds:8.0f} QPS, "
      f"{1000 * t.seconds / 2000:.3f} ms per query")

# --- Serving concurrent requests: many threads calling search(). ------------
# search() releases the GIL while it runs, so request-handler threads can
# search one shared index concurrently.
from concurrent.futures import ThreadPoolExecutor

print("\nconcurrent search() calls from a Python thread pool")
for workers in (1, 8, 32, 64):
  with ThreadPoolExecutor(workers) as pool, Timer() as t:
    list(pool.map(loaded.search, queries))
  print(f"  {workers:2} threads: {len(queries) / t.seconds:8.0f} QPS")
