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

"""scann-core from Python: build, search, measure recall, save and reload.

Builds a tree + asymmetric hashing (AH) + reorder index over synthetic
vectors, searches it one query at a time and in batches, checks its recall
against exact brute force, and saves / reloads it with docids.

Run with scann-core installed (`pip install scann-core`), or from a CMake
build tree:
  PYTHONPATH=<build>/python python examples/python/quickstart.py
"""

import os
import tempfile
import time

import numpy as np
import scann

# Synthetic data: 5000 unit vectors of dimension 64, clustered around 50
# random centers (real embeddings are clustered too; uniform noise isn't a
# realistic test of an approximate index). Queries come from the same
# distribution.
N, DIM, NUM_QUERIES, K = 5000, 64, 200, 10
rng = np.random.default_rng(0)


def clustered(n):
  centers = np.random.default_rng(42).standard_normal((50, DIM))
  x = centers[rng.integers(0, 50, n)] + 0.3 * rng.standard_normal((n, DIM))
  # L2-normalize, so the dot product is the cosine similarity. ScaNN needs
  # float32, row-major (C-contiguous) arrays.
  x /= np.linalg.norm(x, axis=1, keepdims=True)
  return x.astype(np.float32)


dataset = clustered(N)
queries = clustered(NUM_QUERIES)

# The full pipeline (see docs/api_reference.md for every option):
#   tree():      k-means partitioning into 70 leaves (about sqrt(N)); each
#                query only scores the points in its 10 nearest leaves.
#   score_ah():  scores those points with 4-bit product-quantization codes,
#                2 dimensions per code; the anisotropic threshold favours
#                the dot-product ranking over plain reconstruction error.
#   reorder():   rescores the best 100 candidates with the exact float32
#                vectors, and returns the top K.
# random_init=False picks k-means++ initialization, which (unlike the
# default random init) trains the same tree on every run.
start = time.perf_counter()
searcher = (
    scann.scann_ops_pybind.builder(dataset, K, "dot_product")
    .tree(num_leaves=70, num_leaves_to_search=10, random_init=False)
    .score_ah(2, anisotropic_quantization_threshold=0.2)
    .reorder(100)
    .build())
print(f"built an index of {searcher.size()} x {DIM} "
      f"in {time.perf_counter() - start:.2f} s")

# One query: indices into `dataset` and their scores (dot products here;
# larger is closer). For "squared_l2" indexes smaller is closer.
indices, distances = searcher.search(queries[0])
print("query 0 top 3:", indices[:3], np.round(distances[:3], 3))

# Search-time overrides: more neighbours, more leaves (slower, better).
indices, _ = searcher.search(queries[0], final_num_neighbors=20,
                             leaves_to_search=30)
assert len(indices) == 20

# A batch: [NUM_QUERIES, K] arrays. search_batched_parallel splits the
# batch across the searcher's thread pool; the results are the same.
found, scores = searcher.search_batched(queries)
assert found.shape == (NUM_QUERIES, K)
found_parallel, _ = searcher.search_batched_parallel(queries, batch_size=32)
np.testing.assert_array_equal(found, found_parallel)

# Recall@K against exact search: the fraction of the true K nearest
# neighbours that the index returns.
true_neighbors = np.argsort(-(queries @ dataset.T), axis=1)[:, :K]
recall = np.mean([len(set(f) & set(t)) / K
                  for f, t in zip(found, true_neighbors)])
print(f"recall@{K}: {recall:.3f}")
assert recall > 0.9, recall

# Docids: build(docids=...) makes searches return your identifiers
# instead of row numbers (and enables upsert/delete, see updating.py).
docids = [f"doc-{i}" for i in range(N)]
searcher = (
    scann.scann_ops_pybind.builder(dataset, K, "dot_product")
    .tree(num_leaves=70, num_leaves_to_search=10, random_init=False)
    .score_ah(2, anisotropic_quantization_threshold=0.2)
    .reorder(100)
    .build(docids=docids))
found_docids, scores = searcher.search_batched(queries)  # lists of docids
print("query 0 top 3 docids:", found_docids[0][:3])

with tempfile.TemporaryDirectory() as tmp:
  # serialize() needs an existing directory. relative_path=True records
  # the asset paths relative to it, so the directory can be moved or copied
  # to another machine as a whole.
  index_dir = os.path.join(tmp, "index")
  os.makedirs(index_dir)
  searcher.serialize(index_dir, relative_path=True)
  print("saved:", ", ".join(sorted(os.listdir(index_dir))))

  moved = os.path.join(tmp, "moved-index")
  os.rename(index_dir, moved)
  # Loading needs no dataset and no training. The docids come back from
  # scann_docids.pkl, a pickle: only load index directories you trust.
  loaded = scann.scann_ops_pybind.load_searcher(moved)

  loaded_docids, loaded_scores = loaded.search_batched(queries)
  assert loaded_docids == found_docids
  np.testing.assert_array_equal(loaded_scores, scores)
  print(f"reloaded {loaded.size()} points from a moved directory: "
        "identical results")
