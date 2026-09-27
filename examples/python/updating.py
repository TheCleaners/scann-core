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

"""Keeping a scann-core index up to date from Python.

Inserts, updates and deletes points by docid, watches the health stats,
retrains with rebalance(), and saves over an existing index directory.

Run with scann-core installed (`pip install scann-core`), or from a CMake
build tree:
  PYTHONPATH=<build>/python python examples/python/updating.py
"""

import os
import tempfile

import numpy as np
import scann

DIM, K = 32, 10
NUM_LEAVES = 60
rng = np.random.default_rng(0)


def clustered(n):
  """n unit vectors around 40 fixed random centers."""
  centers = np.random.default_rng(42).standard_normal((40, DIM))
  x = centers[rng.integers(0, 40, n)] + 0.3 * rng.standard_normal((n, DIM))
  return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def top1(searcher, vectors):
  """Each vector's nearest docid, searching (almost) exhaustively.

  Every leaf is searched and 1000 AH candidates are reordered with the exact
  vectors, so a stored vector always finds itself: a simple check that the
  index holds what we think it holds. (With the defaults, 10 leaves and 100
  candidates, a few points in these dense clusters are missed.)
  """
  found, _ = searcher.search_batched(vectors, final_num_neighbors=1,
                                     pre_reorder_num_neighbors=1000,
                                     leaves_to_search=NUM_LEAVES)
  return [row[0] for row in found]


def health(searcher):
  # initialize_health_stats() computes the statistics from scratch; after
  # that, get_health_stats() reports them as they are kept up to date by
  # upserts and deletes. Imbalance: how uneven the leaf sizes are.
  # Quantization error: how far the AH codes are from the vectors.
  s = searcher.get_health_stats()
  return (f"imbalance {s['partition_avg_relative_positive_imbalance']:.3f}, "
          f"quantization error {s['avg_quantization_error']:.4f}")


# Mutation needs docids: a list of unique strings, one per row. The
# searcher keeps its own copy (searcher.docids) and maps between docids
# and the rows of the index.
initial = clustered(3000)
docids = [f"doc-{i}" for i in range(len(initial))]
searcher = (
    scann.scann_ops_pybind.builder(initial, K, "dot_product")
    .tree(num_leaves=NUM_LEAVES, num_leaves_to_search=10, random_init=False)
    .score_ah(2, anisotropic_quantization_threshold=0.2)
    .reorder(100)
    .build(docids=docids))
searcher.initialize_health_stats()
print(f"built {searcher.size()} points; {health(searcher)}")

# --- Insert --------------------------------------------------------------
# upsert(docids, vectors) adds the docids it doesn't know. batch_size > 1
# assigns the batch to leaves on the searcher's thread pool. A large upsert
# can trigger ScaNN's incremental maintenance, and if that decides the tree
# needs it, a full retrain: an upsert isn't always cheap.
new = clustered(2000)
new_docids = [f"new-{i}" for i in range(len(new))]
searcher.upsert(new_docids, new, batch_size=256)
assert searcher.size() == 5000
assert top1(searcher, new) == new_docids
print(f"inserted {len(new)} points; {health(searcher)}")

# --- Update --------------------------------------------------------------
# An upsert with a docid the index already has replaces that point's vector.
old_vector = initial[0]
moved_to = clustered(1)[0]
searcher.upsert("doc-0", moved_to)  # one docid and a 1-D vector work too
assert searcher.size() == 5000
assert top1(searcher, moved_to[None]) == ["doc-0"]
assert top1(searcher, old_vector[None]) != ["doc-0"]
print("updated doc-0: found at its new vector, not at its old one")

# Each docid may appear only once per upsert call. A repeated one is a
# ValueError, raised before anything changes (upstream ScaNN added a
# repeated new docid twice but mapped it once).
try:
  searcher.upsert(["dup", "dup"], clustered(2))
except ValueError as e:
  print("repeated docid rejected:", e)
else:
  raise AssertionError("expected a ValueError")
assert searcher.size() == 5000 and "dup" not in searcher.docids

# --- Delete --------------------------------------------------------------
# delete() removes by docid. Internally the last row moves into each freed
# slot, so row numbers change; the docid mapping follows. Unknown docids
# are a KeyError, again before anything changes.
deleted = [f"doc-{i}" for i in range(1, 1001)]
searcher.delete(deleted)
assert searcher.size() == 4000
found, _ = searcher.search_batched(initial[1:1001])
assert not set(deleted) & {d for row in found for d in row}
try:
  searcher.delete("no-such-doc")
except KeyError:
  pass
else:
  raise AssertionError("expected a KeyError")
print(f"deleted {len(deleted)} points; {health(searcher)}")

# --- Rebalance -----------------------------------------------------------
# The tree and the AH codebooks were trained on the first 3000 points. After
# many inserts and deletes the leaves can become uneven and the codebooks
# stale; rebalance() retrains both on the current data (a full retrain,
# as slow as a build), and recomputes the health stats. Search results keep
# their docids.
searcher.rebalance()
print(f"rebalanced; {health(searcher)}")
assert top1(searcher, new[:100]) == new_docids[:100]

# --- Save over an existing index -----------------------------------------
with tempfile.TemporaryDirectory() as index_dir:
  # A first save...
  searcher.serialize(index_dir, relative_path=True)
  # ...then more updates, and a save into the same directory. That replaces
  # the saved index as a unit: serialize() writes and fsyncs every file in a
  # staging subdirectory first, then renames the files into place with the
  # manifest (scann_assets.pbtxt) last, removing files the new index doesn't
  # have. If the process dies midway, the directory holds the complete old
  # index, the complete new one, or refuses to load with a clear error;
  # never a mix (see docs/api_reference.md, "Persistence"). Don't load it
  # while another process is saving into it.
  searcher.upsert("late-arrival", clustered(1)[0])
  searcher.serialize(index_dir, relative_path=True)
  assert not any(f.startswith(".scann_staging") for f in os.listdir(index_dir))

  loaded = scann.scann_ops_pybind.load_searcher(index_dir)
  assert loaded.size() == searcher.size() == 4001
  assert loaded.docids == searcher.docids
  queries = clustered(50)
  assert loaded.search_batched(queries)[0] == searcher.search_batched(queries)[0]
  print(f"re-saved and reloaded {loaded.size()} points: identical results")

  # The loaded index can be updated in turn.
  loaded.delete("late-arrival")
  assert loaded.size() == 4000
