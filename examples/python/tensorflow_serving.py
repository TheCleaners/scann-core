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

"""Serving retrieval next to a TensorFlow model, end to end.

A two-tower retrieval model splits cleanly for serving: the query tower is
a TensorFlow model, exported as a SavedModel, and the item embeddings are
indexed with scann-core and saved beside it. The serving side loads both
and answers "features in, docids out". Neither step needs a TensorFlow op,
and the index directory also loads from Rust and C++ services. See
docs/tensorflow.md, "Serving: query tower + scann-core".

Needs TensorFlow (`pip install 'scann-core[tf]'`); exits with status 77
(reported as skipped by ctest) without it. Run with scann-core installed,
or from a CMake build tree:
  PYTHONPATH=<build>/python python examples/python/tensorflow_serving.py
"""

import os
import shutil
import sys
import tempfile

try:
  import tensorflow as tf
except ImportError:
  print("TensorFlow is not installed; skipping (pip install 'scann-core[tf]')")
  sys.exit(77)

import keras
import numpy as np
import scann

NUM_FEATURES, DIM, NUM_ITEMS, K = 16, 32, 5000, 10


def export(export_dir):
  """Training side: write the query tower and the item index to export_dir."""
  rng = np.random.default_rng(0)
  # The query tower's initial weights too: with unseeded ones, the recall
  # checked below varied from run to run (0.61 to 0.99).
  keras.utils.set_random_seed(0)

  # The query tower: user features -> a unit-norm embedding. Untrained here,
  # for brevity; a real one comes out of two-tower training.
  query_model = keras.Sequential([
      keras.Input(shape=(NUM_FEATURES,)),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(DIM),
      keras.layers.UnitNormalization(),
  ])

  # The item embeddings: in a real model, the candidate tower applied to
  # every item, offline. Here, clusters of similar items around 50 random
  # centers. Unit-norm, so dot product = cosine similarity.
  centers = rng.standard_normal((50, DIM))
  items = (centers[rng.integers(0, 50, NUM_ITEMS)] +
           0.3 * rng.standard_normal((NUM_ITEMS, DIM))).astype(np.float32)
  items /= np.linalg.norm(items, axis=1, keepdims=True)
  docids = [f"item-{i}" for i in range(NUM_ITEMS)]

  # A SavedModel with one endpoint, "serve". It can also be loaded by
  # TensorFlow Serving; the index never needs TensorFlow.
  query_model.export(os.path.join(export_dir, "query_model"), verbose=False)

  index = (scann.scann_ops_pybind.builder(items, K, "dot_product")
           .tree(num_leaves=70, num_leaves_to_search=10, random_init=False)
           .score_ah(2, anisotropic_quantization_threshold=0.2)
           .reorder(100)
           .build(docids=docids))
  index_dir = os.path.join(export_dir, "index")
  os.makedirs(index_dir)
  # relative_path=True: the directory can be moved and shipped as a whole.
  index.serialize(index_dir, relative_path=True)
  # Keep what the tests below compare against (not part of the pattern).
  return query_model, items


class Recommender:
  """Serving side: loads a model export and answers queries with docids."""

  def __init__(self, export_dir):
    self.query_model = tf.saved_model.load(os.path.join(export_dir,
                                                        "query_model"))
    # The docids come back from scann_docids.pkl, which only Python reads.
    # It is a pickle: only load index directories you trust. A Rust or C++
    # service gets row indices and ships its own id list instead.
    self.index = scann.scann_ops_pybind.load_searcher(
        os.path.join(export_dir, "index"))

  def recommend(self, features, k=K):
    """features: [batch, NUM_FEATURES] -> (docids, scores), k per row."""
    embeddings = self.query_model.serve(features).numpy()
    # search_batched_parallel spreads a large batch over the index's thread
    # pool; for one request with a handful of queries search_batched is
    # just as fast. See tutorial part 5 for batch size vs. throughput.
    return self.index.search_batched_parallel(embeddings,
                                              final_num_neighbors=k)


with tempfile.TemporaryDirectory() as tmp:
  build_dir = os.path.join(tmp, "build")
  query_model, items = export(build_dir)
  print("exported:", sorted(os.listdir(build_dir)))

  # "Deploy": move the whole export somewhere else.
  deployed = os.path.join(tmp, "serving", "v1")
  shutil.move(build_dir, deployed)

  service = Recommender(deployed)
  features = np.random.default_rng(1).random((100, NUM_FEATURES),
                                              dtype=np.float32)
  docids, scores = service.recommend(features)
  print("request 0 ->", docids[0][:3], np.round(scores[0][:3], 3))
  assert len(docids) == 100 and all(len(row) == K for row in docids)

  # The SavedModel computes the same embeddings as the Keras model...
  embeddings = query_model.predict(features, verbose=0)
  np.testing.assert_allclose(service.query_model.serve(features).numpy(),
                             embeddings, rtol=1e-5, atol=1e-6)
  # ...and the served docids are close to exact search over the items.
  exact = np.argsort(-(embeddings @ items.T), axis=1)[:, :K]
  recall = np.mean([len({f"item-{i}" for i in t} & set(d)) / K
                    for t, d in zip(exact, docids)])
  print(f"recall@{K} of the served results against exact search: {recall:.3f}")
  assert recall > 0.9, recall
