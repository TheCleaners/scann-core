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

"""A retrieval model with the index inside it, as one SavedModel.

Uses scann_tf_ops, scann-core's optional TensorFlow op (source-only: build
with -DSCANN_BUILD_TF_OP=ON; see docs/tensorflow.md). The searcher is
TensorFlow state, so a model that searches saves as a SavedModel,
index included, and loads and searches in a process that has never seen
the index, without Python code of ours in the graph:

  1. build a searcher with the upstream-style builder,
  2. search in a tf.function (unknown batch size, static k),
  3. save a module with a serving signature,
  4. load it in a fresh process and search: the same results.

Exits with status 77 (reported as skipped by ctest) without TensorFlow.
Run from a CMake build tree built with the op:
  PYTHONPATH=<build>/python python examples/python/tensorflow_op.py
"""

import os
import subprocess
import sys
import tempfile

try:
  import tensorflow as tf
except ImportError:
  print("TensorFlow is not installed; skipping (pip install tensorflow-cpu)")
  sys.exit(77)

import numpy as np
import scann_tf_ops

DIM, K = 32, 10


class Retrieval(tf.Module):
  """Query embeddings in, top-k item indices and scores out."""

  def __init__(self, searcher):
    super().__init__()
    # The index lives in the searcher's variables; serialize_to_module()
    # returns the tf.Module holding them.
    self.index = searcher.serialize_to_module()

  @tf.function(input_signature=[
      tf.TensorSpec([None, DIM], tf.float32, name="queries")
  ])
  def retrieve(self, queries):
    searcher = scann_tf_ops.searcher_from_module(self.index)
    indices, scores = searcher.search_batched_parallel(queries, K)
    return {"indices": indices, "scores": scores}


def build_and_save(export_dir):
  rng = np.random.default_rng(0)
  items = rng.standard_normal((20_000, DIM)).astype(np.float32)
  items /= np.linalg.norm(items, axis=1, keepdims=True)

  # 1. The same builder as scann_ops_pybind and upstream's scann_ops.
  searcher = (scann_tf_ops.builder(items, K, "dot_product")
              .tree(num_leaves=150, num_leaves_to_search=15,
                    random_init=False)
              .score_ah(2, anisotropic_quantization_threshold=0.2)
              .reorder(100)
              .build())

  # 2. Searches are graph ops: eager, or in a tf.function.
  model = Retrieval(searcher)
  queries = rng.standard_normal((5, DIM)).astype(np.float32)
  result = model.retrieve(queries)
  print("static shape:",
        model.retrieve.get_concrete_function().structured_outputs["indices"]
        .shape)  # (None, 10)
  print("top 3 for query 0:", result["indices"][0, :3].numpy())

  # The index is searched exactly: the item itself is its own best match.
  own = model.retrieve(items[:4])["indices"][:, 0].numpy()
  assert (own == np.arange(4)).all(), own

  # 3. One SavedModel: graph, signature and index.
  tf.saved_model.save(model, export_dir,
                      signatures={"serving_default": model.retrieve})
  np.save(os.path.join(export_dir, "queries.npy"), queries)
  np.save(os.path.join(export_dir, "expected.npy"), result["indices"].numpy())
  print("saved to", export_dir)


def load_and_search(export_dir):
  # 4. A fresh process: importing scann_tf_ops registers the op, which the
  # loaded graph needs; the index comes from the SavedModel's variables.
  loaded = tf.saved_model.load(export_dir)
  serve = loaded.signatures["serving_default"]
  queries = np.load(os.path.join(export_dir, "queries.npy"))
  out = serve(queries=tf.constant(queries))
  print("signature outputs:", {k: v.shape for k, v in out.items()})
  expected = np.load(os.path.join(export_dir, "expected.npy"))
  assert (out["indices"].numpy() == expected).all()
  # The searcher was built once, on the first call, from the variables.
  print("searchers built in this process:", scann_tf_ops.stats()["builds"])
  print("reloaded model: same results")


def main():
  if len(sys.argv) == 3 and sys.argv[1] == "--load":
    load_and_search(sys.argv[2])
    return
  with tempfile.TemporaryDirectory() as tmp:
    export_dir = os.path.join(tmp, "retrieval")
    build_and_save(export_dir)
    sys.stdout.flush()
    subprocess.run([sys.executable, __file__, "--load", export_dir],
                   check=True)


if __name__ == "__main__":
  main()
