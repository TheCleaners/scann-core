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

"""scann-core from TensorFlow code, with scann.tf.

scann.tf's searchers take and return tensors, eagerly and inside
tf.function and tf.data (see docs/tensorflow.md). This script searches
eagerly, then from a tf.function with an unknown batch size, and maps the
result indices to docids with tf.gather, also after an upsert. It runs the
same on either backend: the TensorFlow op when scann_tf_ops is built, the
Python backend (the wheel) otherwise; tensorflow_op.py saves a model.

Needs TensorFlow (`pip install 'scann-core[tf]'`); exits with status 77
(reported as skipped by ctest) without it. Run with scann-core installed,
or from a CMake build tree:
  PYTHONPATH=<build>/python python examples/python/tensorflow_wrapper.py
"""

import sys

try:
  import tensorflow as tf
except ImportError:
  print("TensorFlow is not installed; skipping (pip install 'scann-core[tf]')")
  sys.exit(77)

import numpy as np
# Upstream: from scann.scann_ops.py import scann_ops. `import scann` alone
# never imports TensorFlow; `scann.tf` does.
from scann import tf as scann_tf

# "op" (scann-core's TensorFlow op, if built) or "python" (the wheel's
# tf.numpy_function backend). The results are the same.
print("scann.tf backend:", scann_tf.backend())

N, DIM, K = 5000, 32, 10
rng = np.random.default_rng(0)
dataset = rng.standard_normal((N, DIM)).astype(np.float32)
docids = [f"item-{i}" for i in range(N)]

# The builder is the same as scann.scann_ops_pybind's; the dataset can be a
# numpy array or an eager tensor. Building happens eagerly, never inside a
# tf.function.
searcher = (
    scann_tf.builder(tf.constant(dataset), K, "dot_product")
    .tree(num_leaves=70, num_leaves_to_search=10, random_init=False)
    .score_ah(2, anisotropic_quantization_threshold=0.2)
    .reorder(100)
    .build(docids=docids))
# The index as a scann_ops_pybind searcher, for everything but searching
# (upsert, delete, config, ...). With the op backend it is a copy.
pybind = searcher.to_pybind()

# --- Eager ---------------------------------------------------------------
queries = tf.random.stateless_normal([8, DIM], seed=[1, 2])
result = searcher.search_batched(queries)  # namedtuple (indices, distances)
print("eager:", result.indices.dtype.name, result.indices.shape,
      result.distances.dtype.name)
assert result.indices.shape == (8, K) and result.indices.dtype == tf.int32

# The results are row indices, never docids, as with upstream's op. The
# pybind searcher, which has docids, returns the same neighbours as docids.
want_docids, want_distances = pybind.search_batched(queries.numpy())
assert [[docids[i] for i in row] for row in result.indices.numpy()] == want_docids
np.testing.assert_array_equal(result.distances.numpy(), want_distances)

one = searcher.search(queries[0], final_num_neighbors=3)  # (index, distance)
print("query 0 top 3:", one.index.numpy(), np.round(one.distance.numpy(), 3))

# --- In a tf.function, with docids ---------------------------------------
# A docid table in the graph maps indices to docids. It is a snapshot of
# the docids: upsert and delete move points to other rows, so a function
# like this one is made again after every change (below).
def make_retrieve(searcher, docids):
  docid_table = tf.constant(docids)

  # The batch dimension is unknown (None): one trace serves every batch size.
  @tf.function(input_signature=[tf.TensorSpec([None, DIM], tf.float32)])
  def retrieve(q):
    # search_batched_parallel spreads the batch over the searcher's threads.
    # k=5 as a Python int makes the result's second dimension static.
    indices, scores = searcher.search_batched_parallel(q,
                                                       final_num_neighbors=5)
    # Rows with fewer than k results (a tiny index) are padded with index 0
    # and a NaN score; this index has enough points for that not to happen.
    return tf.gather(docid_table, indices), scores

  return retrieve


retrieve = make_retrieve(searcher, docids)
for batch in (1, 3, 16):
  found, scores = retrieve(tf.random.stateless_normal([batch, DIM], [3, batch]))
  assert found.shape == (batch, 5) and found.dtype == tf.string
print("tf.function output:", retrieve.get_concrete_function().structured_outputs)
assert retrieve.experimental_get_tracing_count() == 1

found, _ = retrieve(queries)
np.testing.assert_array_equal(found.numpy().astype(str),
                              np.array(want_docids)[:, :5])
print("query 0 top docids:", [d.decode() for d in found[0].numpy()[:3]])

# The same function in a tf.data pipeline.
ds = (tf.data.Dataset.from_tensor_slices(queries).batch(4)
      .map(lambda q: retrieve(q)[0]))
assert [b.shape[0] for b in ds] == [4, 4]

# --- After an update -----------------------------------------------------
# Update the pybind searcher, make a searcher from it again, and a function
# with the new searcher and docids. This works with both backends: the op
# backend's searchers are snapshots of the index (a SavedModel keeps the
# one it was saved with), the Python backend's share the pybind searcher.
new_vector = 5 * queries[0].numpy()  # a large dot product with query 0
pybind.upsert("item-new", new_vector)
searcher = scann_tf.from_pybind(pybind)
retrieve = make_retrieve(searcher, pybind.docids)
found, _ = retrieve(queries[:1])
assert found[0, 0].numpy() == b"item-new", found
print("after upsert, query 0's top docid:", found[0, 0].numpy().decode())
print(f"scann.tf ({scann_tf.backend()} backend): eager and tf.function "
      "results match the pybind searcher")
