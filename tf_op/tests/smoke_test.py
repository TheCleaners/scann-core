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

"""Smoke test of the raw TensorFlow ops: load the library, search an index.

Usage: smoke_test.py <path to _scann_tf_ops.so>, with the build's python
directory on PYTHONPATH. Exits with 77 (skipped) without TensorFlow.
"""

import os
import sys
import tempfile

import numpy as np

try:
  import tensorflow as tf
except ImportError:
  print("TensorFlow isn't installed; skipped.")
  sys.exit(77)

import scann

ops = tf.load_op_library(sys.argv[1])
rng = np.random.default_rng(0)
db = rng.standard_normal((500, 16)).astype(np.float32)
queries = rng.standard_normal((7, 16)).astype(np.float32)

pybind = scann.scann_ops_pybind.builder(db, 10,
                                        "dot_product").score_brute_force().build()
with tempfile.TemporaryDirectory() as d:
  pybind.serialize(d, relative_path=True)
  names = sorted(os.listdir(d))
  contents = [open(os.path.join(d, n), "rb").read() for n in names]

args = dict(asset_names=names, asset_contents=contents,
            final_num_neighbors=5, pre_reordering_num_neighbors=-1,
            leaves_to_search=-1, index_id="smoke")
idx, dist = ops.scann_core_search_batched(queries=queries, parallel=False,
                                          batch_size=256, **args)
want_idx, want_dist = pybind.searcher.search_batched(queries, 5, -1, -1, False,
                                                     256)
assert idx.dtype == tf.int32 and dist.dtype == tf.float32
assert (idx.numpy() == want_idx).all(), (idx, want_idx)
assert (dist.numpy() == want_dist).all()
i, d = ops.scann_core_search(queries=queries[3], **args)
assert (i.numpy() == want_idx[3]).all()
live, builds = (int(x) for x in ops.scann_core_stats())
assert builds == 1, builds
print("TF", tf.__version__, "smoke test OK; searchers live/built:", live,
      builds)
