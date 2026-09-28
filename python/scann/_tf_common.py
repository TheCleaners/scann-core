# coding=utf-8
# Copyright 2026 The Google Research Authors.
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
#
# Modified in 2026 by Elias Benali (@ebenali) and TheCleaners for
# scann-core (a derived work of ScaNN, not an official Google product);
# see NOTICE.

"""What scann.tf's two backends share (private; imports TensorFlow).

scann.tf searches either through scann-core's TensorFlow op (the
scann_tf_ops package, built from source) or through tf.numpy_function
around the pybind searcher (scann._tf_python). Both take their arguments
through these helpers and return these result types, so that their results
are the same namedtuples, with the same dtypes and static shapes.
"""

import collections

import numpy as np

try:
  import tensorflow as tf
except ImportError as e:
  raise ImportError(
      "scann.tf needs TensorFlow, which could not be imported "
      f"({e}). Install it with `pip install 'scann-core[tf]'` (or install "
      "tensorflow / tensorflow-cpu yourself). The rest of the scann package "
      "works without it.") from e

# The results are named like the outputs of upstream's ops (ScannSearch,
# ScannSearchBatched), so result.index / result.indices etc. keep working.
SearchResult = collections.namedtuple("ScannSearch", ["index", "distance"])
BatchedSearchResult = collections.namedtuple("ScannSearchBatched",
                                             ["indices", "distances"])

# Where the op backend comes from, for error messages.
OP_HOWTO = (
    "scann.tf uses the TensorFlow op backend automatically when the "
    "scann_tf_ops package is importable. It isn't part of the wheel: build "
    "scann-core from source with -DSCANN_BUILD_TF_OP=ON against your "
    "TensorFlow and put build/python on PYTHONPATH (or copy "
    "build/python/scann_tf_ops next to this scann package); see "
    "docs/tensorflow.md in scann-core.")


def to_numpy(db):
  """A dataset (tf.Tensor, numpy array or nested lists) as a float32 array."""
  if isinstance(db, tf.Tensor):
    if not hasattr(db, "numpy"):
      raise ValueError(
          "scann.tf builds searchers eagerly: pass a numpy array or an eager "
          "tensor, not a symbolic one (inside tf.function or a v1 graph).")
    db = db.numpy()
  return np.ascontiguousarray(db, dtype=np.float32)


def as_int(value):
  if isinstance(value, tf.Tensor):
    return int(value.numpy())
  return int(value)


def config_text(scann_config):
  """A text config given as str, bytes or an eager string tensor."""
  if isinstance(scann_config, tf.Tensor):
    scann_config = scann_config.numpy()
  if isinstance(scann_config, bytes):
    scann_config = scann_config.decode("utf-8")
  return scann_config


def static_int(value):
  """value as a Python int if it is one (not a tensor), else None."""
  if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
    return int(value)
  return None


def param(value):
  """A search parameter as an int32 scalar tensor; None is -1 (default)."""
  return tf.convert_to_tensor(-1 if value is None else value, dtype=tf.int32)


def queries(q, rank, method):
  """The query tensor, float32, of the given rank (if its rank is known)."""
  q = tf.convert_to_tensor(q)
  if q.dtype != tf.float32:
    q = tf.cast(q, tf.float32)
  if q.shape.rank is not None and q.shape.rank != rank:
    raise ValueError(f"{method}() expects a {rank}-dimensional query "
                     f"tensor, got shape {q.shape}.")
  return q


def batched_result(q, idx, dist, final_num_neighbors):
  """The batched result, with its static shape set.

  Exactly [num_queries, k] with an explicit k (rows with fewer results are
  padded with index 0 and NaN distance); with the default k, as wide as the
  longest row. The static shape has k when it is a Python int.
  """
  k = static_int(final_num_neighbors)
  shape = [q.shape[0], k if k is not None and k > 0 else None]
  idx.set_shape(shape)
  dist.set_shape(shape)
  return BatchedSearchResult(idx, dist)


def search_result(idx, dist):
  idx.set_shape([None])
  dist.set_shape([None])
  return SearchResult(idx, dist)
