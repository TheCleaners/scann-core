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

"""TensorFlow wrapper around scann-core's searcher (no op; see scann_tf_ops).

Adapted from upstream's scann/scann_ops/py/scann_ops.py, with the same
builder() / create_searcher() / ScannSearcher.search*() API, but backed by
the pybind searcher (scann.scann_ops_pybind) through tf.numpy_function
instead of a compiled TensorFlow op. It works in eager mode and inside
tf.function and tf.data, but the searcher can't be saved in a SavedModel:
serialize_to_module() and searcher_from_module() raise NotImplementedError.
See docs/tensorflow.md.

  from scann import tf as scann_tf   # upstream: from scann.scann_ops.py import scann_ops

  searcher = scann_tf.builder(db, 10, "dot_product").score_brute_force().build()
  indices, distances = searcher.search_batched(queries)  # int32, float32

Importing `scann` doesn't import TensorFlow; importing this module does.
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

from scann.scann_ops.py import scann_ops_pybind

__all__ = [
    "BatchedSearchResult", "ScannSearcher", "SearchResult", "builder",
    "create_searcher", "load_searcher", "searcher_from_module"
]

# The results are named like the outputs of upstream's ops (ScannSearch,
# ScannSearchBatched), so result.index / result.indices etc. keep working.
SearchResult = collections.namedtuple("ScannSearch", ["index", "distance"])
BatchedSearchResult = collections.namedtuple("ScannSearchBatched",
                                             ["indices", "distances"])

_NO_SAVEDMODEL = (
    "{} is not supported by scann.tf: the tf.numpy_function it searches "
    "through can't be saved in a SavedModel. Either save the index with "
    "serialize() and load it with load_searcher() next to the model, or "
    "build scann-core's optional TensorFlow op (scann_tf_ops, "
    "-DSCANN_BUILD_TF_OP=ON), whose searchers save in SavedModels; see "
    "docs/tensorflow.md in scann-core.")

_INT32_MAX = np.iinfo(np.int32).max


def _to_numpy(db):
  """A dataset (tf.Tensor, numpy array or nested lists) as a float32 array."""
  if isinstance(db, tf.Tensor):
    if not hasattr(db, "numpy"):
      raise ValueError(
          "scann.tf builds searchers eagerly: pass a numpy array or an eager "
          "tensor, not a symbolic one (inside tf.function or a v1 graph).")
    db = db.numpy()
  return np.ascontiguousarray(db, dtype=np.float32)


def _as_int(value):
  if isinstance(value, tf.Tensor):
    return int(value.numpy())
  return int(value)


def _static_int(value):
  """value as a Python int if it is one (not a tensor), else None."""
  if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
    return int(value)
  return None


def _param(value):
  return tf.convert_to_tensor(-1 if value is None else value, dtype=tf.int32)


def _queries(q, rank, method):
  q = tf.convert_to_tensor(q)
  if q.dtype != tf.float32:
    q = tf.cast(q, tf.float32)
  if q.shape.rank is not None and q.shape.rank != rank:
    raise ValueError(f"{method}() expects a {rank}-dimensional query "
                     f"tensor, got shape {q.shape}.")
  return q


def _to_int32(idx):
  if idx.size and idx.max() > _INT32_MAX:
    raise OverflowError(
        "A result index doesn't fit in int32 (the index has more than 2^31 - "
        "1 points); use the pybind searcher (.searcher) directly.")
  return idx.astype(np.int32)


class ScannSearcher(object):
  """A pybind ScannSearcher whose search methods take and return tensors.

  The search methods return (indices, distances) namedtuples of int32 and
  float32 tensors, as upstream's TensorFlow ops did. Indices are positions
  in the index, not docids, even if the underlying searcher has docids; map
  them with `self.searcher.docids` if needed (see docs/tensorflow.md).

  Searches run as tf.numpy_function(stateful=True): the index can change
  between calls (upsert, delete), so they must not be constant-folded or
  deduplicated. tf.numpy_function rather than tf.py_function: the search
  needs numpy arrays and no gradient, so the eager-tensor conversions of
  tf.py_function would be wasted.

  Attributes:
    searcher: the wrapped scann_ops_pybind.ScannSearcher. Use it for
      everything that isn't a search (upsert, delete, serialize, ...);
      changes are seen by later searches through this wrapper.
  """

  def __init__(self, searcher):
    if not isinstance(searcher, scann_ops_pybind.ScannSearcher):
      raise TypeError(
          "scann.tf.ScannSearcher wraps a scann_ops_pybind.ScannSearcher "
          f"(from builder(), create_searcher() or load_searcher()), got "
          f"{type(searcher).__name__}.")
    self.searcher = searcher

  def search(self,
             q,
             final_num_neighbors=None,
             pre_reorder_num_neighbors=None,
             leaves_to_search=None):
    """Single query; None (or -1) uses the searcher's default.

    Args:
      q: a 1-D query (tensor or array; cast to float32).
      final_num_neighbors: k, or None for the default.
      pre_reorder_num_neighbors: candidates to reorder, or None.
      leaves_to_search: tree leaves to search, or None.

    Returns:
      (index, distance), a namedtuple of int32 and float32 tensors of shape
      [n], n <= k (fewer when the index has fewer points than k).
    """
    q = _queries(q, 1, "search")
    raw = self.searcher.searcher  # ScannNumpy; does its own locking.

    def run(q, final_nn, pre_nn, leaves):
      idx, dist = raw.search(q, int(final_nn), int(pre_nn), int(leaves))
      return _to_int32(idx), dist.astype(np.float32, copy=False)

    with tf.name_scope("scann_search"):
      idx, dist = tf.numpy_function(
          run, [
              q,
              _param(final_num_neighbors),
              _param(pre_reorder_num_neighbors),
              _param(leaves_to_search)
          ], [tf.int32, tf.float32],
          stateful=True)
    idx.set_shape([None])
    dist.set_shape([None])
    return SearchResult(idx, dist)

  def _search_batched(self, q, final_num_neighbors, pre_reorder_num_neighbors,
                      leaves_to_search, parallel, batch_size, method):
    q = _queries(q, 2, method)
    raw = self.searcher.searcher
    batch_size = int(batch_size)

    def run(q, final_nn, pre_nn, leaves):
      idx, dist = raw.search_batched(q, int(final_nn), int(pre_nn),
                                     int(leaves), parallel, batch_size)
      return _to_int32(idx), dist.astype(np.float32, copy=False)

    with tf.name_scope(f"scann_{method}"):
      idx, dist = tf.numpy_function(
          run, [
              q,
              _param(final_num_neighbors),
              _param(pre_reorder_num_neighbors),
              _param(leaves_to_search)
          ], [tf.int32, tf.float32],
          stateful=True)
    # With an explicit k the results are exactly [num_queries, k] (short
    # rows padded with index 0 and NaN distance); with the default k they
    # are as wide as the longest row.
    k = _static_int(final_num_neighbors)
    shape = [q.shape[0], k if k is not None and k > 0 else None]
    idx.set_shape(shape)
    dist.set_shape(shape)
    return BatchedSearchResult(idx, dist)

  def search_batched(self,
                     q,
                     final_num_neighbors=None,
                     pre_reorder_num_neighbors=None,
                     leaves_to_search=None):
    """Queries in a 2-D tensor, searched one after another.

    Args:
      q: [num_queries, dim] queries (tensor or array; cast to float32).
      final_num_neighbors: k, or None for the default.
      pre_reorder_num_neighbors: candidates to reorder, or None.
      leaves_to_search: tree leaves to search, or None.

    Returns:
      (indices, distances), a namedtuple of int32 and float32 tensors of
      shape [num_queries, k]. Rows with fewer than k results are padded with
      index 0 and distance NaN. With the default k (None), k is the longest
      row's length.
    """
    return self._search_batched(q, final_num_neighbors,
                                pre_reorder_num_neighbors, leaves_to_search,
                                False, 0, "search_batched")

  def search_batched_parallel(self,
                              q,
                              final_num_neighbors=None,
                              pre_reorder_num_neighbors=None,
                              leaves_to_search=None,
                              batch_size=256):
    """search_batched() on the searcher's thread pool, batch_size at a time."""
    return self._search_batched(q, final_num_neighbors,
                                pre_reorder_num_neighbors, leaves_to_search,
                                True, batch_size, "search_batched_parallel")

  def serialize(self, artifacts_dir, relative_path=False):
    """Saves the index; load it with load_searcher() here or in any binding."""
    self.searcher.serialize(artifacts_dir, relative_path)

  def serialize_to_module(self):
    raise NotImplementedError(_NO_SAVEDMODEL.format("serialize_to_module()"))


def searcher_from_module(module, db=None):
  del module, db  # Unused.
  raise NotImplementedError(_NO_SAVEDMODEL.format("searcher_from_module()"))


def builder(db, num_neighbors, distance_measure):
  """Creates a ScannBuilder that returns a scann.tf.ScannSearcher on build().

  Args:
    db: the dataset that ScaNN will search over; a 2d array or eager tensor
      of floats (converted to float32) with one data point per row.
    num_neighbors: the default # neighbors the searcher will return per query.
    distance_measure: one of "squared_l2" or "dot_product".

  Returns:
    A ScannBuilder object, which builds the ScaNN config via calls such as
    tree() and score_brute_force(). Calling build() on the ScannBuilder will
    return a scann.tf.ScannSearcher with its specified config. It is
    scann_ops_pybind's builder, so autopilot() and docids work as there.
  """
  b = scann_ops_pybind.builder(_to_numpy(db), num_neighbors, distance_measure)
  build_pybind = b.builder_lambda

  def builder_lambda(db, config, training_threads, **kwargs):
    return ScannSearcher(build_pybind(db, config, training_threads, **kwargs))

  return b.set_builder_lambda(builder_lambda)


def create_searcher(db,
                    scann_config,
                    training_threads=0,
                    docids=None,
                    **unused_kwargs):
  """Creates a scann.tf.ScannSearcher from a dataset and text config proto.

  Upstream's container and shared_name arguments are accepted and ignored.
  """
  if isinstance(scann_config, tf.Tensor):
    scann_config = scann_config.numpy()
  if isinstance(scann_config, bytes):
    scann_config = scann_config.decode("utf-8")
  return ScannSearcher(
      scann_ops_pybind.create_searcher(
          _to_numpy(db),
          scann_config,
          _as_int(training_threads),
          docids=docids))


def load_searcher(artifacts_dir, assets_backcompat_shim=True):
  """Loads an index saved by serialize() (from any binding)."""
  return ScannSearcher(
      scann_ops_pybind.load_searcher(artifacts_dir, assets_backcompat_shim))
