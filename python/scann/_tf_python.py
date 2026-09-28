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

"""scann.tf's Python backend: tf.numpy_function around the pybind searcher.

Use it through scann.tf, which picks it when scann-core's TensorFlow op
(scann_tf_ops) isn't installed, or when SCANN_TF_BACKEND=python; or get it
with scann.tf.get_backend("python").

Adapted from upstream's scann/scann_ops/py/scann_ops.py, with the same
builder() / create_searcher() / ScannSearcher.search*() API, but backed by
the pybind searcher (scann.scann_ops_pybind) instead of a compiled
TensorFlow op. It works in eager mode and inside tf.function and tf.data,
but the searcher can't be saved in a SavedModel: serialize_to_module() and
searcher_from_module() raise NotImplementedError. See docs/tensorflow.md.
"""

import numpy as np

from scann import _tf_common as common
from scann._tf_common import BatchedSearchResult
from scann._tf_common import SearchResult
from scann._tf_common import tf
from scann.scann_ops.py import scann_ops_pybind

__all__ = [
    "BatchedSearchResult", "ScannSearcher", "SearchResult", "builder",
    "create_searcher", "from_pybind", "load_searcher", "searcher_from_module"
]

BACKEND = "python"

_NO_SAVEDMODEL = (
    "{} is not supported by scann.tf's Python backend: the tf.numpy_function "
    "it searches through can't be saved in a SavedModel. " + common.OP_HOWTO +
    " Without the op, save the index with serialize() and load it with "
    "load_searcher() next to the model.")

_INT32_MAX = np.iinfo(np.int32).max


def _to_int32(idx):
  if idx.size and idx.max() > _INT32_MAX:
    raise ValueError(
        "A result index doesn't fit in int32 (the index has more than 2^31 - "
        "1 points); use the pybind searcher (.searcher) directly.")
  return idx.astype(np.int32)


def _search(search_fn, q, params, graph):
  """Runs search_fn(q, *params) -> (idx, dist) eagerly or in a graph.

  Search errors are tf.errors.InvalidArgumentError either way, as with the
  op backend: the pybind searcher raises ValueError or RuntimeError, which
  tf.numpy_function would report as InvalidArgumentError or UnknownError
  in a graph, and pass through unchanged eagerly.
  """

  def run(q, *params):
    try:
      idx, dist = search_fn(q, *(int(p) for p in params))
      return _to_int32(idx), dist.astype(np.float32, copy=False)
    except (ValueError, RuntimeError) as e:
      if graph:
        raise ValueError(str(e)) from e  # -> InvalidArgumentError
      raise tf.errors.InvalidArgumentError(None, None, str(e)) from e

  if not graph:
    idx, dist = run(q.numpy(), *(p.numpy() for p in params))
    return tf.convert_to_tensor(idx), tf.convert_to_tensor(dist)
  # stateful: the index can change between calls (upsert, delete), so the
  # search must not be constant-folded or deduplicated.
  idx, dist = tf.numpy_function(run, [q, *params], [tf.int32, tf.float32],
                                stateful=True)
  return idx, dist


class ScannSearcher(object):
  """A pybind ScannSearcher whose search methods take and return tensors.

  The search methods return (indices, distances) namedtuples of int32 and
  float32 tensors, as upstream's TensorFlow ops did. Indices are positions
  in the index, not docids, even if the underlying searcher has docids; map
  them with `self.searcher.docids` if needed (see docs/tensorflow.md).

  Eagerly, searches call the pybind searcher directly; in a graph
  (tf.function, tf.data), through tf.numpy_function(stateful=True).
  tf.numpy_function rather than tf.py_function: the search needs numpy
  arrays and no gradient, so the eager-tensor conversions of
  tf.py_function would be wasted.

  Attributes:
    searcher: the wrapped scann_ops_pybind.ScannSearcher (shared, not a
      copy; to_pybind() returns it too). Changes made through it (upsert,
      delete, rebalance, ...) are seen by later searches through this
      wrapper, also by tf.functions traced before. Specific to this
      backend: the op backend's searchers are snapshots.
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
    q = common.queries(q, 1, "search")
    params = [
        common.param(final_num_neighbors),
        common.param(pre_reorder_num_neighbors),
        common.param(leaves_to_search)
    ]

    # self.searcher is looked up on every call, so it may be replaced.
    def search_fn(q, final_nn, pre_nn, leaves):
      return self.searcher.searcher.search(q, final_nn, pre_nn, leaves)

    with tf.name_scope("scann_search"):
      idx, dist = _search(search_fn, q, params, not tf.executing_eagerly())
    return common.search_result(idx, dist)

  def _search_batched(self, q, final_num_neighbors, pre_reorder_num_neighbors,
                      leaves_to_search, parallel, batch_size, method):
    q = common.queries(q, 2, method)
    params = [
        common.param(final_num_neighbors),
        common.param(pre_reorder_num_neighbors),
        common.param(leaves_to_search),
        tf.convert_to_tensor(batch_size, tf.int32)
    ]

    def search_fn(q, final_nn, pre_nn, leaves, batch_size):
      return self.searcher.searcher.search_batched(q, final_nn, pre_nn, leaves,
                                                   parallel, batch_size)

    with tf.name_scope(f"scann_{method}"):
      idx, dist = _search(search_fn, q, params, not tf.executing_eagerly())
    return common.batched_result(q, idx, dist, final_num_neighbors)

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
      row's length. The static shape includes k when it is a Python int.
    """
    return self._search_batched(q, final_num_neighbors,
                                pre_reorder_num_neighbors, leaves_to_search,
                                False, 256, "search_batched")

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

  def to_pybind(self):
    """The wrapped scann_ops_pybind.ScannSearcher (self.searcher; shared).

    With this backend, changes made to it are seen by this searcher; the op
    backend's to_pybind() returns a copy instead. Code that mutates the
    returned searcher and then calls scann.tf.from_pybind() on it works
    with both.
    """
    return self.searcher

  def serialize(self, artifacts_dir, relative_path=False):
    """Saves the index; load it with load_searcher() here or in any binding."""
    self.searcher.serialize(artifacts_dir, relative_path)

  def serialize_to_module(self):
    raise NotImplementedError(_NO_SAVEDMODEL.format("serialize_to_module()"))


def searcher_from_module(module, db=None):
  del module, db  # Unused.
  raise NotImplementedError(_NO_SAVEDMODEL.format("searcher_from_module()"))


def from_pybind(searcher, shared_name=None):
  """A ScannSearcher over a scann_ops_pybind.ScannSearcher (shared, not copied).

  shared_name is the op backend's cache key, ignored here.
  """
  del shared_name  # Unused.
  return ScannSearcher(searcher)


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
    scann_ops_pybind's builder, so autopilot(), SOAR and docids work as
    there.
  """
  b = scann_ops_pybind.builder(
      common.to_numpy(db), num_neighbors, distance_measure)
  build_pybind = b.builder_lambda

  def builder_lambda(db, config, training_threads, **kwargs):
    return ScannSearcher(build_pybind(db, config, training_threads, **kwargs))

  return b.set_builder_lambda(builder_lambda)


def create_searcher(db,
                    scann_config,
                    training_threads=0,
                    container="",
                    shared_name=None,
                    docids=None,
                    **unused_kwargs):
  """Creates a scann.tf.ScannSearcher from a dataset and text config proto.

  Upstream's container argument is accepted and ignored, and so is
  shared_name (the op backend's cache key).
  """
  del container, shared_name  # Unused.
  return ScannSearcher(
      scann_ops_pybind.create_searcher(
          common.to_numpy(db),
          common.config_text(scann_config),
          common.as_int(training_threads),
          docids=docids))


def load_searcher(artifacts_dir, assets_backcompat_shim=True,
                  shared_name=None):
  """Loads an index saved by serialize() (from any binding).

  shared_name is the op backend's cache key, ignored here.
  """
  del shared_name  # Unused.
  return ScannSearcher(
      scann_ops_pybind.load_searcher(artifacts_dir, assets_backcompat_shim))
