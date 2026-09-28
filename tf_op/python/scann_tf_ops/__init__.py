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

"""scann-core's TensorFlow op, with upstream's scann_ops API.

Adapted from upstream's scann/scann_ops/py/scann_ops.py: the same
builder() / create_searcher() / ScannSearcher.search*() /
serialize_to_module() / searcher_from_module() API, backed by scann-core's
optional TensorFlow op (tf_op/, built with -DSCANN_BUILD_TF_OP=ON; not part
of the wheel). Unlike scann.tf, searches are graph ops, so a model that
searches can be saved as a SavedModel and served without Python. See
docs/tensorflow.md.

  import scann_tf_ops as scann_ops  # upstream: from scann.scann_ops.py import scann_ops

  searcher = scann_ops.builder(db, 10, "dot_product").score_brute_force().build()
  indices, distances = searcher.search_batched(queries)  # int32, float32

The searcher is a tf.Module holding the index as tf.Variables: the files
serialize() writes (names and contents, as two string vectors), and a
random index id. The op builds the ScaNN searcher from them on first use
and caches it (keyed by the index id and a fingerprint of the variables),
so it is built once per process, also after tf.saved_model.load().
"""

import os
import tempfile
import uuid
import weakref

import numpy as np
import tensorflow as tf
from google.protobuf import text_format

# scann.tf is the tf.numpy_function wrapper; it doesn't use the op. Its
# argument handling and result types are shared, so results of both are
# the same namedtuples.
from scann import tf as _scann_tf
from scann.scann_ops import scann_assets_pb2
from scann.scann_ops.py import scann_ops_pybind

__all__ = [
    "BatchedSearchResult", "ScannSearcher", "SearchResult", "builder",
    "create_searcher", "from_pybind", "load_searcher", "searcher_from_module",
    "stats"
]

_ops = tf.load_op_library(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "_scann_tf_ops.so"))

SearchResult = _scann_tf.SearchResult
BatchedSearchResult = _scann_tf.BatchedSearchResult

_MANIFEST = "scann_assets.pbtxt"
_DOCIDS = "scann_docids.pkl"


def stats():
  """The op library's searcher cache: {"live_searchers", "builds"}.

  live_searchers: searchers currently built and cached in this process.
  builds: searchers built since the library was loaded. A model built once
  and searched many times (eagerly, in tf.functions, or loaded from a
  SavedModel) adds one build.
  """
  live, builds = _ops.scann_core_stats()
  return {"live_searchers": int(live), "builds": int(builds)}


def _read_files(directory, names):
  files = {}
  for name in names:
    with open(os.path.join(directory, name), "rb") as f:
      files[name] = f.read()
  return files


def _relative_manifest(manifest):
  """The manifest with every asset path reduced to its file name."""
  assets = scann_assets_pb2.ScannAssets()
  text_format.Parse(manifest, assets)
  for asset in assets.assets:
    asset.asset_path = os.path.basename(asset.asset_path)
  return assets


class ScannSearcher(tf.Module):
  """A ScaNN index as TensorFlow state, searched by scann-core's op.

  Create one with builder(), create_searcher(), from_pybind(),
  load_searcher() or searcher_from_module().

  The search methods return (indices, distances) namedtuples of int32 and
  float32 tensors, as upstream's op did, and work eagerly and inside
  tf.function. Indices are positions in the index, never docids.

  Attributes:
    asset_names: tf.Variable, string [num_files]: the file names of the
      index directory (scann_config.pb, scann_assets.pbtxt, the assets, and
      scann_docids.pkl if the index has docids).
    asset_contents: tf.Variable, string [num_files]: the files' contents.
    index_id: tf.Variable, string scalar: the op's shared_name, a random id
      per index, so that two indexes never share a cached searcher.
  """

  def __init__(self, files, shared_name=None, name=None):
    """Wraps index files.

    Args:
      files: {file name: bytes}, the files serialize() writes to an index
        directory. Prefer builder(), from_pybind() or load_searcher().
      shared_name: the op's cache key; a new random id if None.
      name: the tf.Module name.
    """
    super().__init__(name=name)
    if _MANIFEST not in files:
      raise ValueError(f"The index files lack {_MANIFEST}.")
    names = sorted(files)
    with self.name_scope:
      self.asset_names = tf.Variable(
          tf.constant(names, tf.string),
          trainable=False,
          shape=tf.TensorShape([None]),
          name="asset_names")
      self.asset_contents = tf.Variable(
          tf.constant([files[n] for n in names], tf.string),
          trainable=False,
          shape=tf.TensorShape([None]),
          name="asset_contents")
      self.index_id = tf.Variable(
          shared_name or f"scann_core_{uuid.uuid4().hex}",
          trainable=False,
          name="index_id")
    self._init_functions()

  @classmethod
  def _from_variables(cls, asset_names, asset_contents, index_id):
    self = cls.__new__(cls)
    tf.Module.__init__(self)
    self.asset_names = asset_names
    self.asset_contents = asset_contents
    self.index_id = index_id
    self._init_functions()
    return self

  def _init_functions(self):
    # The op's index_id attr is fixed when a graph is built; it comes from
    # the index_id variable's value now (read eagerly, also while a
    # tf.function is being traced).
    with tf.init_scope():
      self._shared_name = self.index_id.numpy().decode("utf-8")
    self._eager = _EagerFunctions(self.asset_names, self.asset_contents,
                                  self._shared_name)

  @property
  def shared_name(self):
    return self._shared_name

  def _batched(self, q, final_num_neighbors, pre_reorder_num_neighbors,
               leaves_to_search, parallel, batch_size, method):
    q = _scann_tf._queries(q, 2, method)  # pylint: disable=protected-access
    args = (q, _scann_tf._param(final_num_neighbors),  # pylint: disable=protected-access
            _scann_tf._param(pre_reorder_num_neighbors),  # pylint: disable=protected-access
            _scann_tf._param(leaves_to_search),  # pylint: disable=protected-access
            tf.constant(parallel), tf.convert_to_tensor(batch_size, tf.int32))
    with tf.name_scope(f"scann_{method}"):
      if tf.executing_eagerly():
        idx, dist = self._eager.search_batched(*args)
      else:
        idx, dist = _ops.scann_core_search_batched(
            self.asset_names, self.asset_contents, *args,
            index_id=self._shared_name)
    # Exactly [num_queries, k] with an explicit k (rows with fewer results
    # are padded with index 0 and NaN distance); with the default k, as
    # wide as the longest row.
    k = _scann_tf._static_int(final_num_neighbors)  # pylint: disable=protected-access
    shape = [q.shape[0], k if k is not None and k > 0 else None]
    idx.set_shape(shape)
    dist.set_shape(shape)
    return BatchedSearchResult(idx, dist)

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
    q = _scann_tf._queries(q, 1, "search")  # pylint: disable=protected-access
    args = (q, _scann_tf._param(final_num_neighbors),  # pylint: disable=protected-access
            _scann_tf._param(pre_reorder_num_neighbors),  # pylint: disable=protected-access
            _scann_tf._param(leaves_to_search))  # pylint: disable=protected-access
    with tf.name_scope("scann_search"):
      if tf.executing_eagerly():
        idx, dist = self._eager.search(*args)
      else:
        idx, dist = _ops.scann_core_search(
            self.asset_names, self.asset_contents, *args,
            index_id=self._shared_name)
    return SearchResult(idx, dist)

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
    return self._batched(q, final_num_neighbors, pre_reorder_num_neighbors,
                         leaves_to_search, False, 256, "search_batched")

  def search_batched_parallel(self,
                              q,
                              final_num_neighbors=None,
                              pre_reorder_num_neighbors=None,
                              leaves_to_search=None,
                              batch_size=256):
    """search_batched() on the searcher's thread pool, batch_size at a time."""
    return self._batched(q, final_num_neighbors, pre_reorder_num_neighbors,
                         leaves_to_search, True, batch_size,
                         "search_batched_parallel")

  def files(self):
    """The index files, {name: bytes} (eagerly)."""
    names = [n.decode("utf-8") for n in self.asset_names.numpy()]
    return dict(zip(names, self.asset_contents.numpy()))

  def to_pybind(self):
    """The index as a scann_ops_pybind.ScannSearcher (with its docids).

    For mutating it (upsert, delete, rebalance): mutate the pybind searcher,
    then make a new op searcher with from_pybind().
    """
    with tempfile.TemporaryDirectory() as d:
      for name, content in self.files().items():
        with open(os.path.join(d, name), "wb") as f:
          f.write(content)
      return scann_ops_pybind.load_searcher(d)

  def serialize(self, artifacts_dir, relative_path=False):
    """Saves the index; load it with load_searcher() here or in any binding."""
    self.to_pybind().serialize(artifacts_dir, relative_path)

  def serialize_to_module(self):
    """The searcher's state as a tf.Module, for checkpoints and SavedModels.

    It is the searcher itself: its variables hold the whole index. Attach it
    to a model and save that; searcher_from_module() recreates the searcher
    from the loaded object.
    """
    return self


class _EagerFunctions(object):
  """Per-searcher tf.functions for eager searches.

  Calling the op eagerly would create a kernel in TensorFlow's eager kernel
  cache, which lives as long as the process, and with it the cached
  searcher. Through these functions the kernels belong to the functions,
  which are freed with the ScannSearcher, and the searcher with them.
  (Inside a tf.function the op is called directly: its kernels belong to
  that function.) Not a trackable object, so SavedModel ignores them.
  """

  def __init__(self, names, contents, shared_name):

    @tf.function(autograph=False, reduce_retracing=True)
    def search(q, final_nn, pre_nn, leaves):
      return _ops.scann_core_search(
          names, contents, q, final_nn, pre_nn, leaves,
          index_id=shared_name)

    @tf.function(autograph=False, reduce_retracing=True)
    def search_batched(q, final_nn, pre_nn, leaves, parallel, batch_size):
      return _ops.scann_core_search_batched(
          names, contents, q, final_nn, pre_nn, leaves, parallel, batch_size,
          index_id=shared_name)

    self.search = search
    self.search_batched = search_batched


def from_pybind(searcher, shared_name=None):
  """A ScannSearcher from a scann_ops_pybind.ScannSearcher (copies the index).

  The pybind searcher isn't referenced afterwards; delete it to free its
  memory. Its docids are kept (in scann_docids.pkl) for to_pybind() and
  serialize(), but searches return indices.
  """
  if not isinstance(searcher, scann_ops_pybind.ScannSearcher):
    raise TypeError(
        "from_pybind() takes a scann_ops_pybind.ScannSearcher, got "
        f"{type(searcher).__name__}.")
  with tempfile.TemporaryDirectory() as d:
    searcher.serialize(d, relative_path=True)
    names = [n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n))]
    return ScannSearcher(_read_files(d, names), shared_name)


def load_searcher(artifacts_dir, shared_name=None):
  """Loads an index saved by serialize() (from any binding).

  Reads the files the manifest lists (and scann_docids.pkl); the searcher is
  built, and the files checked, on the first search. A directory that
  doesn't load fails there with the same error as
  scann_ops_pybind.load_searcher().
  """
  with open(os.path.join(artifacts_dir, _MANIFEST), "rb") as f:
    manifest = f.read()
  try:
    assets = _relative_manifest(manifest)
  except text_format.ParseError:
    # An incomplete or damaged directory: raise the loader's error.
    scann_ops_pybind.load_searcher(artifacts_dir)
    raise
  orig = scann_assets_pb2.ScannAssets()
  text_format.Parse(manifest, orig)
  files = {
      _MANIFEST: text_format.MessageToString(assets).encode("utf-8"),
      "scann_config.pb": _read_files(artifacts_dir,
                                     ["scann_config.pb"])["scann_config.pb"],
  }
  for asset in orig.assets:
    path = asset.asset_path
    if not os.path.isabs(path):
      path = os.path.join(artifacts_dir, path)
    with open(path, "rb") as f:
      files[os.path.basename(path)] = f.read()
  if os.path.exists(os.path.join(artifacts_dir, _DOCIDS)):
    files.update(_read_files(artifacts_dir, [_DOCIDS]))
  return ScannSearcher(files, shared_name)


def searcher_from_module(module, db=None):
  """The ScannSearcher whose state serialize_to_module() returned.

  Args:
    module: a ScannSearcher, or the object tf.saved_model.load() restored
      for one (anything with its asset_names, asset_contents and index_id
      variables). The variables are shared, not copied.
    db: unused (upstream's signature).
  """
  del db  # Unused.
  if isinstance(module, ScannSearcher):
    return module
  try:
    return _from_module_cache[module]
  except (KeyError, TypeError):
    pass
  searcher = ScannSearcher._from_variables(  # pylint: disable=protected-access
      module.asset_names, module.asset_contents, module.index_id)
  try:
    _from_module_cache[module] = searcher
  except TypeError:  # Not weakly referenceable.
    pass
  return searcher


# One ScannSearcher per restored module, so that calling
# searcher_from_module() in a tf.function body (traced more than once) or
# in a loop doesn't make a new one each time.
_from_module_cache = weakref.WeakKeyDictionary()


def builder(db, num_neighbors, distance_measure):
  """Creates a ScannBuilder that returns a scann_tf_ops.ScannSearcher on build().

  Args:
    db: the dataset that ScaNN will search over; a 2d array or eager tensor
      of floats (converted to float32) with one data point per row.
    num_neighbors: the default # neighbors the searcher will return per query.
    distance_measure: one of "squared_l2" or "dot_product".

  Returns:
    A ScannBuilder object, which builds the ScaNN config via calls such as
    tree() and score_brute_force(). Calling build() on the ScannBuilder will
    return a scann_tf_ops.ScannSearcher with its specified config. It is
    scann_ops_pybind's builder, so autopilot(), SOAR and docids work as
    there.
  """
  b = scann_ops_pybind.builder(_scann_tf._to_numpy(db), num_neighbors,  # pylint: disable=protected-access
                               distance_measure)
  build_pybind = b.builder_lambda

  def builder_lambda(db, config, training_threads, **kwargs):
    return from_pybind(build_pybind(db, config, training_threads, **kwargs))

  return b.set_builder_lambda(builder_lambda)


def create_searcher(db,
                    scann_config,
                    training_threads=0,
                    container="",
                    shared_name=None,
                    docids=None,
                    **unused_kwargs):
  """Creates a ScannSearcher from a dataset and text config proto.

  Upstream's container argument is accepted and ignored; shared_name is the
  op's cache key (a new random id if None).
  """
  del container  # Unused.
  if isinstance(scann_config, tf.Tensor):
    scann_config = scann_config.numpy()
  if isinstance(scann_config, bytes):
    scann_config = scann_config.decode("utf-8")
  return from_pybind(
      scann_ops_pybind.create_searcher(
          _scann_tf._to_numpy(db),  # pylint: disable=protected-access
          scann_config,
          _scann_tf._as_int(training_threads),  # pylint: disable=protected-access
          docids=docids),
      shared_name)
