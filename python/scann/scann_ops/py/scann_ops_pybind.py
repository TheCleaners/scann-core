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

"""Wrapper around pybind module that provides convenience functions for instantiating ScaNN searchers."""

import collections
import contextlib
import os
import pickle as pkl
import numpy as np
import sys
import threading

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cc/python"))
import scann_pybind
from scann.scann_ops.py import scann_builder
from scann.scann_ops.py import scann_ops_pybind_backcompat


def _open(path, mode):
  return open(path, mode)


def _host_array(x):
  """scann-core: an array from any library as something numpy can read.

  numpy arrays pass through. Anything else goes through np.asarray, which
  reads CPU tensors (PyTorch, JAX, TensorFlow) through the array protocol
  without a copy. First, a tensor that requires grad is detached, one on a
  GPU (or other device) is copied to host memory, and a dtype numpy lacks
  (bfloat16) is converted to float32. Duck-typed: nothing is imported.
  """
  if isinstance(x, np.ndarray):
    return x
  if getattr(x, "requires_grad", False) and hasattr(x, "detach"):
    x = x.detach()
  if getattr(getattr(x, "device", None), "type", "cpu") != "cpu" and hasattr(
      x, "cpu"):
    x = x.cpu()
  try:
    return np.asarray(x)
  except TypeError:
    if hasattr(x, "float"):  # e.g. torch.bfloat16, which numpy lacks
      return np.asarray(x.float())
    raise

class _ReadWriteLock:
  """Many readers or one writer; waiting writers go first. Not reentrant.

  scann-core: guards the docid bookkeeping. A search maps the index's ids
  to docids after the C++ search returns, and delete() moves the last
  docid into the freed slot, so without this a search overlapping a
  delete (possible even with the GIL, which searches release) could
  return the wrong docids.
  """

  def __init__(self):
    self._cond = threading.Condition(threading.Lock())
    self._readers = 0
    self._writer = False
    self._writers_waiting = 0

  @contextlib.contextmanager
  def read(self):
    with self._cond:
      while self._writer or self._writers_waiting:
        self._cond.wait()
      self._readers += 1
    try:
      yield
    finally:
      with self._cond:
        self._readers -= 1
        if not self._readers:
          self._cond.notify_all()

  @contextlib.contextmanager
  def write(self):
    with self._cond:
      self._writers_waiting += 1
      while self._writer or self._readers:
        self._cond.wait()
      self._writers_waiting -= 1
      self._writer = True
    try:
      yield
    finally:
      with self._cond:
        self._writer = False
        self._cond.notify_all()


class ScannSearcher(object):
  """Wrapper class around pybind module that provides a cleaner interface."""

  def __init__(self, searcher, docids=None):
    self.searcher = searcher
    # Simple docid mapping.
    # scann-core: a copy. Upstream kept the caller's list, which upsert()
    # then appended to and delete() reordered.
    self.docids = None if docids is None else list(docids)
    self._docids_lock = _ReadWriteLock()
    # scann-core: bumped by every call that may change the index (upsert,
    # delete, rebalance), so that wrappers holding a copy of the index
    # (scann.torch's native backend) know to refresh it.
    self._generation = 0
    if docids is not None:
      self.docid_to_id = {docid: id for id, docid in enumerate(docids)}
      if len(docids) != len(self.docid_to_id):
        raise ValueError("Duplicates found in docids.")

  def _reading_docids(self):
    # Searchers without docids keep no Python-side state; the C++ searcher
    # does its own locking.
    if self.docids is None:
      return contextlib.nullcontext()
    return self._docids_lock.read()

  def _batched_docids(self, idx, dist):
    """Maps batched result indices to docids (if any)."""
    if self.docids is None:
      return idx
    # scann-core: a query with fewer than k results is padded with index 0
    # and a NaN distance. Upstream mapped that padding to docids[0], a real
    # but wrong docid; map it to None instead.
    padded = np.isnan(dist)
    if not padded.any():
      return [[self.docids[j] for j in i] for i in idx]
    return [[None if p else self.docids[j]
             for j, p in zip(i, pi)]
            for i, pi in zip(idx, padded)]

  def search(
      self,
      q,
      final_num_neighbors=-1,
      pre_reorder_num_neighbors=-1,
      leaves_to_search=-1,
  ):
    """Single-query search; -1 for a param uses the searcher's default value."""
    if self.docids is None:
      # scann-core: without docids there is no bookkeeping to lock or map:
      # straight to the C++ searcher, which does its own locking.
      return self.searcher.search(
          q if type(q) is np.ndarray else _host_array(q), final_num_neighbors,
          pre_reorder_num_neighbors, leaves_to_search)
    q = _host_array(q)
    with self._reading_docids():
      idx, dist = self.searcher.search(q, final_num_neighbors,
                                       pre_reorder_num_neighbors,
                                       leaves_to_search)
      idx = idx if self.docids is None else [self.docids[j] for j in idx]
    return idx, dist

  def search_batched(
      self,
      queries,
      final_num_neighbors=None,
      pre_reorder_num_neighbors=None,
      leaves_to_search=None,
  ):
    """Search method for multiple queries."""
    final_nn = -1 if final_num_neighbors is None else final_num_neighbors
    pre_nn = (-1 if pre_reorder_num_neighbors is None else
              pre_reorder_num_neighbors)
    leaves = -1 if leaves_to_search is None else leaves_to_search
    queries = _host_array(queries)
    with self._reading_docids():
      idx, dist = self.searcher.search_batched(
          queries,
          final_nn,
          pre_nn,
          leaves,
          False,
          0,  # Ignored when parallel=False.
      )
      idx = self._batched_docids(idx, dist)
    return idx, dist

  def search_batched_parallel(
      self,
      queries,
      final_num_neighbors=None,
      pre_reorder_num_neighbors=None,
      leaves_to_search=None,
      batch_size=256,
  ):
    """Search method for multiple queries with multiple threads."""
    final_nn = -1 if final_num_neighbors is None else final_num_neighbors
    pre_nn = (-1 if pre_reorder_num_neighbors is None else
              pre_reorder_num_neighbors)
    leaves = -1 if leaves_to_search is None else leaves_to_search
    queries = _host_array(queries)
    with self._reading_docids():
      idx, dist = self.searcher.search_batched(queries, final_nn, pre_nn,
                                               leaves, True, batch_size)
      idx = self._batched_docids(idx, dist)
    return idx, dist

  def serialize(self, artifacts_dir, relative_path=False):
    # scann-core: the docids are written with the index, in one commit (see
    # ScannInterface::SerializeToDirectory); upstream wrote them afterwards,
    # and left a previous index's scann_docids.pkl in place when there were
    # none.
    with self._reading_docids():
      docids_pkl = None if self.docids is None else pkl.dumps(self.docids)
      self.searcher.serialize(artifacts_dir, relative_path, docids_pkl)

  def get_health_stats(self):
    return self.searcher.get_health_stats()

  def initialize_health_stats(self):
    return self.searcher.initialize_health_stats()

  def upsert(self, docids, database, batch_size=1):
    """Insert or update datapoints into the searcher."""
    if not isinstance(docids, list):
      docids = [docids]
    database = _host_array(database)
    if database.ndim == 1:
      database = np.expand_dims(database, 0)
    if len(docids) != database.shape[0]:
      raise ValueError(
          "Number of items mismatch in docids and database vectors:"
          f" {len(docids)} != {database.shape[0]}")
    if self.docids is None:
      raise ValueError("Cannot upsert because docids have not been specified "
                       "when initializing.")
    # scann-core: upstream accepted a docid listed twice. A new one was added
    # to the index twice but mapped only once, leaving a duplicate in docids;
    # an existing one was updated twice. Reject both, as delete() does.
    if len(set(docids)) != len(docids):
      repeated = [d for d, n in collections.Counter(docids).items() if n > 1]
      raise ValueError(f"Docids to upsert are not unique: {repeated[:10]}")
    with self._docids_lock.write():
      indices = [self.docid_to_id.get(docid) for docid in docids]
      # scann-core: update the docid bookkeeping only once the index has
      # accepted the vectors; upstream updated it first, so a failed upsert
      # left docids out of sync with the index.
      try:
        _ = self.searcher.upsert(indices, database, batch_size)
      finally:
        self._generation += 1

      for idx, docid in zip(indices, docids):
        if idx is not None:
          self.docids[idx] = docid
        else:
          self.docids.append(docid)
          self.docid_to_id[docid] = len(self.docids) - 1

  def delete(self, docids):
    """Delete datapoints from searcher."""
    if not isinstance(docids, list):
      docids = [docids]
    # scann-core: upstream failed here with AttributeError (no docid_to_id)
    # on a searcher built without docids; say why, as upsert() does.
    if self.docids is None:
      raise ValueError("Cannot delete because docids have not been specified "
                       "when initializing.")
    with self._docids_lock.write():
      # scann-core: validate before changing anything; upstream raised midway
      # through the loop below, after updating the bookkeeping for earlier
      # docids that were then never deleted from the index.
      for docid in docids:
        if docid not in self.docid_to_id:
          raise KeyError(f"Docid not found: {docid} ")
      if len(set(docids)) != len(docids):
        raise KeyError(f"Docids to delete are not unique: {docids}")
      indices = []
      for docid in docids:
        idx = self.docid_to_id[docid]
        indices.append(idx)
        old_idx = len(self.docids) - 1  # pyrefly: ignore[bad-argument-type]
        if idx != old_idx:
          old_docid = self.docids[
              old_idx]  # pyrefly: ignore[unsupported-operation]
          self.docids[idx] = old_docid  # pyrefly: ignore[unsupported-operation]
          self.docid_to_id[old_docid] = idx
        self.docids.pop()  # pyrefly: ignore[missing-attribute]
        self.docid_to_id.pop(docid)
      try:
        _ = self.searcher.delete(indices)
      finally:
        self._generation += 1

  def rebalance(self, config=None):
    """Rebalances the searcher."""
    # TODO(guorq): currently, this performs a full retrain based on the initial
    # config.
    config = "" if config is None else config
    try:
      return self.searcher.rebalance(config)
    finally:
      self._generation += 1

  def reserve(self, num_datapoints):
    return self.searcher.reserve(num_datapoints)

  def size(self):
    return self.searcher.size()

  def set_num_threads(self, num_threads):
    """Threads for search_batched_parallel and mutations, the caller included.

    scann-core: num_threads - 1 pool threads are started on first use; 0 or
    1 means everything runs on the calling thread. (Upstream started
    num_threads pool threads.) The default is the number of CPUs the process
    may use (affinity mask and cgroup CPU quota), or SCANN_NUM_THREADS.
    """
    self.searcher.set_num_threads(num_threads)

  def config(self):
    """Returns the config."""
    return self.searcher.config()


def builder(db, num_neighbors, distance_measure):
  """pybind analogue of builder() in scann_ops.py; see docstring there."""
  db = _host_array(db)

  class ScannBuilder(scann_builder.ScannBuilder):

    def create_config(self):
      """Create a config with autopilot rewrite without actually building the searcher."""
      config = super().create_config()
      if self.params.get("autopilot") is not None:
        config = scann_pybind.ScannNumpy.suggest_autopilot(
            config, self.db.shape[0], self.db.shape[1])
      return config

  def builder_lambda(db, config, training_threads, **kwargs):
    return create_searcher(db, config, training_threads, **kwargs)

  return ScannBuilder(db, num_neighbors,
                      distance_measure).set_builder_lambda(builder_lambda)


def create_searcher(db,
                    scann_config,
                    training_threads=0,
                    docids=None,
                    **unused_kwargs):
  """Creates a searcher object wrapping a ScannNumpy object."""
  db = _host_array(db)
  if docids is not None:
    if len(docids) != db.shape[0]:
      raise ValueError(
          f"docid and database size mismatch: {len(docids)} != {db.shape[0]}.")
  if isinstance(db, np.ndarray) and db.shape[0] == 0:
    scann_config += f"""
      input_output {{
        pure_dynamic_config {{
          vector_type: DENSE
          dimensionality: {db.shape[1]}
        }}
      }}
    """
  return ScannSearcher(
      scann_pybind.ScannNumpy(db, scann_config, training_threads),
      docids=docids)


def load_searcher(artifacts_dir, assets_backcompat_shim=True):
  """Loads searcher assets from artifacts_dir and returns a ScaNN searcher."""
  is_dir = os.path.isdir(artifacts_dir)
  if not is_dir:
    raise ValueError(f"{artifacts_dir} is not a directory.")

  assets_pbtxt = os.path.join(artifacts_dir, "scann_assets.pbtxt")
  if not scann_ops_pybind_backcompat.path_exists(assets_pbtxt):
    if not assets_backcompat_shim:
      raise ValueError("No scann_assets.pbtxt found.")
    print("No scann_assets.pbtxt found. ScaNN assumes this searcher was from an"
          " earlier release, and is calling `populate_and_save_assets_proto`"
          "from `scann_ops_pybind_backcompat` to create a scann_assets.pbtxt. "
          "Note this compatibility shim may be removed in the future.")
    scann_ops_pybind_backcompat.populate_and_save_assets_proto(artifacts_dir)

  docids_path = os.path.join(artifacts_dir, "scann_docids.pkl")
  exists = os.path.isfile(docids_path)
  docid = pkl.load(_open(docids_path, "rb")) if exists else None

  with _open(assets_pbtxt, "r") as f:
    searcher = scann_pybind.ScannNumpy(artifacts_dir, f.read())
  # scann-core: a scann_docids.pkl from another index was accepted, mapping
  # results to the wrong docids.
  if docid is not None and len(docid) != searcher.size():
    raise ValueError(
        f"{docids_path} has {len(docid)} docids, but the index has "
        f"{searcher.size()} datapoints; the directory may mix files of "
        "different indexes.")
  return ScannSearcher(searcher, docid)
