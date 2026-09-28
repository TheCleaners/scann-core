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

"""PyTorch wrapper around scann-core's searcher.

scann.torch.Searcher is a torch.nn.Module over a pybind searcher
(scann.scann_ops_pybind): its search methods take query tensors on any
device and return (indices, distances) tensors on the same device. The
search is a PyTorch custom op (scann_py::search, scann_py::search_batched)
with exact output shapes, so torch.compile(fullgraph=True) compiles models
that search without a graph break, also with dynamic batch sizes. See
docs/integrations.md.

  import scann.torch as scann_torch

  searcher = scann_torch.builder(db, 10, "dot_product").score_brute_force().build()
  indices, distances = searcher.search_batched(queries)  # int64, float32

The custom op runs Python code (the pybind searcher, which releases the
GIL while searching), and finds the searcher through an integer handle into
a registry of this process. A torch.export program would hold only that
handle, not the index, so exporting a model that searches raises; export
support is planned with a native op.

Importing `scann` doesn't import PyTorch; importing this module does.
"""

# This module is scann.torch; `import torch` below is an absolute import, so
# it is PyTorch, never this file.
import secrets
import threading
import weakref

import numpy as np

try:
  import torch
except ImportError as e:
  raise ImportError(
      "scann.torch needs PyTorch, which could not be imported "
      f"({e}). Install it with `pip install 'scann-core[torch]'` (or install "
      "torch yourself). The rest of the scann package works without it.") from e

from google.protobuf import text_format

from scann.proto import scann_pb2
from scann.scann_ops.py import scann_ops_pybind

__all__ = ["Searcher", "builder", "create_searcher", "load_searcher"]

_NO_EXPORT = (
    "scann.torch searches can't be exported: the exported program would "
    "hold only a handle to a searcher in this process, not the index. "
    "torch.export (and AOTInductor) support is planned with the native op "
    "in scann-core 0.2.1. Meanwhile, export the model without the search "
    "and load the index next to it with scann.torch.load_searcher(); see "
    "docs/integrations.md in scann-core.")

# handle -> ScannNumpy (the C++ searcher; it does its own locking). Handles
# are random 62-bit numbers, so a handle baked into a graph that reaches
# another process (torch.jit.trace, a hand-edited export) finds nothing
# there instead of some other index.
_registry = {}
_registry_lock = threading.Lock()


def _register(raw):
  with _registry_lock:
    while True:
      handle = secrets.randbits(62)
      if handle and handle not in _registry:
        _registry[handle] = raw
        return handle


def _unregister(handle):
  with _registry_lock:
    _registry.pop(handle, None)


def _lookup(handle):
  raw = _registry.get(handle)
  if raw is None:
    raise RuntimeError(
        f"scann.torch: no searcher with handle {handle} in this process. "
        "Its Searcher has been deleted, or the graph calling the op was "
        "made in another process (exported programs don't carry the "
        "index; torch.export support is planned with the native op in "
        "scann-core 0.2.1).")
  return raw


def _num_live_searchers():
  """Searchers registered in this process (for tests)."""
  with _registry_lock:
    return len(_registry)


def _host_float32(t):
  """A tensor as a C-contiguous float32 numpy array in host memory."""
  t = t.detach()
  if t.device.type != "cpu":
    t = t.cpu()
  if t.dtype != torch.float32:
    t = t.float()
  return t.contiguous().numpy()


def _to_output(idx, dist, k, device):
  """pybind results -> (int64, float32) tensors padded to k with -1 / NaN."""
  idx = np.asarray(idx).astype(np.int64)
  dist = np.asarray(dist, dtype=np.float32)
  width = idx.shape[-1]
  if width < k:  # single-query search: n <= k results
    pad = [(0, 0)] * (idx.ndim - 1) + [(0, k - width)]
    idx = np.pad(idx, pad, constant_values=-1)
    dist = np.pad(dist, pad, constant_values=np.nan)
  elif width > k:
    idx, dist = idx[..., :k], dist[..., :k]
  # The pybind searcher pads short batched rows with index 0 and NaN.
  idx[np.isnan(dist)] = -1
  out_i, out_d = torch.from_numpy(idx), torch.from_numpy(dist)
  if device.type != "cpu":
    out_i, out_d = out_i.to(device), out_d.to(device)
  return out_i, out_d


def _check_not_exporting():
  if torch.compiler.is_exporting():
    raise RuntimeError(_NO_EXPORT)


# Not safe to capture in a CUDA graph: the search copies to and from host
# memory and runs on the CPU.
_TAGS = ((torch.Tag.cudagraph_unsafe,)
         if hasattr(torch.Tag, "cudagraph_unsafe") else ())


@torch.library.custom_op("scann_py::search", mutates_args=(), tags=_TAGS)
def _search_op(query: torch.Tensor, handle: int, k: int,
               pre_reorder_num_neighbors: int,
               leaves_to_search: int) -> tuple[torch.Tensor, torch.Tensor]:
  raw = _lookup(handle)
  idx, dist = raw.search(_host_float32(query), k, pre_reorder_num_neighbors,
                         leaves_to_search)
  return _to_output(idx, dist, k, query.device)


@_search_op.register_fake
def _(query, handle, k, pre_reorder_num_neighbors, leaves_to_search):
  del handle, pre_reorder_num_neighbors, leaves_to_search  # Unused.
  _check_not_exporting()
  return (query.new_empty((k,), dtype=torch.int64),
          query.new_empty((k,), dtype=torch.float32))


@torch.library.custom_op(
    "scann_py::search_batched", mutates_args=(), tags=_TAGS)
def _search_batched_op(queries: torch.Tensor, handle: int, k: int,
                       pre_reorder_num_neighbors: int, leaves_to_search: int,
                       parallel: bool,
                       batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
  raw = _lookup(handle)
  idx, dist = raw.search_batched(
      _host_float32(queries), k, pre_reorder_num_neighbors, leaves_to_search,
      parallel, batch_size)
  return _to_output(idx, dist, k, queries.device)


@_search_batched_op.register_fake
def _(queries, handle, k, pre_reorder_num_neighbors, leaves_to_search,
      parallel, batch_size):
  del handle, pre_reorder_num_neighbors, leaves_to_search  # Unused.
  del parallel, batch_size  # Unused.
  _check_not_exporting()
  shape = (queries.shape[0], k)
  return (queries.new_empty(shape, dtype=torch.int64),
          queries.new_empty(shape, dtype=torch.float32))


def _default_num_neighbors(pybind_searcher):
  config = text_format.Parse(pybind_searcher.config(), scann_pb2.ScannConfig())
  return int(config.num_neighbors)


def _param(value, name):
  if value is None:
    return -1
  if isinstance(value, bool) or not isinstance(value, int):
    raise TypeError(f"{name} must be an int or None, got {value!r}.")
  return value


class Searcher(torch.nn.Module):
  """A scann-core searcher as a torch.nn.Module.

  search(), search_batched() and search_batched_parallel() take a query
  tensor (any float dtype, on any device) and return (indices, distances):
  int64 and float32 tensors on the queries' device. The search runs on the
  CPU; queries on a GPU are copied to host memory and the results back.

  Every row has exactly k entries, k being final_num_neighbors or, if that
  is None, the index's default. Missing results (fewer than k points in the
  index) are index -1 with distance NaN. Indices are positions in the
  index, never docids; map them with `self.searcher.docids` if the index
  has docids.

  The searches don't participate in autograd: queries that require grad
  are detached, and the results don't require grad.

  Calling the module is search_batched(). It has no parameters or buffers,
  so moving it (.to(), .cuda()) changes nothing, and the index isn't part
  of state_dict(); save it with serialize() and load it with
  load_searcher().

  Attributes:
    searcher: the wrapped scann_ops_pybind.ScannSearcher. Use it for
      everything that isn't a search (upsert, delete, docids, ...); later
      searches through this module see the changes.
  """

  def __init__(self, searcher):
    super().__init__()
    if not isinstance(searcher, scann_ops_pybind.ScannSearcher):
      raise TypeError(
          "scann.torch.Searcher wraps a scann_ops_pybind.ScannSearcher (from "
          "builder(), create_searcher() or load_searcher()), got "
          f"{type(searcher).__name__}.")
    self.searcher = searcher
    # The default k is read once: rebalance(config) with another
    # num_neighbors isn't seen by searches with final_num_neighbors=None.
    self._default_k = _default_num_neighbors(searcher)
    self._handle = _register(searcher.searcher)
    weakref.finalize(self, _unregister, self._handle)

  @classmethod
  def from_pybind(cls, searcher):
    """A Searcher over an existing scann_ops_pybind searcher (shared)."""
    return cls(searcher)

  def to_pybind(self):
    """The underlying scann_ops_pybind.ScannSearcher (shared, not a copy)."""
    return self.searcher

  @property
  def default_num_neighbors(self):
    """k when final_num_neighbors is None."""
    return self._default_k

  def _k(self, final_num_neighbors):
    k = _param(final_num_neighbors, "final_num_neighbors")
    if k == -1:
      return self._default_k
    if k <= 0:
      raise ValueError(
          f"final_num_neighbors must be > 0 (or None), got {k}.")
    return k

  def search(self,
             q,
             final_num_neighbors=None,
             pre_reorder_num_neighbors=None,
             leaves_to_search=None):
    """One query; None (or -1) for a parameter uses the index's default.

    Args:
      q: a 1-D query tensor (any float dtype, any device).
      final_num_neighbors: k, or None for the index's default.
      pre_reorder_num_neighbors: candidates to reorder, or None.
      leaves_to_search: tree leaves to search, or None.

    Returns:
      (indices, distances): int64 and float32 tensors of shape [k] on q's
      device; missing results are -1 / NaN.
    """
    if not isinstance(q, torch.Tensor):
      q = torch.as_tensor(q)
    if q.dim() != 1:
      raise ValueError(
          f"search() expects a 1-dimensional query, got shape {tuple(q.shape)}.")
    return torch.ops.scann_py.search(
        q.detach(), self._handle, self._k(final_num_neighbors),
        _param(pre_reorder_num_neighbors, "pre_reorder_num_neighbors"),
        _param(leaves_to_search, "leaves_to_search"))

  def _batched(self, queries, final_num_neighbors, pre_reorder_num_neighbors,
               leaves_to_search, parallel, batch_size, method):
    if not isinstance(queries, torch.Tensor):
      queries = torch.as_tensor(queries)
    if queries.dim() != 2:
      raise ValueError(f"{method}() expects 2-dimensional queries, got "
                       f"shape {tuple(queries.shape)}.")
    return torch.ops.scann_py.search_batched(
        queries.detach(), self._handle, self._k(final_num_neighbors),
        _param(pre_reorder_num_neighbors, "pre_reorder_num_neighbors"),
        _param(leaves_to_search, "leaves_to_search"), parallel, batch_size)

  def search_batched(self,
                     queries,
                     final_num_neighbors=None,
                     pre_reorder_num_neighbors=None,
                     leaves_to_search=None):
    """Queries in a 2-D tensor, searched one after another.

    Args:
      queries: [num_queries, dim] (any float dtype, any device).
      final_num_neighbors: k, or None for the index's default.
      pre_reorder_num_neighbors: candidates to reorder, or None.
      leaves_to_search: tree leaves to search, or None.

    Returns:
      (indices, distances): int64 and float32 tensors of shape
      [num_queries, k] on the queries' device; missing results are -1 /
      NaN.
    """
    return self._batched(queries, final_num_neighbors,
                         pre_reorder_num_neighbors, leaves_to_search, False, 0,
                         "search_batched")

  def search_batched_parallel(self,
                              queries,
                              final_num_neighbors=None,
                              pre_reorder_num_neighbors=None,
                              leaves_to_search=None,
                              batch_size=256):
    """search_batched() on the searcher's thread pool, batch_size at a time."""
    return self._batched(queries, final_num_neighbors,
                         pre_reorder_num_neighbors, leaves_to_search, True,
                         _param(batch_size, "batch_size"),
                         "search_batched_parallel")

  forward = search_batched

  def serialize(self, artifacts_dir, relative_path=False):
    """Saves the index; load it with load_searcher() here or in any binding."""
    self.searcher.serialize(artifacts_dir, relative_path)

  def extra_repr(self):
    return (f"size={self.searcher.size()}, "
            f"default_num_neighbors={self._default_k}")

  def __getstate__(self):
    raise TypeError(
        "a scann.torch.Searcher can't be pickled or deep-copied (so neither "
        "can a model holding one, e.g. with torch.save(model)): the index "
        "lives in the pybind searcher. Save the model's state_dict() and the "
        "index (serialize()) separately, and load the index with "
        "scann.torch.load_searcher().")


def builder(db, num_neighbors, distance_measure):
  """Creates a ScannBuilder that returns a scann.torch.Searcher on build().

  Args:
    db: the dataset; a 2-D tensor (any float dtype, any device) or array
      with one data point per row.
    num_neighbors: the default # neighbors the searcher will return per query.
    distance_measure: one of "squared_l2" or "dot_product".

  Returns:
    scann_ops_pybind's ScannBuilder (tree(), score_ah(), autopilot(), ...),
    whose build() (also build(docids=...)) returns a scann.torch.Searcher.
  """
  b = scann_ops_pybind.builder(db, num_neighbors, distance_measure)
  build_pybind = b.builder_lambda

  def builder_lambda(db, config, training_threads, **kwargs):
    return Searcher(build_pybind(db, config, training_threads, **kwargs))

  return b.set_builder_lambda(builder_lambda)


def create_searcher(db, scann_config, training_threads=0, docids=None):
  """Creates a scann.torch.Searcher from a dataset and a text config proto."""
  return Searcher(
      scann_ops_pybind.create_searcher(
          db, scann_config, training_threads, docids=docids))


def load_searcher(artifacts_dir, assets_backcompat_shim=True):
  """Loads an index saved by serialize() (from any binding)."""
  return Searcher(
      scann_ops_pybind.load_searcher(artifacts_dir, assets_backcompat_shim))
