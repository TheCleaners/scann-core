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

scann.torch.Searcher is a torch.nn.Module whose search methods take query
tensors on any device and return (indices, distances) tensors on the same
device. The searches are PyTorch custom ops with exact output shapes, so
torch.compile(fullgraph=True) compiles models that search without a graph
break, also with dynamic batch sizes. See docs/integrations.md.

  import scann.torch as scann_torch

  searcher = scann_torch.builder(db, 10, "dot_product").score_brute_force().build()
  indices, distances = searcher.search_batched(queries)  # int64, float32

Two backends, with the same API and the same results:

- "native", when the scann-core-torch package is installed: the searches
  are C++ ops (torch.ops.scann.*, from the scann_torch_ops package) and the
  Searcher holds the index in buffers, so state_dict() contains it, and
  torch.export / AOTInductor programs of a model that searches carry the
  index and run in other processes; a Searcher (and a model holding one)
  can be pickled (torch.save(model)).
- "python", otherwise (scann-core 0.2.0's): the ops (scann_py::search,
  scann_py::search_batched) run Python code, calling the pybind searcher
  (which releases the GIL while searching), found through an integer
  handle into a registry of this process. Exporting or pickling raises.

backend() says which one new Searchers use; SCANN_TORCH_BACKEND=native,
python or auto (the default) chooses, and so does the `backend` argument
of Searcher, builder(), create_searcher() and load_searcher().

Importing `scann` doesn't import PyTorch; importing this module does.
"""

# This module is scann.torch; `import torch` below is an absolute import, so
# it is PyTorch, never this file.
import io
import os
import pickle
import secrets
import tempfile
import threading
import uuid
import warnings
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
from scann.version import __version__ as _scann_version

__all__ = ["Searcher", "backend", "builder", "create_searcher",
           "load_searcher"]

# --- Backends -----------------------------------------------------------------

_BACKENDS = ("native", "python")
_native_ops = None  # the scann_torch_ops module, once loaded
_native_error = None  # why it isn't available, if it isn't
_native_lock = threading.Lock()


def _load_native():
  """The scann_torch_ops module, or None (and _native_error says why)."""
  global _native_ops, _native_error
  with _native_lock:
    if _native_ops is not None or _native_error is not None:
      return _native_ops
    try:
      import scann_torch_ops  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      _native_error = (
          "the native backend needs the scann-core-torch package (`pip "
          f"install scann-core-torch=={_scann_version}`), which could not "
          f"be imported: {e}")
      return None
    except Exception as e:  # pylint: disable=broad-except
      _native_error = (f"scann-core-torch is installed but failed to load: "
                       f"{type(e).__name__}: {e}")
      return None
    if scann_torch_ops.__version__ != _scann_version:
      _native_error = (
          f"scann-core-torch {scann_torch_ops.__version__} doesn't match "
          f"scann-core {_scann_version}; install scann-core-torch=="
          f"{_scann_version}")
      return None
    _native_ops = scann_torch_ops
    return _native_ops


def _default_backend():
  choice = os.environ.get("SCANN_TORCH_BACKEND", "").strip().lower() or "auto"
  if choice not in ("auto",) + _BACKENDS:
    raise ImportError(
        f"SCANN_TORCH_BACKEND must be auto, native or python, got {choice!r}.")
  if choice == "python":
    return "python"
  if _load_native() is not None:
    return "native"
  if choice == "native":
    raise ImportError(f"SCANN_TORCH_BACKEND=native, but {_native_error}.")
  if _native_error and "failed to load" in _native_error or (
      _native_error and "doesn't match" in _native_error):
    warnings.warn(f"scann.torch: using the Python backend: {_native_error}.",
                  RuntimeWarning, stacklevel=3)
  return "python"


_DEFAULT_BACKEND = _default_backend()


def backend():
  """The backend new Searchers use by default: "native" or "python".

  "native" when the scann-core-torch package is installed (and
  SCANN_TORCH_BACKEND isn't "python"); see the module docstring.
  """
  return _DEFAULT_BACKEND


def _resolve_backend(choice):
  if choice is None:
    return _DEFAULT_BACKEND
  if choice not in _BACKENDS:
    raise ValueError(
        f"backend must be 'native', 'python' or None, got {choice!r}.")
  if choice == "native" and _load_native() is None:
    raise RuntimeError(f"scann.torch: {_native_error}.")
  return choice


# --- The Python backend (scann-core 0.2.0) ------------------------------------

_NO_EXPORT = (
    "scann.torch searches on the Python backend can't be exported: the "
    "exported program would hold only a handle to a searcher in this "
    "process, not the index. The native backend (`pip install "
    f"scann-core-torch=={_scann_version}`) keeps the index in the module's "
    "buffers, so exported (and AOTInductor) programs carry it; see "
    "docs/integrations.md in scann-core. Or export the model without the "
    "search and load the index next to it with scann.torch.load_searcher().")

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
        "made in another process (programs using the Python backend don't "
        "carry the index; the native backend's do: pip install "
        "scann-core-torch).")
  return raw


def _num_live_searchers():
  """Python-backend searchers registered in this process (for tests)."""
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


# --- The native backend's index state -----------------------------------------

# The index as three tensors, the files SerializeToDirectory writes (as the
# op library takes them; see torch_op/scann_torch_ops.cc):
#   index_data     uint8 [total bytes]    the files' contents, concatenated
#   index_offsets  int64 [num_files + 1]  file i is data[offsets[i]:offsets[i+1]]
#   index_names    uint8 [...]            the file names, each followed by '\n'
_STATE = ("index_data", "index_offsets", "index_names")
_MANIFEST = "scann_assets.pbtxt"
_CONFIG = "scann_config.pb"
_DOCIDS = "scann_docids.pkl"


def _snapshot(pybind):
  """A pybind searcher's index as the three state tensors (CPU)."""
  with tempfile.TemporaryDirectory(prefix="scann_torch_") as d:
    pybind.serialize(d, relative_path=True)
    names = sorted(n for n in os.listdir(d)
                   if os.path.isfile(os.path.join(d, n)))
    sizes = [os.path.getsize(os.path.join(d, n)) for n in names]
    data = np.empty(sum(sizes), np.uint8)
    view = memoryview(data)
    pos = 0
    for name, size in zip(names, sizes):
      with open(os.path.join(d, name), "rb", buffering=0) as f:
        end = pos + size
        while pos < end:
          n = f.readinto(view[pos:end])
          if not n:
            raise IOError(f"scann.torch: {name} is shorter than expected.")
          pos += n
  offsets = np.zeros(len(names) + 1, np.int64)
  np.cumsum(sizes, out=offsets[1:])
  blob = "".join(n + "\n" for n in names).encode("utf-8")
  return (torch.from_numpy(data), torch.from_numpy(offsets),
          torch.from_numpy(np.frombuffer(blob, np.uint8).copy()))


def _check_state(data, offsets, names):
  """Validated CPU state tensors, and {file name: memoryview}."""
  for t, n in zip((data, offsets, names), _STATE):
    if not isinstance(t, torch.Tensor):
      raise TypeError(f"{n} must be a tensor, got {type(t).__name__}.")
  if (data.dim() != 1 or data.dtype != torch.uint8 or offsets.dim() != 1 or
      offsets.dtype != torch.int64 or names.dim() != 1 or
      names.dtype != torch.uint8):
    raise ValueError(
        "the index state must be index_data (uint8 [bytes]), index_offsets "
        "(int64 [num_files + 1]) and index_names (uint8 [bytes]) vectors.")
  data, offsets, names = (
      t.detach().to("cpu").contiguous() for t in (data, offsets, names))
  file_names = bytes(names.numpy()).decode("utf-8").split("\n")
  if file_names[-1] != "":
    raise ValueError("index_names must end with a newline.")
  file_names = file_names[:-1]
  off = offsets.tolist()
  if (len(off) != len(file_names) + 1 or off[0] != 0 or
      off[-1] != data.numel() or any(b < a for a, b in zip(off, off[1:]))):
    raise ValueError("index_offsets doesn't match index_data and index_names.")
  view = memoryview(data.numpy())
  files = {n: view[a:b] for n, a, b in zip(file_names, off, off[1:])}
  for required in (_MANIFEST, _CONFIG):
    if required not in files:
      raise ValueError(f"the index state has no {required}.")
  return (data, offsets, names), files


class _DocidsUnpickler(pickle.Unpickler):
  """Unpickles plain data only (str, bytes, numbers, lists, tuples, ...).

  scann_docids.pkl in an index state may come from a state_dict that
  torch.load(weights_only=True) loaded; unpickling it must not run code.
  """

  def find_class(self, module, name):
    raise pickle.UnpicklingError(
        f"scann.torch: the docids in the index state reference {module}."
        f"{name}; only plain data (str, bytes, numbers, lists, tuples) is "
        "loaded from index state.")


def _pybind_from_files(files):
  """A scann_ops_pybind searcher from the index state's files."""
  with tempfile.TemporaryDirectory(prefix="scann_torch_") as d:
    for name, content in files.items():
      if name == _DOCIDS or os.path.basename(name) != name:
        continue
      with open(os.path.join(d, name), "wb") as f:
        f.write(content)
    searcher = scann_ops_pybind.load_searcher(d)
  if _DOCIDS in files:
    docids = _DocidsUnpickler(io.BytesIO(files[_DOCIDS])).load()
    if len(docids) != searcher.size():
      raise ValueError(
          f"the index state has {len(docids)} docids, but the index has "
          f"{searcher.size()} datapoints.")
    searcher = scann_ops_pybind.ScannSearcher(searcher.searcher, docids)
  return searcher


def _new_shared_name():
  return f"scann_{uuid.uuid4().hex}"


def _release(ops, shared_name):
  try:
    ops.release(shared_name)
  except Exception:  # pylint: disable=broad-except
    pass  # e.g. during interpreter shutdown


def _generation(pybind):
  return getattr(pybind, "_generation", 0)


# --- The module -----------------------------------------------------------------


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

  Calling the module is search_batched().

  With the native backend (`backend == "native"`) the index is in the
  buffers index_data, index_offsets and index_names (in state_dict(),
  exported programs and pickles), always in host memory: .to() / .cuda()
  leave them there. With the Python backend it has no parameters or
  buffers.

  Attributes:
    searcher: the scann_ops_pybind.ScannSearcher. Use it for everything
      that isn't a search (upsert, delete, docids, ...); later searches
      through this module see the changes (with the native backend: eager
      searches, state_dict() and pickling copy them into the buffers
      first; call sync() before running compiled or exported code after a
      change). With the native backend it is loaded from the buffers on
      first use unless the Searcher was made around one.
    backend: "native" or "python".
  """

  def __init__(self, searcher, backend=None, *, _owned=False):  # pylint: disable=redefined-outer-name
    """Wraps a scann_ops_pybind searcher.

    Args:
      searcher: a scann_ops_pybind.ScannSearcher (from builder(),
        create_searcher() or load_searcher()); shared, not copied.
      backend: "native", "python", or None for backend().
      _owned: the native backend drops `searcher` once the index is in the
        buffers (builder(), create_searcher() and load_searcher() made it).
    """
    super().__init__()
    if not isinstance(searcher, scann_ops_pybind.ScannSearcher):
      raise TypeError(
          "scann.torch.Searcher wraps a scann_ops_pybind.ScannSearcher (from "
          "builder(), create_searcher() or load_searcher()), got "
          f"{type(searcher).__name__}.")
    self._backend = _resolve_backend(backend)
    # The default k is read once: rebalance(config) with another
    # num_neighbors isn't seen by searches with final_num_neighbors=None.
    self._default_k = _default_num_neighbors(searcher)
    self._size = searcher.size()
    if self._backend == "python":
      self._pybind = searcher
      self._handle = _register(searcher.searcher)
      weakref.finalize(self, _unregister, self._handle)
      return
    self._index_in_state_dict = True
    self._init_native()
    self._pybind = searcher
    self._set_state(*_snapshot(searcher), generation=_generation(searcher))
    if _owned:
      self._pybind = None

  def _init_native(self):
    """Per-object native state (also after unpickling)."""
    self._lock = threading.RLock()
    self._shared_name = _new_shared_name()
    ops = _load_native()
    finalizer = weakref.finalize(self, _release, ops, self._shared_name)
    finalizer.atexit = False

  def _set_state(self, data, offsets, names, generation=None):
    with self._lock:
      for name, t in zip(_STATE, (data, offsets, names)):
        self.register_buffer(name, t, persistent=self._index_in_state_dict)
      self._synced_generation = generation

  @classmethod
  def from_pybind(cls, searcher, backend=None):  # pylint: disable=redefined-outer-name
    """A Searcher over an existing scann_ops_pybind searcher (shared)."""
    return cls(searcher, backend)

  def to_pybind(self):
    """The underlying scann_ops_pybind.ScannSearcher (shared, not a copy)."""
    return self.searcher

  @property
  def backend(self):
    """"native" or "python"."""
    return self._backend

  @property
  def searcher(self):
    """The scann_ops_pybind.ScannSearcher (see the class docstring)."""
    if self._pybind is None:
      with self._lock:
        if self._pybind is None:
          _, files = _check_state(*(getattr(self, n) for n in _STATE))
          pybind = _pybind_from_files(files)
          self._synced_generation = _generation(pybind)
          self._pybind = pybind
    return self._pybind

  @property
  def default_num_neighbors(self):
    """k when final_num_neighbors is None."""
    return self._default_k

  @property
  def index_in_state_dict(self):
    """Native backend: whether state_dict() includes the index (default).

    Set it to False to leave the index out of state_dict() (e.g. of
    frequent training checkpoints); load_state_dict() of a state_dict
    without it keeps the index the module has. Exported programs and
    pickles carry the index either way.
    """
    return self._backend == "native" and self._index_in_state_dict

  @index_in_state_dict.setter
  def index_in_state_dict(self, value):
    if self._backend != "native":
      raise ValueError(
          "index_in_state_dict: only the native backend keeps the index in "
          "buffers.")
    self._index_in_state_dict = bool(value)
    for name in _STATE:
      if self._index_in_state_dict:
        self._non_persistent_buffers_set.discard(name)
      else:
        self._non_persistent_buffers_set.add(name)

  def sync(self):
    """Native backend: copies changes made through `searcher` to the buffers.

    Eager searches, state_dict() and pickling do this themselves; compiled
    and exported code doesn't (it runs no Python), so call sync() after
    changing the index (upsert, delete, rebalance through `searcher`)
    before running it. A no-op when nothing changed, and with the Python
    backend.
    """
    if self._backend != "native":
      return
    pybind = self._pybind
    if pybind is None or _generation(pybind) == self._synced_generation:
      return
    with self._lock:
      pybind = self._pybind
      generation = _generation(pybind)
      if pybind is not None and generation != self._synced_generation:
        self._set_state(*_snapshot(pybind), generation=generation)

  def _k(self, final_num_neighbors):
    k = _param(final_num_neighbors, "final_num_neighbors")
    if k == -1:
      return self._default_k
    if k <= 0:
      raise ValueError(
          f"final_num_neighbors must be > 0 (or None), got {k}.")
    return k

  def _native_state(self):
    """The index tensors to search (synced first, eagerly)."""
    # Compiled and exported code runs no Python here: it searches the
    # buffers as they are (see sync()).
    if torch.compiler.is_compiling():
      return self.index_data, self.index_offsets, self.index_names
    self.sync()
    with self._lock:  # all three from the same state
      return self.index_data, self.index_offsets, self.index_names

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
    k = self._k(final_num_neighbors)
    pre = _param(pre_reorder_num_neighbors, "pre_reorder_num_neighbors")
    leaves = _param(leaves_to_search, "leaves_to_search")
    if self._backend == "native":
      return torch.ops.scann.search(*self._native_state(), self._shared_name,
                                    q.detach(), k, pre, leaves)
    return torch.ops.scann_py.search(q.detach(), self._handle, k, pre, leaves)

  def _batched(self, queries, final_num_neighbors, pre_reorder_num_neighbors,
               leaves_to_search, parallel, batch_size, method):
    if not isinstance(queries, torch.Tensor):
      queries = torch.as_tensor(queries)
    if queries.dim() != 2:
      raise ValueError(f"{method}() expects 2-dimensional queries, got "
                       f"shape {tuple(queries.shape)}.")
    k = self._k(final_num_neighbors)
    pre = _param(pre_reorder_num_neighbors, "pre_reorder_num_neighbors")
    leaves = _param(leaves_to_search, "leaves_to_search")
    if self._backend == "native":
      return torch.ops.scann.search_batched(
          *self._native_state(), self._shared_name, queries.detach(), k, pre,
          leaves, parallel, batch_size)
    return torch.ops.scann_py.search_batched(queries.detach(), self._handle,
                                             k, pre, leaves, parallel,
                                             batch_size)

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
    if self._backend == "python":
      return (f"size={self._pybind.size()}, "
              f"default_num_neighbors={self._default_k}")
    size = self._pybind.size() if self._pybind is not None else self._size
    return ", ".join(
        ([f"size={size}"] if size is not None else []) +
        [f"default_num_neighbors={self._default_k}", "backend=native",
         f"index_bytes={self.index_data.numel()}"])

  # --- Moving, state_dict, pickling ---

  def _apply(self, fn, recurse=True):
    if self._backend == "native":
      # The index stays in host memory, where the search reads it: .to(),
      # .cuda(), .half(), share_memory() ... leave the buffers alone.
      return self
    return super()._apply(fn, recurse)

  def _save_to_state_dict(self, destination, prefix, keep_vars):
    if self._backend == "native" and not torch.compiler.is_compiling():
      self.sync()
    super()._save_to_state_dict(destination, prefix, keep_vars)

  def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):
    del local_metadata, missing_keys  # Unused.
    keys = [prefix + n for n in _STATE]
    present = [k for k in keys if k in state_dict]
    if strict:
      unexpected_keys.extend(
          k for k in state_dict if k.startswith(prefix) and k not in keys)
    if not present:
      # A state_dict without the index (scann-core 0.2.0, the Python
      # backend, index_in_state_dict=False): keep this module's index.
      return
    if len(present) != len(keys):
      error_msgs.append(
          f"scann.torch.Searcher: the state_dict has {present} but not "
          f"{sorted(set(keys) - set(present))}; the index needs all of "
          f"{list(_STATE)}.")
      return
    try:
      self._load_state(*(state_dict[k] for k in keys))
    except Exception as e:  # pylint: disable=broad-except
      error_msgs.append(
          f"scann.torch.Searcher ({prefix or 'the module'}): can't load the "
          f"index from the state_dict: {type(e).__name__}: {e}")

  def _load_state(self, data, offsets, names):
    """Replaces the index with one from state tensors (load_state_dict)."""
    (data, offsets, names), files = _check_state(data, offsets, names)
    config = scann_pb2.ScannConfig.FromString(bytes(files[_CONFIG]))
    if self._backend == "native":
      with self._lock:
        self._default_k = int(config.num_neighbors)
        self._size = None
        self._pybind = None  # loaded from the new state on first use
        self._set_state(data, offsets, names)
      return
    pybind = _pybind_from_files(files)
    handle = _register(pybind.searcher)
    old = self._handle
    self._pybind, self._handle = pybind, handle
    self._default_k = _default_num_neighbors(pybind)
    self._size = pybind.size()
    weakref.finalize(self, _unregister, handle)
    _unregister(old)

  def __getstate__(self):
    if self._backend != "native":
      raise TypeError(
          "a scann.torch.Searcher on the Python backend can't be pickled or "
          "deep-copied (so neither can a model holding one, e.g. with "
          "torch.save(model)): the index lives in the pybind searcher. "
          "Install scann-core-torch for the native backend, which can; or "
          "save the model's state_dict() and the index (serialize()) "
          "separately, and load the index with scann.torch.load_searcher().")
    self.sync()
    state = self.__dict__.copy()
    state["_buffers"] = dict(self._buffers)
    for name in ("_lock", "_pybind", "_shared_name", "_synced_generation"):
      state.pop(name, None)
    return state

  def __setstate__(self, state):
    super().__setstate__(state)
    self._pybind = None
    self._synced_generation = None
    if _DEFAULT_BACKEND == "python" or _load_native() is None:
      # Pickled with the native backend, unpickled where the default is the
      # Python backend (scann-core-torch not installed, or
      # SCANN_TORCH_BACKEND=python): the same index on the Python backend.
      _, files = _check_state(*(self._buffers[n] for n in _STATE))
      for n in _STATE:
        del self._buffers[n]
        self._non_persistent_buffers_set.discard(n)
      pybind = _pybind_from_files(files)
      self._backend = "python"
      self._pybind = pybind
      self._handle = _register(pybind.searcher)
      weakref.finalize(self, _unregister, self._handle)
      return
    self._init_native()


def builder(db, num_neighbors, distance_measure, backend=None):  # pylint: disable=redefined-outer-name
  """Creates a ScannBuilder that returns a scann.torch.Searcher on build().

  Args:
    db: the dataset; a 2-D tensor (any float dtype, any device) or array
      with one data point per row.
    num_neighbors: the default # neighbors the searcher will return per query.
    distance_measure: one of "squared_l2" or "dot_product".
    backend: "native", "python", or None for backend().

  Returns:
    scann_ops_pybind's ScannBuilder (tree(), score_ah(), autopilot(), ...),
    whose build() (also build(docids=...)) returns a scann.torch.Searcher.
  """
  choice = _resolve_backend(backend)
  b = scann_ops_pybind.builder(db, num_neighbors, distance_measure)
  build_pybind = b.builder_lambda

  def builder_lambda(db, config, training_threads, **kwargs):
    return Searcher(build_pybind(db, config, training_threads, **kwargs),
                    choice, _owned=True)

  return b.set_builder_lambda(builder_lambda)


def create_searcher(db, scann_config, training_threads=0, docids=None,
                    backend=None):  # pylint: disable=redefined-outer-name
  """Creates a scann.torch.Searcher from a dataset and a text config proto."""
  choice = _resolve_backend(backend)
  return Searcher(
      scann_ops_pybind.create_searcher(
          db, scann_config, training_threads, docids=docids),
      choice, _owned=True)


def load_searcher(artifacts_dir, assets_backcompat_shim=True, backend=None):  # pylint: disable=redefined-outer-name
  """Loads an index saved by serialize() (from any binding)."""
  choice = _resolve_backend(backend)
  return Searcher(
      scann_ops_pybind.load_searcher(artifacts_dir, assets_backcompat_shim),
      choice, _owned=True)
