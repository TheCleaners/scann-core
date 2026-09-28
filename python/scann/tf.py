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

"""scann-core for TensorFlow code, with upstream's scann_ops API.

The same builder() / create_searcher() / ScannSearcher.search*() /
serialize_to_module() / searcher_from_module() API as upstream's
scann/scann_ops/py/scann_ops.py, on one of two backends:

  "op"      scann-core's TensorFlow op (the scann_tf_ops package, built
            from source with -DSCANN_BUILD_TF_OP=ON). Searches are graph
            ops; searchers save in SavedModels.
  "python"  tf.numpy_function around the pybind searcher; ships in the
            wheel. Works eagerly, in tf.function and tf.data, but can't be
            saved in a SavedModel (serialize_to_module() raises).

The op backend is used when scann_tf_ops is importable, the Python backend
otherwise. Both return the same results (int32 indices, float32
distances, the same padding and static shapes). See docs/tensorflow.md.

  import scann.tf   # upstream: from scann.scann_ops.py import scann_ops

  searcher = scann.tf.builder(db, 10, "dot_product").score_brute_force().build()
  indices, distances = searcher.search_batched(queries)  # int32, float32
  scann.tf.backend()  # "op" or "python"

The SCANN_TF_BACKEND environment variable, read when scann.tf is first
imported, overrides the choice: "op" (an ImportError if the op can't be
imported), "python", or "auto" (the default). get_backend(name) returns
either backend's module, with this API, regardless of the choice.

Importing `scann` doesn't import TensorFlow; importing this module does.
"""

import importlib
import importlib.util
import os
import warnings

# Imports TensorFlow, with an ImportError that says how to install it.
from scann import _tf_common
from scann import _tf_python

__all__ = [
    "BatchedSearchResult", "ScannSearcher", "SearchResult", "available_backends",
    "backend", "builder", "create_searcher", "from_pybind", "get_backend",
    "load_searcher", "searcher_from_module"
]

BACKEND_ENV_VAR = "SCANN_TF_BACKEND"
_OP_PACKAGE = "scann_tf_ops"


def get_backend(name):
  """The module implementing scann.tf's API with the named backend.

  Args:
    name: "op" (scann-core's TensorFlow op, the scann_tf_ops package) or
      "python" (tf.numpy_function around the pybind searcher).

  Returns:
    A module with builder(), create_searcher(), load_searcher(),
    from_pybind(), searcher_from_module(), ScannSearcher, SearchResult and
    BatchedSearchResult, and BACKEND, its name. Its searchers are that
    backend's, whichever backend scann.tf itself uses.

  Raises:
    ImportError: "op" when scann_tf_ops isn't installed or fails to import.
    ValueError: an unknown name.
  """
  if name == "python":
    return _tf_python
  if name == "op":
    if importlib.util.find_spec(_OP_PACKAGE) is None:
      raise ImportError(
          "scann.tf's op backend needs the scann_tf_ops package, which is "
          "not installed. " + _tf_common.OP_HOWTO,
          name=_OP_PACKAGE)
    return importlib.import_module(_OP_PACKAGE)
  raise ValueError(
      f"Unknown scann.tf backend {name!r}; expected 'op' or 'python'.")


def available_backends():
  """The backends that can be used here: ("op", "python") or ("python",)."""
  try:
    get_backend("op")
  except Exception:  # pylint: disable=broad-except
    return ("python",)
  return ("op", "python")


def _select():
  choice = os.environ.get(BACKEND_ENV_VAR, "").strip().lower() or "auto"
  if choice in ("op", "python"):
    return get_backend(choice)
  if choice != "auto":
    raise ValueError(f"{BACKEND_ENV_VAR}={choice!r}: expected 'auto', 'op' "
                     "or 'python'.")
  if importlib.util.find_spec(_OP_PACKAGE) is None:
    return _tf_python
  try:
    return get_backend("op")
  except Exception as e:  # pylint: disable=broad-except
    # Installed but broken, e.g. built for another TensorFlow or copied
    # from another scann-core version.
    warnings.warn(
        "scann.tf: scann-core's TensorFlow op (scann_tf_ops) is installed "
        f"but failed to import ({type(e).__name__}: {e}); using the Python "
        f"backend. Set {BACKEND_ENV_VAR}=python to silence this warning, or "
        f"{BACKEND_ENV_VAR}=op to make it an error.",
        RuntimeWarning,
        stacklevel=2)
    return _tf_python


_backend = _select()


def backend():
  """The backend scann.tf uses: "op" or "python"."""
  return _backend.BACKEND


SearchResult = _tf_common.SearchResult
BatchedSearchResult = _tf_common.BatchedSearchResult
ScannSearcher = _backend.ScannSearcher
builder = _backend.builder
create_searcher = _backend.create_searcher
load_searcher = _backend.load_searcher
from_pybind = _backend.from_pybind
searcher_from_module = _backend.searcher_from_module
