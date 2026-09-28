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

"""scann-core's native PyTorch ops (the scann-core-torch package).

Importing this package loads the op library and registers the ops
torch.ops.scann.search and torch.ops.scann.search_batched, with fake
(meta) implementations for torch.compile and torch.export. scann.torch
uses them when this package is installed (its "native" backend); use
scann.torch rather than the ops directly. Import scann.torch (or this
package) before torch.export.load() or loading an AOTInductor package of a
model that searches, so that the ops exist in the process.

The library is built against LibTorch's stable ABI only, so it runs with
any torch >= 2.10 (CPU, CUDA and ROCm builds) and any Python. A C++
program that runs an AOTInductor package without Python loads
`library_path` itself (dlopen) before loading the package.

See docs/integrations.md in scann-core.
"""

import os

import torch

from scann_torch_ops._version import __version__
from scann_torch_ops._version import built_with_torch

__all__ = ["__version__", "built_with_torch", "library_path", "release",
           "stats"]

library_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_scann_torch_ops.so")


def _torch_version():
  parts = torch.__version__.split("+")[0].split(".")
  try:
    return int(parts[0]), int(parts[1])
  except (IndexError, ValueError):
    return (0, 0)


if _torch_version() < (2, 10):
  raise ImportError(
      f"scann-core-torch needs torch >= 2.10 (LibTorch's stable ABI); this "
      f"is torch {torch.__version__}.")

torch.ops.load_library(library_path)


def _tag_cudagraph_unsafe():
  """Tags the searches as unsafe to capture in a CUDA graph.

  They copy to and from host memory and run on the CPU, so Inductor must
  keep them out of CUDA graphs (torch.compile(mode="reduce-overhead")), as
  it does for scann.torch's Python-backend ops, which are defined with
  this tag. LibTorch's stable ABI (2.10) can't define an op with tags, so
  the tag is added to the Python op objects, where Inductor reads it.
  """
  tag = getattr(torch.Tag, "cudagraph_unsafe", None)
  if tag is None:
    return
  for name in ("search", "search_batched"):
    op = getattr(torch.ops.scann, name).default
    tags = list(getattr(op, "_tags", ()))
    if tag not in tags:
      op._tags = tags + [tag]  # pylint: disable=protected-access


_tag_cudagraph_unsafe()


@torch.library.register_fake("scann::search")
def _search_fake(index_data, index_offsets, index_names, shared_name, query,
                 k, pre_reorder_num_neighbors, leaves_to_search):
  del index_data, index_offsets, index_names, shared_name  # Unused.
  del pre_reorder_num_neighbors, leaves_to_search  # Unused.
  return (query.new_empty((k,), dtype=torch.int64),
          query.new_empty((k,), dtype=torch.float32))


@torch.library.register_fake("scann::search_batched")
def _search_batched_fake(index_data, index_offsets, index_names, shared_name,
                         queries, k, pre_reorder_num_neighbors,
                         leaves_to_search, parallel, batch_size):
  del index_data, index_offsets, index_names, shared_name  # Unused.
  del pre_reorder_num_neighbors, leaves_to_search, parallel  # Unused.
  del batch_size  # Unused.
  shape = (queries.shape[0], k)
  return (queries.new_empty(shape, dtype=torch.int64),
          queries.new_empty(shape, dtype=torch.float32))


def stats():
  """The op library's searcher cache: {"live_searchers", "builds"}.

  live_searchers: searchers currently built and cached in this process.
  builds: searchers built since the library was loaded. A Searcher built
  once and searched many times (eagerly, compiled, exported) adds one.
  """
  live, builds = torch.ops.scann.stats(torch.empty(0)).tolist()
  return {"live_searchers": live, "builds": builds}


def release(shared_name):
  """Drops the cached searcher for shared_name; True if there was one.

  scann.torch.Searcher calls this when it is deleted. A later search with
  that name builds the searcher again.
  """
  return bool(torch.ops.scann.release(torch.empty(0), shared_name)[0])
