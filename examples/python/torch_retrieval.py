#!/usr/bin/env python3
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

"""End-to-end retrieval in PyTorch with scann.torch.

A toy encoder (an nn.Module) embeds a catalogue of items; scann-core indexes
the embeddings; a retrieval model whose forward() encodes queries and
searches the index is compiled whole with torch.compile(fullgraph=True).
Runs on a GPU if there is one: the encoder runs there, the search on the
CPU, and the results come back on the GPU (see docs/integrations.md).

Needs PyTorch (`pip install 'scann-core[torch]'`); exits with status 77
(reported as skipped by ctest) without it. Run with scann-core installed,
or from a CMake build tree:
  PYTHONPATH=<build>/python python examples/python/torch_retrieval.py
"""

import sys
import time

try:
  import torch
except ImportError:
  print("PyTorch is not installed; skipping (pip install 'scann-core[torch]')")
  sys.exit(77)

# `import scann` alone never imports PyTorch; `scann.torch` does.
import scann.torch as scann_torch

NUM_ITEMS, FEATURES, DIM, K = 20_000, 24, 32, 10
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


class Encoder(torch.nn.Module):
  """Features -> unit-length embeddings (untrained, for the example)."""

  def __init__(self):
    super().__init__()
    self.mlp = torch.nn.Sequential(
        torch.nn.Linear(FEATURES, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, DIM))

  def forward(self, features):
    return torch.nn.functional.normalize(self.mlp(features), dim=-1)


class Retriever(torch.nn.Module):
  """Encodes queries and returns their K nearest items."""

  def __init__(self, encoder, index):
    super().__init__()
    self.encoder = encoder
    self.index = index  # a scann.torch.Searcher, itself an nn.Module

  def forward(self, features):
    return self.index.search_batched_parallel(self.encoder(features), K)


encoder = Encoder().to(device).eval()

# --- Index the catalogue ----------------------------------------------------
item_features = torch.randn(NUM_ITEMS, FEATURES, device=device)
with torch.no_grad():
  item_embeddings = encoder(item_features)  # on `device`; copied to host
item_ids = [f"item-{i}" for i in range(NUM_ITEMS)]
index = (scann_torch.builder(item_embeddings, K, "dot_product")
         .tree(num_leaves=140, num_leaves_to_search=20)
         .score_ah(2, anisotropic_quantization_threshold=0.2)
         .reorder(100)
         .build(docids=item_ids))
print(f"indexed {index.searcher.size()} items on the CPU; encoder on {device}")

# --- A compiled model that encodes and searches -------------------------------
retriever = Retriever(encoder, index).eval()
compiled = torch.compile(retriever, fullgraph=True, dynamic=True)

# Queries: slightly perturbed features of known items, in batches of any
# size (dynamic=True: one compilation for all of them).
rng = torch.Generator(device=device).manual_seed(1)
targets = torch.randint(NUM_ITEMS, (512,), device=device, generator=rng)
queries = item_features[targets] + 0.05 * torch.randn(
    512, FEATURES, device=device, generator=rng)

start = time.perf_counter()
with torch.no_grad():
  batches = [compiled(queries[i:i + n])
             for i, n in ((0, 7), (7, 64), (71, 441))]
first = time.perf_counter() - start
indices = torch.cat([b[0] for b in batches])
scores = torch.cat([b[1] for b in batches])
assert indices.shape == (512, K) and indices.dtype == torch.int64
assert indices.device.type == torch.device(device).type
print(f"compiled and searched 512 queries in 3 batches: {first:.1f} s "
      "(mostly compilation)")

# Each query is a perturbed item: that item should be its top result.
hit = (indices[:, 0] == targets).float().mean().item()
print(f"query's own item ranked first: {hit:.1%}")
assert hit > 0.9, hit

# Recall against exact search over the same embeddings.
with torch.no_grad():
  exact = (encoder(queries) @ item_embeddings.T).topk(K).indices
recall = sum(len(set(a.tolist()) & set(b.tolist()))
             for a, b in zip(indices.cpu(), exact.cpu())) / exact.numel()
print(f"recall@{K} against exact search: {recall:.3f}")
assert recall > 0.9, recall

# Results are row indices; the pybind searcher (index.searcher) has the
# docids. -1 marks a missing result (only when the index has fewer than K
# points), so check before mapping: docids[-1] would be the last item.
docids = index.searcher.docids
top = [[docids[i] if i >= 0 else None for i in row] for row in indices[:2].tolist()]
print("query 0:", top[0][:3], scores[0, :3].tolist())
assert top[0][0] == item_ids[int(targets[0])]

# Eager and compiled search the same index; only the encoder's rounding can
# differ between them.
with torch.no_grad():
  eager_indices, _ = retriever(queries)
agree = (eager_indices == indices).float().mean().item()
print(f"compiled vs eager: {agree:.1%} of results identical")
assert agree > 0.95, agree
