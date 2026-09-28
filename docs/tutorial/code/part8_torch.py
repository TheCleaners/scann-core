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

"""Part 8, PyTorch: the part 3 index as a torch.nn.Module (scann.torch).

Needs PyTorch (pip install 'scann-core[torch]'). The export section needs
scann.torch's native backend (pip install scann-core-torch, or a CMake
build with -DSCANN_BUILD_TORCH_OP=ON), and is skipped without it. The GPU
section runs when torch sees a CUDA (or ROCm) device.
"""

import os
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import scann.torch as scann_torch

from tutorial_data import Timer, load_glove, recall

dataset, queries, true_neighbors = load_glove()
print(f"torch {torch.__version__}; scann.torch backend: "
      f"{scann_torch.backend()}")

# --- 1. The part 3 index, as a module ---------------------------------------
# The same builder as scann_ops_pybind; build() returns a scann.torch.Searcher.
# A float32 CPU tensor is read without a copy.
with Timer() as t:
  searcher = (scann_torch.builder(torch.from_numpy(dataset), 10, "dot_product")
              .tree(num_leaves=2000, num_leaves_to_search=100,
                    training_sample_size=250000)
              .score_ah(2, anisotropic_quantization_threshold=0.2)
              .reorder(100)
              .build())
print(f"built in {t.seconds:.1f} s: {searcher}")

q = torch.from_numpy(queries)
indices, distances = searcher.search_batched_parallel(q)
print(f"search_batched_parallel: {indices.dtype} {tuple(indices.shape)}, "
      f"{distances.dtype}; recall@10 {recall(indices.numpy(), true_neighbors):.4f}")

# The scann_ops_pybind searcher underneath, for everything that isn't a
# search (upsert, delete, docids, serialize), and for comparison.
pybind = searcher.searcher
p_idx, p_dist = pybind.search_batched_parallel(queries)
print("identical to the pybind searcher:",
      np.array_equal(indices.numpy(), p_idx) and
      np.array_equal(distances.numpy(), p_dist))

# The Python backend (scann-core 0.2.0's): same API, same results; the op
# calls the pybind searcher from Python.
py_searcher = scann_torch.Searcher(pybind, backend="python")
py_idx, _ = py_searcher.search_batched_parallel(q)
print("Python backend identical:", torch.equal(py_idx, indices))

# --- 2. A model that encodes, then searches ---------------------------------
# A real query tower is trained. This stand-in is built so that we know the
# right answers: the model's input is a GloVe query hidden behind a random
# rotation, and the encoder (one linear layer + normalization) undoes it.
rotation, _ = torch.linalg.qr(
    torch.randn(100, 100, generator=torch.Generator().manual_seed(0)))
features = q @ rotation


class QueryEncoder(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.proj = torch.nn.Linear(100, 100, bias=False)

  def forward(self, x):
    return torch.nn.functional.normalize(self.proj(x), dim=-1)


class Retriever(torch.nn.Module):
  """Features in, the 10 nearest GloVe words out."""

  def __init__(self, encoder, index):
    super().__init__()
    self.encoder = encoder
    self.index = index  # a Searcher is a submodule like any other

  def forward(self, features):
    return self.index.search_batched_parallel(self.encoder(features), 10)


encoder = QueryEncoder().eval()
with torch.no_grad():
  encoder.proj.weight.copy_(rotation)  # x @ rotation.T undoes the rotation
retriever = Retriever(encoder, searcher).eval()
compiled = torch.compile(retriever, fullgraph=True, dynamic=True)

with torch.no_grad(), Timer() as t:
  found = torch.cat([compiled(features[i:i + n])[0]
                     for i, n in ((0, 7), (7, 993), (1000, 9000))])
print(f"\ncompiled Retriever, 3 batch sizes, first calls: {t.seconds:.1f} s "
      "(mostly compiling)")
print(f"  recall@10 {recall(found.numpy(), true_neighbors):.4f}; "
      f"{(found != indices).sum().item()} of 100,000 results differ from "
      "searching the GloVe queries directly")

# --- 3. Queries on a GPU ----------------------------------------------------
# The encoder runs on the GPU; the search runs on the CPU (queries copied to
# host memory, results copied back); the index stays in host memory.
cuda = torch.cuda.is_available()
if cuda:
  gpu_retriever = Retriever(QueryEncoder().cuda().eval(), searcher).eval()
  with torch.no_grad():
    gpu_retriever.encoder.proj.weight.copy_(rotation)
  compiled_gpu = torch.compile(gpu_retriever, fullgraph=True, dynamic=True)
  gpu_features = features.cuda()
  with torch.no_grad():
    gpu_idx, gpu_dist = compiled_gpu(gpu_features)
  print(f"\nGPU ({torch.cuda.get_device_name()}): results on "
        f"{gpu_idx.device}, recall@10 "
        f"{recall(gpu_idx.cpu().numpy(), true_neighbors):.4f}; "
        f"{(gpu_idx.cpu() != indices).sum().item()} of 100,000 results differ")
  print("  index buffers stay on:", searcher.index_data.device
        if scann_torch.backend() == "native" else "(no buffers)")

# --- 4. What a call costs ---------------------------------------------------
# Median time per call, measured round-robin so that all columns see the
# same machine load. "encoder, then pybind": the model runs without the
# search, and the pybind searcher searches its output (a numpy array).
py_retriever = Retriever(encoder, py_searcher).eval()
py_compiled = torch.compile(py_retriever, fullgraph=True, dynamic=True)


def encode_then_pybind(model):
  def run(x):
    emb = model.encoder(x).cpu().numpy()
    return pybind.search_batched_parallel(emb)
  return run


def median_us(fns, reps, sync=False):
  """{name: median µs per call}, interleaving the functions' calls."""
  times = {name: [] for name in fns}
  for fn in fns.values():  # warm up (and compile, for a new batch size)
    for _ in range(3):
      fn()
  for _ in range(reps):
    for name, fn in fns.items():
      start = time.perf_counter()
      fn()
      if sync:
        torch.cuda.synchronize()
      times[name].append(time.perf_counter() - start)
  return {name: 1e6 * statistics.median(t) for name, t in times.items()}


def row(label, x, model, py_model, comp, py_comp, reps):
  fns = {"encoder, then pybind": lambda: encode_then_pybind(model)(x),
         "native": lambda: model(x),
         "python": lambda: py_model(x),
         "compiled native": lambda: comp(x),
         "compiled python": lambda: py_comp(x)}
  with torch.no_grad():
    us = median_us(fns, reps, sync=x.is_cuda)
  cells = "".join(f"{v:>10.0f}" if v >= 1000 else f"{v:>10.1f}"
                  for v in us.values())
  print(f"  {label:28}{cells}")


print("\nµs per call, median (Retriever = encoder + search_batched_parallel)")
print(f"  {'':28}{'pybind':>10}{'native':>10}{'python':>10}"
      f"{'comp. nat':>10}{'comp. py':>10}")
if scann_torch.backend() == "native":
  row("1 query, CPU", features[:1], retriever, py_retriever, compiled,
      py_compiled, 2000)
  row("1,000 queries, CPU", features[:1000], retriever, py_retriever,
      compiled, py_compiled, 200)
  if cuda:
    py_gpu = Retriever(gpu_retriever.encoder, py_searcher).eval()
    py_gpu_compiled = torch.compile(py_gpu, fullgraph=True, dynamic=True)
    row("1 query, GPU queries", gpu_features[:1], gpu_retriever, py_gpu,
        compiled_gpu, py_gpu_compiled, 2000)
    row("1,000 queries, GPU queries", gpu_features[:1000], gpu_retriever,
        py_gpu, compiled_gpu, py_gpu_compiled, 200)

# --- 5. Exporting the model (native backend) --------------------------------
# The native backend keeps the index in the Searcher's buffers, so it is in
# state_dict(), and torch.export captures it with the model.
if scann_torch.backend() != "native":
  print("\nnative backend not installed: skipping torch.export")
  sys.exit(0)

state = retriever.state_dict()
print(f"\nstate_dict: {list(state)}; index_data "
      f"{state['index.index_data'].numel() / 2**20:.1f} MiB")

try:
  torch.export.export(py_retriever, (features[:8],))
except Exception as e:  # pylint: disable=broad-except
  print(f"Python backend, torch.export: {type(e).__name__}: "
        f"{str(e).splitlines()[0][:72]}...")

with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, "retriever.pt2")
  with Timer() as t:
    program = torch.export.export(
        retriever, (features[:8],),
        dynamic_shapes={"features": {0: torch.export.Dim("batch")}})
    torch.export.save(program, path)
  print(f"exported and saved in {t.seconds:.1f} s: retriever.pt2, "
        f"{os.path.getsize(path) / 2**20:.1f} MiB")
  del program
  np.save(os.path.join(tmp, "features.npy"), features.numpy())
  # A fresh process: import scann.torch (it registers the ops), load, run.
  # The index comes from the .pt2 file only.
  child = """
import sys, time
import numpy as np, torch
import scann.torch
start = time.perf_counter()
model = torch.export.load(sys.argv[1]).module()
features = torch.from_numpy(np.load(sys.argv[2]))
indices, _ = model(features)
print(f"{time.perf_counter() - start:.1f}")
np.save(sys.argv[3], indices.numpy())
"""
  out = os.path.join(tmp, "indices.npy")
  seconds = subprocess.run(
      [sys.executable, "-c", child, path, os.path.join(tmp, "features.npy"),
       out], check=True, capture_output=True, text=True).stdout.split()[-1]
  loaded = np.load(out)
print(f"fresh process: loaded and searched 10,000 queries in {seconds} s; "
      f"recall@10 {recall(loaded, true_neighbors):.4f}; identical to "
      f"eager: {np.array_equal(loaded, retriever(features)[0].numpy())}")
