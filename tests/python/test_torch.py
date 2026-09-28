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

"""scann.torch, the PyTorch wrapper around the pybind searcher.

Runs every check below once per backend, each in a subprocess with
SCANN_TORCH_BACKEND set: "python" always, "native" when scann-core-torch
(scann_torch_ops) is importable (required if SCANN_TEST_REQUIRE_TORCH_NATIVE
is set). Where the backends differ:

- `import scann` doesn't import PyTorch, and `import scann.torch` without
  PyTorch is a clear ImportError (checked in subprocesses).
- search / search_batched / search_batched_parallel (and calling the
  module) return exactly what the wrapped pybind searcher returns, as int64
  / float32 tensors on the queries' device (CPU, and CUDA when present),
  for brute force, AH and a tree; search parameters; input dtypes; queries
  that require grad; short rows padded with -1 / NaN; zero queries.
- torch.compile(fullgraph=True, dynamic=True) with the Inductor backend:
  one compilation for many batch sizes, same results as eager; a model
  that encodes and then searches, compiled whole.
- Concurrent calls from threads, eager and compiled.
- Python backend: torch.export raises (strict and non-strict), and so
  does the op with a handle that isn't registered in this process.
  Native backend: strict and non-strict export give the eager results.
- Python backend: deleting the Searcher unregisters it; pickling it is a
  clear error. Native backend: deleting it releases the op's cached
  searcher; pickle / deepcopy give an equivalent Searcher.
- Docids (and changes through the pybind searcher seen by searches),
  create_searcher / load_searcher / from_pybind.

tests/python/test_torch_native.py (python_torch_native) covers the native
backend in depth: parity for every index type, exported and AOTInductor
programs in fresh processes, state_dict and pickling round trips.

Without PyTorch the test exits with 77, which ctest reports as skipped
(SKIP_RETURN_CODE), unless SCANN_TEST_REQUIRE_TORCH is set.

Run with scann-core's build/python on PYTHONPATH:
  PYTHONPATH=build/python python tests/python/test_torch.py
"""

import concurrent.futures
import copy
import gc
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

SKIP = 77
DIM = 16


def check_import_isolation():
  """`import scann` leaves PyTorch alone; scann.torch says what's missing."""
  code = ("import sys, scann\n"
          "from scann.scann_ops.py import scann_ops_pybind\n"
          "assert 'torch' not in sys.modules, 'import scann imported torch'\n")
  subprocess.run([sys.executable, "-c", code], check=True)
  # A blocked import behaves like a missing package.
  code = ("import sys\n"
          "sys.modules['torch'] = None\n"
          "import scann\n"
          "try:\n"
          "  import scann.torch\n"
          "except ImportError as e:\n"
          "  assert 'scann-core[torch]' in str(e), str(e)\n"
          "else:\n"
          "  raise AssertionError('import scann.torch worked without torch')\n")
  subprocess.run([sys.executable, "-c", code], check=True)


def dataset(n, seed=0):
  return np.random.default_rng(seed).standard_normal((n, DIM)).astype(
      np.float32)


def expect_raises(exc, f, match):
  try:
    f()
  except exc as e:
    assert match in str(e), f"{match!r} not in {str(e)!r}"
    return
  raise AssertionError(f"expected {exc.__name__}({match!r})")


def assert_same(got, want, what, device="cpu"):
  """got: (tensor, tensor) from scann.torch; want: arrays from pybind."""
  gi, gd = got
  wi, wd = (np.asarray(w) for w in want)
  assert str(gi.dtype) == "torch.int64", (what, gi.dtype)
  assert str(gd.dtype) == "torch.float32", (what, gd.dtype)
  assert gi.device.type == device and gd.device.type == device, (what,
                                                                 gi.device)
  assert not gi.requires_grad and not gd.requires_grad, what
  assert tuple(gi.shape) == wi.shape == tuple(gd.shape), (what, gi.shape,
                                                          wi.shape)
  np.testing.assert_array_equal(gi.cpu().numpy(), wi.astype(np.int64),
                                err_msg=what)
  np.testing.assert_array_equal(gd.cpu().numpy(), wd, err_msg=what)


def searchers(st, torch, db):
  return {
      "brute_force": st.builder(torch.from_numpy(db), 10,
                                "dot_product").score_brute_force().build(),
      "ah_reorder": st.builder(db, 10, "squared_l2").score_ah(
          2, anisotropic_quantization_threshold=0.2).reorder(40).build(),
      "tree_ah": st.builder(db, 10, "dot_product").tree(
          20, 5, training_sample_size=len(db)).score_ah(2).reorder(50).build(),
  }


def check_eager(st, torch, db, queries, device):
  q = torch.from_numpy(queries).to(device)
  for name, s in searchers(st, torch, db).items():
    p = s.searcher  # the pybind searcher
    assert s.to_pybind() is p
    assert s.default_num_neighbors == 10
    assert_same(s.search(q[0]), p.search(queries[0]), f"{name} search",
                device)
    assert_same(s.search_batched(q), p.search_batched(queries),
                f"{name} search_batched", device)
    assert_same(s(q), p.search_batched(queries), f"{name} forward", device)
    assert_same(s.search_batched_parallel(q, batch_size=7),
                p.search_batched_parallel(queries, batch_size=7),
                f"{name} search_batched_parallel", device)
    params = dict(final_num_neighbors=4, pre_reorder_num_neighbors=20)
    if name == "tree_ah":
      params["leaves_to_search"] = 3
    want = p.search_batched(queries, **params)
    assert want[0].shape == (len(queries), 4)
    assert_same(s.search_batched(q, **params), want, f"{name} params", device)
    assert_same(s.search_batched_parallel(q, **params),
                p.search_batched_parallel(queries, **params),
                f"{name} parallel params", device)
    assert_same(s.search(q[1], **params), p.search(queries[1], **params),
                f"{name} search params", device)
    # -1 means the default, as in scann_ops_pybind.
    assert_same(s.search_batched(q, -1, -1, -1), p.search_batched(queries),
                f"{name} -1", device)
    # Other float dtypes are converted to float32; numpy arrays work too.
    for dtype in (torch.float64, torch.float16, torch.bfloat16):
      as_f32 = q.to(dtype).float().cpu().numpy()
      assert_same(s.search_batched(q.to(dtype)), p.search_batched(as_f32),
                  f"{name} {dtype}", device)
    assert_same(s.search_batched(queries), p.search_batched(queries),
                f"{name} numpy", "cpu")
    # Queries that require grad are fine; the results don't require grad.
    g = q.clone().requires_grad_()
    assert_same(s.search_batched(g), p.search_batched(queries),
                f"{name} requires_grad", device)
    with torch.inference_mode():
      assert_same(s.search_batched(q), p.search_batched(queries),
                  f"{name} inference_mode", device)
    # Non-contiguous queries.
    wide = torch.from_numpy(np.repeat(queries, 2, axis=1)).to(device)[:, ::2]
    assert_same(s.search_batched(wide), p.search_batched(queries),
                f"{name} strided", device)
    # Zero queries.
    i, d = s.search_batched(q[:0])
    assert tuple(i.shape) == (0, 10) and tuple(d.shape) == (0, 10)
    assert i.device.type == device
    # Errors.
    expect_raises(ValueError, lambda: s.search(q), "1-dimensional")
    expect_raises(ValueError, lambda: s.search_batched(q[0]), "2-dimensional")
    expect_raises(ValueError, lambda: s.search_batched(q, 0), "> 0")
    expect_raises(TypeError, lambda: s.search_batched(q, 2.5), "int or None")
    expect_raises(Exception, lambda: s.search_batched(q[:, :3]),
                  "dimensionality")
  assert s.backend == st.backend(), (s.backend, st.backend())


def check_padding(st, torch, device):
  """Fewer points than k: index -1, distance NaN, always k wide."""
  db = dataset(3)
  s = st.builder(db, 10, "dot_product").score_brute_force().build()
  p = s.searcher
  q = dataset(4, seed=1)
  qt = torch.from_numpy(q).to(device)
  for i, d in (s.search_batched(qt), s.search_batched_parallel(qt)):
    assert tuple(i.shape) == (4, 10) and tuple(d.shape) == (4, 10)
    assert (i[:, 3:] == -1).all() and torch.isnan(d[:, 3:]).all()
    pi, pd = p.search_batched(q)  # the pybind searcher: 3 wide by default
    np.testing.assert_array_equal(i[:, :3].cpu().numpy(), pi)
    np.testing.assert_array_equal(d[:, :3].cpu().numpy(), pd)
  i, d = s.search_batched(qt, final_num_neighbors=5)
  assert tuple(i.shape) == (4, 5)
  pi, pd = p.search_batched(q, final_num_neighbors=5)  # pybind: 0 / NaN
  assert (pi[:, 3:] == 0).all()
  np.testing.assert_array_equal(i.cpu().numpy(),
                                np.where(np.isnan(pd), -1, pi.astype(np.int64)))
  i, d = s.search(qt[0])
  assert tuple(i.shape) == (10,)
  assert (i[3:] == -1).all() and torch.isnan(d[3:]).all()
  assert_same((i[:3], d[:3]), p.search(q[0]), "search padded", device)
  # An empty index: every entry is padding.
  empty = st.create_searcher(
      np.zeros((0, DIM), np.float32),
      st.builder(db, 10, "dot_product").score_brute_force().create_config())
  i, d = empty.search_batched(qt, 2)
  assert (i == -1).all() and torch.isnan(d).all() and tuple(i.shape) == (4, 2)


def make_model(torch, searcher):
  """A model whose forward encodes, then searches."""

  class Retrieval(torch.nn.Module):

    def __init__(self):
      super().__init__()
      g = torch.Generator().manual_seed(0)
      self.encoder = torch.nn.Linear(8, DIM)
      with torch.no_grad():
        self.encoder.weight.copy_(torch.randn(DIM, 8, generator=g))
        self.encoder.bias.zero_()
      self.index = searcher

    def forward(self, features):
      emb = torch.nn.functional.normalize(self.encoder(features), dim=-1)
      indices, distances = self.index.search_batched(emb, 5)
      return indices, distances, emb

  return Retrieval()


def check_compile(st, torch, db, device):
  from torch._dynamo.testing import CompileCounterWithBackend  # pylint: disable=g-import-not-at-top
  s = st.builder(db, 10, "dot_product").score_ah(2).reorder(40).build()
  p = s.searcher

  # The searcher alone, compiled: dynamic batch sizes, one compilation.
  counter = CompileCounterWithBackend("inductor")
  f = torch.compile(lambda q: s.search_batched_parallel(q, 7),
                    backend=counter, fullgraph=True, dynamic=True)
  for n in (2, 5, 9, 33, 64):
    q = dataset(n, seed=n)
    assert_same(f(torch.from_numpy(q).to(device)),
                p.search_batched_parallel(q, 7), f"compiled n={n}", device)
  assert counter.frame_count == 1, counter.frame_count
  # 1 and 0 are specialized by dynamic shapes: at most one more compilation
  # each.
  for n in (1, 0):
    q = dataset(n, seed=3)
    i, _ = f(torch.from_numpy(q).to(device))
    assert tuple(i.shape) == (n, 7)
  assert counter.frame_count <= 3, counter.frame_count

  one = torch.compile(lambda q: s.search(q, 3), fullgraph=True)
  q = dataset(2, seed=5)
  assert_same(one(torch.from_numpy(q[0]).to(device)), p.search(q[0], 3),
              "compiled search", device)

  # A model that encodes and searches, compiled whole.
  model = make_model(torch, s).to(device)
  counter = CompileCounterWithBackend("inductor")
  compiled = torch.compile(model, backend=counter, fullgraph=True,
                           dynamic=True)
  for n in (3, 17, 40):
    feats = torch.randn(n, 8, generator=torch.Generator().manual_seed(n))
    feats = feats.to(device)
    ci, cd, emb = compiled(feats)
    assert ci.device.type == device and ci.dtype == torch.int64
    assert tuple(ci.shape) == (n, 5)
    # The same neighbours as the pybind searcher for the same embeddings.
    assert_same((ci, cd), p.search_batched(emb.detach().cpu().numpy(), 5),
                f"compiled model n={n}", device)
    ei, _, _ = model(feats)
    # Inductor may round the embeddings differently from eager; the
    # neighbours agree nearly everywhere.
    agree = (ei.cpu() == ci.cpu()).float().mean().item()
    assert agree > 0.9, agree
  assert counter.frame_count == 1, counter.frame_count
  if device == "cuda":
    # CUDA graphs: the op is tagged unsafe to capture, so Inductor leaves it
    # out of the graphs.
    graphed = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    feats = torch.randn(6, 8, device=device)
    for _ in range(3):
      gi, gd, emb = graphed(feats)
      assert_same((gi, gd), p.search_batched(emb.detach().cpu().numpy(), 5),
                  "reduce-overhead", device)
  # Training mode: the encoder gets gradients; the search passes none.
  feats = torch.randn(4, 8, device=device, requires_grad=True)
  ci, cd, emb = compiled(feats)
  assert not cd.requires_grad
  emb.sum().backward()
  assert model.encoder.weight.grad is not None


def check_threads(st, torch, db):
  s = st.builder(db, 1, "squared_l2").score_brute_force().build()
  n = len(db)
  compiled = torch.compile(lambda q: s.search_batched(q)[0][:, 0],
                           fullgraph=True, dynamic=True)
  compiled(torch.from_numpy(db[:2]))  # compile once before the threads

  def worker(seed):
    rng = np.random.default_rng(seed)
    for _ in range(30):
      rows = rng.integers(n, size=int(rng.integers(2, 20)))
      q = torch.from_numpy(db[rows])
      np.testing.assert_array_equal(compiled(q).numpy(), rows)
      np.testing.assert_array_equal(
          s.search_batched_parallel(q)[0][:, 0].numpy(), rows)
      i = int(rng.integers(n))
      assert int(s.search(torch.from_numpy(db[i]))[0][0]) == i
    return True

  with concurrent.futures.ThreadPoolExecutor(8) as pool:
    assert all(pool.map(worker, range(16)))


def check_export(st, torch, db):
  s = st.builder(db, 5, "dot_product").score_brute_force().build()
  model = make_model(torch, s)
  feats = torch.randn(3, 8)
  if st.backend() == "native":
    want = model(feats)
    for strict in (True, False):
      ep = torch.export.export(model, (feats,), strict=strict)
      got = ep.module()(feats)
      for g, w in zip(got, want):
        assert torch.equal(g, w) or torch.allclose(g, w, equal_nan=True), (
            strict, g, w)
    return
  for strict in (True, False):
    expect_raises(Exception,
                  lambda: torch.export.export(model, (feats,), strict=strict),
                  "scann-core-torch")
  # The op with a handle that isn't registered here (a graph from another
  # process, or a deleted searcher).
  expect_raises(RuntimeError,
                lambda: torch.ops.scann_py.search_batched(
                    feats, 12345, 5, -1, -1, False, 0),
                "no searcher with handle 12345")


def check_lifetime(st, torch, db):
  if st.backend() == "native":
    check_lifetime_native(st, torch, db)
    return
  gc.collect()
  before = st._num_live_searchers()  # pylint: disable=protected-access
  s = st.builder(db, 5, "dot_product").score_brute_force().build()
  model = make_model(torch, s)
  compiled = torch.compile(model, fullgraph=True, dynamic=True)
  compiled(torch.randn(3, 8))
  assert st._num_live_searchers() == before + 1  # pylint: disable=protected-access
  handle = s._handle  # pylint: disable=protected-access
  # Pickling (torch.save(model), copy.deepcopy) is a clear error.
  expect_raises(TypeError, lambda: pickle.dumps(s), "serialize()")
  expect_raises(TypeError, lambda: copy.deepcopy(model), "serialize()")
  # Deleting the Searcher (and everything holding it) frees it, even with
  # compiled code around.
  del s, model, compiled
  gc.collect()
  assert st._num_live_searchers() == before  # pylint: disable=protected-access
  expect_raises(RuntimeError,
                lambda: torch.ops.scann_py.search(
                    torch.zeros(DIM), handle, 5, -1, -1),
                "no searcher with handle")


def check_lifetime_native(st, torch, db):
  import scann_torch_ops  # pylint: disable=g-import-not-at-top
  gc.collect()
  before = scann_torch_ops.stats()["live_searchers"]
  s = st.builder(db, 5, "dot_product").score_brute_force().build()
  model = make_model(torch, s)
  compiled = torch.compile(model, fullgraph=True, dynamic=True)
  feats = torch.randn(3, 8)
  compiled(feats)
  # Eager results (Inductor may round the encoder's output differently).
  want = model(feats)
  assert scann_torch_ops.stats()["live_searchers"] == before + 1
  # Deep copies (of the model) and pickles (of the searcher; the model's
  # class is local to this test) work, and search the same index.
  copied = copy.deepcopy(model)
  got = copied(feats)
  assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
  assert copied.index._shared_name != s._shared_name  # pylint: disable=protected-access
  unpickled = pickle.loads(pickle.dumps(s))
  q = torch.from_numpy(db[:7])
  for g, w in zip(unpickled.search_batched(q), s.search_batched(q)):
    assert torch.equal(g, w)
  assert unpickled._shared_name != s._shared_name  # pylint: disable=protected-access
  del copied, unpickled
  # Deleting the Searcher (and everything holding it) releases the op's
  # cached searcher, even with compiled code around.
  del s, model, compiled, got
  gc.collect()
  assert scann_torch_ops.stats()["live_searchers"] == before, (
      scann_torch_ops.stats(), before)


def check_docids(st, torch, db, queries):
  """Results are indices; the pybind searcher maps them to docids."""
  docids = [f"d{i}" for i in range(len(db))]
  s = st.builder(db, 5, "squared_l2").score_brute_force().build(docids=docids)
  p = s.searcher
  q = torch.from_numpy(queries)
  idx, dist = s.search_batched(q)
  want_docids, want_dist = p.search_batched(queries)
  assert [[p.docids[i] for i in row] for row in idx.tolist()] == want_docids
  np.testing.assert_array_equal(dist.numpy(), want_dist)
  # Updates through the pybind searcher are seen by the wrapper.
  p.delete(["d0", "d1"])
  p.upsert(["new"], db[:1] + 100)
  idx, _ = s.search_batched(q)
  assert [[p.docids[i] for i in row] for row in idx.tolist()] == (
      p.search_batched(queries)[0])
  i, _ = s.search(torch.from_numpy(db[0] + 100))
  assert int(i[0]) == p.docid_to_id["new"]


def check_create_and_load(st, torch, db, queries):
  config = st.builder(db, 5, "dot_product").score_ah(2).create_config()
  s = st.create_searcher(torch.from_numpy(db), config, training_threads=2)
  assert isinstance(s, st.Searcher) and isinstance(s, torch.nn.Module)
  want = s.searcher.search_batched(queries)
  assert_same(s.search_batched(torch.from_numpy(queries)), want,
              "create_searcher")
  with tempfile.TemporaryDirectory() as d:
    s.serialize(d)
    loaded = st.load_searcher(d)
    assert_same(loaded.search_batched(torch.from_numpy(queries)), want,
                "load_searcher")
    from scann.scann_ops.py import scann_ops_pybind  # pylint: disable=g-import-not-at-top
    wrapped = st.Searcher.from_pybind(scann_ops_pybind.load_searcher(d))
    assert_same(wrapped.search_batched(torch.from_numpy(queries)), want,
                "from_pybind")
  expect_raises(TypeError, lambda: st.Searcher(s.searcher.searcher),
                "scann_ops_pybind.ScannSearcher")
  assert "default_num_neighbors=5" in repr(s)
  assert not list(s.parameters())
  if st.backend() == "native":
    assert [n for n, _ in s.named_buffers()] == [
        "index_data", "index_offsets", "index_names"]
  else:
    assert not list(s.buffers())
  assert s.to("cpu") is s


def main():
  backend = os.environ.get("SCANN_TORCH_TEST_BACKEND")
  if backend is None:
    check_import_isolation()
  try:
    import torch  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    if os.environ.get("SCANN_TEST_REQUIRE_TORCH"):
      raise
    print(f"SKIPPED: torch is not importable ({e})")
    sys.exit(SKIP)
  if backend is None:
    # Every check, once per backend, each in a fresh process.
    import scann.torch as st  # pylint: disable=g-import-not-at-top
    backends = ["python"]
    if st._load_native() is not None:  # pylint: disable=protected-access
      backends.append("native")
    elif os.environ.get("SCANN_TEST_REQUIRE_TORCH_NATIVE"):
      raise AssertionError(st._native_error)  # pylint: disable=protected-access
    for b in backends:
      env = dict(os.environ, SCANN_TORCH_BACKEND=b, SCANN_TORCH_TEST_BACKEND=b)
      subprocess.run([sys.executable, __file__], env=env, check=True)
    if len(backends) == 1:
      print("(native backend not tested: scann-core-torch isn't importable: "
            f"{st._native_error})")  # pylint: disable=protected-access
    return
  import scann.torch as st  # pylint: disable=g-import-not-at-top
  assert st.backend() == backend, (st.backend(), backend)

  db = dataset(2000)
  queries = dataset(50, seed=1)
  devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
  for device in devices:
    check_eager(st, torch, db, queries, device)
    check_padding(st, torch, device)
    check_compile(st, torch, db, device)
  check_threads(st, torch, db)
  check_export(st, torch, db)
  check_lifetime(st, torch, db)
  check_docids(st, torch, db, queries)
  check_create_and_load(st, torch, db, queries)
  gpu = (f", CUDA: {torch.cuda.get_device_name(0)}" if len(devices) > 1 else
         ", no CUDA device: GPU cases not run")
  print(f"PASSED, {backend} backend (torch {torch.__version__}, Python "
        f"{sys.version.split()[0]}{gpu})")


if __name__ == "__main__":
  main()
