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

"""scann.torch's native backend (scann-core-torch, torch_op/).

- Parity: for every index type tf_op's test covers (brute force float /
  int8 / bfloat16, AH, trees with each, SOAR), every parameter set, and
  search / search_batched / search_batched_parallel, the native backend
  returns exactly (bit for bit) what the Python backend returns for the
  same pybind searcher; SOAR + AH + reordering to the last bit of a
  distance, which varies between identical pybind searches too. Also
  float16 / bfloat16 / float64 / int queries, short rows, zero queries,
  an empty index and trees with every point deleted.
- torch.compile(fullgraph=True, dynamic=True): one compilation for many
  batch sizes, the eager results.
- torch.export (strict and non-strict), saved and loaded in a fresh
  process, and an AOTInductor package (CPU; and GPU when there is one and
  Inductor can compile for it) loaded in a fresh process: the eager
  results, with the index built once per process.
- CUDA / ROCm queries: results on the queries' device; model.to("cuda")
  leaves the index in host memory; compiled with mode="reduce-overhead".
- state_dict: contains the index; loads into another Searcher (native or
  Python backend) and into a fresh process; a 0.2.0-era state_dict (no
  index) loads with strict=True and keeps the index; index_in_state_dict
  = False; an incomplete index is an error.
- Stale state: after load_state_dict() of another index, or changes
  through the pybind searcher (then sync() for compiled code), searches use
  the new index (rebuilt); unchanged state is never rebuilt.
- Pickling: torch.save(model) / torch.load in a fresh process, also one
  without the native backend (SCANN_TORCH_BACKEND=python: the Python
  backend with the same index).
- Errors (bad queries, damaged index tensors, bad arguments to the raw ops)
  are RuntimeErrors, never crashes.
- Concurrency: first searches from 8 threads build once; eager and
  compiled searches from 8 threads.

Exits with 77 (ctest: skipped) without torch, unless SCANN_TEST_REQUIRE_TORCH
is set, or without scann_torch_ops (built with -DSCANN_BUILD_TORCH_OP=ON),
unless SCANN_TEST_REQUIRE_TORCH_NATIVE is set (as ctest does in builds with
the op).

Run with the build's python directory on PYTHONPATH:
  PYTHONPATH=build/python python tests/python/test_torch_native.py
"""

import concurrent.futures
import gc
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np

SKIP = 77
DIM, LEAVES = 8, 12
CONFIGS = ("bf", "bf_int8", "bf_bf16", "ah", "tree_bf", "tree_bf_int8",
           "tree_bf_bf16", "tree_ah", "tree_ah_no_reorder", "tree_ah_soar")
# (final_num_neighbors, pre_reorder_num_neighbors, leaves_to_search)
PARAMS = ((None, None, None), (5, None, None), (5, 50, LEAVES), (3, 20, 2))


def dataset(n, seed=0, dim=DIM):
  return np.random.default_rng(seed).standard_normal((n, dim)).astype(
      np.float32)


def pybind_builder(scann, db, cfg, k=10):
  from scann.scann_ops.py.scann_builder import ReorderType  # pylint: disable=g-import-not-at-top
  b = scann.scann_ops_pybind.builder(db, k, "dot_product")
  tree = lambda **kw: b.tree(LEAVES, 4, training_sample_size=len(db), **kw)
  return {
      "bf": lambda: b.score_brute_force(),
      "bf_int8": lambda: b.score_brute_force(ReorderType.INT8),
      "bf_bf16": lambda: b.score_brute_force(ReorderType.BFLOAT16),
      "ah": lambda: b.score_ah(2).reorder(30),
      "tree_bf": lambda: tree().score_brute_force(),
      "tree_bf_int8": lambda: tree().score_brute_force(ReorderType.INT8),
      "tree_bf_bf16": lambda: tree().score_brute_force(ReorderType.BFLOAT16),
      "tree_ah": lambda: tree().score_ah(2).reorder(30),
      "tree_ah_no_reorder": lambda: tree().score_ah(2),
      "tree_ah_soar": lambda: tree(soar_lambda=1.5).score_ah(2).reorder(30),
  }[cfg]()


def assert_same(got, want, what, exact=True):
  """got, want: (indices, distances) tensors from scann.torch.

  exact=False for SOAR + AH + reordering, whose distances vary in the last
  bit between identical searches with the pybind searcher itself (so
  neighbors with nearly equal distances may swap places): the same
  neighbors per row, distances equal to 1e-6.
  """
  gi, gd = got
  wi, wd = want
  assert gi.dtype == wi.dtype and gd.dtype == wd.dtype, (what, gi.dtype)
  assert gi.device == wi.device and gd.device == wd.device, (what, gi.device,
                                                             wi.device)
  assert gi.shape == wi.shape and gd.shape == wd.shape, (what, gi.shape,
                                                         wi.shape)
  gi, gd, wi, wd = (t.cpu().numpy() for t in (gi, gd, wi, wd))
  if exact:
    np.testing.assert_array_equal(gi, wi, err_msg=what)
    np.testing.assert_array_equal(gd, wd, err_msg=what)  # NaN == NaN here
  else:
    np.testing.assert_array_equal(np.sort(gi, axis=-1), np.sort(wi, axis=-1),
                                  err_msg=what)
    np.testing.assert_allclose(gd, wd, rtol=1e-6, err_msg=what)


def expect_error(f, match, what, exc=RuntimeError):
  try:
    f()
  except exc as e:
    assert match in str(e), (what, match, str(e))
    return
  raise AssertionError(f"{what}: expected {exc.__name__} matching {match!r}")


def builds(ops):
  return ops.stats()["builds"]


def child(mode, *args):
  """Runs this file in a fresh process (mode: see child_main)."""
  r = subprocess.run([sys.executable, __file__, "--child", mode, *args],
                     capture_output=True, text=True)
  out = "\n".join(l for l in (r.stdout + r.stderr).splitlines()
                  if not l.startswith(("I0000", "W0000", "WARNING: All log")))
  if r.returncode != 0:
    raise AssertionError(f"child {mode} failed ({r.returncode}):\n{out}")
  return out


def model_class(torch):
  """The Retrieval model class, defined once torch is imported.

  A module-level class (__main__.Retrieval), so that pickled models load in
  the child processes, which run this file as __main__ too.
  """
  if "Retrieval" in globals():
    return globals()["Retrieval"]

  class Retrieval(torch.nn.Module):
    """A model that encodes, then searches (in parallel batches)."""

    def __init__(self, searcher, k, return_embeddings=False):
      super().__init__()
      g = torch.Generator().manual_seed(0)
      self.encoder = torch.nn.Linear(6, DIM)
      with torch.no_grad():
        self.encoder.weight.copy_(torch.randn(DIM, 6, generator=g))
        self.encoder.bias.zero_()
      self.index = searcher
      self.k = k
      self.return_embeddings = return_embeddings

    def forward(self, features):
      emb = self.encoder(features)
      indices, distances = self.index.search_batched_parallel(
          emb, self.k, batch_size=4)
      if self.return_embeddings:
        return indices, distances, emb
      return indices, distances

  Retrieval.__qualname__ = "Retrieval"
  globals()["Retrieval"] = Retrieval
  return Retrieval


def make_model(torch, searcher, k=5, return_embeddings=False):
  return model_class(torch)(searcher, k, return_embeddings)


# --- Checks -------------------------------------------------------------------


def check_parity(torch, st, scann, db, queries, device):
  q = torch.from_numpy(queries).to(device)
  for cfg in CONFIGS:
    p = pybind_builder(scann, db, cfg).build()
    native = st.Searcher(p, backend="native")
    python = st.Searcher(p, backend="python")
    assert native.backend == "native" and python.backend == "python"
    exact = "soar" not in cfg
    for k, pre, leaves in PARAMS:
      what = f"{cfg} k={k} pre={pre} leaves={leaves} {device}"
      for method, kw in (("search_batched", {}),
                         ("search_batched_parallel", {"batch_size": 3}),
                         ("search_batched_parallel", {})):
        assert_same(
            getattr(native, method)(q, k, pre, leaves, **kw),
            getattr(python, method)(q, k, pre, leaves, **kw),
            f"{what} {method} {kw}", exact)
      for i in (0, 7):
        assert_same(native.search(q[i], k, pre, leaves),
                    python.search(q[i], k, pre, leaves),
                    f"{what} search {i}", exact)
    # Query dtypes are converted to float32 the same way.
    for dtype in (torch.float16, torch.bfloat16, torch.float64, torch.int32):
      assert_same(native(q.to(dtype)), python(q.to(dtype)),
                  f"{cfg} {dtype} {device}", exact)
    assert_same(native(q[:0]), python(q[:0]), f"{cfg} zero queries {device}")
  print(f"parity ({device}): {len(CONFIGS)} configs x {len(PARAMS)} "
        "parameter sets x search/search_batched/search_batched_parallel, "
        "and query dtypes: identical to the Python backend")


def check_edge_cases(torch, st, scann, db, queries):
  q = torch.from_numpy(queries[:4])
  small = st.builder(db[:3], 10, "dot_product", backend="native") \
      .score_brute_force().build()
  i, d = small.search_batched(q, 5)
  assert tuple(i.shape) == (4, 5) and (i[:, 3:] == -1).all()
  assert torch.isnan(d[:, 3:]).all() and not torch.isnan(d[:, :3]).any()
  i, d = small.search(q[0])
  assert tuple(i.shape) == (10,) and (i[3:] == -1).all()

  empty = st.create_searcher(
      np.zeros((0, DIM), np.float32),
      scann.scann_ops_pybind.builder(db, 10, "dot_product")
      .score_brute_force().create_config(), backend="native")
  i, d = empty.search_batched(q, 2)
  assert (i == -1).all() and torch.isnan(d).all() and tuple(i.shape) == (4, 2)

  for cfg in ("tree_bf", "tree_bf_int8", "tree_bf_bf16", "tree_ah_no_reorder",
              "tree_ah_soar", "bf"):
    p = pybind_builder(scann, db[:600], cfg).build(
        docids=[f"d{i}" for i in range(600)])
    p.delete([f"d{i}" for i in range(600)])
    s = st.Searcher(p, backend="native")
    i, d = s.search_batched(q[:3], 5, 50, LEAVES)
    assert tuple(i.shape) == (3, 5) and (i == -1).all(), cfg
    assert torch.isnan(d).all(), cfg
    i, _ = s.search(q[0], 5, 50, LEAVES)
    assert (i == -1).all(), cfg
  print("edge cases: short rows, empty index, all-deleted trees")


def check_compile(torch, st, scann, db, device):
  from torch._dynamo.testing import CompileCounterWithBackend  # pylint: disable=g-import-not-at-top
  s = st.Searcher(pybind_builder(scann, db, "tree_ah").build(), "native")
  model = make_model(torch, s).to(device)
  assert s.index_data.device.type == "cpu"  # .to() leaves the index alone
  counter = CompileCounterWithBackend("inductor")
  compiled = torch.compile(model, backend=counter, fullgraph=True,
                           dynamic=True)
  with torch.no_grad():
    for n in (3, 17, 40, 2):
      feats = torch.randn(n, 6, generator=torch.Generator().manual_seed(n))
      feats = feats.to(device)
      ci, cd = compiled(feats)
      assert ci.device.type == device and tuple(ci.shape) == (n, 5)
      emb = model.encoder(feats)
      assert_same((ci, cd), s.search_batched_parallel(emb, 5, batch_size=4),
                  f"compiled n={n} {device}")
  assert counter.frame_count == 1, counter.frame_count
  if device == "cuda":
    graphed = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    feats = torch.randn(6, 6, device=device)
    with torch.no_grad():
      want = model(feats)
      for _ in range(3):
        assert_same(graphed(feats), want, "reduce-overhead")
  print(f"torch.compile ({device}): fullgraph, dynamic, one compilation")


def check_export_and_aoti(torch, st, scann, ops, db, tmp):
  s = st.Searcher(pybind_builder(scann, db, "tree_ah").build(), "native")
  model = make_model(torch, s).eval()
  feats = torch.randn(5, 6, generator=torch.Generator().manual_seed(1))
  with torch.no_grad():
    want = model(feats)
  torch.save({"feats": feats, "want": want}, os.path.join(tmp, "want.pt"))
  dyn = {"features": {0: torch.export.Dim("n", min=1, max=1024)}}
  for strict in (True, False):
    ep = torch.export.export(model, (feats,), dynamic_shapes=dyn,
                             strict=strict)
    assert_same(ep.module()(feats), want, f"exported strict={strict}")
  path = os.path.join(tmp, "retrieval.pt2")
  torch.export.save(ep, path)
  print("export:", child("export", path, tmp))

  # AOTInductor: the compiled encoder may round differently from eager
  # (on a GPU it does), so the model also returns its embeddings, and the
  # child checks the search results against the Searcher's for those.
  torch.save(s, os.path.join(tmp, "searcher.pt"))
  devices = ["cpu"]
  if torch.cuda.is_available():
    devices.append("cuda")
  for device in devices:
    m = make_model(torch, s, return_embeddings=True).to(device).eval()
    f = feats.to(device)
    with torch.no_grad():
      torch.save({"feats": f, "want": m(f)},
                 os.path.join(tmp, f"want_aoti_{device}.pt"))
    ep = torch.export.export(m, (f,), dynamic_shapes=dyn)
    pkg = os.path.join(tmp, f"retrieval_aoti_{device}.pt2")
    try:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch._inductor.aoti_compile_and_package(ep, package_path=pkg)  # pylint: disable=protected-access
    except Exception as e:  # pylint: disable=broad-except
      if device == "cpu":
        raise
      # CUDA AOTInductor needs a CUDA toolkit (nvcc); ROCm's is part of
      # the ROCm build of torch.
      print(f"AOTInductor ({device}): not compiled here: "
            f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
      continue
    print(f"AOTInductor ({device}):", child("aoti", pkg, tmp, device))
  del ops


def check_devices(torch, st, scann, db, queries):
  if not torch.cuda.is_available():
    print("devices: no CUDA/ROCm device, GPU cases not run")
    return
  s = st.Searcher(pybind_builder(scann, db, "ah").build(), "native")
  q = torch.from_numpy(queries)
  want = s.search_batched(q)
  got = s.cuda().search_batched(q.cuda())
  assert got[0].device.type == "cuda" and s.index_data.device.type == "cpu"
  assert_same(tuple(t.cpu() for t in got), want, "cuda queries")
  # The raw op with the index state on the GPU works too (copied to host).
  gi, gd = torch.ops.scann.search_batched(
      s.index_data.cuda(), s.index_offsets.cuda(), s.index_names.cuda(),
      s._shared_name, q.cuda(), 10, -1, -1, False, 0)  # pylint: disable=protected-access
  assert_same((gi.cpu(), gd.cpu()), want, "state on the GPU")
  name = torch.cuda.get_device_name(0)
  kind = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
  print(f"devices: {kind} ({name}): results on the queries' device")


def check_state_dict(torch, st, scann, ops, db, queries, tmp):
  q = torch.from_numpy(queries)
  a = st.Searcher(pybind_builder(scann, db, "tree_ah").build(
      docids=[f"a{i}" for i in range(len(db))]), "native")
  b = st.Searcher(pybind_builder(scann, dataset(len(db), seed=9), "bf").build(),
                  "native")
  want_a, want_b = a(q), b(q)
  sd = a.state_dict()
  assert sorted(sd) == ["index_data", "index_names", "index_offsets"]

  # Into a Searcher of another index: it searches a's index now (rebuilt).
  n0 = builds(ops)
  b.load_state_dict(sd)
  assert_same(b(q), want_a, "load_state_dict into another index")
  assert builds(ops) == n0 + 1
  assert b.searcher.docids == a.searcher.docids  # loaded from the state
  assert b.default_num_neighbors == a.default_num_neighbors

  # Through a file, in a fresh process (weights_only).
  model = make_model(torch, a)
  model_sd = os.path.join(tmp, "model_state.pt")
  torch.save(model.state_dict(), model_sd)
  feats = torch.randn(4, 6, generator=torch.Generator().manual_seed(3))
  torch.save({"feats": feats, "want": model(feats)},
             os.path.join(tmp, "want_sd.pt"))
  print("state_dict:", child("state_dict", model_sd, tmp))

  # A 0.2.0-era state_dict (no index): strict loading works, keeps the index.
  old = {k: v for k, v in model.state_dict().items()
         if not k.startswith("index.")}
  fresh = make_model(torch, b)
  missing, unexpected = fresh.load_state_dict(old, strict=True)
  assert not missing and not unexpected
  assert_same(b(q), want_a, "0.2.0 state_dict keeps the index")

  # index_in_state_dict = False: the state_dict has no index.
  b.index_in_state_dict = False
  assert not b.state_dict() and not make_model(torch, b).state_dict().get(
      "index.index_data")
  b.index_in_state_dict = True
  assert len(b.state_dict()) == 3

  # Incomplete or damaged index state: errors.
  partial = {"index_data": sd["index_data"], "index_offsets": sd["index_offsets"]}
  expect_error(lambda: b.load_state_dict(partial), "needs all of", "partial")
  bad = dict(sd, index_offsets=sd["index_offsets"][:-1])
  expect_error(lambda: b.load_state_dict(bad), "doesn't match", "bad offsets")
  assert_same(b(q), want_a, "unchanged after failed loads")

  # Across backends: a native state_dict into a Python-backend Searcher,
  # and a Python-backend (index-less) state_dict into a native one.
  py = st.Searcher(pybind_builder(scann, dataset(50, seed=4), "bf").build(),
                   "python")
  py.load_state_dict(sd)
  assert_same(py(q), want_a, "native state_dict into the Python backend")
  assert not py.state_dict()
  b2 = st.Searcher(pybind_builder(scann, dataset(len(db), seed=9), "bf")
                   .build(), "native")
  b2.load_state_dict(py.state_dict())
  assert_same(b2(q), want_b, "Python-backend state_dict: index kept")
  print("state_dict: round trips, 0.2.0-era and index-less dicts, "
        "across backends, errors")


def check_stale_state(torch, st, scann, ops, db, queries):
  q = torch.from_numpy(queries)
  p = scann.scann_ops_pybind.builder(db, 5, "squared_l2").score_brute_force() \
      .build(docids=[f"d{i}" for i in range(len(db))])
  s = st.Searcher(p, "native")
  compiled = torch.compile(lambda x: s.search_batched(x), fullgraph=True,
                           dynamic=True)
  n0 = builds(ops)
  for _ in range(3):
    s(q)
    compiled(q)
  assert builds(ops) == n0 + 1, (builds(ops), n0)  # built once

  # Changes through the pybind searcher: eager searches see them at once.
  p.delete(["d0", "d1"])
  p.upsert(["new"], db[:1] + 100)
  i, _ = s.search(torch.from_numpy(db[0] + 100))
  assert int(i[0]) == p.docid_to_id["new"]
  assert_same(s(q), st.Searcher(p, "python")(q), "after upsert/delete")
  assert builds(ops) == n0 + 2
  # Compiled code after sync() (here the eager search above synced).
  p.upsert(["newer"], db[:1] - 100)
  s.sync()
  i, _ = compiled(torch.from_numpy(db[:1] - 100))
  assert int(i[0, 0]) == p.docid_to_id["newer"]
  print("stale state: rebuilt after changes, never for unchanged state")


def check_pickle(torch, st, scann, db, queries, tmp):
  s = st.Searcher(pybind_builder(scann, db, "tree_ah").build(
      docids=[f"x{i}" for i in range(len(db))]), "native")
  model = make_model(torch, s)
  feats = torch.randn(4, 6, generator=torch.Generator().manual_seed(5))
  want = model(feats)
  back = pickle.loads(pickle.dumps(model))
  assert_same(back(feats), want, "pickle round trip")
  assert back.index.searcher.docids == s.searcher.docids
  path = os.path.join(tmp, "model.pt")
  torch.save(model, path)
  torch.save({"feats": feats, "want": want}, os.path.join(tmp, "want_pk.pt"))
  print("pickle:", child("pickle", path, tmp, "native"))
  print("pickle:", child("pickle", path, tmp, "python"))
  del queries


def check_errors(torch, st, scann, db, queries):
  s = st.Searcher(pybind_builder(scann, db, "ah").build(), "native")
  q = torch.from_numpy(queries)
  expect_error(lambda: s.search_batched(q[:, :3]), "dimensionality", "width")
  expect_error(lambda: s.search(q[0, :3]), "dimensionality", "width 1")
  raw = torch.ops.scann.search_batched
  state = (s.index_data, s.index_offsets, s.index_names)

  def call(*st_, k=5, queries_=q, name="err"):
    return raw(*st_, name, queries_, k, -1, -1, False, 0)

  expect_error(lambda: call(*state, k=0), "k must be > 0", "k=0")
  expect_error(lambda: call(*state, queries_=q[0]), "2-dimensional", "1-D q")
  expect_error(
      lambda: torch.ops.scann.search(*state, "err", q, 5, -1, -1),
      "1-dimensional", "2-D q to search")
  expect_error(lambda: call(s.index_data.float(), *state[1:]), "uint8",
               "dtype")
  data, off, names = state
  expect_error(lambda: call(data[:-10], off, names), "index_offsets", "short")
  bad_off = off.clone()
  bad_off[1] = int(off[2]) + 1
  expect_error(lambda: call(data, bad_off, names), "non-decreasing",
               "offsets order")
  expect_error(lambda: call(data, off, names[:-1]), "newline", "names")
  # A damaged file (the manifest's first bytes): the loader's error.
  files = bytes(names.numpy()).decode().split("\n")[:-1]
  m = files.index("scann_assets.pbtxt")
  damaged = data.clone()
  damaged[int(off[m]):int(off[m]) + 8] = ord("#")
  damaged[int(off[m]) + 8:int(off[m + 1])] = ord("x")
  expect_error(lambda: call(damaged, off, names, name="damaged"),
               "can't build the ScaNN searcher", "damaged manifest")
  # Nothing was cached for the failures; the good state still works.
  assert_same(call(*state, name="ok"), s.search_batched(q, 5), "after errors")
  # Python-level argument checks as with the Python backend.
  expect_error(lambda: s.search_batched(q, 0), "> 0", "k=0", ValueError)
  expect_error(lambda: s.search(q), "1-dimensional", "2-D", ValueError)
  expect_error(lambda: st.Searcher(s.searcher, backend="gpu"), "backend",
               "bad backend", ValueError)
  print("errors: RuntimeError for bad queries and damaged index tensors")


def check_concurrency(torch, st, scann, ops, db):
  s = st.Searcher(scann.scann_ops_pybind.builder(db, 1, "squared_l2")
                  .score_brute_force().build(), "native")
  n = len(db)
  n0 = builds(ops)
  start = concurrent.futures.ThreadPoolExecutor(8)
  firsts = list(start.map(
      lambda i: int(s.search(torch.from_numpy(db[i]))[0][0]), range(16)))
  start.shutdown()
  assert firsts == list(range(16)), firsts
  assert builds(ops) == n0 + 1, (builds(ops), n0)  # concurrent first calls
  compiled = torch.compile(lambda x: s.search_batched(x)[0][:, 0],
                           fullgraph=True, dynamic=True)
  compiled(torch.from_numpy(db[:2]))

  def worker(seed):
    rng = np.random.default_rng(seed)
    for _ in range(30):
      rows = rng.integers(n, size=int(rng.integers(2, 20)))
      x = torch.from_numpy(db[rows])
      np.testing.assert_array_equal(compiled(x).numpy(), rows)
      np.testing.assert_array_equal(
          s.search_batched_parallel(x)[0][:, 0].numpy(), rows)
    return True

  with concurrent.futures.ThreadPoolExecutor(8) as pool:
    assert all(pool.map(worker, range(16)))
  assert builds(ops) == n0 + 1
  print("concurrency: 8 threads, built once")


# --- Fresh-process children ----------------------------------------------------


def child_main(mode, args):
  import torch  # pylint: disable=g-import-not-at-top
  if mode == "pickle" and args[2] == "python":
    os.environ["SCANN_TORCH_BACKEND"] = "python"
  import scann.torch as st  # pylint: disable=g-import-not-at-top  # registers the ops
  import scann_torch_ops as ops  # pylint: disable=g-import-not-at-top
  tmp = args[1]
  if mode == "export":
    ref = torch.load(os.path.join(tmp, "want.pt"))
    m = torch.export.load(args[0]).module()
    for _ in range(3):
      assert_same(m(ref["feats"]), ref["want"], "exported, fresh process")
    assert_same(m(ref["feats"][:2]), tuple(w[:2] for w in ref["want"]),
                "exported, other batch size")
    assert ops.stats()["builds"] == 1, ops.stats()
    return "loaded in a fresh process, same results, built once for 4 calls"
  if mode == "aoti":
    device = args[2]
    ref = torch.load(os.path.join(tmp, f"want_aoti_{device}.pt"))
    # torch 2.10's aoti_load_package() fails in a fresh process unless
    # torch._inductor.codecache has been imported (fixed in later versions).
    import torch._inductor.codecache  # pylint: disable=g-import-not-at-top,unused-import
    m = torch._inductor.aoti_load_package(args[0])  # pylint: disable=protected-access
    feats = ref["feats"]
    runs = [m(feats) for _ in range(3)]
    assert ops.stats()["builds"] == 1, ops.stats()
    s = torch.load(os.path.join(tmp, "searcher.pt"), weights_only=False)
    same_as_eager = True
    for i, d, emb in runs:
      assert i.device == feats.device and d.device == feats.device
      # The search is exact for the program's embeddings ...
      assert_same((i, d), s.search_batched_parallel(emb, 5, batch_size=4),
                  f"AOTI {device}")
      # ... which are the eager model's, up to rounding.
      torch.testing.assert_close(emb, ref["want"][2], rtol=1e-4, atol=1e-5)
      same_as_eager &= bool(torch.equal(i, ref["want"][0]))
    how = "identical to" if same_as_eager else "close to"
    return (f"loaded in a fresh process ({device}), exact search results for "
            f"its embeddings (indices {how} eager), built once for 3 calls")
  if mode == "state_dict":
    ref = torch.load(os.path.join(tmp, "want_sd.pt"))
    sd = torch.load(args[0], weights_only=True)
    db = dataset(30, seed=11)
    s = st.builder(db, 5, "dot_product").score_brute_force().build()
    model = make_model(torch, s)
    model.load_state_dict(sd)
    assert_same(model(ref["feats"]), ref["want"], "state_dict, fresh process")
    return "torch.load(weights_only=True) + load_state_dict: same results"
  if mode == "pickle":
    ref = torch.load(os.path.join(tmp, "want_pk.pt"))
    model_class(torch)  # __main__.Retrieval, for unpickling
    model = torch.load(args[0], weights_only=False)
    assert model.index.backend == args[2], (model.index.backend, args[2])
    assert_same(model(ref["feats"]), ref["want"], f"pickle {args[2]}")
    assert model.index.searcher.docids[:2] == ["x0", "x1"]
    return (f"torch.load(model) in a fresh process, {args[2]} backend: same "
            "results")
  raise ValueError(mode)


def main():
  if len(sys.argv) > 2 and sys.argv[1] == "--child":
    print(child_main(sys.argv[2], sys.argv[3:]))
    return
  try:
    import torch  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    if os.environ.get("SCANN_TEST_REQUIRE_TORCH"):
      raise
    print(f"SKIPPED: torch is not importable ({e})")
    sys.exit(SKIP)
  import scann  # pylint: disable=g-import-not-at-top
  import scann.torch as st  # pylint: disable=g-import-not-at-top
  ops = st._load_native()  # pylint: disable=protected-access
  if ops is None:
    if os.environ.get("SCANN_TEST_REQUIRE_TORCH_NATIVE"):
      raise AssertionError(st._native_error)  # pylint: disable=protected-access
    print(f"SKIPPED: {st._native_error}")  # pylint: disable=protected-access
    sys.exit(SKIP)

  db = dataset(1000)
  queries = dataset(20, seed=1)
  devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
  tmp = tempfile.mkdtemp(prefix="scann_torch_native_")
  try:
    for device in devices:
      check_parity(torch, st, scann, db, queries, device)
      check_compile(torch, st, scann, db, device)
    check_edge_cases(torch, st, scann, db, queries)
    check_devices(torch, st, scann, db, queries)
    check_export_and_aoti(torch, st, scann, ops, db, tmp)
    check_state_dict(torch, st, scann, ops, db, queries, tmp)
    check_stale_state(torch, st, scann, ops, db, queries)
    check_pickle(torch, st, scann, db, queries, tmp)
    check_errors(torch, st, scann, db, queries)
    check_concurrency(torch, st, scann, ops, db)
  finally:
    shutil.rmtree(tmp, ignore_errors=True)
  gc.collect()
  gpu = (f", {torch.cuda.get_device_name(0)}" if len(devices) > 1 else
         ", no GPU")
  print(f"PASSED (torch {torch.__version__}, Python "
        f"{sys.version.split()[0]}{gpu}; op built with torch "
        f"{ops.built_with_torch})")


if __name__ == "__main__":
  main()
