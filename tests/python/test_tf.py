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

"""scann.tf with both of its backends, against the pybind searcher.

- `import scann` imports neither TensorFlow nor scann_tf_ops, and
  `import scann.tf` without TensorFlow is a clear ImportError (checked in
  subprocesses).
- Backend selection: the op backend when scann_tf_ops is importable, the
  Python backend otherwise (also when scann_tf_ops is hidden, silently, or
  broken, with a warning); SCANN_TF_BACKEND=op/python/auto overrides it,
  and get_backend() returns either one.
- Every check below runs against each available backend ("python" always;
  "op" when scann_tf_ops is importable, skipped otherwise). The searches
  return exactly what the pybind searcher returns, as int32 / float32
  tensors named like upstream's op outputs, eagerly and inside tf.function
  (also with an unknown batch dimension), with static shapes where they
  are known. Short rows (fewer points than k) are padded; zero queries;
  docids (results are indices); create_searcher / load_searcher /
  from_pybind / ScannSearcher(pybind); errors are the same TensorFlow
  error classes; tf.data maps and a tf.function called from several
  threads.
- The two backends agree with each other: the same outputs, dtypes and
  static shapes for the same calls.
- SavedModel: the Python backend's serialize_to_module() and
  searcher_from_module() raise NotImplementedError saying how to get the
  op; with the op backend, a model saved through scann.tf loads in a fresh
  process (which picks the op backend by itself) and searches the same.

Without TensorFlow the test exits with 77, which ctest reports as skipped
(SKIP_RETURN_CODE), unless SCANN_TEST_REQUIRE_TF is set. The op backend's
checks are skipped when scann_tf_ops isn't importable, unless
SCANN_TEST_REQUIRE_TF_OP is set. With SCANN_TEST_HIDE_TF_OP=1,
scann_tf_ops is hidden (as if it weren't built), so only the Python
backend is available and must be picked.

Run with scann-core's build/python on PYTHONPATH:
  PYTHONPATH=build/python python tests/python/test_tf.py
"""

import concurrent.futures
import gc
import os
import subprocess
import sys
import tempfile

import numpy as np

SKIP = 77
DIM = 16
HIDE_OP = bool(os.environ.get("SCANN_TEST_HIDE_TF_OP"))
# Prepended to every Python snippet run here (and to this test's own
# imports) when the op is hidden: a None in sys.modules makes it
# unimportable, as if it weren't installed.
HIDE_OP_CODE = "import sys; sys.modules['scann_tf_ops'] = None\n"


def run_py(code, env=None, check=True):
  full_env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
  full_env.pop("SCANN_TF_BACKEND", None)
  full_env.update(env or {})
  r = subprocess.run(
      [sys.executable, "-c", (HIDE_OP_CODE if HIDE_OP else "") + code],
      env=full_env, capture_output=True, text=True)
  if check and r.returncode:
    print(r.stdout[-3000:], r.stderr[-5000:])
    raise AssertionError(f"subprocess failed:\n{code}")
  return r


def check_import_isolation():
  """`import scann` leaves TensorFlow alone; scann.tf says what's missing."""
  run_py("import sys, scann\n"
         "from scann.scann_ops.py import scann_ops_pybind\n"
         "assert 'tensorflow' not in sys.modules, 'import scann imported TF'\n"
         "assert 'scann_tf_ops' not in sys.modules or "
         "sys.modules['scann_tf_ops'] is None, 'imported scann_tf_ops'\n")
  # A blocked import behaves like a missing package.
  run_py("import sys\n"
         "sys.modules['tensorflow'] = None\n"
         "import scann\n"
         "try:\n"
         "  import scann.tf\n"
         "except ImportError as e:\n"
         "  assert 'scann-core[tf]' in str(e), str(e)\n"
         "else:\n"
         "  raise AssertionError('import scann.tf worked without TF')\n")
  print("import isolation: import scann imports neither TensorFlow nor "
        "scann_tf_ops")


def check_backend_selection(op_available, tmp):
  auto = "op" if op_available else "python"
  # Prints the backend, the available ones, the searcher class's module,
  # and scann.tf's warnings about the op (none expected but when broken).
  show = ("import warnings\n"
          "with warnings.catch_warnings(record=True) as w:\n"
          "  warnings.simplefilter('always')\n"
          "  import scann.tf\n"
          "print(scann.tf.backend(), scann.tf.available_backends(), "
          "scann.tf.ScannSearcher.__module__, [str(x.message) for x in w "
          "if 'scann_tf_ops' in str(x.message)])\n")
  r = run_py(show)
  want_avail = ("op", "python") if op_available else ("python",)
  want_module = "scann_tf_ops" if op_available else "scann._tf_python"
  assert r.stdout.strip() == f"{auto} {want_avail!r} {want_module} []", (
      r.stdout)
  for value in ("auto", "AUTO", ""):
    r = run_py(show, env={"SCANN_TF_BACKEND": value})
    assert r.stdout.startswith(auto + " "), (value, r.stdout)
  r = run_py(show, env={"SCANN_TF_BACKEND": "python"})
  assert r.stdout.startswith("python "), r.stdout
  assert "scann._tf_python" in r.stdout, r.stdout
  r = run_py(show, env={"SCANN_TF_BACKEND": "op"}, check=False)
  if op_available:
    assert r.returncode == 0 and r.stdout.startswith("op "), (r.stdout,
                                                             r.stderr)
  else:
    assert r.returncode and "ImportError" in r.stderr, r.stderr
    assert "-DSCANN_BUILD_TF_OP=ON" in r.stderr, r.stderr
  r = run_py(show, env={"SCANN_TF_BACKEND": "tpu"}, check=False)
  assert r.returncode and "SCANN_TF_BACKEND='tpu'" in r.stderr, r.stderr

  # get_backend: either backend, whatever scann.tf uses.
  run_py("import scann.tf, scann._tf_python\n"
         "assert scann.tf.backend() == 'python'\n"
         "assert scann.tf.get_backend('python') is scann._tf_python\n"
         "try:\n"
         "  scann.tf.get_backend('gpu')\n"
         "except ValueError:\n"
         "  pass\n"
         "else:\n"
         "  raise AssertionError('get_backend(gpu)')\n"
         "try:\n"
         "  op = scann.tf.get_backend('op')\n"
         "except ImportError as e:\n"
         f"  assert not {op_available}\n"
         "  assert '-DSCANN_BUILD_TF_OP=ON' in str(e), str(e)\n"
         "else:\n"
         f"  assert {op_available}\n"
         "  assert op.BACKEND == 'op' and op.__name__ == 'scann_tf_ops'\n",
         env={"SCANN_TF_BACKEND": "python"})

  # Not installed (hidden): the Python backend, silently.
  r = run_py(HIDE_OP_CODE + show)
  assert r.stdout.strip() == "python ('python',) scann._tf_python []", (
      r.stdout)
  if HIDE_OP:
    print("backend selection: auto -> python (scann_tf_ops hidden); "
          "SCANN_TF_BACKEND and get_backend override it")
    return

  # Installed but broken: the Python backend with a warning, or the error
  # with SCANN_TF_BACKEND=op.
  broken = os.path.join(tmp, "broken", "scann_tf_ops")
  os.makedirs(broken)
  with open(os.path.join(broken, "__init__.py"), "w") as f:
    f.write("raise RuntimeError('built for another TensorFlow')\n")
  prepend = f"import sys; sys.path.insert(0, {os.path.dirname(broken)!r})\n"
  r = run_py(prepend + show)
  assert r.stdout.startswith("python ('python',) scann._tf_python ["), (
      r.stdout)
  assert "built for another TensorFlow" in r.stdout, r.stdout
  assert "SCANN_TF_BACKEND=python" in r.stdout, r.stdout
  r = run_py(prepend + "import scann.tf\n", env={"SCANN_TF_BACKEND": "op"},
             check=False)
  assert r.returncode and "built for another TensorFlow" in r.stderr, r.stderr
  print(f"backend selection: auto -> {auto}; SCANN_TF_BACKEND and "
        "get_backend override it; hidden -> python; broken -> python with a "
        "warning")


def dataset(n, seed=0):
  return np.random.default_rng(seed).standard_normal((n, DIM)).astype(
      np.float32)


def assert_same(got, want, what):
  """got: (tensor, tensor) from scann.tf; want: arrays from pybind."""
  gi, gd = got
  wi, wd = want
  assert gi.dtype.name == "int32" and gd.dtype.name == "float32", what
  wi = np.asarray(wi)
  assert gi.shape == wi.shape and gd.shape == wi.shape, (what, gi.shape,
                                                         wi.shape)
  np.testing.assert_array_equal(gi.numpy(), wi.astype(np.int32), err_msg=what)
  np.testing.assert_array_equal(gd.numpy(), np.asarray(wd), err_msg=what)


def expect_raises(exc, f, match):
  try:
    f()
  except exc as e:
    assert match in str(e), f"{match!r} not in {str(e)!r}"
    return
  raise AssertionError(f"expected {exc.__name__}({match!r})")


def pybind_searchers(scann, db):
  pb = scann.scann_ops_pybind
  return {
      "brute_force":
          pb.builder(db, 10, "dot_product").score_brute_force().build(),
      "tree_ah_reorder":
          pb.builder(db, 10, "squared_l2").tree(
              20, 5, training_sample_size=len(db)).score_ah(2).reorder(
                  50).build(),
  }


def check_eager(m, tf, scann, db, queries):
  for name, p in pybind_searchers(scann, db).items():
    s = m.from_pybind(p)
    assert isinstance(s, m.ScannSearcher)
    r = s.search(queries[0])
    assert isinstance(r, m.SearchResult)
    assert r.index is r[0] and r.distance is r[1]
    assert_same(r, p.search(queries[0]), f"{name} search")
    r = s.search_batched(queries)
    assert isinstance(r, m.BatchedSearchResult)
    assert r.indices is r[0] and r.distances is r[1]
    assert_same(r, p.search_batched(queries), f"{name} search_batched")
    assert_same(s.search_batched_parallel(queries, batch_size=7),
                p.search_batched_parallel(queries, batch_size=7),
                f"{name} search_batched_parallel")
    # Every search parameter, as Python ints and as tensors.
    params = dict(final_num_neighbors=4, pre_reorder_num_neighbors=20,
                  leaves_to_search=3)
    if name == "brute_force":
      del params["leaves_to_search"]
    tparams = {k: tf.constant(v) for k, v in params.items()}
    want = p.search_batched(queries, **params)
    assert want[0].shape == (len(queries), 4)
    assert_same(s.search_batched(queries, **params), want, f"{name} params")
    assert_same(s.search_batched(queries, **tparams), want, f"{name} tparams")
    assert_same(s.search(queries[1], **params), p.search(queries[1], **params),
                f"{name} search params")
    # float64 queries are cast, as tensors or arrays.
    assert_same(s.search_batched(tf.constant(queries.astype(np.float64))),
                p.search_batched(queries), f"{name} float64")
    assert_same(s.search(queries[2].astype(np.float64)), p.search(queries[2]),
                f"{name} float64 single")
    # Zero queries.
    r = s.search_batched(queries[:0])
    assert r.indices.shape == (0, 10), r.indices.shape
    r = s.search_batched(queries[:0], 3)
    assert r.indices.shape == (0, 3), r.indices.shape
    # Wrong ranks are caught before searching.
    expect_raises(ValueError, lambda: s.search(queries), "1-dimensional")
    expect_raises(ValueError, lambda: s.search_batched(queries[0]),
                  "2-dimensional")

  # The builder, from an eager tensor.
  s = m.builder(tf.constant(db), 10, "dot_product").score_brute_force().build()
  assert isinstance(s, m.ScannSearcher)
  assert_same(s.search_batched(queries),
              pybind_searchers(scann, db)["brute_force"].search_batched(queries),
              "builder")


def check_errors(m, tf, scann, db, queries):
  """Search errors are tf.errors.InvalidArgumentError, eager or not."""
  s = m.from_pybind(pybind_searchers(scann, db)["tree_ah_reorder"])
  nan = queries.copy()
  nan[3, 2] = np.nan
  cases = [
      ("dimensionality", lambda f: f(lambda q: s.search_batched(q))(
          queries[:, :3])),
      ("dimensionality", lambda f: f(lambda q: s.search(q))(queries[0, :3])),
      ("NaN", lambda f: f(lambda q: s.search_batched_parallel(q))(nan)),
      ("", lambda f: f(lambda q: s.search_batched(q, 0))(queries)),
      ("", lambda f: f(lambda q: s.search_batched_parallel(
          q, 5, batch_size=0))(queries)),
      ("> 0", lambda f: f(lambda q: s.search_batched(q, 0))(queries[:0])),
  ]
  for match, call in cases:
    for wrap in (lambda fn: fn, tf.function):  # eagerly, and in a graph
      expect_raises(tf.errors.InvalidArgumentError, lambda: call(wrap), match)
  # Still fine afterwards.
  assert s.search_batched(queries, 5).indices.shape == (len(queries), 5)


def check_tf_function(m, tf, scann, db, queries):
  p = pybind_searchers(scann, db)["tree_ah_reorder"]
  s = m.from_pybind(p)
  spec = [tf.TensorSpec([None, DIM], tf.float32)]
  f_k = tf.function(lambda q: s.search_batched_parallel(q, 5),
                    input_signature=spec)
  f_default = tf.function(lambda q: s.search_batched(q), input_signature=spec)
  f_one = tf.function(lambda q: s.search(q, leaves_to_search=4),
                      input_signature=[tf.TensorSpec([DIM], tf.float32)])

  # Static shapes: batch unknown; k known only when it's a Python int.
  out = f_k.get_concrete_function().structured_outputs
  assert out.indices.shape.as_list() == [None, 5], out
  assert out.indices.dtype == tf.int32 and out.distances.dtype == tf.float32
  out = f_default.get_concrete_function().structured_outputs
  assert out.indices.shape.as_list() == [None, None], out
  out = f_one.get_concrete_function().structured_outputs
  assert out.index.shape.as_list() == [None], out
  fixed = tf.function(lambda q: s.search_batched(q, 3)).get_concrete_function(
      tf.TensorSpec([8, DIM], tf.float32)).structured_outputs
  assert fixed.indices.shape.as_list() == [8, 3], fixed
  k_tensor = tf.function(lambda q, k: s.search_batched(q, k)).get_concrete_function(
      tf.TensorSpec([8, DIM], tf.float32),
      tf.TensorSpec([], tf.int32)).structured_outputs
  assert k_tensor.indices.shape.as_list() == [8, None], k_tensor

  for n in (1, 7, len(queries), 0):
    q = queries[:n]
    assert_same(f_k(q), p.search_batched_parallel(q, 5), f"tf.function k n={n}")
    if n:
      assert_same(f_default(q), p.search_batched(q), f"tf.function n={n}")
  assert_same(f_one(queries[2]), p.search(queries[2], leaves_to_search=4),
              "tf.function search")
  # One trace each, whatever the batch size.
  assert f_k.experimental_get_tracing_count() == 1
  assert f_default.experimental_get_tracing_count() == 1

  # Building needs concrete data.
  expect_raises(ValueError,
                lambda: tf.function(lambda d: m.builder(d, 5, "dot_product")
                                    .score_brute_force().build())(db),
                "eagerly")


def check_short_rows(m, tf):
  db = dataset(3)  # fewer points than k
  s = m.builder(db, 10, "dot_product").score_brute_force().build()
  q = dataset(4, seed=1)
  r = s.search_batched(q)  # default k: as wide as the longest row
  assert r.indices.shape == (4, 3), r.indices.shape
  r = s.search_batched(q, final_num_neighbors=5)  # explicit k: padded to it
  assert r.indices.shape == (4, 5), r.indices.shape
  assert np.isnan(r.distances.numpy()[:, 3:]).all()
  assert (r.indices.numpy()[:, 3:] == 0).all()
  assert_same(r, s.to_pybind().search_batched(q, final_num_neighbors=5),
              "padded")
  f = tf.function(lambda q: s.search_batched(q, 5),
                  input_signature=[tf.TensorSpec([None, DIM], tf.float32)])
  assert_same(f(q), (r.indices.numpy(), r.distances.numpy()),
              "padded, tf.function")
  r = s.search(q[0])
  assert r.index.shape == (3,), r.index.shape
  r = s.search(q[0], 5)
  assert r.index.shape == (3,), r.index.shape


def check_docids(m, tf, scann, db, queries):
  """Results are indices; the pybind searcher maps them to docids."""
  docids = [f"d{i}" for i in range(len(db))]
  s = (m.builder(db, 5, "squared_l2").score_brute_force()
       .build(docids=docids))
  p = s.to_pybind()
  assert p.docids == docids
  idx, dist = s.search_batched(queries)
  want_docids, want_dist = p.search_batched(queries)
  assert [[p.docids[i] for i in row] for row in idx.numpy()] == want_docids
  np.testing.assert_array_equal(dist.numpy(), want_dist)
  # Mapping to docids in the graph, as docs/tensorflow.md shows.
  table = tf.constant(p.docids)
  f = tf.function(lambda q: tf.gather(table, s.search_batched(q).indices))
  assert f(queries).numpy().astype(str).tolist() == want_docids

  # Updating, portably: mutate to_pybind()'s searcher, then from_pybind().
  before = s.search_batched(queries).indices.numpy()
  p.delete(["d0", "d1"])
  p.upsert(["new"], db[:1] + 100)
  s2 = m.from_pybind(p)
  idx, _ = s2.search_batched(queries)
  assert [[p.docids[i] for i in row] for row in idx.numpy()] == (
      p.search_batched(queries)[0])
  assert s2.search(db[:1][0] + 100).index.numpy()[0] == p.docid_to_id["new"]
  after = s.search_batched(queries).indices.numpy()
  if m.BACKEND == "python":
    # The Python backend's searcher shares its pybind searcher: the old
    # searcher (and functions traced with it) see the change too.
    assert s.searcher is p and s2.searcher is p
    np.testing.assert_array_equal(after, idx.numpy())
  else:
    # The op backend's searchers are snapshots.
    np.testing.assert_array_equal(after, before)


def check_create_and_load(m, tf, scann, db, queries):
  b = scann.scann_ops_pybind.builder(db, 5, "dot_product").tree(
      10, 3).score_ah(2)
  config = b.create_config()
  # Upstream's positional signature: db, config, training_threads,
  # container, shared_name.
  s = m.create_searcher(tf.constant(db), tf.constant(config),
                        tf.constant(2), "", "custom_id")
  assert isinstance(s, m.ScannSearcher)
  want = s.to_pybind().search_batched(queries)
  assert_same(s.search_batched(queries), want, "create_searcher")
  if m.BACKEND == "op":
    assert s.shared_name == "custom_id"
  s = m.create_searcher(db, config.encode(), docids=[str(i) for i in
                                                     range(len(db))])
  assert s.to_pybind().docids[:2] == ["0", "1"]
  with tempfile.TemporaryDirectory() as d:
    s.serialize(d)
    want = scann.scann_ops_pybind.load_searcher(d).searcher.search_batched(
        queries, -1, -1, -1, False, 256)
    for loaded in (m.load_searcher(d), m.load_searcher(d, True),
                   m.load_searcher(d, shared_name="loaded")):
      assert isinstance(loaded, m.ScannSearcher)
      assert_same(loaded.search_batched(queries), want, "load_searcher")
      assert loaded.to_pybind().docids == s.to_pybind().docids
    # A pybind searcher can be wrapped directly.
    wrapped = m.ScannSearcher(scann.scann_ops_pybind.load_searcher(d))
    assert_same(wrapped.search_batched(queries), want, "wrapped")
  expect_raises(ValueError, lambda: m.load_searcher(os.path.join(d, "gone")),
                "is not a directory")
  expect_raises(TypeError,
                lambda: m.ScannSearcher(s.to_pybind().searcher),
                "scann_ops_pybind.ScannSearcher")
  expect_raises(TypeError, lambda: m.from_pybind(s),
                "scann_ops_pybind.ScannSearcher")


def check_concurrency(m, tf, db):
  s = (m.builder(db, 1, "squared_l2")
       .tree(20, 20, random_init=False).score_brute_force().build())
  # Every leaf is searched and scoring is exact: each point finds itself.
  n = len(db)
  ds = tf.data.Dataset.from_tensor_slices(db).enumerate().map(
      lambda i, v: (i, s.search(v).index[0]), num_parallel_calls=8,
      deterministic=False)
  got = dict((int(i), int(j)) for i, j in ds.as_numpy_iterator())
  assert got == {i: i for i in range(n)}, "tf.data map"

  ds = tf.data.Dataset.from_tensor_slices(db).batch(37).map(
      lambda q: s.search_batched_parallel(q).indices[:, 0],
      num_parallel_calls=4)
  np.testing.assert_array_equal(np.concatenate(list(ds.as_numpy_iterator())),
                                np.arange(n))

  f = tf.function(lambda q: s.search_batched(q).indices[:, 0],
                  input_signature=[tf.TensorSpec([None, DIM], tf.float32)])

  def worker(seed):
    rng = np.random.default_rng(seed)
    for _ in range(50):
      rows = rng.integers(n, size=int(rng.integers(1, 20)))
      np.testing.assert_array_equal(f(db[rows]).numpy(), rows)
      i = int(rng.integers(n))
      assert s.search(db[i]).index.numpy()[0] == i
    return True

  f(db[:1])  # trace once first; concurrent first calls would each trace
  with concurrent.futures.ThreadPoolExecutor(8) as pool:
    assert all(pool.map(worker, range(16)))


def check_backends_agree(tf, scann, op, py, db, queries):
  """The same calls through both backends: identical results and shapes."""
  for name, p in pybind_searchers(scann, db).items():
    a, b = op.from_pybind(p), py.from_pybind(p)

    def calls(s):
      spec = [tf.TensorSpec([None, DIM], tf.float32)]
      return [
          lambda: s.search(queries[0]),
          lambda: s.search(queries[1], 3, 30),
          lambda: s.search_batched(queries),
          lambda: s.search_batched(queries, 7, 40),
          lambda: s.search_batched(queries[:0]),
          lambda: s.search_batched_parallel(queries, 12, batch_size=5),
          lambda: s.search_batched(queries, tf.constant(4)),
          lambda: tf.function(lambda q: s.search_batched_parallel(q, 5),
                              input_signature=spec)(queries),
          lambda: tf.function(lambda q: s.search_batched(q),
                              input_signature=spec)(queries[:3]),
          lambda: tf.function(lambda q: s.search(q, 2))(queries[4]),
      ]

    for i, (ca, cb) in enumerate(zip(calls(a), calls(b))):
      ra, rb = ca(), cb()
      assert type(ra) is type(rb), (name, i)
      for ta, tb in zip(ra, rb):
        assert ta.dtype == tb.dtype and ta.shape == tb.shape, (name, i)
        np.testing.assert_array_equal(ta.numpy(), tb.numpy(),
                                      err_msg=f"{name} call {i}")

    def static(s):
      fns = [
          (lambda q: s.search_batched_parallel(q, 5), [None, DIM]),
          (lambda q: s.search_batched(q), [None, DIM]),
          (lambda q: s.search_batched(q, 3), [8, DIM]),
          (lambda q: s.search(q, 3), [DIM]),
      ]
      return [
          [(t.dtype, t.shape.as_list()) for t in tf.function(f)
           .get_concrete_function(tf.TensorSpec(shape, tf.float32))
           .structured_outputs] for f, shape in fns
      ]

    assert static(a) == static(b), (static(a), static(b))
  print("backends agree: identical outputs, dtypes and static shapes")


def make_model(tf, scann_tf, searcher):

  class Model(tf.Module):

    def __init__(self):
      super().__init__()
      self.index = searcher.serialize_to_module()

    @tf.function(input_signature=[
        tf.TensorSpec([None, DIM], tf.float32, name="queries")
    ])
    def serve(self, queries):
      return scann_tf.searcher_from_module(self.index).search_batched_parallel(
          queries, 5)

  return Model()


def check_savedmodel(m, tf, scann, db, queries, tmp):
  s = m.from_pybind(pybind_searchers(scann, db)["tree_ah_reorder"])
  if m.BACKEND == "python":
    for f in (s.serialize_to_module,
              lambda: m.searcher_from_module(tf.Module())):
      expect_raises(NotImplementedError, f, "-DSCANN_BUILD_TF_OP=ON")
      expect_raises(NotImplementedError, f, "docs/tensorflow.md")
    print("  SavedModel: serialize_to_module() / searcher_from_module() "
          "raise NotImplementedError pointing to the op")
    return
  model = make_model(tf, m, s)
  want = model.serve(queries)
  assert_same(want, s.to_pybind().search_batched_parallel(queries, 5),
              "model")
  d = os.path.join(tmp, "savedmodel")
  os.makedirs(d)
  np.save(os.path.join(d, "queries.npy"), queries)
  np.save(os.path.join(d, "want_i.npy"), want.indices.numpy())
  np.save(os.path.join(d, "want_d.npy"), want.distances.numpy())
  tf.saved_model.save(model, os.path.join(d, "model"),
                      signatures={"serving_default": model.serve})
  del model
  gc.collect()
  env = dict(os.environ)
  env.pop("SCANN_TF_BACKEND", None)
  r = subprocess.run([sys.executable, __file__, "--load", d], env=env,
                     capture_output=True, text=True)
  if r.returncode:
    print(r.stdout[-3000:], r.stderr[-5000:])
    raise AssertionError("loading the SavedModel in a fresh process failed")
  print("  " + r.stdout.strip())


def load_savedmodel(d):
  """Run in a fresh process by check_savedmodel: only scann.tf imported."""
  import tensorflow as tf  # pylint: disable=g-import-not-at-top
  import scann.tf  # pylint: disable=g-import-not-at-top  # registers the op
  assert scann.tf.backend() == "op", scann.tf.backend()
  queries = np.load(os.path.join(d, "queries.npy"))
  want = (np.load(os.path.join(d, "want_i.npy")),
          np.load(os.path.join(d, "want_d.npy")))
  loaded = tf.saved_model.load(os.path.join(d, "model"))
  assert_same(loaded.serve(tf.constant(queries)), want, "loaded serve")
  out = loaded.signatures["serving_default"](queries=tf.constant(queries[:4]))
  assert sorted(out) == ["distances", "indices"], sorted(out)
  assert_same((out["indices"], out["distances"]),
              (want[0][:4], want[1][:4]), "serving signature")
  s = scann.tf.searcher_from_module(loaded.index)
  assert isinstance(s, scann.tf.ScannSearcher)
  assert_same(s.search_batched_parallel(queries, 5), want,
              "searcher_from_module of the loaded model")
  print("SavedModel through scann.tf: loaded in a fresh process (op backend "
        "picked automatically); functions, signature and "
        "searcher_from_module match")


def main():
  if len(sys.argv) == 3 and sys.argv[1] == "--load":
    load_savedmodel(sys.argv[2])
    return
  if HIDE_OP:
    sys.modules["scann_tf_ops"] = None
  check_import_isolation()
  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
  try:
    import tensorflow as tf  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    if os.environ.get("SCANN_TEST_REQUIRE_TF"):
      raise
    print(f"SKIPPED: TensorFlow is not importable ({e})")
    sys.exit(SKIP)
  import scann  # pylint: disable=g-import-not-at-top
  os.environ.pop("SCANN_TF_BACKEND", None)
  import scann.tf as scann_tf  # pylint: disable=g-import-not-at-top

  backends = scann_tf.available_backends()
  op_available = "op" in backends
  assert scann_tf.backend() == backends[0], (scann_tf.backend(), backends)
  if HIDE_OP:
    assert not op_available
  elif os.environ.get("SCANN_TEST_REQUIRE_TF_OP") and not op_available:
    scann_tf.get_backend("op")  # raises the reason
  db = dataset(2000)
  queries = dataset(50, seed=1)
  with tempfile.TemporaryDirectory() as tmp:
    check_backend_selection(op_available, tmp)
    for name in ("op", "python"):
      if name not in backends:
        print(f"backend {name}: SKIPPED (scann_tf_ops is not importable; "
              "build with -DSCANN_BUILD_TF_OP=ON)")
        continue
      m = scann_tf.get_backend(name)
      assert m.BACKEND == name
      check_eager(m, tf, scann, db, queries)
      check_errors(m, tf, scann, db, queries)
      check_tf_function(m, tf, scann, db, queries)
      check_short_rows(m, tf)
      check_docids(m, tf, scann, db, queries)
      check_create_and_load(m, tf, scann, db, queries)
      check_concurrency(m, tf, db)
      print(f"backend {name}: eager, tf.function, errors, short rows, docids, "
            "create/load, tf.data and threads match the pybind searcher")
      check_savedmodel(m, tf, scann, db, queries, tmp)
    if op_available:
      check_backends_agree(tf, scann, scann_tf.get_backend("op"),
                           scann_tf.get_backend("python"), db, queries)
  print(f"PASSED (backends {', '.join(backends)}; TensorFlow "
        f"{tf.__version__}, Python {sys.version.split()[0]})")


if __name__ == "__main__":
  main()
