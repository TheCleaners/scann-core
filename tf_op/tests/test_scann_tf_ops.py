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

"""scann_tf_ops, scann-core's TensorFlow op (tf_op/).

- Parity: search / search_batched / search_batched_parallel return exactly
  (bit for bit) what the pybind searcher they were made from returns, for
  every config tests/python/test_serialization.py round-trips (brute force
  float/int8/bfloat16, AH, trees with each, SOAR), with default and
  explicit k, pre-reorder and leaves, eagerly and in tf.function (with an
  unknown batch dimension; static shapes). SOAR + AH + reordering only up
  to the last bit of a distance, which varies between identical searches
  with the pybind searcher too. Trees with every point deleted.
- The searcher is built once: repeated eager calls, several tf.functions,
  a reloaded SavedModel's functions and signature, and a
  searcher_from_module() of it share one build; deleting the searcher
  frees it.
- Stale state: after assigning other values to the variables (another
  index, also one with identical file sizes) or restoring a checkpoint of
  another index, searches use the new index (rebuilt).
- SavedModel: save, then load in a fresh process; the functions, the
  serving signature and searcher_from_module() match the original.
- Round trips: to_pybind() / serialize() / load_searcher() (with docids;
  relative and absolute manifests), create_searcher().
- Bad inputs and damaged index tensors: tf.errors.InvalidArgumentError (or
  another OpError), never a crash.
- Concurrency: one tf.function called from several threads.

Without TensorFlow the test exits with 77, which ctest reports as skipped
(SKIP_RETURN_CODE), unless SCANN_TEST_REQUIRE_TF is set.

Run with the build's python directory (built with -DSCANN_BUILD_TF_OP=ON)
on PYTHONPATH:
  PYTHONPATH=build/python python tf_op/tests/test_scann_tf_ops.py
"""

import concurrent.futures
import gc
import os
import subprocess
import sys
import tempfile

import numpy as np

SKIP = 77
DIM, LEAVES = 8, 12


def dataset(n, seed=0):
  return np.random.default_rng(seed).standard_normal((n, DIM)).astype(
      np.float32)


def pybind_builder(scann, db, cfg):
  from scann.scann_ops.py.scann_builder import ReorderType  # pylint: disable=g-import-not-at-top
  b = scann.scann_ops_pybind.builder(db, 10, "dot_product")
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


CONFIGS = ("bf", "bf_int8", "bf_bf16", "ah", "tree_bf", "tree_bf_int8",
           "tree_bf_bf16", "tree_ah", "tree_ah_no_reorder", "tree_ah_soar")

# (final_num_neighbors, pre_reorder_num_neighbors, leaves_to_search)
PARAMS = ((None, None, None), (5, None, None), (5, 50, LEAVES), (3, 20, 2))


def _assert_same(got, want, what, exact=True):
  """got: (tensor, tensor) from the op; want: arrays from pybind.

  exact=False for SOAR + AH + reordering, whose distances vary in the last
  bit between identical searches with the pybind searcher itself (so
  neighbors with nearly equal distances may swap places): the same
  neighbors per row, distances equal to 1e-6.
  """
  gi, gd = (np.asarray(t) for t in got)
  wi, wd = (np.asarray(a) for a in want)
  assert gi.dtype == np.int32 and gd.dtype == np.float32, (what, gi.dtype,
                                                           gd.dtype)
  assert gi.shape == wi.shape and gd.shape == wd.shape, (what, gi.shape,
                                                         wi.shape)
  wi = wi.astype(np.int64)
  if exact:
    assert (gi == wi).all(), (what, gi, wi)
    np.testing.assert_array_equal(gd, wd, err_msg=what)  # NaN == NaN here
  else:
    assert (np.sort(gi, axis=-1) == np.sort(wi, axis=-1)).all(), (what, gi, wi)
    np.testing.assert_allclose(gd, wd, rtol=1e-6, err_msg=what)


assert_same = _assert_same


def expect_error(tf, f, match, what, exc=None):
  exc = exc or tf.errors.InvalidArgumentError
  try:
    f()
  except exc as e:
    assert match in str(e), (what, match, str(e))
    return
  raise AssertionError(f"{what}: expected {exc.__name__} matching {match!r}")


def check_parity(tf, ops, scann, db, queries):
  """Every config, every parameter set, eager and in tf.function."""
  for cfg in CONFIGS:
    p = pybind_builder(scann, db, cfg).build()
    raw = p.searcher
    s = ops.from_pybind(p)
    exact = "soar" not in cfg

    def assert_same(got, want, what, exact=exact):  # pylint: disable=redefined-outer-name
      _assert_same(got, want, what, exact)

    @tf.function(input_signature=[tf.TensorSpec([None, DIM], tf.float32)])
    def batched_fn(q, s=s):
      return s.search_batched_parallel(q, 5, 50, LEAVES, batch_size=4)

    for k, pre, leaves in PARAMS:
      args = [-1 if v is None else v for v in (k, pre, leaves)]
      what = f"{cfg} k={k} pre={pre} leaves={leaves}"
      assert_same(s.search_batched(queries, k, pre, leaves),
                  raw.search_batched(queries, *args, False, 256), what)
      for bs in (256, 3):
        assert_same(
            s.search_batched_parallel(queries, k, pre, leaves, batch_size=bs),
            raw.search_batched(queries, *args, True, bs),
            f"{what} parallel batch_size={bs}")
      for i in (0, 7):
        assert_same(s.search(queries[i], k, pre, leaves),
                    raw.search(queries[i], *args), f"{what} single {i}")
      # Search parameters as tensors (no static k).
      assert_same(
          s.search_batched(queries, tf.constant(args[0]), tf.constant(args[1]),
                           tf.constant(args[2])),
          raw.search_batched(queries, *args, False, 256), f"{what} tensors")
    for n in (len(queries), 1):
      assert_same(batched_fn(queries[:n]),
                  raw.search_batched(queries[:n], 5, 50, LEAVES, True, 4),
                  f"{cfg} tf.function {n} queries")
    shape = batched_fn.get_concrete_function().structured_outputs[0].shape
    assert shape.as_list() == [None, 5], (cfg, shape)
  print("parity:", len(CONFIGS), "configs x", len(PARAMS),
        "parameter sets, eager and tf.function: identical")


def check_edge_cases(tf, ops, scann, db, queries):
  """Short rows, zero queries, empty (all-deleted) indexes, docids."""
  small = ops.builder(db[:3], 10, "dot_product").score_brute_force().build()
  r = small.search_batched(queries[:4], 5)
  assert r.indices.shape == (4, 5), r.indices.shape
  assert np.isnan(r.distances.numpy()[:, 3:]).all()
  assert (r.indices.numpy()[:, 3:] == 0).all()
  assert small.search_batched(queries[:4]).indices.shape == (4, 3)
  assert small.search(queries[0], 5).index.shape == (3,)
  r = small.search_batched(queries[:0], 5)
  assert r.indices.shape == (0, 5), r.indices.shape
  assert small.search_batched(queries[:0]).indices.shape == (0, 10)

  for cfg in ("tree_bf", "tree_bf_int8", "tree_bf_bf16", "tree_ah_no_reorder",
              "tree_ah_soar", "bf"):
    p = pybind_builder(scann, db[:600], cfg).build(
        docids=[f"d{i}" for i in range(600)])
    p.delete([f"d{i}" for i in range(600)])
    s = ops.from_pybind(p)
    got = s.search_batched(queries[:3], 5, 50, LEAVES)
    assert got.indices.shape == (3, 5) and np.isnan(
        got.distances.numpy()).all(), cfg
    assert s.search(queries[0], 5, 50, LEAVES).index.shape == (0,), cfg
    assert s.to_pybind().size() == 0, cfg

  docids = [f"doc{i}" for i in range(len(db))]
  p = scann.scann_ops_pybind.builder(db, 10, "squared_l2").score_ah(
      2).reorder(40).build(docids=docids)
  s = ops.from_pybind(p)
  assert "scann_docids.pkl" in s.files()
  back = s.to_pybind()
  assert back.docids == docids
  assert_same(s.search_batched(queries), back.searcher.search_batched(
      queries, -1, -1, -1, False, 256), "docids: indices, not docids")
  print("edge cases: short rows, zero queries, all-deleted trees, docids")


def check_round_trips(tf, ops, scann, db, queries, tmp):
  p = pybind_builder(scann, db, "tree_ah_soar").build(
      docids=[str(i) for i in range(len(db))])
  want = p.searcher.search_batched(queries, 5, -1, -1, False, 256)
  for relative in (True, False):
    d = os.path.join(tmp, f"index_{relative}")
    os.mkdir(d)
    p.serialize(d, relative_path=relative)
    s = ops.load_searcher(d)
    assert_same(s.search_batched(queries, 5), want, f"load relative={relative}",
                exact=False)
    assert s.to_pybind().docids == p.docids
    d2 = d + "_again"
    os.mkdir(d2)
    s.serialize(d2, relative_path=relative)
    t = scann.scann_ops_pybind.load_searcher(d2)
    assert t.docids == p.docids
    assert_same(ops.load_searcher(d2).search_batched(queries, 5), want,
                f"re-serialized relative={relative}", exact=False)

  # (Brute force: two trainings of a tree needn't agree.)
  config = pybind_builder(scann, db, "bf_int8").create_config()
  s = ops.create_searcher(db, config, shared_name="custom_id")
  assert s.shared_name == "custom_id"
  t = scann.scann_ops_pybind.create_searcher(db, config)
  assert_same(s.search_batched(queries), t.searcher.search_batched(
      queries, -1, -1, -1, False, 256), "create_searcher")

  # An interrupted serialize: load_searcher raises the loader's error.
  d = os.path.join(tmp, "incomplete")
  os.mkdir(d)
  with open(os.path.join(d, "scann_assets.pbtxt"), "w") as f:
    f.write("scann_core_incomplete_serialization: true\n")
  open(os.path.join(d, "scann_config.pb"), "wb").close()
  try:
    ops.load_searcher(d)
    raise AssertionError("loaded an incomplete directory")
  except (RuntimeError, ValueError) as e:
    assert "incomplete" in str(e), str(e)
  print("round trips: load_searcher (relative and absolute), serialize, "
        "to_pybind, docids, create_searcher")


def check_build_once_and_lifetime(tf, ops, scann, db, queries):
  before = ops.stats()
  s = ops.builder(db, 10, "dot_product").tree(
      LEAVES, 4, training_sample_size=len(db)).score_ah(2).reorder(30).build()
  assert ops.stats()["builds"] == before["builds"], "built before a search"
  for _ in range(3):
    s.search_batched(queries)
    s.search(queries[0])
  s.search_batched_parallel(queries[:5], 3)

  @tf.function
  def f1(q):
    return s.search_batched(q, 5)

  @tf.function(input_signature=[tf.TensorSpec([None, DIM], tf.float32)])
  def f2(q):
    return s.search_batched_parallel(q), s.search(q[0], 2)

  f1(queries)
  f1(queries[:2])
  f2(queries)
  after = ops.stats()
  assert after["builds"] == before["builds"] + 1, (before, after)
  assert after["live_searchers"] == before["live_searchers"] + 1

  del s, f1, f2
  gc.collect()
  assert ops.stats()["live_searchers"] == before["live_searchers"], (
      "deleting the searcher and its functions didn't free it",
      ops.stats())
  print("build once: eager calls and two tf.functions built 1 searcher; "
        "freed with them")


def check_stale_state(tf, ops, scann, db, queries, tmp):
  """New variable values are never searched with the old searcher."""
  a_p = scann.scann_ops_pybind.builder(db, 10,
                                       "dot_product").score_brute_force().build()
  # Same config and shapes, other data: every file has the same size.
  b_p = scann.scann_ops_pybind.builder(dataset(len(db), seed=5), 10,
                                       "dot_product").score_brute_force().build()
  c_p = pybind_builder(scann, db, "tree_ah").build()
  a, b, c, a2 = (ops.from_pybind(x) for x in (a_p, b_p, c_p, a_p))
  assert {n: len(v) for n, v in a.files().items()} == {
      n: len(v) for n, v in b.files().items()}

  def want(p, k=5):
    return p.searcher.search_batched(queries, k, -1, -1, False, 256)

  @tf.function(input_signature=[tf.TensorSpec([None, DIM], tf.float32)])
  def f(q):
    return a.search_batched(q, 5)

  assert_same(f(queries), want(a_p), "before")
  builds = ops.stats()["builds"]
  for src, src_p, what in ((b, b_p, "same sizes"), (c, c_p, "other config"),
                           (a2, a_p, "back")):
    a.asset_names.assign(src.asset_names)
    a.asset_contents.assign(src.asset_contents)
    assert_same(f(queries), want(src_p), f"assigned: {what}")
    assert_same(a.search_batched(queries, 5), want(src_p),
                f"assigned, eager: {what}")
  assert ops.stats()["builds"] > builds

  # A checkpoint of another index restored into a searcher's module.
  ckpt = tf.train.Checkpoint(index=b).save(os.path.join(tmp, "ckpt", "b"))
  tf.train.Checkpoint(index=a).restore(ckpt).assert_consumed()
  assert_same(f(queries), want(b_p), "restored checkpoint")
  print("stale state: assigned variables (also with identical file sizes) "
        "and a restored checkpoint are rebuilt")


def check_errors(tf, ops, scann, db, queries):
  s = ops.builder(db, 10, "dot_product").score_ah(2).reorder(30).build()
  raw_ops = ops._ops  # pylint: disable=protected-access
  names, contents = s.asset_names, s.asset_contents

  def raw_batched(q=queries, k=5, parallel=False, batch_size=256,
                  names=names, contents=contents):
    return raw_ops.scann_core_search_batched(names, contents, q, k, -1, -1,
                                             parallel, batch_size,
                                             index_id="errors")

  expect_error(tf, lambda: raw_batched(q=queries[0]), "two-dimensional",
               "1-D queries to the batched op")
  expect_error(tf, lambda: raw_ops.scann_core_search(
      names, contents, queries, 5, -1, -1, index_id="errors"),
               "one-dimensional", "2-D query to the single op")
  expect_error(tf, lambda: raw_batched(q=queries[:, :4]), "dimensionality",
               "wrong width")
  expect_error(tf, lambda: s.search(queries[0, :4]), "dimensionality",
               "wrong width, single")
  bad = queries.copy()
  bad[2, 1] = np.nan
  expect_error(tf, lambda: raw_batched(q=bad), "NaN", "NaN query")
  expect_error(tf, lambda: raw_batched(k=[5, 5]), "scalar", "vector k")
  expect_error(tf, lambda: raw_batched(k=0), "", "k = 0")
  expect_error(tf, lambda: raw_batched(q=queries[:0], k=0), "> 0",
               "k = 0, no queries")
  expect_error(tf, lambda: raw_batched(parallel=True, batch_size=0),
               "batch_size", "batch_size 0")
  expect_error(tf, lambda: raw_batched(contents=contents[:-1]),
               "same length", "names and contents of different lengths")
  expect_error(tf, lambda: raw_batched(names=[["a"]], contents=[["b"]]),
               "", "matrix of names")
  files = s.files()
  for victim, damage in (("dataset.npy", lambda b: b[:len(b) // 2]),
                         ("hashed_dataset.npy", lambda b: b[:100]),
                         ("scann_config.pb", lambda b: b"\xff\xff garbage"),
                         ("scann_assets.pbtxt", lambda b: b"nonsense {"),
                         ("ah_codebook.pb", None)):
    damaged = dict(files)
    if damage is None:
      del damaged[victim]
    else:
      damaged[victim] = damage(files[victim])
    names_t = tf.constant(list(damaged))
    contents_t = tf.constant(list(damaged.values()))
    expect_error(tf, lambda: raw_batched(names=names_t, contents=contents_t),
                 "Can't build the ScaNN searcher", f"damaged {victim}",
                 exc=tf.errors.OpError)
  # Still fine afterwards.
  assert raw_batched()[0].shape == (len(queries), 5)
  print("errors: bad queries, parameters and damaged index tensors raise "
        "OpErrors")


def check_concurrency(tf, ops, db):
  s = ops.builder(db, 10, "dot_product").tree(
      LEAVES, 4, training_sample_size=len(db)).score_ah(2).reorder(30).build()
  raw = s.to_pybind().searcher

  @tf.function(input_signature=[tf.TensorSpec([None, DIM], tf.float32)])
  def f(q):
    return s.search_batched(q, 5), s.search_batched_parallel(q, 7)

  def worker(seed):
    q = dataset(40, seed=seed)
    for _ in range(10):
      (i1, d1), (i2, d2) = f(q)
      assert_same((i1, d1), raw.search_batched(q, 5, -1, -1, False, 256),
                  "concurrent")
      assert_same((i2, d2), raw.search_batched(q, 7, -1, -1, True, 256),
                  "concurrent parallel")
    return True

  with concurrent.futures.ThreadPoolExecutor(8) as pool:
    assert all(pool.map(worker, range(100, 116)))
  print("concurrency: 16 threads x 10 calls of one tf.function: correct")


def make_model(tf, ops, searcher):

  class Model(tf.Module):

    def __init__(self):
      super().__init__()
      self.index = searcher.serialize_to_module()

    @tf.function(input_signature=[
        tf.TensorSpec([None, DIM], tf.float32, name="queries")
    ])
    def serve(self, queries):
      return ops.searcher_from_module(self.index).search_batched_parallel(
          queries, 5)

    @tf.function(input_signature=[tf.TensorSpec([DIM], tf.float32)])
    def one(self, query):
      return ops.searcher_from_module(self.index).search(query, 3)

  return Model()


def check_savedmodel(tf, ops, scann, db, queries, tmp):
  p = pybind_builder(scann, db, "tree_ah").build()
  m = make_model(tf, ops, ops.from_pybind(p))
  want = m.serve(queries)
  assert_same(want, p.searcher.search_batched(queries, 5, -1, -1, True, 256),
              "model")
  np.save(os.path.join(tmp, "want_i.npy"), want.indices.numpy())
  np.save(os.path.join(tmp, "want_d.npy"), want.distances.numpy())
  np.save(os.path.join(tmp, "queries.npy"), queries)
  saved = os.path.join(tmp, "saved_model")
  tf.saved_model.save(m, saved, signatures={"serving_default": m.serve})
  del m
  gc.collect()
  r = subprocess.run([sys.executable, __file__, "--load", tmp],
                     capture_output=True, text=True)
  if r.returncode:
    print(r.stdout[-3000:], r.stderr[-5000:])
    raise AssertionError("loading the SavedModel in a fresh process failed")
  print(r.stdout.strip())


def load_savedmodel(tmp):
  """Run in a fresh process by check_savedmodel."""
  import tensorflow as tf  # pylint: disable=g-import-not-at-top
  import scann_tf_ops as ops  # pylint: disable=g-import-not-at-top
  queries = np.load(os.path.join(tmp, "queries.npy"))
  want = (np.load(os.path.join(tmp, "want_i.npy")),
          np.load(os.path.join(tmp, "want_d.npy")))
  m = tf.saved_model.load(os.path.join(tmp, "saved_model"))
  assert ops.stats() == {"live_searchers": 0, "builds": 0}, ops.stats()
  for n in (len(queries), 2):
    got = m.serve(tf.constant(queries[:n]))
    assert_same(got, (want[0][:n], want[1][:n]), f"loaded serve, {n}")
  out = m.signatures["serving_default"](queries=tf.constant(queries))
  assert sorted(out) == ["distances", "indices"], sorted(out)
  assert_same((out["indices"], out["distances"]), want, "serving signature")
  s = ops.searcher_from_module(m.index)
  assert_same(s.search_batched_parallel(queries, 5), want,
              "searcher_from_module of the loaded model")
  i, _ = m.one(tf.constant(queries[3]))
  assert (i.numpy() == want[0][3][:3]).all()
  assert ops.stats() == {"live_searchers": 1, "builds": 1}, ops.stats()
  print("SavedModel: loaded in a fresh process; functions, serving signature "
        "and searcher_from_module match, 1 build")


def main():
  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
  if len(sys.argv) == 3 and sys.argv[1] == "--load":
    load_savedmodel(sys.argv[2])
    return
  try:
    import tensorflow as tf  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    if os.environ.get("SCANN_TEST_REQUIRE_TF"):
      raise
    print(f"SKIPPED: TensorFlow is not importable ({e})")
    sys.exit(SKIP)
  import scann  # pylint: disable=g-import-not-at-top
  import scann_tf_ops as ops  # pylint: disable=g-import-not-at-top

  db = dataset(1200)
  queries = dataset(20, seed=1)
  with tempfile.TemporaryDirectory() as tmp:
    check_parity(tf, ops, scann, db, queries)
    check_edge_cases(tf, ops, scann, db, queries)
    check_round_trips(tf, ops, scann, db, queries, tmp)
    check_build_once_and_lifetime(tf, ops, scann, db, queries)
    check_stale_state(tf, ops, scann, db, queries, tmp)
    check_errors(tf, ops, scann, db, queries)
    check_concurrency(tf, ops, db)
    check_savedmodel(tf, ops, scann, db, queries, tmp)
  print(f"PASSED (TensorFlow {tf.__version__}, Python "
        f"{sys.version.split()[0]})")


if __name__ == "__main__":
  main()
