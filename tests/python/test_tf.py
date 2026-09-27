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

"""scann.tf, the TensorFlow wrapper around the pybind searcher.

- `import scann` doesn't import TensorFlow, and `import scann.tf` without
  TensorFlow is a clear ImportError (checked in subprocesses, with or
  without TensorFlow installed).
- search / search_batched / search_batched_parallel return exactly what the
  wrapped pybind searcher returns, as int32 / float32 tensors named like
  upstream's op outputs, eagerly and inside tf.function (also with an
  unknown batch dimension), with static shapes where they are known.
- Short rows (fewer points than k), zero queries, docids (results are
  indices, as with upstream's op), create_searcher / load_searcher.
- serialize_to_module() / searcher_from_module() raise NotImplementedError.
- Concurrent use: a tf.data map with parallel calls, and a tf.function
  called from several Python threads.

Without TensorFlow the test exits with 77, which ctest reports as skipped
(SKIP_RETURN_CODE), unless SCANN_TEST_REQUIRE_TF is set.

Run with scann-core's build/python on PYTHONPATH:
  PYTHONPATH=build/python python tests/python/test_tf.py
"""

import concurrent.futures
import os
import subprocess
import sys
import tempfile

import numpy as np

SKIP = 77
DIM = 16


def check_import_isolation():
  """`import scann` leaves TensorFlow alone; scann.tf says what's missing."""
  code = ("import sys, scann\n"
          "from scann.scann_ops.py import scann_ops_pybind\n"
          "assert 'tensorflow' not in sys.modules, 'import scann imported TF'\n")
  subprocess.run([sys.executable, "-c", code], check=True)
  # A blocked import behaves like a missing package.
  code = ("import sys\n"
          "sys.modules['tensorflow'] = None\n"
          "import scann\n"
          "try:\n"
          "  import scann.tf\n"
          "except ImportError as e:\n"
          "  assert 'scann-core[tf]' in str(e), str(e)\n"
          "else:\n"
          "  raise AssertionError('import scann.tf worked without TF')\n")
  subprocess.run([sys.executable, "-c", code], check=True)


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


def searchers(scann_tf, tf, db):
  bf = scann_tf.builder(tf.constant(db), 10, "dot_product").score_brute_force()
  tree = (scann_tf.builder(db, 10, "squared_l2")
          .tree(20, 5, training_sample_size=len(db)).score_ah(2).reorder(50))
  return {"brute_force": bf.build(), "tree_ah_reorder": tree.build()}


def check_eager(scann_tf, tf, db, queries):
  for name, s in searchers(scann_tf, tf, db).items():
    p = s.searcher  # the pybind searcher
    r = s.search(queries[0])
    assert r.index is r[0] and r.distance is r[1]
    assert_same(r, p.search(queries[0]), f"{name} search")
    r = s.search_batched(queries)
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
    # Zero queries.
    r = s.search_batched(queries[:0])
    assert r.indices.shape == (0, 10), r.indices.shape
    # Wrong ranks are caught before searching.
    expect_raises(ValueError, lambda: s.search(queries), "1-dimensional")
    expect_raises(ValueError, lambda: s.search_batched(queries[0]),
                  "2-dimensional")
    # Wrong dimensionality: the pybind searcher's error.
    expect_raises(Exception, lambda: s.search_batched(queries[:, :3]),
                  "dimensionality")


def check_tf_function(scann_tf, tf, db, queries):
  s = searchers(scann_tf, tf, db)["tree_ah_reorder"]
  p = s.searcher
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

  # Errors from inside the graph are TF errors carrying ScaNN's message.
  expect_raises(tf.errors.OpError,
                lambda: tf.function(lambda q: s.search_batched(q))(
                    queries[:, :3]), "dimensionality")

  # Building needs concrete data.
  expect_raises(ValueError,
                lambda: tf.function(lambda d: scann_tf.builder(d, 5, "dot_product")
                                    .score_brute_force().build())(db),
                "eagerly")


def check_short_rows(scann_tf, tf):
  db = dataset(3)  # fewer points than k
  s = scann_tf.builder(db, 10, "dot_product").score_brute_force().build()
  q = dataset(4, seed=1)
  r = s.search_batched(q)  # default k: as wide as the longest row
  assert r.indices.shape == (4, 3), r.indices.shape
  r = s.search_batched(q, final_num_neighbors=5)  # explicit k: padded to it
  assert r.indices.shape == (4, 5), r.indices.shape
  assert np.isnan(r.distances.numpy()[:, 3:]).all()
  assert (r.indices.numpy()[:, 3:] == 0).all()
  assert_same(r, s.searcher.search_batched(q, final_num_neighbors=5), "padded")
  r = s.search(q[0])
  assert r.index.shape == (3,), r.index.shape


def check_docids(scann_tf, tf, db, queries):
  """Results are indices; the pybind searcher maps them to docids."""
  docids = [f"d{i}" for i in range(len(db))]
  s = (scann_tf.builder(db, 5, "squared_l2").score_brute_force()
       .build(docids=docids))
  p = s.searcher
  idx, dist = s.search_batched(queries)
  want_docids, want_dist = p.search_batched(queries)
  assert [[p.docids[i] for i in row] for row in idx.numpy()] == want_docids
  np.testing.assert_array_equal(dist.numpy(), want_dist)
  # Mapping to docids in the graph, as docs/tensorflow.md shows.
  table = tf.constant(p.docids)
  f = tf.function(lambda q: tf.gather(table, s.search_batched(q).indices))
  assert f(queries).numpy().astype(str).tolist() == want_docids
  # Updates through the pybind searcher are seen by the wrapper.
  p.delete(["d0", "d1"])
  p.upsert(["new"], db[:1] + 100)
  idx, _ = s.search_batched(queries)
  assert [[p.docids[i] for i in row] for row in idx.numpy()] == (
      p.search_batched(queries)[0])
  assert s.search(db[:1][0] + 100).index.numpy()[0] == p.docid_to_id["new"]


def check_create_and_load(scann_tf, tf, db, queries):
  b = scann_tf.builder(db, 5, "dot_product").tree(10, 3).score_ah(2)
  config = b.create_config()
  s = scann_tf.create_searcher(tf.constant(db), tf.constant(config),
                               training_threads=tf.constant(2),
                               container="", shared_name="ignored")
  assert isinstance(s, scann_tf.ScannSearcher)
  want = s.searcher.search_batched(queries)
  assert_same(s.search_batched(queries), want, "create_searcher")
  with tempfile.TemporaryDirectory() as d:
    s.serialize(d)
    loaded = scann_tf.load_searcher(d)
    assert_same(loaded.search_batched(queries), want, "load_searcher")
    # A pybind searcher can be wrapped directly.
    from scann.scann_ops.py import scann_ops_pybind  # pylint: disable=g-import-not-at-top
    wrapped = scann_tf.ScannSearcher(scann_ops_pybind.load_searcher(d))
    assert_same(wrapped.search_batched(queries), want, "wrapped")
  expect_raises(TypeError, lambda: scann_tf.ScannSearcher(s.searcher.searcher),
                "scann_ops_pybind.ScannSearcher")


def check_no_savedmodel(scann_tf, tf, db):
  s = scann_tf.builder(db, 5, "dot_product").score_brute_force().build()
  expect_raises(NotImplementedError, s.serialize_to_module, "docs/tensorflow.md")
  expect_raises(NotImplementedError,
                lambda: scann_tf.searcher_from_module(tf.Module()),
                "docs/tensorflow.md")


def check_concurrency(scann_tf, tf, db):
  s = (scann_tf.builder(db, 1, "squared_l2")
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


def main():
  check_import_isolation()
  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
  try:
    import tensorflow as tf  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    if os.environ.get("SCANN_TEST_REQUIRE_TF"):
      raise
    print(f"SKIPPED: TensorFlow is not importable ({e})")
    sys.exit(SKIP)
  from scann import tf as scann_tf  # pylint: disable=g-import-not-at-top

  db = dataset(2000)
  queries = dataset(50, seed=1)
  check_eager(scann_tf, tf, db, queries)
  check_tf_function(scann_tf, tf, db, queries)
  check_short_rows(scann_tf, tf)
  check_docids(scann_tf, tf, db, queries)
  check_create_and_load(scann_tf, tf, db, queries)
  check_no_savedmodel(scann_tf, tf, db)
  check_concurrency(scann_tf, tf, db)
  print(f"PASSED (TensorFlow {tf.__version__}, Python "
        f"{sys.version.split()[0]})")


if __name__ == "__main__":
  main()
