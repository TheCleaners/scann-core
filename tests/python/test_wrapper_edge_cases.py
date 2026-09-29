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

"""Regression tests for edge cases of the Python wrapper (upstream defects):

- upsert() with a docid listed twice: upstream added a new docid to the
  index twice but mapped it once, leaving a duplicate in `docids`. Now a
  ValueError, before anything changes.
- build(docids=ids) kept the caller's list as the searcher's bookkeeping,
  so upsert() appended to it and delete() reordered it. Now a copy.
- search_batched() with zero queries: upstream failed with "Queries have
  dimensionality 0" (or a bare RET_CHECK failure in the parallel path). Now
  empty results shaped like a normal call's.
- A dataset with more rows than 32-bit datapoint indices can address:
  upstream truncated the row count. Now a ValueError. (Tested with zero-width
  rows, which need no memory.)
- spherical=True with vectors that aren't unit-norm: upstream normalized
  points added later but not the ones the index was built from, so the same
  vector scored differently depending on when it was added.
- rebalance() of an index that keeps no float data (tree + bfloat16 brute
  force): a clear error, and the index keeps working.
- upsert() inputs: a 2-D array of any float dtype, a list of rows, one 1-D
  vector, and the pybind searcher's upsert() with a float32 2-D array (read
  in place) or a list of rows, all give the same index; a wrong-sized row
  is an error naming it.

Run with scann-core's build/python on PYTHONPATH:
  PYTHONPATH=build/python python tests/python/test_wrapper_edge_cases.py
"""

import numpy as np
import scann
from scann.scann_ops.py import scann_ops_pybind
from scann.scann_ops.py.scann_builder import ReorderType

DIM = 8


def dataset(n, seed=0, scale=1.0):
  rng = np.random.default_rng(seed)
  return (rng.standard_normal((n, DIM)) * scale).astype(np.float32)


def expect_raises(exc, f, match):
  try:
    f()
  except exc as e:
    assert match in str(e), f"{match!r} not in {str(e)!r}"
    return
  raise AssertionError(f"expected {exc.__name__}({match!r})")


def check_repeated_docids():
  db = dataset(300)
  ids = [f"d{i}" for i in range(300)]
  for name, build in [
      ("bf", lambda b: b.score_brute_force()),
      ("tree_ah", lambda b: b.tree(6, 3, training_sample_size=300)
       .score_ah(2).reorder(30)),
  ]:
    s = build(scann_ops_pybind.builder(db, 5, "squared_l2")).build(
        docids=list(ids))
    before = s.search_batched(db[:20], leaves_to_search=6)
    new = dataset(3, seed=1)
    for docids in (["new", "new"], ["d3", "d3"], ["new", "d3", "new"],
                   ["d3", "new", "d3"]):
      rows = new[:len(docids)]
      for batch_size in (1, 3):
        expect_raises(ValueError, lambda: s.upsert(docids, rows, batch_size),
                      "not unique")
      assert s.size() == 300 and s.docids == ids, name
      assert "new" not in s.docid_to_id, name
      after = s.search_batched(db[:20], leaves_to_search=6)
      assert after[0] == before[0], name
      np.testing.assert_array_equal(after[1], before[1])
    # Distinct docids still work, new and existing mixed.
    s.upsert(["new", "d3", "new2"], new)
    assert s.size() == 302 and len(s.docids) == 302
    assert s.docids[300:] == ["new", "new2"] and s.docid_to_id["new2"] == 301
    for docid, vec in zip(["new", "d3", "new2"], new):
      got, dist = s.search(vec, 1, 300, 6)
      assert got[0] == docid and abs(dist[0]) < 1e-3, (name, docid, got, dist)


def check_docids_not_aliased():
  # The searcher keeps its own copy: upstream appended to the caller's list
  # on upsert() and reordered it on delete().
  db = dataset(10)
  ids = [f"d{i}" for i in range(10)]
  s = scann_ops_pybind.builder(db, 3, "squared_l2").score_brute_force().build(
      docids=ids)
  s.upsert(["new"], dataset(1, seed=3))
  s.delete(["d0"])
  assert ids == [f"d{i}" for i in range(10)], ids
  assert s.docids == ["new"] + [f"d{i}" for i in range(1, 10)], s.docids


def check_zero_queries():
  db = dataset(200)
  empty = np.zeros((0, DIM), np.float32)
  for name, build in [
      ("bf", lambda b: b.score_brute_force()),
      ("ah_reorder", lambda b: b.score_ah(2).reorder(30)),
      ("tree", lambda b: b.tree(5, 2, training_sample_size=200).score_ah(2)),
  ]:
    for docids in (None, [str(i) for i in range(200)]):
      s = build(scann_ops_pybind.builder(db, 7, "squared_l2")).build(
          docids=docids)
      for search in (s.search_batched, s.search_batched_parallel):
        for k, width in ((None, 7), (3, 3)):
          idx, dist = search(empty, final_num_neighbors=k)
          ref_idx, ref_dist = search(db[:2], final_num_neighbors=k)
          assert dist.shape == (0, width) and dist.dtype == ref_dist.dtype
          if docids is None:
            assert idx.shape == (0, width) and idx.dtype == ref_idx.dtype, (
                name, idx.shape, idx.dtype)
          else:
            assert idx == [], (name, idx)
        expect_raises(ValueError, lambda: search(empty, final_num_neighbors=0),
                      "final_num_neighbors")
        expect_raises(ValueError, lambda: search(np.zeros((0, DIM + 1),
                                                          np.float32)),
                      "dimensionality")
        # A wrong dimensionality is the same clear error for non-empty
        # batches, in both paths.
        expect_raises(RuntimeError, lambda: search(np.zeros((2, DIM + 1),
                                                            np.float32)),
                      "dimensionality")


def check_too_many_rows():
  db = dataset(100)
  config = scann_ops_pybind.builder(db, 5, "dot_product").score_brute_force(
  ).create_config()
  # Zero-width rows: a 2**32-row array that needs no memory.
  huge = np.zeros((2**32, 0), np.float32)
  expect_raises(ValueError,
                lambda: scann_ops_pybind.create_searcher(huge, config),
                "at most 4294967295")


def check_spherical_is_consistent():
  # Not unit-norm on purpose. Before, the index stored these as given, and
  # points upserted later normalized in some configurations (float data) and
  # as given in others.
  db = dataset(600, scale=3.0)
  q = dataset(20, seed=5)
  unit = db / np.linalg.norm(db, axis=1, keepdims=True)
  # (name, scoring, can rebalance, tolerance). int8 quantizes a point
  # slightly differently at build time than when it's added later.
  configs = [
      ("bf", lambda b: b.score_brute_force(), True, 1e-4),
      ("int8", lambda b: b.score_brute_force(quantize=True), False, 1e-2),
      ("ah_bf16_reorder",
       lambda b: b.score_ah(2).reorder(40, quantize=ReorderType.BFLOAT16),
       True, 1e-4),
  ]
  for distance in ("squared_l2", "dot_product"):
    for name, build, can_rebalance, tol in configs:
      s = build(scann_ops_pybind.builder(db, 10, distance).tree(
          12, 4, training_sample_size=600, spherical=True)).build(
              docids=[str(i) for i in range(600)])
      # An upserted copy of a point scores like the original, also after a
      # rebalance (which failed upstream for these configs).
      s.upsert(["copy_of_3"], db[3][None])
      for rebalanced in (False, True):
        if rebalanced:
          if not can_rebalance:
            break
          s.rebalance()
        idx, dist = s.search(db[3], 601, 700, 12)
        d = dict(zip(idx, dist))
        np.testing.assert_allclose(d["copy_of_3"], d["3"], rtol=tol,
                                   atol=tol, err_msg=f"{distance} {name}")
    # Stored vectors are the normalized ones: exact distances match brute
    # force over the normalized dataset.
    s = (scann_ops_pybind.builder(db, 10, distance)
         .tree(12, 4, training_sample_size=600, spherical=True)
         .score_brute_force().build())
    idx, dist = s.search_batched(q, 5, leaves_to_search=12)
    for qi in range(len(q)):
      ref = (((unit[idx[qi]] - q[qi]) ** 2).sum(1) if distance == "squared_l2"
             else unit[idx[qi]] @ q[qi])
      np.testing.assert_allclose(dist[qi], ref, rtol=1e-4, atol=1e-4)

  # Already unit-norm input is stored as given, bit for bit, as upstream did:
  # exhaustive searches agree exactly with a non-spherical tree's, which
  # stores its input unchanged.
  results = []
  for spherical in (True, False):
    s = (scann_ops_pybind.builder(unit, 10, "dot_product")
         .tree(12, 4, training_sample_size=600, spherical=spherical)
         .score_brute_force().build())
    results.append(s.search_batched(q, 5, leaves_to_search=12))
  np.testing.assert_array_equal(results[0][0], results[1][0])
  np.testing.assert_array_equal(results[0][1], results[1][1])


def check_rebalance_without_float_data():
  db = dataset(600)
  ids = [str(i) for i in range(600)]
  s = (scann_ops_pybind.builder(db, 5, "dot_product")
       .tree(8, 3, training_sample_size=600)
       .score_brute_force(quantize=ReorderType.BFLOAT16).build(docids=ids))
  before = s.search_batched(db[:30], leaves_to_search=8)
  expect_raises(RuntimeError, s.rebalance, "keeps only quantized data")
  after = s.search_batched(db[:30], leaves_to_search=8)
  assert after[0] == before[0]
  np.testing.assert_array_equal(after[1], before[1])
  s.upsert(["new"], dataset(1, seed=9))
  s.delete(["0"])
  assert s.size() == 600


def check_upsert_inputs():
  db = dataset(300)
  new = dataset(5, seed=2)
  ids = [f"d{i}" for i in range(300)]
  results = []
  for kind in ("float32", "float64", "list", "raw2d", "rawlist", "rows"):
    s = (scann_ops_pybind.builder(db, 5, "squared_l2")
         .tree(6, 3, training_sample_size=300).score_ah(2).reorder(30)
         .build(docids=list(ids)))
    docids = ["n0", "d3", "n1", "n2", "d7"]
    if kind == "float32":
      s.upsert(docids, new)
    elif kind == "float64":
      s.upsert(docids, new.astype(np.float64))
    elif kind == "list":
      s.upsert(docids, [list(map(float, r)) for r in new])
    elif kind == "rows":  # one docid and one 1-D vector per call
      for d, r in zip(docids, new):
        s.upsert(d, r)
    else:  # the pybind searcher directly; indices None = add
      idx = [None, 3, None, None, 7]
      vecs = new if kind == "raw2d" else [r for r in new]
      s.searcher.upsert(idx, vecs, 256)
      s.docids.extend(["n0", "n1", "n2"])
      s.docids[3], s.docids[7] = "d3", "d7"
    assert s.size() == 303, (kind, s.size())
    results.append(s.search_batched(new, leaves_to_search=6))
  for i, d in results[1:]:
    assert i == results[0][0]
    np.testing.assert_array_equal(d, results[0][1])
  expect_raises(ValueError, lambda: s.upsert(["x", "y"], dataset(2)[:, :5]),
                "Upsert vector has dimensionality 5")
  expect_raises(ValueError,
                lambda: s.searcher.upsert([None], [np.zeros(5, np.float32)],
                                          256),
                "Upsert vector has dimensionality 5")
  expect_raises(ValueError,
                lambda: s.searcher.upsert([None], np.zeros(8, np.float32),
                                          256),
                "two-dimensional")


def main():
  check_repeated_docids()
  check_docids_not_aliased()
  check_zero_queries()
  check_too_many_rows()
  check_spherical_is_consistent()
  check_rebalance_without_float_data()
  check_upsert_inputs()
  print("PASSED")


if __name__ == "__main__":
  main()
