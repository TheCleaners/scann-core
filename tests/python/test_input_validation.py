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

"""Regression test: bad inputs are clean errors, not crashes or garbage.

Upstream crashed the process on
  * NaN/infinity data when building a tree (QCHECK abort in k-means);
  * upserting a NaN/infinity vector into a tree (read of leaf_mutators_[-1]);
  * leaves_to_search on a brute-force int8 index (bad parameter cast);
and silently returned garbage for NaN/infinity queries in batched search. A
batch upsert failing on a later row left the earlier rows in the index but
not in the docid bookkeeping, and short batched results with docids mapped
their padding to docids[0].

Each check below is a clean-error check: a regression crashes this script,
which fails the test.

Run with scann-core's build/python on PYTHONPATH:
  PYTHONPATH=build/python python tests/python/test_input_validation.py
"""

import numpy as np
import scann
from scann.scann_ops.py.scann_builder import ReorderType

N, DIM = 600, 8


def dataset(n=N, seed=0):
  return np.random.default_rng(seed).standard_normal((n, DIM)).astype(
      np.float32)


def builder(db, cfg, distance="dot_product"):
  b = scann.scann_ops_pybind.builder(db, 10, distance)
  tree = lambda **kw: b.tree(12, 4, training_sample_size=len(db), **kw)
  if cfg == "bf":
    return b.score_brute_force()
  if cfg == "bf_int8":
    return b.score_brute_force(ReorderType.INT8)
  if cfg == "ah":
    return b.score_ah(2).reorder(30)
  if cfg == "tree_bf":
    return tree().score_brute_force()
  if cfg == "tree_bf_int8":
    return tree().score_brute_force(ReorderType.INT8)
  if cfg == "tree_ah":
    return tree().score_ah(2).reorder(30)
  if cfg == "tree_ah_soar":
    return tree(soar_lambda=1.5).score_ah(2).reorder(30)
  if cfg == "tree_ah_incr":
    return tree(incremental_threshold=0.2).score_ah(2).reorder(30)
  raise ValueError(cfg)


CONFIGS = ("bf", "bf_int8", "ah", "tree_bf", "tree_bf_int8", "tree_ah",
           "tree_ah_soar", "tree_ah_incr")


def expect_raises(exc, f, match):
  try:
    f()
  except exc as e:
    assert match in str(e), (match, str(e))
    return
  raise AssertionError(f"expected {exc.__name__} matching {match!r}")


def check_non_finite_build():
  for cfg in ("bf", "ah", "tree_bf", "tree_ah", "tree_ah_soar"):
    for bad in (np.nan, np.inf):
      db = dataset()
      db[7, 2] = bad
      expect_raises(RuntimeError, lambda: builder(db, cfg).build(),
                    "dataset row 7 contains NaN or infinity")


def check_non_finite_upsert_and_query():
  db = dataset()
  rng = np.random.default_rng(1)
  good = rng.standard_normal(DIM).astype(np.float32)
  for cfg in CONFIGS:
    s = builder(db, cfg).build(docids=[str(i) for i in range(N)])
    for bad_value in (np.nan, np.inf, -np.inf):
      bad = np.full(DIM, bad_value, np.float32)
      for batch_size in (1, 2):
        for docids, vecs, row in ((["new"], bad[None], 0),
                                  (["5"], bad[None], 0),
                                  (["a", "b"], np.stack([good, bad]), 1),
                                  (["3", "b"], np.stack([good, bad]), 1)):
          expect_raises(ValueError,
                        lambda: s.upsert(docids, vecs, batch_size=batch_size),
                        f"row {row} contains NaN or infinity")
          # Nothing was applied: the index and the docids agree.
          assert s.size() == N == len(s.docids), (cfg, s.size(), len(s.docids))
          assert "a" not in s.docid_to_id and "new" not in s.docid_to_id

      queries = rng.standard_normal((4, DIM)).astype(np.float32)
      queries[1] = bad
      expect_raises(RuntimeError, lambda: s.search_batched(queries),
                    "query row 1 contains NaN or infinity")
      expect_raises(RuntimeError,
                    lambda: s.search_batched_parallel(queries, batch_size=1),
                    "query row 1 contains NaN or infinity")
      expect_raises(RuntimeError, lambda: s.search(bad), "NaN")

    # The searcher still works and good upserts land where docids say.
    target = rng.standard_normal(DIM).astype(np.float32) * 10
    s.upsert(["c"], target[None])
    assert s.size() == N + 1 == len(s.docids)
    assert s.docid_to_id["c"] == N
    if cfg == "bf":
      got, _ = s.search(target, final_num_neighbors=1)
      assert got[0] == "c", got


def check_partial_upsert_is_atomic():
  db = dataset(100)
  s = builder(db, "bf", "squared_l2").build(docids=[str(i) for i in range(100)])
  good = np.full(DIM, 5.0, np.float32)
  expect_raises(ValueError,
                lambda: s.upsert(["a", "b"], np.stack(
                    [good, np.full(DIM, np.nan, np.float32)])),
                "row 1")
  assert s.size() == 100 == len(s.docids)
  c = np.full(DIM, -7.0, np.float32)
  s.upsert(["c"], c[None])
  assert s.search(c, 1)[0] == ["c"]
  assert s.search(good, 1)[0] != ["a"]


def check_leaves_to_search_on_non_tree():
  db = dataset(300)
  for cfg in ("bf", "bf_int8", "ah"):
    s = builder(db, cfg).build()
    want, _ = s.search(db[0])
    got, _ = s.search(db[0], leaves_to_search=5)
    assert list(got) == list(want), (cfg, got, want)
    got, _ = s.search_batched(db[:4], leaves_to_search=5)
    assert list(got[0]) == list(want), (cfg, got[0], want)
    got, _ = s.search_batched_parallel(db[:4], leaves_to_search=5)
    assert list(got[0]) == list(want), (cfg, got[0], want)


def check_padding():
  db = dataset(5)
  q = dataset(3, seed=2)
  with_docids = builder(db, "bf").build(docids=[f"d{i}" for i in range(5)])
  for search in (with_docids.search_batched,
                 with_docids.search_batched_parallel):
    idx, dist = search(q, final_num_neighbors=8)
    for row, drow in zip(idx, dist):
      assert len(row) == 8
      assert all(d is not None for d in row[:5]), row
      assert row[5:] == [None] * 3, row
      assert np.isnan(drow[5:]).all() and not np.isnan(drow[:5]).any()
  # Without docids the padding stays upstream's: index 0, NaN distance.
  plain = builder(db, "bf").build()
  idx, dist = plain.search_batched(q, final_num_neighbors=8)
  assert (idx[:, 5:] == 0).all() and np.isnan(dist[:, 5:]).all()


def main():
  check_non_finite_build()
  check_non_finite_upsert_and_query()
  check_partial_upsert_is_atomic()
  check_leaves_to_search_on_non_tree()
  check_padding()
  print("PASSED")


if __name__ == "__main__":
  main()
