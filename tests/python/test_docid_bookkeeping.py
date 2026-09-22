#!/usr/bin/env python3
"""Regression test: a failed upsert()/delete() must leave the searcher's
docid bookkeeping in sync with the index.

Upstream updated `docids`/`docid_to_id` before the C++ call (upsert) or
while validating (delete), so an error left docids pointing at the wrong
vectors from then on.

Run with scann-core's build/python on PYTHONPATH:
  PYTHONPATH=build/python python tests/python/test_docid_bookkeeping.py
"""

import numpy as np
import scann


def make_searcher(n=2000, dim=16):
  db = np.random.default_rng(0).standard_normal((n, dim)).astype(np.float32)
  s = (scann.scann_ops_pybind.builder(db, 1, "squared_l2")
       .tree(20, 20, random_init=False).score_brute_force().build(
           docids=[f"d{i}" for i in range(n)]))
  return s, db


def expect_raises(exc, f):
  try:
    f()
  except exc:
    return
  raise AssertionError(f"expected {exc.__name__}")


def check_consistent(s, db_rows):
  """Every docid still finds its own vector."""
  assert len(s.docids) == s.size(), (len(s.docids), s.size())
  for docid, vec in db_rows.items():
    got, _ = s.search(vec, final_num_neighbors=1)
    assert got[0] == docid, (docid, got)


def main():
  s, db = make_searcher()
  rows = {f"d{i}": db[i] for i in (0, 1, 1999)}
  new = np.full(16, 3.0, np.float32)

  # Failing upsert: nothing may change.
  expect_raises(ValueError, lambda: s.upsert(["new"], new[None], batch_size=0))
  expect_raises(Exception, lambda: s.upsert(["new"], np.zeros((1, 15), np.float32)))
  assert "new" not in s.docid_to_id
  check_consistent(s, rows)
  s.upsert(["new"], new[None])
  rows["new"] = new
  check_consistent(s, rows)

  # Failing delete (unknown / repeated docid): nothing may change.
  expect_raises(KeyError, lambda: s.delete(["d0", "missing"]))
  expect_raises(KeyError, lambda: s.delete(["d1", "d1"]))
  check_consistent(s, rows)
  s.delete(["d0"])
  del rows["d0"]
  check_consistent(s, rows)
  print("PASSED")


if __name__ == "__main__":
  main()
