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

"""Regression test: mutating a tree whose partitioner projects (PCA/TRUNCATE).

Upstream's health-stats collector skipped the quantization error at
initialization when the centroids were projected, but on every upsert/delete
took the distance between the unprojected datapoint and the projected
centroid: a heap out-of-bounds read, after which avg_quantization_error was
garbage (typically inf).

Upstream also reported avg_quantization_error = 0 for such trees (not
computed). It is now the distance between the projected datapoints and their
(projected) centroids, tracked incrementally like a non-projected tree's: after
every mutation, all statistics must match what a fresh
initialize_health_stats() computes. A full-dimensional PCA is a rotation,
which k-means doesn't see: that tree's quantization error must match the
unprojected tree's. (tests/cpp/mutation_regressions.cc also checks an identity
TRUNCATE tree against an unprojected one.)

Run with scann-core's build/python on PYTHONPATH:
  PYTHONPATH=build/python python tests/python/test_projection_mutation.py
"""

import math

import numpy as np
import scann

N, D, LEAVES = 2000, 16, 20


def build(db, projection):
  b = (scann.scann_ops_pybind.builder(db, 10, "dot_product")
       .tree(LEAVES, LEAVES, training_sample_size=N))
  if projection == "pca":
    b = b.pca(reduction_dim=8, pca_significance_threshold=None)
  elif projection == "pca_full":  # Projected but not dimension-reducing.
    b = b.pca(reduction_dim=D, pca_significance_threshold=None)
  elif projection == "truncate":
    b = b.truncate(8)
  return b.score_ah(2).reorder(200).build(docids=[f"d{i}" for i in range(N)])


def check_health(s, what):
  """Incremental stats are sane and agree with a fresh initialization."""
  inc = s.get_health_stats()
  s.initialize_health_stats()
  fresh = s.get_health_stats()
  assert all(math.isfinite(v) for v in inc.values()), (what, inc)
  assert inc["sum_partition_sizes"] == s.size(), (what, inc, s.size())
  assert inc["sum_partition_sizes"] == fresh["sum_partition_sizes"], what
  for k in ("partition_weighted_avg_relative_imbalance",
            "partition_avg_relative_positive_imbalance"):
    assert math.isclose(inc[k], fresh[k], rel_tol=1e-9, abs_tol=1e-12), (
        what, k, inc, fresh)
  assert inc["avg_quantization_error"] > 0.0, (what, inc)
  assert math.isclose(inc["avg_quantization_error"],
                      fresh["avg_quantization_error"], rel_tol=1e-4), (
                          what, inc, fresh)


def unit_rows(rng, n):
  """Unit-norm rows, so that under dot_product each is its own top match."""
  x = rng.standard_normal((n, D))
  return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def check_found(s, rows, what):
  """Every stored vector finds its own docid.

  AH scores only the projected dimensions, so the exact reordering considers
  every candidate: this checks that each vector is in some searched leaf under
  the right docid, not the recall of a projected tree.
  """
  assert len(s.docids) == s.size() == len(rows), (what, len(s.docids),
                                                   s.size(), len(rows))
  docids = list(rows)
  found, _ = s.search_batched(np.stack([rows[d] for d in docids]),
                              final_num_neighbors=1,
                              pre_reorder_num_neighbors=s.size())
  missing = [d for d, f in zip(docids, found) if f[0] != d]
  assert not missing, (what, missing[:5])


def run(projection):
  rng = np.random.default_rng(0)
  db = unit_rows(rng, N)
  rows = {f"d{i}": db[i] for i in range(N)}
  s = build(db, projection)
  check_health(s, "build")
  built_stats = s.get_health_stats()

  def mutate(tag):
    # Insert new docids, update existing ones, delete some.
    new = {f"{tag}n{i}": v for i, v in enumerate(unit_rows(rng, 50))}
    s.upsert(list(new), np.stack(list(new.values())), batch_size=16)
    rows.update(new)
    check_health(s, f"{tag} insert")
    upd = list(rows)[5:305:10]
    vecs = unit_rows(rng, len(upd))
    s.upsert(upd, vecs)
    rows.update(zip(upd, vecs))
    check_health(s, f"{tag} update")
    gone = list(rows)[7:407:8]
    s.delete(gone)
    for d in gone:
      del rows[d]
    check_health(s, f"{tag} delete")
    check_found(s, rows, f"{tag} mutate")

  mutate("a")
  s.rebalance()
  check_health(s, "rebalance")
  check_found(s, rows, "rebalance")
  mutate("b")
  return built_stats, s.get_health_stats()


def main():
  built = {}
  for projection in (None, "pca", "pca_full", "truncate"):
    built[projection], stats = run(projection)
    print(f"{projection}: {stats}")
  # A rotation: the same partitioning, the same quantization error.
  assert math.isclose(built["pca_full"]["avg_quantization_error"],
                      built[None]["avg_quantization_error"], rel_tol=1e-3), (
                          built["pca_full"], built[None])
  # Projecting to 8 of 16 dimensions discards half of the variance: the
  # projected datapoints are closer to their centroids.
  for projection in ("pca", "truncate"):
    assert (built[projection]["avg_quantization_error"] <
            built[None]["avg_quantization_error"]), (projection, built)
  print("PASSED")


if __name__ == "__main__":
  main()
