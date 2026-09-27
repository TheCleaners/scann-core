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

"""Regression test: raw config values are errors, not crashes.

create_searcher() takes a config as text. Upstream aborted the process on
single-field changes to builder-made configs (num_dims_per_block = 0:
SIGFPE; num_blocks = 0: CHECK; Hamming distance on float data and bfloat16
brute force with L1 / LimitedInnerProduct / AbsDotProduct: LOG(FATAL)),
overflowed the heap (LUT16 with fewer than 16 clusters per block) and hit
undefined behavior (fixed_point_multiplier_quantile outside (0, 1]).

The builder: tree() + pca()/truncate() + score_ah() without residual
quantization (every squared_l2 tree) failed with a bare "SCANN_RET_CHECK
failure" and now builds; incremental_threshold together with pca(),
truncate() or upper_tree() raises ValueError from create_config() instead of
failing in the C++ initialization.

Each check is a clean-error check: a regression crashes this script, which
fails the test.

Run with scann-core's build/python on PYTHONPATH:
  PYTHONPATH=build/python python tests/python/test_config_validation.py
"""

import numpy as np
import scann
from scann.scann_ops.py.scann_builder import ReorderType

N, DIM = 600, 16


def dataset(n=N, seed=0):
  return np.random.default_rng(seed).standard_normal((n, DIM)).astype(
      np.float32)


def builder(db, distance="dot_product"):
  return scann.scann_ops_pybind.builder(db, 10, distance)


def edit(config, old, new):
  assert old in config, (old, config)
  return config.replace(old, new, 1)


def expect_raises(exc, f, match=""):
  try:
    f()
  except exc as e:
    assert match in str(e), (match, str(e))
    return
  raise AssertionError(f"expected {exc.__name__} matching {match!r}")


def check_raw_configs():
  db = dataset()
  ah = builder(db).score_ah(2).reorder(30).create_config()
  tree_ah = builder(db).tree(
      12, 4, training_sample_size=N).score_ah(2).reorder(30).create_config()
  bf = builder(db).score_brute_force().create_config()
  bf16 = builder(db).score_brute_force(ReorderType.BFLOAT16).create_config()
  tree_int8 = builder(db).tree(
      12, 4, training_sample_size=N).score_brute_force(
          ReorderType.INT8).create_config()
  cases = [
      (edit(ah, "num_dims_per_block: 2", "num_dims_per_block: 0"),
       "num_dims_per_block"),
      (edit(ah, "num_blocks: 8", "num_blocks: 0"), "num_blocks"),
      (edit(tree_ah, "num_clusters_per_block: 16", "num_clusters_per_block: 1"),
       "INT8_LUT16"),
      (edit(bf, '"DotProductDistance"', '"BinaryHammingDistance"'), "binary"),
      (edit(tree_int8, "fixed_point {",
            "fixed_point { fixed_point_multiplier_quantile: 2.0"), "quantile"),
  ]
  for d in ("L1Distance", "LimitedInnerProductDistance",
            "AbsDotProductDistance"):
    cases.append((edit(bf16, '"DotProductDistance"', f'"{d}"'), "bfloat16"))
  for config, match in cases:
    expect_raises(
        RuntimeError,
        lambda: scann.scann_ops_pybind.create_searcher(db, config, 1), match)


def check_projected_tree_ah_builds():
  db = dataset(3000, 1)
  queries = dataset(40, 2)
  d2 = ((queries[:, None, :] - db[None]) ** 2).sum(-1)
  want = np.argsort(d2, axis=1)[:, :10]
  for proj in ("pca", "truncate"):
    b = builder(db, "squared_l2").tree(12, 4, training_sample_size=3000)
    b = (b.pca(reduction_dim=12, pca_significance_threshold=None)
         if proj == "pca" else b.truncate(12))
    s = b.score_ah(2).reorder(100).build()
    got = s.search_batched(queries, leaves_to_search=12)[0]
    recall = np.mean([len(set(g) & set(w)) / 10 for g, w in zip(got, want)])
    assert recall > 0.8, (proj, recall)


def check_builder_rejects_incremental_with_projection():
  db = dataset()
  tree = lambda b: b.tree(12, 4, training_sample_size=N,
                          incremental_threshold=0.2)
  for extra in (lambda b: b.pca(reduction_dim=8, pca_significance_threshold=None),
                lambda b: b.truncate(8),
                lambda b: b.upper_tree(4, 2)):
    b = extra(tree(builder(db))).score_ah(2).reorder(30)
    expect_raises(ValueError, b.create_config, "incremental_threshold")
  # Without them, incremental training still works.
  s = tree(builder(db, "squared_l2")).score_ah(2).reorder(30).build()
  assert s.search(db[3], leaves_to_search=12)[0][0] == 3


if __name__ == "__main__":
  check_raw_configs()
  check_projected_tree_ah_builds()
  check_builder_rejects_incremental_with_projection()
  print("OK")
