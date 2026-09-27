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

"""Regression test: grow an index from empty, then retrain it with a new config.

The flow of big-ann-benchmarks' ScaNN entry (neurips23/ood/scann/scann.py),
scaled down: an empty brute-force searcher, reserve(), batched upserts
through the pybind object, rebalance(config) into a SOAR-spilled tree with
AVQ, AH and bfloat16 reordering, more upserts, then search. Also the
builder's own SOAR options, and the error for a tree with more leaves than
points (see google-research/google-research#2712).

Run with scann-core's build/python on PYTHONPATH:
  PYTHONPATH=build/python python tests/python/test_rebalance_flow.py
"""

import numpy as np
import scann

D, N, LEAVES, SEARCHED = 32, 20000, 200, 10


def make_data():
  rng = np.random.default_rng(0)
  centers = rng.standard_normal((100, D))
  data = centers[rng.integers(100, size=N)] + 0.4 * rng.standard_normal((N, D))
  data = (data / np.linalg.norm(data, axis=1, keepdims=True)).astype(np.float32)
  queries = (data[rng.integers(N, size=200)] +
             0.05 * rng.standard_normal((200, D))).astype(np.float32)
  truth = np.argsort(-(queries @ data.T), axis=1)[:, :10]
  return data, queries, truth


def recall(found, truth):
  return np.mean([len(set(f) & set(t)) / 10 for f, t in zip(found, truth)])


def soar_config(num_children):
  """big-ann-benchmarks' OOD config, sized for this data.

  Its custom_search_method line is kept as in the original, although
  open-source ScaNN never reads that field.
  """
  return f"""
num_neighbors: 10
distance_measure {{ distance_measure: "DotProductDistance" }}
partitioning {{
  num_children: {num_children}
  max_clustering_iterations: 40
  min_cluster_size: 10
  partitioning_distance {{ distance_measure: "SquaredL2Distance" }}
  database_spilling {{
    spilling_type: TWO_CENTER_ORTHOGONALITY_AMPLIFIED
    orthogonality_amplification_lambda: 1.3
    overretrieve_factor: 1.2
  }}
  query_spilling {{ spilling_type: FIXED_NUMBER_OF_CENTERS max_spill_centers: {SEARCHED} }}
  partitioning_type: GENERIC
  query_tokenization_distance_override {{ distance_measure: "DotProductDistance" }}
  query_tokenization_type: FLOAT
  balancing_type: UNBALANCED_FLOAT32
  single_machine_center_initialization: DEFAULT_KMEANS_PLUS_PLUS
  avq: 1.6
}}
hash {{ asymmetric_hash {{
  projection {{ projection_type: CHUNK num_blocks: {D // 2} num_dims_per_block: 2 input_dim: {D} }}
  num_clusters_per_block: 16
  max_clustering_iterations: 30
  quantization_distance {{ distance_measure: "SquaredL2Distance" }}
  lookup_type: INT8_LUT16
  use_residual_quantization: true
  noise_shaping_threshold: 0.1
  expected_sample_size: 100000
  use_global_topn: true
}} }}
exact_reordering {{
  approx_num_neighbors: 150
  bfloat16 {{ enabled: true noise_shaping_threshold: 0.2 }}
}}
custom_search_method: "experimental_top_level_partitioner:40,10,3.0,2.5,1.8"
"""


def empty_searcher():
  s = (scann.scann_ops_pybind.builder(np.zeros((0, D), np.float32), 10,
                                      "dot_product")
       .score_brute_force().build())
  assert s.size() == 0
  return s


def main():
  data, queries, truth = make_data()

  # Grow from empty, retrain into a SOAR tree, keep growing.
  s = empty_searcher()
  s.reserve(N)
  for start in range(0, 15000, 5000):
    s.searcher.upsert([None] * 5000, data[start:start + 5000], 5000)
  assert s.size() == 15000
  s.rebalance(soar_config(LEAVES))
  assert "TWO_CENTER_ORTHOGONALITY_AMPLIFIED" in s.config()
  s.searcher.upsert([None] * 5000, data[15000:], 5000)
  assert s.size() == N
  found, _ = s.search_batched_parallel(queries, leaves_to_search=SEARCHED,
                                       pre_reorder_num_neighbors=100)
  r = recall(found, truth)
  assert r >= 0.9, f"recall after grow + rebalance: {r:.3f}"

  # The builder's SOAR options give the same kind of index directly.
  b = (scann.scann_ops_pybind.builder(data, 10, "dot_product")
       .tree(LEAVES, SEARCHED, avq=1.6, soar_lambda=1.3, overretrieve_factor=1.2,
             random_init=False)
       .score_ah(2, anisotropic_quantization_threshold=0.1).reorder(150))
  built = b.build()
  assert "TWO_CENTER_ORTHOGONALITY_AMPLIFIED" in built.config()
  r_built = recall(built.search_batched(queries)[0], truth)
  assert r_built >= 0.9, f"recall with builder SOAR: {r_built:.3f}"

  # More leaves than points: a clear error, and the searcher is unchanged.
  small = empty_searcher()
  small.searcher.upsert([None] * 1000, data[:1000], 1000)
  try:
    small.rebalance(soar_config(40000))
    raise AssertionError("rebalance with 40000 leaves on 1000 points succeeded")
  except RuntimeError as e:
    assert "less than the number of clusters" in str(e), e
  assert small.size() == 1000
  assert small.search(data[3])[0][0] == 3

  print(f"PASSED (grow + rebalance recall {r:.3f}, builder SOAR recall "
        f"{r_built:.3f})")


if __name__ == "__main__":
  main()
