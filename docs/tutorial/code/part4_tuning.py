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

"""Part 4: trading speed for recall, at query time and at build time."""

import scann

from tutorial_data import Timer, evaluate, load_glove

dataset, queries, true_neighbors = load_glove()


def builder():
  return scann.scann_ops_pybind.builder(dataset, 10, "dot_product")


# --- Query time: one index, different search parameters. -----------------
searcher = (builder()
            .tree(num_leaves=2000, num_leaves_to_search=100,
                  training_sample_size=250000)
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(100)
            .build())
print("query-time sweep (index: 2000 leaves, AH, reorder)")
for leaves in (25, 50, 100, 200, 400):
  for pre_reorder in (50, 100, 200, 400):
    evaluate(f"leaves {leaves:3} pre_reorder {pre_reorder:3}", searcher,
             queries, true_neighbors, latency_queries=200,
             leaves_to_search=leaves, pre_reorder_num_neighbors=pre_reorder)

# --- Build time: variations on the index itself. --------------------------
# Each variant is swept over three leaves_to_search values (pre_reorder 200),
# so they can be compared as curves, not single points.
print("\nbuild-time variations (leaves_to_search sweep, pre_reorder 200)")


def variant(name, num_leaves, make):
  with Timer() as t:
    s = make().build()
  print(f"{name}: built in {t.seconds:.1f} s")
  for fraction in (0.025, 0.05, 0.1):  # of the leaves searched per query
    leaves = int(fraction * num_leaves)
    evaluate(f"  leaves {leaves:4}", s, queries, true_neighbors,
             latency_queries=200, leaves_to_search=leaves,
             pre_reorder_num_neighbors=200)


def tree(b, leaves, soar=None):
  return b.tree(num_leaves=leaves, num_leaves_to_search=leaves // 20,
                training_sample_size=250000, soar_lambda=soar)


def aq(b, dims=2):
  return b.score_ah(dims, anisotropic_quantization_threshold=0.2)


VARIANTS = [
    ("1000 leaves", 1000, lambda: aq(tree(builder(), 1000)).reorder(200)),
    ("2000 leaves", 2000, lambda: aq(tree(builder(), 2000)).reorder(200)),
    ("4000 leaves", 4000, lambda: aq(tree(builder(), 4000)).reorder(200)),
    ("2000 leaves, plain PQ (no AQ)", 2000,
     lambda: tree(builder(), 2000).score_ah(2).reorder(200)),
    ("2000 leaves, 4 dims/block", 2000,
     lambda: aq(tree(builder(), 2000), dims=4).reorder(200)),
    ("2000 leaves, SOAR lambda 1.5", 2000,
     lambda: aq(tree(builder(), 2000, soar=1.5)).reorder(200)),
]
for v in VARIANTS:
  variant(*v)

# --- Where anisotropic quantization matters: little or no reordering. -----
print("\nAQ vs plain PQ with less reordering (2000 leaves, 100 searched)")
for name, make in [
    ("AQ", lambda: aq(tree(builder(), 2000))),
    ("plain PQ", lambda: tree(builder(), 2000).score_ah(2)),
]:
  s = make().build()  # no reorder(): AH ranks the final results
  evaluate(f"{name}, no reordering", s, queries, true_neighbors,
           latency_queries=200, leaves_to_search=100)
  s = make().reorder(200).build()
  for pre_reorder in (20, 50):
    evaluate(f"{name}, reorder {pre_reorder}", s, queries, true_neighbors,
             latency_queries=200, leaves_to_search=100,
             pre_reorder_num_neighbors=pre_reorder)
