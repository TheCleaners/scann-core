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

"""Part 3: partitioning, asymmetric hashing and reordering, one at a time."""

import scann

from tutorial_data import Timer, evaluate, load_glove

dataset, queries, true_neighbors = load_glove()


def builder():
  return scann.scann_ops_pybind.builder(dataset, 10, "dot_product")


def tree(b):
  # ~sqrt(1.18M) = 1088 leaves would be the rule of thumb; 2000 is a common
  # choice for ~1M points. Search 100 of them (5%) per query.
  return b.tree(num_leaves=2000, num_leaves_to_search=100,
                training_sample_size=250000)


def ah(b):
  return b.score_ah(2, anisotropic_quantization_threshold=0.2)


configs = [
    ("1. tree + brute force", lambda: tree(builder()).score_brute_force()),
    ("2. AH only", lambda: ah(builder())),
    ("3. AH + reorder 100", lambda: ah(builder()).reorder(100)),
    ("4. tree + AH", lambda: ah(tree(builder()))),
    ("5. tree + AH + reorder 100", lambda: ah(tree(builder())).reorder(100)),
]
for name, make in configs:
  with Timer() as t:
    searcher = make().build()
  print(f"{name}: built in {t.seconds:.1f} s")
  evaluate(name, searcher, queries, true_neighbors)
