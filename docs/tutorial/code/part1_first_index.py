# Copyright 2026 ebenali and TheCleaners.
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

"""Part 1: a first index -- exact (brute-force) search."""

import numpy as np
import scann

from tutorial_data import Timer, load_glove

dataset, queries, true_neighbors = load_glove()
print(f"dataset {dataset.shape} {dataset.dtype}, queries {queries.shape}")

# 10 neighbours by dot product, scored exactly.
with Timer() as t:
  searcher = (scann.scann_ops_pybind.builder(dataset, 10, "dot_product")
              .score_brute_force()
              .build())
print(f"built in {t.seconds:.2f} s")

# One query.
neighbors, distances = searcher.search(queries[0])
print("neighbors:", neighbors)
print("distances:", np.round(distances, 4))

# The same thing by hand: the dot products of query 0 with every point.
scores = dataset @ queries[0]
print("by hand:  ", np.argsort(-scores)[:10])
print("          ", np.round(np.sort(scores)[::-1][:10], 4))

# Ask for more (or fewer) neighbours than the config's default of 10.
neighbors, _ = searcher.search(queries[0], final_num_neighbors=3)
print("top 3:", neighbors)

# All queries at once.
with Timer() as t:
  neighbors, distances = searcher.search_batched(queries)
print(f"batched: {neighbors.shape} in {t.seconds:.2f} s")
print("same as the ground truth for query 0:",
      set(neighbors[0]) == set(true_neighbors[0][:10]))
