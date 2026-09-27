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

"""Part 2: measuring recall and speed, on brute force at three precisions."""

import scann

from tutorial_data import evaluate, load_glove

dataset, queries, true_neighbors = load_glove()


def builder():
  return scann.scann_ops_pybind.builder(dataset, 10, "dot_product")


for name, quantize in [("float32", scann.ReorderType.FLOAT32),
                       ("bfloat16", scann.ReorderType.BFLOAT16),
                       ("int8", scann.ReorderType.INT8)]:
  searcher = builder().score_brute_force(quantize=quantize).build()
  evaluate(f"brute force, {name}", searcher, queries, true_neighbors)
