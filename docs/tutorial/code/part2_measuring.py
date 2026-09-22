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
