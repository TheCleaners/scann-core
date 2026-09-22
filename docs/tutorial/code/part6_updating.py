"""Part 6: keeping an index up to date: upsert, delete, rebalance."""

import os

import numpy as np
import scann

from tutorial_data import Timer, load_glove, recall

dataset, queries, true_neighbors = load_glove()
n = len(dataset)
docids = [str(i) for i in range(n)]  # docid = row number in the full dataset


def measure(name, searcher):
  searcher.set_num_threads(os.cpu_count())
  found, _ = searcher.search_batched_parallel(queries)
  found = np.array([[int(d) for d in row] for row in found])
  stats = searcher.get_health_stats()
  print(f"{name:40} size {searcher.size():8}  recall@10 "
        f"{recall(found, true_neighbors):.4f}  "
        f"imbalance {stats['partition_avg_relative_positive_imbalance']:.3f}  "
        f"quantization error {stats['avg_quantization_error']:.4f}")
  return found


def builder(data):
  return (scann.scann_ops_pybind.builder(data, 10, "dot_product")
          .tree(num_leaves=2000, num_leaves_to_search=100,
                training_sample_size=250000)
          .score_ah(2, anisotropic_quantization_threshold=0.2)
          .reorder(100))


# Build on the first 70% of the data only...
first = int(0.7 * n)
with Timer() as t:
  searcher = builder(dataset[:first]).build(docids=docids[:first])
print(f"built on {first} points in {t.seconds:.1f} s")
searcher.initialize_health_stats()
measure("70% indexed", searcher)

# ...then stream the remaining 30% in.
with Timer() as t:
  for start in range(first, n, 10000):
    end = min(start + 10000, n)
    searcher.upsert(docids[start:end], dataset[start:end], batch_size=10000)
print(f"upserted {n - first} points in {t.seconds:.1f} s "
      f"({(n - first) / t.seconds:.0f} points/s)")
measure("after upserting the other 30%", searcher)

# Retrain the partitioning and quantization on everything.
with Timer() as t:
  searcher.rebalance()
print(f"rebalanced in {t.seconds:.1f} s")
searcher.initialize_health_stats()
measure("after rebalance()", searcher)

with Timer() as t:
  fresh = builder(dataset).build(docids=docids)
fresh.initialize_health_stats()
print(f"(fresh build on everything: {t.seconds:.1f} s)")
measure("built from scratch on 100%", fresh)

# Update in place: move point 0 far away; delete query 0's top result.
top = searcher.search(queries[0])[0]
print("\nquery 0 top-3 before:", top[:3])
searcher.delete(top[0])
searcher.upsert(top[1], -queries[0])  # same docid, new vector
print("query 0 top-3 after deleting", top[0], "and moving", top[1], ":",
      searcher.search(queries[0])[0][:3])
