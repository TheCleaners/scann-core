#!/usr/bin/env python3
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

"""Records what upstream's Python ScannBuilder emits for a matrix of options.

Writes <out>/cases.txt, one case per block:
    case <name> <num_neighbors> <distance> <dimensionality> <num_points>
    call <method> <key=value ...>
    ...
    expected <file>
and <out>/<name>.pbtxt with Python's create_config() output.
tests/cpp/config_builder_test.cc replays the calls through the C++
ConfigBuilder and requires a semantically identical ScannConfig.

Run with an interpreter where `scann` is importable (scann-core's
build/python on PYTHONPATH, or the upstream wheel -- the builder code is
byte-identical).
"""

import argparse
import os

import numpy as np
import scann

RT = scann.ReorderType
DIMS = (128, 100, 768)  # 100: not divisible by 3 -> VARIABLE_CHUNK blocks

# name -> (distance, [(method, kwargs), ...])
CASES = {
    "bf_float": ("dot_product", [("score_brute_force", {})]),
    "bf_int8": ("dot_product", [("score_brute_force", {"quantize": RT.INT8})]),
    "bf_bf16": ("squared_l2", [("score_brute_force", {"quantize": RT.BFLOAT16})]),
    "ah_plain": ("dot_product", [("score_ah", {"dimensions_per_block": 2})]),
    "ah_aq_reorder": ("dot_product", [
        ("score_ah", {"dimensions_per_block": 2, "anisotropic_quantization_threshold": 0.2}),
        ("reorder", {"reordering_num_neighbors": 100})]),
    "ah_lut256_dpb3": ("squared_l2", [
        ("score_ah", {"dimensions_per_block": 3, "hash_type": "lut256",
                      "training_sample_size": 5000, "training_iterations": 7})]),
    "ah_int8_reorder": ("dot_product", [
        ("score_ah", {"dimensions_per_block": 2}),
        ("reorder", {"reordering_num_neighbors": 50, "quantize": RT.INT8,
                     "anisotropic_quantization_threshold": 0.25})]),
    "ah_bf16_reorder": ("dot_product", [
        ("score_ah", {"dimensions_per_block": 2}),
        ("reorder", {"reordering_num_neighbors": 50, "quantize": RT.BFLOAT16})]),
    "tree_ah_dot": ("dot_product", [
        ("tree", {"num_leaves": 64, "num_leaves_to_search": 8, "training_sample_size": 5000}),
        ("score_ah", {"dimensions_per_block": 2, "anisotropic_quantization_threshold": 0.2}),
        ("reorder", {"reordering_num_neighbors": 100})]),
    "tree_ah_l2_kmpp": ("squared_l2", [
        ("tree", {"num_leaves": 32, "num_leaves_to_search": 4, "random_init": False,
                  "min_partition_size": 10, "training_iterations": 5}),
        ("score_ah", {"dimensions_per_block": 2})]),
    "tree_spherical_quantized": ("dot_product", [
        ("tree", {"num_leaves": 16, "num_leaves_to_search": 2, "spherical": True,
                  "quantize_centroids": True}),
        ("score_brute_force", {})]),
    "tree_avq_soar": ("dot_product", [
        ("tree", {"num_leaves": 64, "num_leaves_to_search": 8, "avq": 2.5,
                  "soar_lambda": 1.5, "overretrieve_factor": 2.0}),
        ("score_ah", {"dimensions_per_block": 2}),
        ("reorder", {"reordering_num_neighbors": 40})]),
    "tree_soar_no_overretrieve": ("dot_product", [
        ("tree", {"num_leaves": 64, "num_leaves_to_search": 8, "soar_lambda": 1.0}),
        ("score_ah", {"dimensions_per_block": 2})]),
    "tree_incremental_points": ("dot_product", [
        ("tree", {"num_leaves": 64, "num_leaves_to_search": 8, "incremental_threshold": 1000}),
        ("score_ah", {"dimensions_per_block": 2})]),
    "tree_incremental_fraction": ("dot_product", [
        ("tree", {"num_leaves": 64, "num_leaves_to_search": 8, "incremental_threshold": 0.3}),
        ("score_ah", {"dimensions_per_block": 2})]),
    "tree_ah_residual_forced_off": ("dot_product", [
        ("tree", {"num_leaves": 64, "num_leaves_to_search": 8}),
        ("score_ah", {"dimensions_per_block": 2, "residual_quantization": False})]),
    "upper_tree": ("dot_product", [
        ("tree", {"num_leaves": 1000, "num_leaves_to_search": 50}),
        ("upper_tree", {"num_leaves": 40, "num_leaves_to_search": 10}),
        ("score_ah", {"dimensions_per_block": 2}),
        ("reorder", {"reordering_num_neighbors": 100})]),
    "upper_tree_soar_bf16": ("dot_product", [
        ("tree", {"num_leaves": 1000, "num_leaves_to_search": 50}),
        ("upper_tree", {"num_leaves": 40, "num_leaves_to_search": 10, "avq": 1.0,
                        "soar_lambda": 1.2, "overretrieve_factor": 1.8,
                        "scoring_mode": RT.BFLOAT16,
                        "anisotropic_quantization_threshold": 0.2}),
        ("score_ah", {"dimensions_per_block": 2})]),
    "pca_significance": ("dot_product", [
        ("pca", {}),
        ("tree", {"num_leaves": 64, "num_leaves_to_search": 8}),
        ("score_ah", {"dimensions_per_block": 2})]),
    "pca_reduction_dim": ("dot_product", [
        ("pca", {"reduction_dim": 32, "pca_significance_threshold": None}),
        ("tree", {"num_leaves": 64, "num_leaves_to_search": 8}),
        ("score_ah", {"dimensions_per_block": 2})]),
    "truncate": ("squared_l2", [
        ("truncate", {"reduction_dim": 64}),
        ("tree", {"num_leaves": 64, "num_leaves_to_search": 8}),
        ("score_ah", {"dimensions_per_block": 2})]),
    "autopilot_default": ("dot_product", [("autopilot", {})]),
    "autopilot_online_int8": ("squared_l2", [
        ("autopilot", {"mode": scann.scann_ops.py.scann_builder.IncrementalMode.ONLINE,
                       "quantize": RT.INT8})]),
}
NUM_POINTS = (5000, 200000)


def render(v):
  if isinstance(v, RT):
    return v.name
  if hasattr(v, "name") and type(v).__name__ == "IncrementalMode":
    return v.name
  if v is None:
    return "None"
  if isinstance(v, bool):
    return "true" if v else "false"
  if isinstance(v, float):
    return repr(v) + "f"   # marks a float, so 0.3f != int 3
  return str(v)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", required=True)
  args = p.parse_args()
  os.makedirs(args.out, exist_ok=True)
  lines = []
  n_cases = 0
  for dim in DIMS:
    for num_points in NUM_POINTS:
      for name, (distance, calls) in CASES.items():
        if not name.startswith("autopilot") and num_points != NUM_POINTS[0]:
          continue  # num_points only matters for autopilot
        # The builder only reads db.shape; a broadcast view costs no memory.
        db = np.broadcast_to(np.float32(0), (num_points, dim))
        b = scann.scann_ops_pybind.builder(db, 10, distance)
        for method, kwargs in calls:
          b = getattr(b, method)(**kwargs)
        text = b.create_config()
        case = f"{name}_d{dim}_n{num_points}"
        with open(os.path.join(args.out, case + ".pbtxt"), "w") as f:
          f.write(text)
        lines.append(f"case {case} 10 {distance} {dim} {num_points}")
        for method, kwargs in calls:
          kv = " ".join(f"{k}={render(v)}" for k, v in kwargs.items())
          lines.append(f"call {method} {kv}".rstrip())
        lines.append(f"expected {case}.pbtxt")
        n_cases += 1
  with open(os.path.join(args.out, "cases.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
  print(f"wrote {n_cases} cases to {args.out}")


if __name__ == "__main__":
  main()
