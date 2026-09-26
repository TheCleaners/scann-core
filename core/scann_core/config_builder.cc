// Copyright 2026 The Google Research Authors.
// Copyright 2026 ebenali and TheCleaners.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Port of scann/scann_ops/py/scann_builder.py to C++.

#include "scann_core/config_builder.h"

#include <charconv>
#include <cmath>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "google/protobuf/text_format.h"
#include "scann/utils/single_machine_autopilot.h"

namespace scann_core {
namespace {

// Shortest representation that round-trips; "nan" for unset thresholds
// (the text format parser accepts nan/inf).
std::string Num(double v) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof(buf), v);
  return std::string(buf, r.ptr);
}

const char* Bool(bool b) { return b ? "true" : "false"; }

const char* DistanceName(DistanceMeasure d) {
  return d == DistanceMeasure::kDotProduct ? "DotProductDistance"
                                           : "SquaredL2Distance";
}

// Python: {"bfloat16" if quantize == BFLOAT16 else "fixed_point"}
const char* ReorderBlock(Quantization q) {
  return q == Quantization::kBfloat16 ? "bfloat16" : "fixed_point";
}

const char* QuantizationName(Quantization q) {
  switch (q) {
    case Quantization::kFloat32: return "FLOAT32";
    case Quantization::kInt8: return "INT8";
    case Quantization::kBfloat16: return "BFLOAT16";
  }
  return "FLOAT32";
}

// Python: upper_tree's scoring_mode mapping.
const char* UpperTreeQuantization(Quantization q) {
  switch (q) {
    case Quantization::kFloat32: return "FLOAT32";
    case Quantization::kInt8: return "FIXED8";
    case Quantization::kBfloat16: return "BFLOAT16";
  }
  return "FIXED8";
}

const char* IncrementalModeName(IncrementalMode m) {
  switch (m) {
    case IncrementalMode::kNone: return "NONE";
    case IncrementalMode::kOnline: return "ONLINE";
    case IncrementalMode::kOnlineIncremental: return "ONLINE_INCREMENTAL";
  }
  return "NONE";
}

}  // namespace

ConfigBuilder::ConfigBuilder(int32_t num_neighbors, DistanceMeasure distance,
                             uint32_t dimensionality)
    : num_neighbors_(num_neighbors),
      distance_(distance),
      dimensionality_(dimensionality) {}

void ConfigBuilder::Set(const char* what, bool& flag) {
  // Python: Exception(f"{key} has already been configured")
  if (flag) errors_.push_back(absl::StrCat(what, " has already been configured"));
  flag = true;
}

ConfigBuilder& ConfigBuilder::Tree(const TreeOptions& o) {
  Set("tree", tree_set_);
  tree_ = o;
  return *this;
}
ConfigBuilder& ConfigBuilder::UpperTree(const UpperTreeOptions& o) {
  Set("upper_tree", upper_tree_set_);
  upper_tree_ = o;
  return *this;
}
ConfigBuilder& ConfigBuilder::ScoreAh(const AhOptions& o) {
  Set("score_ah", ah_set_);
  ah_ = o;
  return *this;
}
ConfigBuilder& ConfigBuilder::ScoreBruteForce(Quantization q) {
  Set("score_bf", bf_set_);
  brute_force_ = q;
  return *this;
}
ConfigBuilder& ConfigBuilder::Reorder(const ReorderOptions& o) {
  Set("reorder", reorder_set_);
  reorder_ = o;
  return *this;
}
ConfigBuilder& ConfigBuilder::Pca(const PcaOptions& o) {
  Set("pca", pca_set_);
  pca_ = o;
  return *this;
}
ConfigBuilder& ConfigBuilder::Truncate(int32_t reduction_dim) {
  Set("truncate", truncate_set_);
  truncate_ = reduction_dim;
  return *this;
}
ConfigBuilder& ConfigBuilder::Autopilot(IncrementalMode mode, Quantization q) {
  Set("autopilot", autopilot_set_);
  autopilot_ = {mode, q};
  return *this;
}

absl::StatusOr<std::string> ConfigBuilder::BuildText(uint64_t num_points) const {
  if (!errors_.empty())
    return absl::InvalidArgumentError(absl::StrJoin(errors_, "; "));
  if (dimensionality_ == 0)
    return absl::InvalidArgumentError("dimensionality must be positive");
  const bool dot = distance_ == DistanceMeasure::kDotProduct;
  const std::string distance_msg =
      absl::StrCat("{distance_measure: \"", DistanceName(distance_), "\"}");

  std::string config = absl::StrCat("num_neighbors: ", num_neighbors_, "\n",
                                    "distance_measure ", distance_msg, "\n");

  if (autopilot_) {
    if (tree_ || upper_tree_ || ah_ || brute_force_ || reorder_ || pca_ || truncate_)
      return absl::InvalidArgumentError(
          "Autopilot() cannot be combined with other options: it chooses the "
          "whole configuration (the Python builder silently ignores the rest)");
    absl::StrAppend(&config, "autopilot { tree_ah { incremental_mode: ",
                    IncrementalModeName(autopilot_->first),
                    " reordering_dtype: ", QuantizationName(autopilot_->second),
                    " } }\n");
    if (num_points == 0)
      return absl::InvalidArgumentError("Autopilot() needs num_points");
    research_scann::ScannConfig proto;
    if (!google::protobuf::TextFormat::ParseFromString(config, &proto))
      return absl::InternalError("failed to parse generated autopilot config");
    auto tuned = research_scann::Autopilot(proto, nullptr, num_points, dimensionality_);
    if (!tuned.ok()) return tuned.status();
    std::string out;
    google::protobuf::TextFormat::PrintToString(*tuned, &out);
    return out;
  }

  // Projection (only emitted inside the partitioning stanza, as in Python).
  std::string projection;
  if (pca_ && truncate_)
    return absl::InvalidArgumentError("Exactly 1 of pca or truncate must be set");
  if (pca_) {
    const bool by_dim = pca_->reduction_dim.has_value();
    if (by_dim && pca_->pca_significance_threshold.has_value())
      return absl::InvalidArgumentError(
          "pca: set either reduction_dim or pca_significance_threshold, not both");
    projection = absl::StrCat("projection { projection_type: PCA input_dim: ",
                              dimensionality_, " ");
    if (by_dim) {
      absl::StrAppend(&projection, "num_dims_per_block: ", *pca_->reduction_dim);
    } else {
      absl::StrAppend(&projection, "pca_significance_threshold: ",
                      Num(pca_->pca_significance_threshold.value_or(0.8)),
                      " pca_truncation_threshold: ",
                      Num(pca_->pca_truncation_threshold));
    }
    absl::StrAppend(&projection, " }\n");
  } else if (truncate_) {
    if (*truncate_ >= static_cast<int64_t>(dimensionality_))
      return absl::InvalidArgumentError(
          absl::StrCat("reduction_dim must be less than ", dimensionality_));
    projection = absl::StrCat("projection { projection_type: TRUNCATE num_dims_per_block: ",
                              *truncate_, " input_dim: ", dimensionality_, " }\n");
  }
  if (!projection.empty() && !tree_)
    return absl::InvalidArgumentError(
        "Pca()/Truncate() require Tree(): the projection is part of the "
        "partitioning config (the Python builder silently drops it)");
  if (upper_tree_ && !tree_)
    return absl::InvalidArgumentError("UpperTree() requires Tree()");

  if (tree_) {
    const TreeOptions& t = *tree_;
    if (t.avq && !dot)
      return absl::InvalidArgumentError("AVQ only applies to dot product distance.");
    if (t.soar_lambda && !dot)
      return absl::InvalidArgumentError("SOAR requires dot product distance.");
    if (t.incremental_threshold_points && t.incremental_threshold_fraction)
      return absl::InvalidArgumentError(
          "set at most one of incremental_threshold_points / _fraction");
    absl::StrAppend(
        &config, "partitioning {\n",
        "  num_children: ", t.num_leaves, "\n",
        "  min_cluster_size: ", t.min_partition_size, "\n",
        "  max_clustering_iterations: ", t.training_iterations, "\n",
        "  single_machine_center_initialization: ",
        t.random_init ? "RANDOM_INITIALIZATION" : "DEFAULT_KMEANS_PLUS_PLUS", "\n",
        "  partitioning_distance { distance_measure: \"SquaredL2Distance\" }\n",
        "  query_spilling { spilling_type: FIXED_NUMBER_OF_CENTERS max_spill_centers: ",
        t.num_leaves_to_search, " }\n",
        "  expected_sample_size: ", t.training_sample_size, "\n",
        "  query_tokenization_distance_override ", distance_msg, "\n",
        "  partitioning_type: ", t.spherical ? "SPHERICAL" : "GENERIC", "\n",
        "  query_tokenization_type: ",
        t.quantize_centroids ? "FIXED_POINT_INT8" : "FLOAT", "\n");
    if (t.incremental_threshold_points)
      absl::StrAppend(&config, "  incremental_training_config { number_of_datapoints: ",
                      *t.incremental_threshold_points, " }\n");
    if (t.incremental_threshold_fraction)
      absl::StrAppend(&config, "  incremental_training_config { fraction: ",
                      Num(*t.incremental_threshold_fraction), " }\n");
    if (t.avq) absl::StrAppend(&config, "  avq: ", Num(*t.avq), "\n");
    if (t.soar_lambda) {
      absl::StrAppend(&config,
                      "  database_spilling { spilling_type: TWO_CENTER_ORTHOGONALITY_AMPLIFIED"
                      " orthogonality_amplification_lambda: ", Num(*t.soar_lambda));
      if (t.overretrieve_factor)
        absl::StrAppend(&config, " overretrieve_factor: ", Num(*t.overretrieve_factor));
      absl::StrAppend(&config, " }\n");
    }
    if (!projection.empty()) absl::StrAppend(&config, "  ", projection);
    if (upper_tree_) {
      const UpperTreeOptions& u = *upper_tree_;
      absl::StrAppend(
          &config, "  bottom_up_top_level_partitioner {\n",
          "    enabled: true\n",
          "    num_centroids: ", u.num_leaves, "\n",
          "    num_centroids_to_search: ", u.num_leaves_to_search, "\n",
          "    avq: ", Num(u.avq), "\n",
          "    soar { enabled: ", Bool(u.soar_lambda.has_value()),
          " lambda: ", Num(u.soar_lambda.value_or(1.5)),
          " overretrieve_factor: ", Num(u.overretrieve_factor.value_or(2.0)), " }\n",
          "    quantization: ", UpperTreeQuantization(u.scoring_mode), "\n",
          "    noise_shaping_threshold: ", Num(u.anisotropic_quantization_threshold), "\n",
          "  }\n");
    }
    absl::StrAppend(&config, "}\n");
  }

  if (ah_ && brute_force_)
    return absl::InvalidArgumentError(
        "Exactly 1 of score_ah or score_brute_force must be set");
  if (ah_) {
    const AhOptions& a = *ah_;
    if (a.dimensions_per_block <= 0)
      return absl::InvalidArgumentError("dimensions_per_block must be positive");
    const bool lut16 = a.hash_type == HashType::kLut16;
    const bool residual = a.residual_quantization.value_or(tree_.has_value() && dot);
    const uint32_t full_blocks = dimensionality_ / a.dimensions_per_block;
    const uint32_t partial_dims = dimensionality_ % a.dimensions_per_block;
    // "global top-N requires (1) LUT16, (2) int16 accumulators, and
    // (3) residual quantization" (scann_builder.py).
    const bool global_topn =
        lut16 && (full_blocks + (partial_dims > 0 ? 1 : 0)) <= 256 && residual;
    std::string proj;
    if (!projection.empty()) {
      // Python: "Projection will be set by C++ logic."
      proj = absl::StrCat("projection_type: CHUNK num_dims_per_block: ",
                          a.dimensions_per_block);
    } else if (partial_dims == 0) {
      proj = absl::StrCat("input_dim: ", dimensionality_,
                          " projection_type: CHUNK num_blocks: ", full_blocks,
                          " num_dims_per_block: ", a.dimensions_per_block);
    } else {
      proj = absl::StrCat("input_dim: ", dimensionality_,
                          " projection_type: VARIABLE_CHUNK"
                          " variable_blocks { num_blocks: ", full_blocks,
                          " num_dims_per_block: ", a.dimensions_per_block, " }"
                          " variable_blocks { num_blocks: 1 num_dims_per_block: ",
                          partial_dims, " }");
    }
    absl::StrAppend(
        &config, "hash {\n  asymmetric_hash {\n",
        "    lookup_type: ", lut16 ? "INT8_LUT16" : "INT8", "\n",
        "    use_residual_quantization: ", Bool(residual), "\n",
        "    use_global_topn: ", Bool(global_topn), "\n",
        "    quantization_distance { distance_measure: \"SquaredL2Distance\" }\n",
        "    num_clusters_per_block: ", lut16 ? 16 : 256, "\n",
        "    projection { ", proj, " }\n",
        "    fixed_point_lut_conversion_options { float_to_int_conversion_method: ROUND }\n",
        "    noise_shaping_threshold: ", Num(a.anisotropic_quantization_threshold), "\n",
        "    expected_sample_size: ", a.training_sample_size, "\n",
        "    max_clustering_iterations: ", a.training_iterations, "\n",
        "  }\n}\n");
  } else if (brute_force_) {
    absl::StrAppend(&config, "brute_force { ", ReorderBlock(*brute_force_),
                    " { enabled: ", Bool(*brute_force_ != Quantization::kFloat32),
                    " } }\n");
  } else {
    return absl::InvalidArgumentError(
        "Exactly 1 of score_ah or score_brute_force must be set");
  }

  if (reorder_) {
    const ReorderOptions& r = *reorder_;
    absl::StrAppend(&config, "exact_reordering {\n",
                    "  approx_num_neighbors: ", r.reordering_num_neighbors, "\n",
                    "  ", ReorderBlock(r.quantize), " { enabled: ",
                    Bool(r.quantize != Quantization::kFloat32),
                    " noise_shaping_threshold: ",
                    Num(r.anisotropic_quantization_threshold), " }\n",
                    "}\n");
  }
  return config;
}

absl::StatusOr<research_scann::ScannConfig> ConfigBuilder::Build(
    uint64_t num_points) const {
  auto text = BuildText(num_points);
  if (!text.ok()) return text.status();
  research_scann::ScannConfig config;
  if (!google::protobuf::TextFormat::ParseFromString(*text, &config))
    return absl::InternalError(absl::StrCat("generated config failed to parse:\n", *text));
  return config;
}

}  // namespace scann_core
