// Copyright 2026 The Google Research Authors.
// Copyright 2026 Elias Benali (@ebenali) and TheCleaners.
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

// C++ equivalent of upstream's Python ScannBuilder (scann_builder.py): turns
// high-level options (tree / score_ah / reorder / ...) into a ScannConfig.
//
// It produces the same configuration as the Python builder for the same
// options (tests/cpp/config_builder_test.cc compares against configs emitted
// by the Python builder), except where the Python builder silently does
// something other than what was asked; those cases are errors here:
//   * UpperTree() without Tree()                (Python: ignored)
//   * Pca()/Truncate() without Tree()           (Python: projection dropped)
//   * Autopilot() combined with manual options  (Python: others ignored)
// and three Python quirks are fixed:
//   * UpperTree soar_lambda = 0.0 is kept (Python replaces it with 1.5)
//   * Pca(reduction_dim) doesn't also require clearing the significance
//     threshold (Python raises unless pca_significance_threshold=None)
//   * the incremental threshold is typed (Python treats `True` as 1)

#ifndef SCANN_CORE_CONFIG_BUILDER_H_
#define SCANN_CORE_CONFIG_BUILDER_H_

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "scann/proto/scann.pb.h"

namespace scann_core {

enum class DistanceMeasure { kDotProduct, kSquaredL2 };
// Python: scann.ReorderType.
enum class Quantization { kFloat32, kInt8, kBfloat16 };
enum class HashType { kLut16, kLut256 };
// Python: scann_builder.IncrementalMode.
enum class IncrementalMode { kNone, kOnline, kOnlineIncremental };

inline constexpr double kUnset = std::numeric_limits<double>::quiet_NaN();

// Python: ScannBuilder.tree(). Defaults match Python's.
struct TreeOptions {
  int32_t num_leaves = 0;
  int32_t num_leaves_to_search = 0;
  int64_t training_sample_size = 100000;
  int32_t min_partition_size = 50;
  int32_t training_iterations = 12;
  bool spherical = false;
  bool quantize_centroids = false;
  // Python's default is random initialization. Note that upstream's random
  // initialization makes training depend on hash-table iteration order, so
  // it is not reproducible run to run; false (k-means++) is.
  bool random_init = true;
  // At most one of these (Python: int -> points, float -> fraction).
  std::optional<int64_t> incremental_threshold_points;
  std::optional<double> incremental_threshold_fraction;
  std::optional<double> avq;           // dot_product only
  std::optional<double> soar_lambda;   // dot_product only
  std::optional<double> overretrieve_factor;
};

// Python: ScannBuilder.upper_tree(). Requires Tree().
struct UpperTreeOptions {
  int32_t num_leaves = 0;
  int32_t num_leaves_to_search = 0;
  double avq = kUnset;
  std::optional<double> soar_lambda;
  std::optional<double> overretrieve_factor;
  Quantization scoring_mode = Quantization::kInt8;
  double anisotropic_quantization_threshold = kUnset;
};

// Python: ScannBuilder.score_ah().
struct AhOptions {
  int32_t dimensions_per_block = 2;
  double anisotropic_quantization_threshold = kUnset;
  int64_t training_sample_size = 100000;
  HashType hash_type = HashType::kLut16;
  int32_t training_iterations = 10;
  // Unset: true iff Tree() is configured and the distance is dot product
  // (the Python default).
  std::optional<bool> residual_quantization;
};

// Python: ScannBuilder.reorder().
struct ReorderOptions {
  int32_t reordering_num_neighbors = 0;
  Quantization quantize = Quantization::kFloat32;
  double anisotropic_quantization_threshold = kUnset;
};

// Python: ScannBuilder.pca(). Exactly one of reduction_dim /
// pca_significance_threshold; the threshold defaults to 0.8 only when
// reduction_dim is unset.
struct PcaOptions {
  std::optional<int32_t> reduction_dim;
  std::optional<double> pca_significance_threshold;
  double pca_truncation_threshold = 0.6;
};

class ConfigBuilder {
 public:
  ConfigBuilder(int32_t num_neighbors, DistanceMeasure distance,
                uint32_t dimensionality);

  ConfigBuilder& Tree(const TreeOptions& options);
  ConfigBuilder& UpperTree(const UpperTreeOptions& options);
  ConfigBuilder& ScoreAh(const AhOptions& options);
  ConfigBuilder& ScoreBruteForce(Quantization quantize = Quantization::kFloat32);
  ConfigBuilder& Reorder(const ReorderOptions& options);
  ConfigBuilder& Pca(const PcaOptions& options);
  ConfigBuilder& Truncate(int32_t reduction_dim);
  ConfigBuilder& Autopilot(IncrementalMode mode = IncrementalMode::kNone,
                           Quantization quantize = Quantization::kFloat32);

  // The config as ScaNN text format. num_points is only needed with
  // Autopilot(), which sizes the config to the dataset.
  absl::StatusOr<std::string> BuildText(uint64_t num_points = 0) const;
  absl::StatusOr<research_scann::ScannConfig> Build(uint64_t num_points = 0) const;

 private:
  void Set(const char* what, bool& flag);

  int32_t num_neighbors_;
  DistanceMeasure distance_;
  uint32_t dimensionality_;
  std::optional<TreeOptions> tree_;
  std::optional<UpperTreeOptions> upper_tree_;
  std::optional<AhOptions> ah_;
  std::optional<Quantization> brute_force_;
  std::optional<ReorderOptions> reorder_;
  std::optional<PcaOptions> pca_;
  std::optional<int32_t> truncate_;
  std::optional<std::pair<IncrementalMode, Quantization>> autopilot_;
  bool tree_set_ = false, upper_tree_set_ = false, ah_set_ = false,
       bf_set_ = false, reorder_set_ = false, pca_set_ = false,
       truncate_set_ = false, autopilot_set_ = false;
  std::vector<std::string> errors_;
};

}  // namespace scann_core

#endif  // SCANN_CORE_CONFIG_BUILDER_H_
