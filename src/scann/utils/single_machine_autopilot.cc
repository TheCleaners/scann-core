// Copyright 2026 The Google Research Authors.
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
// Modified in 2026 by Elias Benali (@ebenali) and TheCleaners for
// scann-core (a derived work of ScaNN, not an official Google product);
// see NOTICE.

#include "scann/utils/single_machine_autopilot.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "absl/log/log.h"
#include "scann/data_format/dataset.h"
#include "scann/proto/auto_tuning.pb.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/proto/exact_reordering.pb.h"
#include "scann/proto/hash.pb.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/proto/projection.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

StatusOr<ScannConfig> AutopilotTreeAh(const ScannConfig& config,
                                      shared_ptr<const Dataset> dataset,
                                      DatapointIndex n, DimensionIndex dim) {
  const auto dist = config.distance_measure().distance_measure();
  if (dist != "SquaredL2Distance" && dist != "DotProductDistance") {
    return UnimplementedError(
        "Autopilot is currently implemented only for DotProductDistance "
        "and SquaredL2Distance.");
  }
  auto result = config;
  result.clear_brute_force();
  result.clear_hash();
  result.clear_partitioning();
  result.clear_exact_reordering();
  if (dataset != nullptr) {
    dim = dataset->dimensionality();
    n = dataset->size();
  }
  if (dim == 0) {
    return FailedPreconditionError("Not supported: dim == 0.");
  }

  const int ah_size = 2;

  const int kmeans_stable_size = 100;

  const int safety = 2;

  const int magic = 42;

  const int l1_size = config.autopilot().tree_ah().l1_size();
  const int k = config.num_neighbors();
  const int l3_size_bound =
      std::ceil(config.autopilot().tree_ah().l3_size() / dim / sizeof(float));

  int ah2_leaf_size = std::ceil(ah_size * 2 * l1_size / dim);
  ah2_leaf_size = std::max(ah2_leaf_size, safety * kmeans_stable_size);

  const int approx_num_neighbors =
      std::ceil(std::max(1.0 * safety * k, 100 * sqrt(k)));

  int treeah_bound =
      std::max(safety * approx_num_neighbors, magic * ah2_leaf_size);
  VLOG(2) << "Minimal Tree AH Size: " << treeah_bound;

  if (n < treeah_bound) {
    result.mutable_brute_force();
    return result;
  }

  result.mutable_exact_reordering()->set_approx_num_neighbors(
      approx_num_neighbors);
  if (config.autopilot().tree_ah().reordering_dtype() ==
      AutopilotTreeAH::INT8) {
    result.mutable_exact_reordering()->mutable_fixed_point()->set_enabled(true);
  } else if (config.autopilot().tree_ah().reordering_dtype() ==
             AutopilotTreeAH::BFLOAT16) {
    result.mutable_exact_reordering()->mutable_bfloat16()->set_enabled(true);
  }

  auto ah = result.mutable_hash()->mutable_asymmetric_hash();
  ah->mutable_quantization_distance()->set_distance_measure(
      "SquaredL2Distance");
  ah->set_num_clusters_per_block(16);
  ah->set_max_clustering_iterations(10);
  ah->set_lookup_type(ah->INT8_LUT16);

  ah->set_expected_sample_size(16 * kmeans_stable_size * safety * 10);
  if (dist == "DotProductDistance") {
    ah->set_use_residual_quantization(true);
    ah->set_use_global_topn(true);
    ah->set_noise_shaping_threshold(0.2);
  }

  int full_blocks = dim / ah_size;
  int partial_block_dims = dim % ah_size;
  auto ah_proj = ah->mutable_projection();
  ah_proj->set_input_dim(dim);
  ah_proj->set_num_dims_per_block(ah_size);
  if (partial_block_dims == 0) {
    ah_proj->set_projection_type(ah_proj->CHUNK);
    ah_proj->set_num_blocks(full_blocks);
    ah_proj->set_num_dims_per_block(ah_size);
  } else {
    ah_proj->set_projection_type(ah_proj->VARIABLE_CHUNK);
    auto vblock = ah_proj->add_variable_blocks();
    vblock->set_num_blocks(full_blocks);
    vblock->set_num_dims_per_block(ah_size);
    vblock = ah_proj->add_variable_blocks();
    vblock->set_num_blocks(1);
    vblock->set_num_dims_per_block(partial_block_dims);
  }

  VLOG(2) << "AH2 Leaf Size: " << ah2_leaf_size;
  int tree_size = n / ah2_leaf_size;

  const int train_size_bound = std::ceil(
      std::sqrt(60.0 * 32 * 2e9 / dim / (safety * kmeans_stable_size)));
  VLOG(2) << "L1 recommended tree size: " << tree_size;
  VLOG(2) << "L3 recommended tree size: " << l3_size_bound;
  VLOG(2) << "Training time recommended tree size: " << train_size_bound;
  tree_size = std::min(tree_size, l3_size_bound);
  tree_size = std::min(tree_size, train_size_bound);

  auto part = result.mutable_partitioning();

  if (config.autopilot().tree_ah().has_num_leaf_partitions()) {
    part->set_num_children(config.autopilot().tree_ah().num_leaf_partitions());
  } else {
    part->set_num_children(tree_size);
  }

  if (config.autopilot().tree_ah().has_fraction_leaf_partitions()) {
    auto n = part->num_children();
    part->set_num_children(
        config.autopilot().tree_ah().fraction_leaf_partitions() * n);
  }

  if (config.autopilot().tree_ah().has_partitioning_expected_sample_size()) {
    part->set_expected_sample_size(
        config.autopilot().tree_ah().partitioning_expected_sample_size());
  } else {
    part->set_expected_sample_size(tree_size * kmeans_stable_size * safety);
  }

  if (config.autopilot()
          .tree_ah()
          .has_first_n_leaf_partitions_for_assignment()) {
    part->set_first_n_centroids(
        config.autopilot().tree_ah().first_n_leaf_partitions_for_assignment());
  }

  part->set_min_cluster_size(10);
  part->set_max_clustering_iterations(10);
  part->set_single_machine_center_initialization(part->RANDOM_INITIALIZATION);
  part->mutable_partitioning_distance()->set_distance_measure(
      "SquaredL2Distance");
  part->mutable_query_spilling()->set_spilling_type(
      part->query_spilling().FIXED_NUMBER_OF_CENTERS);

  const int leaves_to_search =
      std::ceil(magic * std::pow(2.0, std::log(1.0 * tree_size / magic) /
                                          std::log(10.0)));

  part->mutable_query_spilling()->set_max_spill_centers(
      std::min(tree_size, leaves_to_search));
  part->mutable_query_tokenization_distance_override()->set_distance_measure(
      dist);
  part->set_partitioning_type(part->GENERIC);

  part->set_query_tokenization_type(part->FLOAT);

  switch (config.autopilot().tree_ah().incremental_mode()) {
    case (AutopilotTreeAH::ONLINE_INCREMENTAL):
      part->mutable_incremental_training_config()->set_autopilot(true);
      part->mutable_incremental_training_config()->set_fraction(0.5);
      break;
    case (AutopilotTreeAH::ONLINE):
      part->mutable_incremental_training_config()->set_fraction(0.5);
      break;
    default:
      break;
  }
  VLOG(1) << "Autopilot Tree AH Result: " << result.DebugString();
  return result;
}

// scann-core: the TUNED_V1 rules (AutopilotTreeAH.rules), from a tuning
// study of single-query search on GloVe-100, SIFT-128 and 768-dimensional
// text embeddings, checked on more datasets (docs/tuning.md, "Defaults and
// autopilot"). They keep upstream's brute-force cutoff, reordering count and
// reordering precision, and change the tree-AH index:
//   * leaves: upstream's count, but at most about sqrt(n) (upstream's leaves
//     hold 4 * L1 / dim points: at high dimension thousands of tiny leaves,
//     each costing a centre to compare the query with);
//   * leaves_to_search: upstream's count, times sqrt(leaves / upstream's) when
//     there are fewer leaves (coarser leaves find fewer neighbours per
//     scanned point at high recall: a quarter of the leaves needed half as
//     many searched for the same recall at the default settings);
//   * dot product: tree AVQ 2.5; 2 dimensions per AH block up to 384
//     dimensions, then about 192 blocks (3 at 512, 4 at 768); an anisotropic
//     threshold that scales with the norms and falls with the dimension
//     above 128 (upstream: 0.2 whatever both, which caps recall at 768
//     dimensions with 4 dimensions per block);
//   * rounded (not truncated) AH lookup tables;
//   * squared L2 through l2_as_dot_product where the norms are nearly
//     constant (decided in AutopilotChoosesL2AsDotProduct, before the index
//     is built).
namespace {

// Upstream's constants (AutopilotTreeAh).
constexpr int kAhSize = 2;
constexpr int kKmeansStableSize = 100;
constexpr int kSafety = 2;
constexpr int kMagic = 42;

constexpr double kTunedTreeAvq = 2.5;
// Blocks of kAhSize dimensions up to this many blocks; above, about this many.
constexpr int kTunedTargetBlocks = 192;
// The anisotropic threshold relative to the norm: upstream's 0.2 up to 128
// dimensions, then 0.2 (128 / d)^0.75 (0.071 at 512, 0.052 at 768). The
// penalty ratio it makes, (d - 1) t^2 / (1 - t^2) for t = T / |x|
// (ComputeParallelCostMultiplier), is 4.1 at d=100, 2.6 at 512 and 2.1 at
// 768. The study's good values: t = 0.15-0.3 at d=100, 0.2-0.39 on SIFT
// through l2_as_dot_product (d=129), 0.05 at d=768 (0.1 capped recall at
// 0.97, 0.2 at 0.795).
constexpr double kTunedThresholdAtLowDims = 0.2;
constexpr double kTunedThresholdDims = 128;
constexpr double kTunedThresholdExponent = 0.75;
// The threshold is relative to this quantile of the datapoints' norms, so
// that 95% of them get at most the intended penalty ratio (it falls with the
// norm, and grows without bound as the norm approaches the threshold).
constexpr double kTunedNormQuantile = 0.05;
// l2_as_dot_product only where squared L2 and the inner product nearly
// agree: the squared norms vary by at most this much (their coefficient of
// variation). SIFT (0.0026) gained 33-47% QPS at every recall; synthetic
// Gaussian clusters whose squared norms vary by 0.20 lost 0.04 recall at the
// default settings (the stored vectors' extra coordinate then varies a lot).
constexpr double kTunedMaxSquaredNormCv = 0.05;
// And not for data far from the origin relative to its spread (|mean|^2 /
// spread above this; the study's datasets: 0.13-1.23): l2_as_dot_product's
// scores cancel terms of the order of |x|^2.
constexpr double kTunedMaxOffsetRatio = 4.0;

// Upstream's sizes for (n, dim), the same formulas as AutopilotTreeAh.
struct UpstreamTreeAhSizes {
  int approx_num_neighbors;
  int64_t brute_force_below;
  int64_t tree_size;
  int64_t leaves_to_search;
};

UpstreamTreeAhSizes ComputeUpstreamTreeAhSizes(const ScannConfig& config,
                                               DatapointIndex n,
                                               DimensionIndex dim) {
  const int l1_size = config.autopilot().tree_ah().l1_size();
  const int k = config.num_neighbors();
  const int l3_size_bound =
      std::ceil(config.autopilot().tree_ah().l3_size() / dim / sizeof(float));
  int ah2_leaf_size = std::ceil(kAhSize * 2 * l1_size / dim);
  ah2_leaf_size = std::max(ah2_leaf_size, kSafety * kKmeansStableSize);
  const int approx_num_neighbors =
      std::ceil(std::max(1.0 * kSafety * k, 100 * sqrt(k)));
  const int treeah_bound =
      std::max(kSafety * approx_num_neighbors, kMagic * ah2_leaf_size);
  int tree_size = n / ah2_leaf_size;
  const int train_size_bound = std::ceil(
      std::sqrt(60.0 * 32 * 2e9 / dim / (kSafety * kKmeansStableSize)));
  tree_size = std::min(tree_size, l3_size_bound);
  tree_size = std::min(tree_size, train_size_bound);
  const int leaves_to_search =
      std::ceil(kMagic * std::pow(2.0, std::log(1.0 * tree_size / kMagic) /
                                           std::log(10.0)));
  return {approx_num_neighbors, treeah_bound, tree_size,
          std::min(tree_size, leaves_to_search)};
}

bool IsTunedTreeAh(const ScannConfig& config) {
  return config.has_autopilot() &&
         (config.autopilot().autopilot_option_case() ==
              AutopilotConfig::kTreeAh ||
          config.autopilot().autopilot_option_case() ==
              AutopilotConfig::AUTOPILOT_OPTION_NOT_SET) &&
         config.autopilot().tree_ah().rules() == AutopilotTreeAH::TUNED_V1;
}

std::optional<AutopilotDataStats> DataStatsIfFloat(const Dataset* dataset) {
  if (dataset == nullptr || dataset->IsSparse() ||
      dataset->TypeTag() != TagForType<float>() || dataset->empty())
    return std::nullopt;
  const auto* dense = static_cast<const DenseDataset<float>*>(dataset);
  return ComputeAutopilotDataStats(dense->data(), dense->size(),
                                   dense->dimensionality());
}

void SetChunkProjection(DimensionIndex dim, int dims_per_block,
                        ProjectionConfig* proj) {
  const int full_blocks = dim / dims_per_block;
  const int partial_block_dims = dim % dims_per_block;
  proj->set_input_dim(dim);
  proj->set_num_dims_per_block(dims_per_block);
  if (partial_block_dims == 0) {
    proj->set_projection_type(ProjectionConfig::CHUNK);
    proj->set_num_blocks(full_blocks);
  } else {
    proj->set_projection_type(ProjectionConfig::VARIABLE_CHUNK);
    auto* vblock = proj->add_variable_blocks();
    vblock->set_num_blocks(full_blocks);
    vblock->set_num_dims_per_block(dims_per_block);
    vblock = proj->add_variable_blocks();
    vblock->set_num_blocks(1);
    vblock->set_num_dims_per_block(partial_block_dims);
  }
}

StatusOr<ScannConfig> AutopilotTreeAhTuned(const ScannConfig& config,
                                           const Dataset* dataset,
                                           DatapointIndex n,
                                           DimensionIndex dim) {
  const auto dist = config.distance_measure().distance_measure();
  if (dist != "SquaredL2Distance" && dist != "DotProductDistance") {
    return UnimplementedError(
        "Autopilot is currently implemented only for DotProductDistance "
        "and SquaredL2Distance.");
  }
  auto result = config;
  result.clear_brute_force();
  result.clear_hash();
  result.clear_partitioning();
  result.clear_exact_reordering();
  if (dataset != nullptr) {
    dim = dataset->dimensionality();
    n = dataset->size();
  }
  if (dim == 0) {
    return FailedPreconditionError("Not supported: dim == 0.");
  }
  const UpstreamTreeAhSizes up = ComputeUpstreamTreeAhSizes(config, n, dim);
  if (n < up.brute_force_below) {
    result.mutable_brute_force();
    return result;
  }
  const bool dot = dist == "DotProductDistance";
  const AutopilotTreeAH& opts = config.autopilot().tree_ah();

  // As upstream: unset, the proto's default (bfloat16); scann-core's
  // builders set it (float32 unless asked otherwise).
  result.mutable_exact_reordering()->set_approx_num_neighbors(
      up.approx_num_neighbors);
  if (opts.reordering_dtype() == AutopilotTreeAH::INT8) {
    result.mutable_exact_reordering()->mutable_fixed_point()->set_enabled(true);
  } else if (opts.reordering_dtype() == AutopilotTreeAH::BFLOAT16) {
    result.mutable_exact_reordering()->mutable_bfloat16()->set_enabled(true);
  }

  auto ah = result.mutable_hash()->mutable_asymmetric_hash();
  ah->mutable_quantization_distance()->set_distance_measure(
      "SquaredL2Distance");
  ah->set_num_clusters_per_block(16);
  ah->set_max_clustering_iterations(10);
  ah->set_lookup_type(ah->INT8_LUT16);
  ah->set_expected_sample_size(16 * kKmeansStableSize * kSafety * 10);
  // Rounded lookup tables (the Python builder's; upstream's autopilot
  // truncates): the truncation bias adds up over the blocks, and cost recall
  // at the default settings with 3 dimensions per block at 512 dimensions.
  ah->mutable_fixed_point_lut_conversion_options()
      ->set_float_to_int_conversion_method(
          AsymmetricHasherConfig::FixedPointLUTConversionOptions::ROUND);
  int dims_per_block = kAhSize;
  if (dot) {
    dims_per_block =
        std::max<int>(kAhSize, (dim + kTunedTargetBlocks - 1) /
                                   kTunedTargetBlocks);
    ah->set_use_residual_quantization(true);
    ah->set_use_global_topn(true);
    double threshold;
    if (opts.has_noise_shaping_threshold()) {
      threshold = opts.noise_shaping_threshold();
    } else {
      // Measured from the data when it is available as floats (when an
      // index is built), and recorded; a reloaded index's dataset may be
      // bfloat16 or int8, but it has it recorded. Without data (a config
      // preview), unit norms are assumed and nothing is recorded.
      const std::optional<AutopilotDataStats> stats =
          DataStatsIfFloat(dataset);
      const double norm = stats ? stats->norm_quantile : 1.0;
      threshold = std::numeric_limits<double>::quiet_NaN();
      if (dim >= 2 && norm > 0 && std::isfinite(norm))
        threshold =
            norm * kTunedThresholdAtLowDims *
            std::pow(std::min(1.0, kTunedThresholdDims / dim),
                     kTunedThresholdExponent);
      if (stats) {
        result.mutable_autopilot()
            ->mutable_tree_ah()
            ->set_noise_shaping_threshold(threshold);
      }
    }
    if (!std::isnan(threshold)) ah->set_noise_shaping_threshold(threshold);
  }
  SetChunkProjection(dim, dims_per_block, ah->mutable_projection());

  auto part = result.mutable_partitioning();
  const int64_t tree_size = std::max<int64_t>(
      1, std::min<int64_t>(up.tree_size,
                           std::llround(std::sqrt(static_cast<double>(n)))));
  if (opts.has_num_leaf_partitions()) {
    part->set_num_children(opts.num_leaf_partitions());
  } else {
    part->set_num_children(tree_size);
  }
  if (opts.has_fraction_leaf_partitions()) {
    part->set_num_children(std::max<int64_t>(
        1, opts.fraction_leaf_partitions() * part->num_children()));
  }
  if (opts.has_partitioning_expected_sample_size()) {
    part->set_expected_sample_size(opts.partitioning_expected_sample_size());
  } else {
    part->set_expected_sample_size(tree_size * kKmeansStableSize * kSafety);
  }
  if (opts.has_first_n_leaf_partitions_for_assignment()) {
    part->set_first_n_centroids(opts.first_n_leaf_partitions_for_assignment());
  }
  part->set_min_cluster_size(10);
  part->set_max_clustering_iterations(10);
  part->set_single_machine_center_initialization(part->RANDOM_INITIALIZATION);
  part->mutable_partitioning_distance()->set_distance_measure(
      "SquaredL2Distance");
  part->mutable_query_spilling()->set_spilling_type(
      part->query_spilling().FIXED_NUMBER_OF_CENTERS);
  // Upstream's leaves_to_search for its leaf count, scaled to this one: by
  // sqrt(ratio) for fewer leaves, by the ratio (the same fraction) for more.
  const int64_t num_children = part->num_children();
  const double ratio =
      up.tree_size > 0 ? 1.0 * num_children / up.tree_size : 1.0;
  const double scale = ratio < 1 ? std::sqrt(ratio) : ratio;
  part->mutable_query_spilling()->set_max_spill_centers(std::clamp<int64_t>(
      std::ceil(up.leaves_to_search * scale - 1e-9), 1, num_children));
  part->mutable_query_tokenization_distance_override()->set_distance_measure(
      dist);
  part->set_partitioning_type(part->GENERIC);
  part->set_query_tokenization_type(part->FLOAT);
  if (dot) part->set_avq(kTunedTreeAvq);

  switch (opts.incremental_mode()) {
    case (AutopilotTreeAH::ONLINE_INCREMENTAL):
      part->mutable_incremental_training_config()->set_autopilot(true);
      part->mutable_incremental_training_config()->set_fraction(0.5);
      break;
    case (AutopilotTreeAH::ONLINE):
      part->mutable_incremental_training_config()->set_fraction(0.5);
      break;
    default:
      break;
  }
  VLOG(1) << "Autopilot Tree AH (TUNED_V1) Result: " << result.DebugString();
  return result;
}

}  // namespace

AutopilotDataStats ComputeAutopilotDataStats(ConstSpan<float> data,
                                             DatapointIndex n,
                                             DimensionIndex dim) {
  AutopilotDataStats stats;
  if (n == 0 || dim == 0 || data.size() < static_cast<size_t>(n) * dim)
    return stats;
  const size_t m = std::min<size_t>(n, kAutopilotStatsSampleSize);
  std::vector<double> norms(m);
  std::vector<double> mean(dim, 0.0);
  double sum_sq = 0;
  for (size_t j = 0; j < m; ++j) {
    const size_t row = j * static_cast<size_t>(n) / m;
    const float* x = data.data() + row * dim;
    double sq = 0;
    for (size_t i = 0; i < dim; ++i) {
      sq += static_cast<double>(x[i]) * x[i];
      mean[i] += x[i];
    }
    norms[j] = std::sqrt(sq);
    sum_sq += sq;
  }
  double mean_sq = 0;
  for (double& v : mean) {
    v /= m;
    mean_sq += v * v;
  }
  const size_t q = static_cast<size_t>(kTunedNormQuantile * (m - 1));
  std::nth_element(norms.begin(), norms.begin() + q, norms.end());
  stats.norm_quantile = norms[q];
  const double spread = sum_sq / m - mean_sq;
  stats.offset_ratio = spread > 1e-12 * (sum_sq / m)
                           ? mean_sq / spread
                           : std::numeric_limits<double>::infinity();
  // norms[] is reordered by nth_element; the squared norms' order doesn't
  // matter here.
  const double mean_norm_sq = sum_sq / m;
  double var_norm_sq = 0;
  for (double v : norms) {
    const double d = v * v - mean_norm_sq;
    var_norm_sq += d * d;
  }
  var_norm_sq /= m;
  if (mean_norm_sq > 0)
    stats.squared_norm_cv = std::sqrt(var_norm_sq) / mean_norm_sq;
  return stats;
}

bool AutopilotChoosesL2AsDotProduct(const ScannConfig& config,
                                    ConstSpan<float> data, DatapointIndex n) {
  if (!IsTunedTreeAh(config) ||
      !config.autopilot().tree_ah().allow_l2_as_dot_product() ||
      config.has_l2_as_dot_product() ||
      config.distance_measure().distance_measure() != "SquaredL2Distance" ||
      std::isfinite(config.epsilon_distance()) || config.has_min_distance() ||
      n == 0 || data.size() % n != 0)
    return false;
  const DimensionIndex dim = data.size() / n;
  if (dim == 0) return false;
  // Only where the rules build a tree: brute force stays exact squared L2.
  if (n < ComputeUpstreamTreeAhSizes(config, n, dim + 1).brute_force_below)
    return false;
  const AutopilotDataStats stats = ComputeAutopilotDataStats(data, n, dim);
  return stats.squared_norm_cv <= kTunedMaxSquaredNormCv &&
         stats.offset_ratio <= kTunedMaxOffsetRatio;
}

bool ApplyAutopilotL2AsDotProduct(ScannConfig* config, ConstSpan<float> data,
                                  DatapointIndex n) {
  if (!AutopilotChoosesL2AsDotProduct(*config, data, n)) return false;
  config->mutable_l2_as_dot_product();
  config->clear_brute_force();
  config->clear_hash();
  config->clear_partitioning();
  config->clear_exact_reordering();
  return true;
}

StatusOr<ScannConfig> Autopilot(const ScannConfig& config,
                                shared_ptr<const Dataset> dataset,
                                DatapointIndex n, DimensionIndex dim) {
  if (!config.has_autopilot())
    return FailedPreconditionError("Autopilot config is not present.");
  auto ds = std::static_pointer_cast<const DenseDataset<float>>(dataset);
  if (ds == nullptr &&
      (n == kInvalidDatapointIndex || dim == kInvalidDimension))
    return FailedPreconditionError(
        "Autopilot requires either original, uncompressed Dataset, or "
        "explicitly specified dimensionality and size.");
  switch (config.autopilot().autopilot_option_case()) {
    case (AutopilotConfig::AutopilotOptionCase::AUTOPILOT_OPTION_NOT_SET):
    case (AutopilotConfig::AutopilotOptionCase::kTreeAh):
      // scann-core: the tuned rules, when asked for (AutopilotTreeAH.rules).
      if (config.autopilot().tree_ah().rules() == AutopilotTreeAH::TUNED_V1)
        return AutopilotTreeAhTuned(config, dataset.get(), n, dim);
      return AutopilotTreeAh(config, ds, n, dim);
    default:
      return FailedPreconditionError("Autopilot option not supported: %s",
                                     config.autopilot().DebugString());
  }
}

}  // namespace research_scann
