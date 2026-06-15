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

#include "scann/hashes/internal/asymmetric_hashing_impl_neon.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/noise_shaping_utils.h"

#if defined(__aarch64__)

namespace research_scann {
namespace asymmetric_hashing_internal {
namespace neon {

using fallback::ComputeParallelResidualComponent;
using fallback::OptimizeSingleSubspace;
using fallback::SubspaceResidualStats;
using fallback::ValidateNoiseShapingParams;

namespace {

SubspaceResidualStats ComputeResidualStatsForCluster(
    ConstSpan<float> maybe_residual_dptr, ConstSpan<float> original_dptr,
    double inv_norm, ConstSpan<float> quantized) {
  DCHECK_EQ(maybe_residual_dptr.size(), quantized.size());
  const size_t dims = maybe_residual_dptr.size();
  SubspaceResidualStats result;
  for (size_t i : Seq(dims)) {
    const double residual_coordinate =
        static_cast<double>(maybe_residual_dptr[i]) -
        static_cast<double>(quantized[i]);
    result.residual_norm += Square(residual_coordinate);
    result.parallel_residual_component +=
        residual_coordinate * original_dptr[i] * inv_norm;
  }
  return result;
}

StatusOr<vector<std::vector<SubspaceResidualStats>>> ComputeResidualStats(
    DatapointPtr<float> maybe_residual_dptr,
    DatapointPtr<float> original_dptr, ConstSpan<DenseDataset<float>> centers,
    const ChunkingProjection<float>& projection) {
  const size_t num_subspaces = centers.size();
  DCHECK_GE(num_subspaces, 1);
  vector<std::vector<SubspaceResidualStats>> residual_stats(num_subspaces);
  const size_t num_clusters_per_block = centers[0].size();

  ChunkedDatapoint<float> maybe_residual_dptr_chunked;
  ChunkedDatapoint<float> original_dptr_chunked;
  SCANN_RETURN_IF_ERROR(projection.ProjectInput(maybe_residual_dptr,
                                                &maybe_residual_dptr_chunked));
  SCANN_RETURN_IF_ERROR(
      projection.ProjectInput(original_dptr, &original_dptr_chunked));
  SCANN_RET_CHECK_EQ(maybe_residual_dptr_chunked.size(), num_subspaces);
  SCANN_RET_CHECK_EQ(original_dptr_chunked.size(), num_subspaces);
  double chunked_norm = 0.0;
  for (size_t subspace_idx : Seq(num_subspaces)) {
    for (float x : original_dptr_chunked[subspace_idx].values_span()) {
      chunked_norm += Square<double>(x);
    }
  }
  chunked_norm = std::sqrt(chunked_norm);
  const double inverse_chunked_norm = 1.0 / chunked_norm;

  for (size_t subspace_idx : Seq(num_subspaces)) {
    auto& cur_subspace_residual_stats = residual_stats[subspace_idx];
    cur_subspace_residual_stats.resize(num_clusters_per_block);
    const DenseDataset<float>& cur_subspace_centers = centers[subspace_idx];
    for (size_t cluster_idx : Seq(num_clusters_per_block)) {
      ConstSpan<float> center =
          cur_subspace_centers[cluster_idx].values_span();
      ConstSpan<float> maybe_residual_dptr_span =
          maybe_residual_dptr_chunked[subspace_idx].values_span();
      ConstSpan<float> original_dptr_span =
          original_dptr_chunked[subspace_idx].values_span();
      cur_subspace_residual_stats[cluster_idx] = ComputeResidualStatsForCluster(
          maybe_residual_dptr_span, original_dptr_span, inverse_chunked_norm,
          center);
    }
  }
  return residual_stats;
}

void InitializeToMinResidualNorm(
    ConstSpan<std::vector<SubspaceResidualStats>> residual_stats,
    MutableSpan<uint8_t> result) {
  DCHECK_EQ(result.size(), residual_stats.size());
  for (size_t subspace_idx : IndicesOf(residual_stats)) {
    auto it = std::min_element(
        residual_stats[subspace_idx].begin(),
        residual_stats[subspace_idx].end(),
        [](const SubspaceResidualStats& a, const SubspaceResidualStats& b) {
          return a.residual_norm < b.residual_norm;
        });
    result[subspace_idx] = it - residual_stats[subspace_idx].begin();
  }
}

}  // namespace

template <>
Status IndexDatapointNoiseShaped<float>(
    const DatapointPtr<float>& maybe_residual_dptr,
    const DatapointPtr<float>& original_dptr,
    const ChunkingProjection<float>& projection,
    ConstSpan<DenseDataset<float>> centers, double threshold, double eta,
    MutableSpan<uint8_t> result) {
  SCANN_RET_CHECK_EQ(result.size(), centers.size());
  SCANN_RET_CHECK_EQ(maybe_residual_dptr.dimensionality(),
                     original_dptr.dimensionality());
  SCANN_RETURN_IF_ERROR(ValidateNoiseShapingParams(threshold, eta));
  SCANN_ASSIGN_OR_RETURN(
      auto residual_stats,
      ComputeResidualStats(maybe_residual_dptr, original_dptr, centers,
                           projection));

  const double parallel_cost_multiplier =
      std::isnan(eta) ? ComputeParallelCostMultiplier(
                            threshold, SquaredL2Norm(original_dptr),
                            original_dptr.dimensionality())
                      : eta;
  InitializeToMinResidualNorm(residual_stats, result);
  double parallel_residual_component =
      ComputeParallelResidualComponent(result, residual_stats);

  vector<uint16_t> subspace_idxs(result.size());
  std::iota(subspace_idxs.begin(), subspace_idxs.end(), 0U);
  vector<double> subspace_residual_norms(result.size());
  for (size_t subspace_idx : IndicesOf(result)) {
    const uint8_t cluster_idx = result[subspace_idx];
    subspace_residual_norms[subspace_idx] =
        residual_stats[subspace_idx][cluster_idx].residual_norm;
  }
  std::vector<uint8_t> result_sorted(result.begin(), result.end());
  ZipSortBranchOptimized(
      std::greater<double>(), subspace_residual_norms.begin(),
      subspace_residual_norms.end(), result_sorted.begin(), result_sorted.end(),
      subspace_idxs.begin(), subspace_idxs.end());

  enum { kMaxRounds = 10 };
  bool cur_round_changes = true;
  for (int round = 0; cur_round_changes && round < kMaxRounds; ++round) {
    cur_round_changes = false;
    for (size_t i : IndicesOf(subspace_idxs)) {
      const size_t subspace_idx = subspace_idxs[i];
      ConstSpan<SubspaceResidualStats> cur_subspace_residual_stats =
          residual_stats[subspace_idx];
      const uint8_t cur_center_idx = result_sorted[i];
      auto subspace_result = OptimizeSingleSubspace(
          cur_subspace_residual_stats, cur_center_idx,
          parallel_residual_component, parallel_cost_multiplier);
      if (subspace_result.new_center_idx != cur_center_idx) {
        parallel_residual_component =
            subspace_result.new_parallel_residual_component;
        result_sorted[i] = subspace_result.new_center_idx;
        cur_round_changes = true;
      }
    }
  }

  for (size_t i : IndicesOf(result_sorted)) {
    const size_t subspace_idx = subspace_idxs[i];
    const uint8_t center_idx = result_sorted[i];
    result[subspace_idx] = center_idx;
  }
  return OkStatus();
}

}  // namespace neon
}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#endif  // defined(__aarch64__)
