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

#include <arm_neon.h>

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

  const float* maybe_ptr = maybe_residual_dptr.data();
  const float* original_ptr = original_dptr.data();
  const float* quantized_ptr = quantized.data();

  const float64x2_t inv_norm_v = vdupq_n_f64(inv_norm);
  float64x2_t residual_norm_acc = vdupq_n_f64(0.0);
  float64x2_t parallel_acc = vdupq_n_f64(0.0);

  size_t i = 0;
  for (; i + 2 <= dims; i += 2) {
    const float32x2_t maybe_f32 = vld1_f32(maybe_ptr + i);
    const float32x2_t quantized_f32 = vld1_f32(quantized_ptr + i);
    const float32x2_t original_f32 = vld1_f32(original_ptr + i);

    const float64x2_t maybe_f64 = vcvt_f64_f32(maybe_f32);
    const float64x2_t quantized_f64 = vcvt_f64_f32(quantized_f32);
    const float64x2_t original_f64 = vcvt_f64_f32(original_f32);

    const float64x2_t residual_coordinate = vsubq_f64(maybe_f64, quantized_f64);
    residual_norm_acc =
        vfmaq_f64(residual_norm_acc, residual_coordinate, residual_coordinate);
    parallel_acc = vfmaq_f64(
        parallel_acc, vmulq_f64(residual_coordinate, original_f64), inv_norm_v);
  }

  result.residual_norm = vaddvq_f64(residual_norm_acc);
  result.parallel_residual_component = vaddvq_f64(parallel_acc);

  for (; i < dims; ++i) {
    const double residual_coordinate = static_cast<double>(maybe_ptr[i]) -
                                       static_cast<double>(quantized_ptr[i]);
    result.residual_norm += Square(residual_coordinate);
    result.parallel_residual_component +=
        residual_coordinate * original_ptr[i] * inv_norm;
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
  const float* original_chunked_ptr = original_dptr_chunked.storage().data();
  const size_t original_chunked_size = original_dptr_chunked.storage().size();

  float64x2_t chunked_norm_acc0 = vdupq_n_f64(0.0f);
  float64x2_t chunked_norm_acc1 = vdupq_n_f64(0.0f);

  size_t i = 0;
  for (; i + 8 <= original_chunked_size; i += 8) {
    float32x4_t o0_s32 = vld1q_f32(original_chunked_ptr + i);
    float32x4_t o1_s32 = vld1q_f32(original_chunked_ptr + i + 4);
    float64x2_t o0_s64_lo = vcvt_f64_f32(vget_low_f32(o0_s32));
    float64x2_t o0_s64_hi = vcvt_f64_f32(vget_high_f32(o0_s32));
    float64x2_t o1_s64_lo = vcvt_f64_f32(vget_low_f32(o1_s32));
    float64x2_t o1_s64_hi = vcvt_f64_f32(vget_high_f32(o1_s32));

    chunked_norm_acc0 = vfmaq_f64(chunked_norm_acc0, o0_s64_lo, o0_s64_lo);
    chunked_norm_acc1 = vfmaq_f64(chunked_norm_acc1, o0_s64_hi, o0_s64_hi);
    chunked_norm_acc0 = vfmaq_f64(chunked_norm_acc0, o1_s64_lo, o1_s64_lo);
    chunked_norm_acc1 = vfmaq_f64(chunked_norm_acc1, o1_s64_hi, o1_s64_hi);
  }

  for (; i + 4 <= original_chunked_size; i += 4) {
    float32x4_t o_s32 = vld1q_f32(original_chunked_ptr + i);
    float64x2_t o_s64_lo = vcvt_f64_f32(vget_low_f32(o_s32));
    float64x2_t o_s64_hi = vcvt_f64_f32(vget_high_f32(o_s32));

    chunked_norm_acc0 = vfmaq_f64(chunked_norm_acc0, o_s64_lo, o_s64_lo);
    chunked_norm_acc1 = vfmaq_f64(chunked_norm_acc1, o_s64_hi, o_s64_hi);
  }

  double chunked_norm =
      vaddvq_f64(vaddq_f64(chunked_norm_acc0, chunked_norm_acc1));

  for (; i < original_chunked_size; ++i) {
    const double v = original_chunked_ptr[i];
    chunked_norm += Square<double>(v);
  }

  // Identity chunking repeats the full vector for each subspace, so scale the
  // squared norm by the number of subspaces.
  if (projection.is_identity_chunk_impl()) {
    chunked_norm *= num_subspaces;
  }

  chunked_norm = std::sqrt(chunked_norm);
  const double inverse_chunked_norm = 1.0 / chunked_norm;

  for (size_t subspace_idx : Seq(num_subspaces)) {
    auto& cur_subspace_residual_stats = residual_stats[subspace_idx];
    cur_subspace_residual_stats.resize(num_clusters_per_block);
    const DenseDataset<float>& cur_subspace_centers = centers[subspace_idx];
    const size_t cur_dims = cur_subspace_centers[0].values_span().size();
    ConstSpan<float> maybe_residual_dptr_span =
        maybe_residual_dptr_chunked[subspace_idx].values_span();
    ConstSpan<float> original_dptr_span =
        original_dptr_chunked[subspace_idx].values_span();

    if (cur_dims == 1) {
      const double maybe_val = static_cast<double>(maybe_residual_dptr_span[0]);
      const double original_val = static_cast<double>(original_dptr_span[0]);
      const float* center_raw = cur_subspace_centers.data().data();
      SubspaceResidualStats* stats = cur_subspace_residual_stats.data();
      const float64x2_t maybe = vdupq_n_f64(maybe_val);
      const float64x2_t original = vdupq_n_f64(original_val);
      const float64x2_t inv_norm = vdupq_n_f64(inverse_chunked_norm);
      const float64x2_t orig_inv_norm = vmulq_f64(original, inv_norm);

      size_t cluster_idx = 0;
      for (; cluster_idx + 4 <= num_clusters_per_block; cluster_idx += 4) {
        const float32x4_t quantized_f32 = vld1q_f32(center_raw + cluster_idx);
        const float64x2_t quantized_lo =
            vcvt_f64_f32(vget_low_f32(quantized_f32));
        const float64x2_t quantized_hi =
            vcvt_f64_f32(vget_high_f32(quantized_f32));

        const float64x2_t coor_lo = vsubq_f64(maybe, quantized_lo);
        const float64x2_t coor_hi = vsubq_f64(maybe, quantized_hi);

        const float64x2_t norm_lo = vmulq_f64(coor_lo, coor_lo);
        const float64x2_t norm_hi = vmulq_f64(coor_hi, coor_hi);

        const float64x2_t parallel_lo = vmulq_f64(coor_lo, orig_inv_norm);
        const float64x2_t parallel_hi = vmulq_f64(coor_hi, orig_inv_norm);

        const float64x2x2_t results_lo = {norm_lo, parallel_lo};
        const float64x2x2_t results_hi = {norm_hi, parallel_hi};
        vst2q_f64(&stats[cluster_idx + 0].residual_norm, results_lo);
        vst2q_f64(&stats[cluster_idx + 2].residual_norm, results_hi);
      }
      for (; cluster_idx < num_clusters_per_block; ++cluster_idx) {
        stats[cluster_idx] = ComputeResidualStatsForCluster(
            maybe_residual_dptr_span, original_dptr_span, inverse_chunked_norm,
            cur_subspace_centers[cluster_idx].values_span());
      }
    } else if (cur_dims == 2) {
      const double maybe_val0 =
          static_cast<double>(maybe_residual_dptr_span[0]);
      const double maybe_val1 =
          static_cast<double>(maybe_residual_dptr_span[1]);
      const double original_val0 = static_cast<double>(original_dptr_span[0]);
      const double original_val1 = static_cast<double>(original_dptr_span[1]);
      const float* center_raw = cur_subspace_centers.data().data();
      SubspaceResidualStats* stats = cur_subspace_residual_stats.data();

      const float64x2_t maybe0 = vdupq_n_f64(maybe_val0);
      const float64x2_t maybe1 = vdupq_n_f64(maybe_val1);
      const float64x2_t original0 = vdupq_n_f64(original_val0);
      const float64x2_t original1 = vdupq_n_f64(original_val1);
      const float64x2_t inv_norm = vdupq_n_f64(inverse_chunked_norm);

      int cluster_idx = 0;
      for (; cluster_idx + 4 <= num_clusters_per_block; cluster_idx += 4) {
        const float32x4x2_t quantized_f32 =
            vld2q_f32(center_raw + cluster_idx * 2);
        const float64x2_t quantized0_lo =
            vcvt_f64_f32(vget_low_f32(quantized_f32.val[0]));
        const float64x2_t quantized0_hi =
            vcvt_f64_f32(vget_high_f32(quantized_f32.val[0]));
        const float64x2_t quantized1_lo =
            vcvt_f64_f32(vget_low_f32(quantized_f32.val[1]));
        const float64x2_t quantized_hi =
            vcvt_f64_f32(vget_high_f32(quantized_f32.val[1]));

        const float64x2_t coor0_lo = vsubq_f64(maybe0, quantized0_lo);
        const float64x2_t coor0_hi = vsubq_f64(maybe0, quantized0_hi);
        const float64x2_t coor1_lo = vsubq_f64(maybe1, quantized1_lo);
        const float64x2_t coor1_hi = vsubq_f64(maybe1, quantized_hi);

        float64x2_t norm_lo = vmulq_f64(coor0_lo, coor0_lo);
        float64x2_t norm_hi = vmulq_f64(coor0_hi, coor0_hi);

        norm_lo = vfmaq_f64(norm_lo, coor1_lo, coor1_lo);
        norm_hi = vfmaq_f64(norm_hi, coor1_hi, coor1_hi);

        float64x2_t parallel_lo = vmulq_f64(coor0_lo, original0);
        float64x2_t parallel_hi = vmulq_f64(coor0_hi, original0);

        parallel_lo = vfmaq_f64(parallel_lo, coor1_lo, original1);
        parallel_hi = vfmaq_f64(parallel_hi, coor1_hi, original1);
        parallel_lo = vmulq_f64(parallel_lo, inv_norm);
        parallel_hi = vmulq_f64(parallel_hi, inv_norm);

        const float64x2x2_t results_lo = {norm_lo, parallel_lo};
        const float64x2x2_t results_hi = {norm_hi, parallel_hi};

        vst2q_f64(&stats[cluster_idx + 0].residual_norm, results_lo);
        vst2q_f64(&stats[cluster_idx + 2].residual_norm, results_hi);
      }
      for (; cluster_idx < num_clusters_per_block; ++cluster_idx) {
        stats[cluster_idx] = ComputeResidualStatsForCluster(
            maybe_residual_dptr_span, original_dptr_span, inverse_chunked_norm,
            cur_subspace_centers[cluster_idx].values_span());
      }
    } else {
      for (size_t cluster_idx : Seq(num_clusters_per_block)) {
        ConstSpan<float> center =
            cur_subspace_centers[cluster_idx].values_span();
        cur_subspace_residual_stats[cluster_idx] =
            ComputeResidualStatsForCluster(maybe_residual_dptr_span,
                                           original_dptr_span,
                                           inverse_chunked_norm, center);
      }
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
