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
#include <float.h>

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
using fallback::CoordinateDescentResult;
using fallback::SubspaceResidualStats;
using fallback::ValidateNoiseShapingParams;

namespace {

CoordinateDescentResult OptimizeSingleSubspace(
    ConstSpan<SubspaceResidualStats> cur_subspace_residual_stats,
    uint8_t cur_center_idx, double parallel_residual_component,
    double parallel_cost_multiplier) {
  CoordinateDescentResult result;
  result.new_center_idx = cur_center_idx;
  result.new_parallel_residual_component = parallel_residual_component;

  const double old_subspace_residual_norm =
      cur_subspace_residual_stats[cur_center_idx].residual_norm;
  const double old_subspace_parallel_component =
      cur_subspace_residual_stats[cur_center_idx].parallel_residual_component;

  const float64x2_t parallel_sq_const =
      vdupq_n_f64(-parallel_residual_component * parallel_residual_component);
  const float64x2_t parallel_const = vdupq_n_f64(
      parallel_residual_component - old_subspace_parallel_component);
  const float64x2_t cost_const = vdupq_n_f64(parallel_cost_multiplier - 1.0);
  const float64x2_t residual_const =
      vdupq_n_f64(cur_subspace_residual_stats[cur_center_idx].residual_norm);
  const float64x2_t max_cost = vdupq_n_f64(DBL_MAX);
  const uint64x2_t cur_center_idx_v = vdupq_n_u64(cur_center_idx);

  const size_t num_clusters = cur_subspace_residual_stats.size();
  const SubspaceResidualStats* stats = cur_subspace_residual_stats.data();

  uint64x2_t idx01 = vcombine_u64(vcreate_u64(0), vcreate_u64(1));
  uint64x2_t idx23 = vcombine_u64(vcreate_u64(2), vcreate_u64(3));
  const uint64x2_t idx_step = vdupq_n_u64(4);

  float64x2_t best_cost01 = vdupq_n_f64(result.cost_delta);
  float64x2_t best_cost23 = vdupq_n_f64(result.cost_delta);
  uint64x2_t best_idx01 = vdupq_n_u64(result.new_center_idx);
  uint64x2_t best_idx23 = vdupq_n_u64(result.new_center_idx);
  float64x2_t best_parallel01 =
      vdupq_n_f64(result.new_parallel_residual_component);
  float64x2_t best_parallel23 =
      vdupq_n_f64(result.new_parallel_residual_component);

  size_t new_center_idx = 0;
  for (; new_center_idx + 3 < num_clusters; new_center_idx += 4) {
    float64x2x2_t rs01 = vld2q_f64(&stats[new_center_idx + 0].residual_norm);
    float64x2x2_t rs23 = vld2q_f64(&stats[new_center_idx + 2].residual_norm);

    float64x2_t rs01_residual = vsubq_f64(rs01.val[0], residual_const);
    float64x2_t rs23_residual = vsubq_f64(rs23.val[0], residual_const);

    float64x2_t parallel01 = vaddq_f64(rs01.val[1], parallel_const);
    float64x2_t parallel23 = vaddq_f64(rs23.val[1], parallel_const);

    float64x2_t p01_delta =
        vfmaq_f64(parallel_sq_const, parallel01, parallel01);
    float64x2_t cost01 = vfmaq_f64(rs01_residual, p01_delta, cost_const);
    uint64x2_t invalid01_mask = vcgtzq_f64(p01_delta);

    float64x2_t p23_delta =
        vfmaq_f64(parallel_sq_const, parallel23, parallel23);
    float64x2_t cost23 = vfmaq_f64(rs23_residual, p23_delta, cost_const);
    uint64x2_t invalid23_mask = vcgtzq_f64(p23_delta);

    invalid01_mask =
        vorrq_u64(invalid01_mask, vceqq_u64(idx01, cur_center_idx_v));
    invalid23_mask =
        vorrq_u64(invalid23_mask, vceqq_u64(idx23, cur_center_idx_v));
    cost01 = vbslq_f64(invalid01_mask, max_cost, cost01);
    cost23 = vbslq_f64(invalid23_mask, max_cost, cost23);

    const uint64x2_t better01_mask = vcltq_f64(cost01, best_cost01);
    best_cost01 = vbslq_f64(better01_mask, cost01, best_cost01);
    best_parallel01 = vbslq_f64(better01_mask, parallel01, best_parallel01);
    best_idx01 = vbslq_u64(better01_mask, idx01, best_idx01);

    const uint64x2_t better23_mask = vcltq_f64(cost23, best_cost23);
    best_cost23 = vbslq_f64(better23_mask, cost23, best_cost23);
    best_parallel23 = vbslq_f64(better23_mask, parallel23, best_parallel23);
    best_idx23 = vbslq_u64(better23_mask, idx23, best_idx23);

    idx01 = vaddq_u64(idx01, idx_step);
    idx23 = vaddq_u64(idx23, idx_step);
  }

  const uint64x2_t use23_mask =
      vorrq_u64(vcltq_f64(best_cost23, best_cost01),
                vandq_u64(vceqq_f64(best_cost23, best_cost01),
                          vcgtq_u64(best_idx01, best_idx23)));
  const float64x2_t best_cost = vbslq_f64(use23_mask, best_cost23, best_cost01);
  const uint64x2_t best_idx = vbslq_u64(use23_mask, best_idx23, best_idx01);
  const float64x2_t best_parallel =
      vbslq_f64(use23_mask, best_parallel23, best_parallel01);

  const double best_cost0 = vgetq_lane_f64(best_cost, 0);
  const uint64_t best_idx0 = vgetq_lane_u64(best_idx, 0);
  if (best_cost0 < result.cost_delta ||
      (best_cost0 == result.cost_delta && best_idx0 < result.new_center_idx)) {
    result.new_center_idx = static_cast<uint8_t>(best_idx0);
    result.cost_delta = best_cost0;
    result.new_parallel_residual_component = vgetq_lane_f64(best_parallel, 0);
  }
  const double best_cost1 = vgetq_lane_f64(best_cost, 1);
  const uint64_t best_idx1 = vgetq_lane_u64(best_idx, 1);
  if (best_cost1 < result.cost_delta ||
      (best_cost1 == result.cost_delta && best_idx1 < result.new_center_idx)) {
    result.new_center_idx = static_cast<uint8_t>(best_idx1);
    result.cost_delta = best_cost1;
    result.new_parallel_residual_component = vgetq_lane_f64(best_parallel, 1);
  }

  for (; new_center_idx < num_clusters; ++new_center_idx) {
    if (new_center_idx == cur_center_idx) {
      continue;
    }
    const SubspaceResidualStats& rs = stats[new_center_idx];
    const double new_parallel_residual_component =
        parallel_residual_component - old_subspace_parallel_component +
        rs.parallel_residual_component;
    const double parallel_norm_delta =
        new_parallel_residual_component * new_parallel_residual_component -
        parallel_residual_component * parallel_residual_component;
    if (parallel_norm_delta > 0.0) {
      continue;
    }
    const double residual_norm_delta =
        rs.residual_norm - old_subspace_residual_norm;
    const double perpendicular_norm_delta =
        residual_norm_delta - parallel_norm_delta;
    const double cost_delta = parallel_cost_multiplier * parallel_norm_delta +
                              perpendicular_norm_delta;
    if (cost_delta < result.cost_delta) {
      result.new_center_idx = static_cast<uint8_t>(new_center_idx);
      result.cost_delta = cost_delta;
      result.new_parallel_residual_component = new_parallel_residual_component;
    }
  }

  return result;
}

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

static inline void UpdateMinResidualIdx(float64x2_t cur_val, uint64x2_t cur_idx,
                                        float64x2_t* best_val,
                                        uint64x2_t* best_idx) {
  const uint64x2_t best_mask = vcltq_f64(cur_val, *best_val);
  *best_idx = vbslq_u64(best_mask, cur_idx, *best_idx);
  *best_val = vminq_f64(cur_val, *best_val);
}

StatusOr<vector<std::vector<SubspaceResidualStats>>>
ComputeResidualStatsAndInitialize(DatapointPtr<float> maybe_residual_dptr,
                                  DatapointPtr<float> original_dptr,
                                  ConstSpan<DenseDataset<float>> centers,
                                  const ChunkingProjection<float>& projection,
                                  MutableSpan<uint8_t> result,
                                  vector<double>& subspace_residual_norms) {
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
    double best_norm = DBL_MAX;
    uint8_t best_idx = 0;

    if (cur_dims == 1) {
      const double maybe_val = static_cast<double>(maybe_residual_dptr_span[0]);
      const double original_val = static_cast<double>(original_dptr_span[0]);
      const float* center_raw = cur_subspace_centers.data().data();
      SubspaceResidualStats* stats = cur_subspace_residual_stats.data();
      const float64x2_t maybe = vdupq_n_f64(maybe_val);
      const float64x2_t original = vdupq_n_f64(original_val);
      const float64x2_t inv_norm = vdupq_n_f64(inverse_chunked_norm);
      const float64x2_t orig_inv_norm = vmulq_f64(original, inv_norm);

      float64x2_t best_norm_v0 = vdupq_n_f64(DBL_MAX);
      float64x2_t best_norm_v1 = vdupq_n_f64(DBL_MAX);
      uint64x2_t best_idx_v0 = vdupq_n_u64(0);
      uint64x2_t best_idx_v1 = vdupq_n_u64(0);

      uint64x2_t idx0 = vcombine_u64(vcreate_u64(0), vcreate_u64(1));
      uint64x2_t idx1 = vcombine_u64(vcreate_u64(2), vcreate_u64(3));
      const uint64x2_t idx_step = vdupq_n_u64(4);

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

        UpdateMinResidualIdx(norm_lo, idx0, &best_norm_v0, &best_idx_v0);
        UpdateMinResidualIdx(norm_hi, idx1, &best_norm_v1, &best_idx_v1);

        idx0 = vaddq_u64(idx0, idx_step);
        idx1 = vaddq_u64(idx1, idx_step);
      }

      if (cluster_idx > 0) {
        UpdateMinResidualIdx(best_norm_v1, best_idx_v1, &best_norm_v0,
                             &best_idx_v0);

        const double norm0 = vgetq_lane_f64(best_norm_v0, 0);
        const double norm1 = vgetq_lane_f64(best_norm_v0, 1);

        if (norm0 == norm1) {
          const uint64_t idx0 = vgetq_lane_u64(best_idx_v0, 0);
          const uint64_t idx1 = vgetq_lane_u64(best_idx_v0, 1);

          best_norm = norm0;
          best_idx = idx0 < idx1 ? idx0 : idx1;
        } else {
          const bool is_lane1_best = norm1 < norm0;

          best_norm = is_lane1_best ? norm1 : norm0;
          best_idx = static_cast<uint8_t>(is_lane1_best
                                              ? vgetq_lane_u64(best_idx_v0, 1)
                                              : vgetq_lane_u64(best_idx_v0, 0));
        }
      }

      for (; cluster_idx < num_clusters_per_block; ++cluster_idx) {
        stats[cluster_idx] = ComputeResidualStatsForCluster(
            maybe_residual_dptr_span, original_dptr_span, inverse_chunked_norm,
            cur_subspace_centers[cluster_idx].values_span());
        if (stats[cluster_idx].residual_norm < best_norm) {
          best_norm = stats[cluster_idx].residual_norm;
          best_idx = static_cast<uint8_t>(cluster_idx);
        }
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

      float64x2_t best_norm_v0 = vdupq_n_f64(DBL_MAX);
      float64x2_t best_norm_v1 = vdupq_n_f64(DBL_MAX);
      uint64x2_t best_idx_v0 = vdupq_n_u64(0);
      uint64x2_t best_idx_v1 = vdupq_n_u64(0);

      uint64x2_t idx0 = vcombine_u64(vcreate_u64(0), vcreate_u64(1));
      uint64x2_t idx1 = vcombine_u64(vcreate_u64(2), vcreate_u64(3));
      const uint64x2_t idx_step = vdupq_n_u64(4);

      size_t cluster_idx = 0;
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

        UpdateMinResidualIdx(norm_lo, idx0, &best_norm_v0, &best_idx_v0);
        UpdateMinResidualIdx(norm_hi, idx1, &best_norm_v1, &best_idx_v1);

        idx0 = vaddq_u64(idx0, idx_step);
        idx1 = vaddq_u64(idx1, idx_step);
      }

      if (cluster_idx > 0) {
        UpdateMinResidualIdx(best_norm_v1, best_idx_v1, &best_norm_v0,
                             &best_idx_v0);

        const double norm0 = vgetq_lane_f64(best_norm_v0, 0);
        const double norm1 = vgetq_lane_f64(best_norm_v0, 1);

        if (norm0 == norm1) {
          const uint64_t idx0 = vgetq_lane_u64(best_idx_v0, 0);
          const uint64_t idx1 = vgetq_lane_u64(best_idx_v0, 1);

          best_norm = norm0;
          best_idx = idx0 < idx1 ? idx0 : idx1;
        } else {
          const bool is_lane1_best = norm1 < norm0;

          best_norm = is_lane1_best ? norm1 : norm0;
          best_idx = static_cast<uint8_t>(is_lane1_best
                                              ? vgetq_lane_u64(best_idx_v0, 1)
                                              : vgetq_lane_u64(best_idx_v0, 0));
        }
      }

      for (; cluster_idx < num_clusters_per_block; ++cluster_idx) {
        stats[cluster_idx] = ComputeResidualStatsForCluster(
            maybe_residual_dptr_span, original_dptr_span, inverse_chunked_norm,
            cur_subspace_centers[cluster_idx].values_span());
        if (stats[cluster_idx].residual_norm < best_norm) {
          best_norm = stats[cluster_idx].residual_norm;
          best_idx = static_cast<uint8_t>(cluster_idx);
        }
      }
    } else {
      for (size_t cluster_idx : Seq(num_clusters_per_block)) {
        ConstSpan<float> center =
            cur_subspace_centers[cluster_idx].values_span();
        cur_subspace_residual_stats[cluster_idx] =
            ComputeResidualStatsForCluster(maybe_residual_dptr_span,
                                           original_dptr_span,
                                           inverse_chunked_norm, center);
        const auto& stats = cur_subspace_residual_stats[cluster_idx];
        if (stats.residual_norm < best_norm) {
          best_norm = stats.residual_norm;
          best_idx = static_cast<uint8_t>(cluster_idx);
        }
      }
    }
    result[subspace_idx] = best_idx;
    subspace_residual_norms[subspace_idx] = best_norm;
  }
  return residual_stats;
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
  vector<double> subspace_residual_norms(result.size());
  SCANN_ASSIGN_OR_RETURN(auto residual_stats,
                         ComputeResidualStatsAndInitialize(
                             maybe_residual_dptr, original_dptr, centers,
                             projection, result, subspace_residual_norms));

  const double parallel_cost_multiplier =
      std::isnan(eta) ? ComputeParallelCostMultiplier(
                            threshold, SquaredL2Norm(original_dptr),
                            original_dptr.dimensionality())
                      : eta;
  double parallel_residual_component =
      ComputeParallelResidualComponent(result, residual_stats);

  vector<uint16_t> subspace_idxs(result.size());
  std::iota(subspace_idxs.begin(), subspace_idxs.end(), 0U);
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
