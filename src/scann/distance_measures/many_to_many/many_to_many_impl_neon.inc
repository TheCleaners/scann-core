template <bool kIsSquaredL2, typename FloatT>
SCANN_SIMD_INLINE void M2MTransposer<kIsSquaredL2, FloatT>::
    TransposeDatabaseBlockNeon(const size_t dimensionality,
                               const FloatT* database, size_t first_dp_idx,
                               size_t n_to_transpose) {
  static_assert(std::is_same_v<FloatT, float>);
  if (ABSL_PREDICT_FALSE(n_to_transpose != kElementsPerRegister ||
                         dimensionality == 1 || dimensionality > 4)) {
    TransposeDatabaseBlockHighway(dimensionality, database, first_dp_idx,
                                  n_to_transpose);
    return;
  }
  FloatT* __restrict__ transposed_ptr0 = GetTransposedPtr0(dimensionality);
  FloatT* __restrict__ transposed_ptr1 = GetTransposedPtr1(dimensionality);

  TransposeDatabaseBlockImplNeon(dimensionality, database, first_dp_idx,
                                 kElementsPerRegister, transposed_ptr0);
  if constexpr (kIsSquaredL2) {
    AugmentWithL2Norms(transposed_ptr0, transposed_ptr1, dimensionality);
  }
}

template <bool kIsSquaredL2, typename FloatT>
SCANN_SIMD_INLINE void M2MTransposer<kIsSquaredL2, FloatT>::
    TransposeDatabaseBlockImplNeon(const size_t dimensionality,
                                   const float* database, size_t first_dp_idx,
                                   size_t n_to_transpose,
                                   float* __restrict__ transposed_storage) {
  const float* __restrict__ src = database + first_dp_idx * dimensionality;

  if (dimensionality == 4) {
    float32x4_t s0, s1, s2, s3;
    ::research_scann::neon::load_f32_4x4(src, dimensionality, s0, s1, s2, s3);

    float32x4_t d0, d1, d2, d3;
    ::research_scann::neon::transpose_f32_4x4(s0, s1, s2, s3, &d0, &d1, &d2,
                                              &d3);

    ::research_scann::neon::store_f32_4x4(transposed_storage, 4, d0, d1, d2,
                                          d3);

    return;
  }

  for (size_t i = 0; i < dimensionality; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      transposed_storage[j] = src[j * dimensionality];
    }
    transposed_storage += 4;
    src++;
  }
}

template <bool kIsSquaredL2, typename FloatT>
SCANN_SIMD_INLINE void M2MTransposer<kIsSquaredL2, FloatT>::
    ExpandPretransposedFP8BlockNeon(
        const FP8SimdBlockTransposedDatabase& fp8_db, size_t first_dp_idx,
        size_t n_to_transpose) {
  static_assert(std::is_same_v<FloatT, float>);
  DCHECK_EQ(kElementsPerRegister, fp8_db.simd_block_size());
  const size_t dimensionality = fp8_db.dimensionality();
  const size_t first_block_idx = first_dp_idx / kElementsPerRegister;
  FloatT* __restrict__ transposed_ptr0 = GetTransposedPtr0(dimensionality);
  FloatT* __restrict__ transposed_ptr1 = GetTransposedPtr1(dimensionality);
  if (ABSL_PREDICT_FALSE(n_to_transpose <= kElementsPerRegister)) {
    ExpandPretransposedFP8BlockImplNeon(
        fp8_db.GetBlock(first_block_idx), dimensionality, n_to_transpose,
        fp8_db.inverse_fp8_multipliers().data(), transposed_ptr0);
  } else {
    ExpandPretransposedFP8BlockImplNeon(
        fp8_db.GetBlock(first_block_idx), dimensionality, kElementsPerRegister,
        fp8_db.inverse_fp8_multipliers().data(), transposed_ptr0);
    ExpandPretransposedFP8BlockImplNeon(
        fp8_db.GetBlock(first_block_idx + 1), dimensionality,
        n_to_transpose - kElementsPerRegister,
        fp8_db.inverse_fp8_multipliers().data(), transposed_ptr1);
  }
  if constexpr (kIsSquaredL2) {
    AugmentWithL2Norms(transposed_ptr0, transposed_ptr1, dimensionality);
  }
}

// clang-format off
alignas(16) constexpr uint8_t kInt8ToInt32TblIndices[4][16] = {
    {255, 255, 255, 0,  255, 255, 255, 1,  255, 255, 255, 2,  255, 255, 255, 3 },
    {255, 255, 255, 4,  255, 255, 255, 5,  255, 255, 255, 6,  255, 255, 255, 7 },
    {255, 255, 255, 8,  255, 255, 255, 9,  255, 255, 255, 10, 255, 255, 255, 11},
    {255, 255, 255, 12, 255, 255, 255, 13, 255, 255, 255, 14, 255, 255, 255, 15},
};
// clang-format on

template <bool kIsSquaredL2, typename FloatT>
SCANN_SIMD_INLINE void M2MTransposer<kIsSquaredL2, FloatT>::
    ExpandPretransposedFP8BlockImplNeon(
        ConstSpan<int8_t> block, size_t dimensionality, size_t n_to_transpose,
        const float* __restrict__ inverse_multipliers_or_null,
        float* __restrict__ transposed_storage) {
  static_assert(kElementsPerRegister == 4);
  DCHECK_EQ(n_to_transpose * dimensionality, block.size());
  const int8_t* __restrict__ src = block.data();

  const uint8x16_t index0 = vld1q_u8(kInt8ToInt32TblIndices[0]);
  const uint8x16_t index1 = vld1q_u8(kInt8ToInt32TblIndices[1]);
  const uint8x16_t index2 = vld1q_u8(kInt8ToInt32TblIndices[2]);
  const uint8x16_t index3 = vld1q_u8(kInt8ToInt32TblIndices[3]);

  if (n_to_transpose != kElementsPerRegister) {
    for (size_t i = 0; i < dimensionality; ++i) {
      const float inverse_multiplier =
          inverse_multipliers_or_null ? inverse_multipliers_or_null[i] : 1.0f;
      for (size_t j = 0; j < n_to_transpose; ++j) {
        transposed_storage[j] = src[j] * inverse_multiplier;
      }
      src += n_to_transpose;
      transposed_storage += kElementsPerRegister;
    }
  } else {
    if (inverse_multipliers_or_null == nullptr) {
      size_t i = 0;
      for (; i + 4 <= dimensionality; i += 4) {
        int8x16_t s_s8 = vld1q_s8(src);

        int32x4_t s0_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(s_s8, index0));
        int32x4_t s1_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(s_s8, index1));
        int32x4_t s2_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(s_s8, index2));
        int32x4_t s3_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(s_s8, index3));

        float32x4_t s0_f32 = vcvtq_n_f32_s32(s0_s32, 24);
        float32x4_t s1_f32 = vcvtq_n_f32_s32(s1_s32, 24);
        float32x4_t s2_f32 = vcvtq_n_f32_s32(s2_s32, 24);
        float32x4_t s3_f32 = vcvtq_n_f32_s32(s3_s32, 24);

        ::research_scann::neon::store_f32_4x4(transposed_storage, 4, s0_f32,
                                              s1_f32, s2_f32, s3_f32);

        src += 16;
        transposed_storage += 16;
      }

      if (dimensionality & 2) {
        int8x16_t s_s8 = vcombine_s8(vld1_s8(src), vdup_n_s8(0));

        int32x4_t s0_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(s_s8, index0));
        int32x4_t s1_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(s_s8, index1));

        float32x4_t s0_f32 = vcvtq_n_f32_s32(s0_s32, 24);
        float32x4_t s1_f32 = vcvtq_n_f32_s32(s1_s32, 24);

        vst1q_f32(transposed_storage, s0_f32);
        vst1q_f32(transposed_storage + 4, s1_f32);

        src += 8;
        transposed_storage += 8;
      }

      if (dimensionality & 1) {
        transposed_storage[0] = static_cast<float>(src[0]);
        transposed_storage[1] = static_cast<float>(src[1]);
        transposed_storage[2] = static_cast<float>(src[2]);
        transposed_storage[3] = static_cast<float>(src[3]);
      }
    } else {
      const float* inverse_multipliers = inverse_multipliers_or_null;
      size_t i = 0;
      for (; i + 4 <= dimensionality; i += 4) {
        int8x16_t s_s8 = vld1q_s8(src);
        float32x4_t inv_multipliers = vld1q_f32(inverse_multipliers);

        int32x4_t s0_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(s_s8, index0));
        int32x4_t s1_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(s_s8, index1));
        int32x4_t s2_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(s_s8, index2));
        int32x4_t s3_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(s_s8, index3));

        float32x4_t s0_f32 = vcvtq_n_f32_s32(s0_s32, 24);
        float32x4_t s1_f32 = vcvtq_n_f32_s32(s1_s32, 24);
        float32x4_t s2_f32 = vcvtq_n_f32_s32(s2_s32, 24);
        float32x4_t s3_f32 = vcvtq_n_f32_s32(s3_s32, 24);

        s0_f32 = vmulq_laneq_f32(s0_f32, inv_multipliers, 0);
        s1_f32 = vmulq_laneq_f32(s1_f32, inv_multipliers, 1);
        s2_f32 = vmulq_laneq_f32(s2_f32, inv_multipliers, 2);
        s3_f32 = vmulq_laneq_f32(s3_f32, inv_multipliers, 3);

        ::research_scann::neon::store_f32_4x4(transposed_storage, 4, s0_f32,
                                              s1_f32, s2_f32, s3_f32);

        src += 16;
        transposed_storage += 16;
        inverse_multipliers += 4;
      }

      if (dimensionality & 2) {
        int8x16_t s_s8 = vcombine_s8(vld1_s8(src), vdup_n_s8(0));
        float32x2_t inv_multipliers = vld1_f32(inverse_multipliers);

        int32x4_t s0_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(s_s8, index0));
        int32x4_t s1_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(s_s8, index1));

        float32x4_t s0_f32 = vcvtq_n_f32_s32(s0_s32, 24);
        float32x4_t s1_f32 = vcvtq_n_f32_s32(s1_s32, 24);

        s0_f32 = vmulq_lane_f32(s0_f32, inv_multipliers, 0);
        s1_f32 = vmulq_lane_f32(s1_f32, inv_multipliers, 1);

        vst1q_f32(transposed_storage, s0_f32);
        vst1q_f32(transposed_storage + 4, s1_f32);

        src += 8;
        transposed_storage += 8;
        inverse_multipliers += 2;
      }

      if (dimensionality & 1) {
        const float inverse_multiplier = *inverse_multipliers;
        transposed_storage[0] = static_cast<float>(src[0]) * inverse_multiplier;
        transposed_storage[1] = static_cast<float>(src[1]) * inverse_multiplier;
        transposed_storage[2] = static_cast<float>(src[2]) * inverse_multiplier;
        transposed_storage[3] = static_cast<float>(src[3]) * inverse_multiplier;
      }
    }
  }
}

namespace m2m_neon_internal {

SCANN_SIMD_INLINE void InitAccumulatorsWithQueryNorms(
    float32x4_t db_norm, const float* query_norms, float32x4_t& acc0,
    float32x4_t& acc1, float32x4_t& acc2, float32x4_t& acc3, float32x4_t& acc4,
    float32x4_t& acc5, float32x4_t& acc6, float32x4_t& acc7, float32x4_t& acc8,
    float32x4_t& acc9) {
  acc0 = vaddq_f32(db_norm, vdupq_n_f32(query_norms[0]));
  acc1 = vaddq_f32(db_norm, vdupq_n_f32(query_norms[1]));
  acc2 = vaddq_f32(db_norm, vdupq_n_f32(query_norms[2]));
  acc3 = vaddq_f32(db_norm, vdupq_n_f32(query_norms[3]));
  acc4 = vaddq_f32(db_norm, vdupq_n_f32(query_norms[4]));
  acc5 = vaddq_f32(db_norm, vdupq_n_f32(query_norms[5]));
  acc6 = vaddq_f32(db_norm, vdupq_n_f32(query_norms[6]));
  acc7 = vaddq_f32(db_norm, vdupq_n_f32(query_norms[7]));
  acc8 = vaddq_f32(db_norm, vdupq_n_f32(query_norms[8]));
  acc9 = vaddq_f32(db_norm, vdupq_n_f32(query_norms[9]));
}

SCANN_SIMD_INLINE void InitAccumulators(float32x4_t value, float32x4_t& acc0,
                                        float32x4_t& acc1, float32x4_t& acc2,
                                        float32x4_t& acc3, float32x4_t& acc4,
                                        float32x4_t& acc5, float32x4_t& acc6,
                                        float32x4_t& acc7, float32x4_t& acc8,
                                        float32x4_t& acc9) {
  acc0 = value;
  acc1 = value;
  acc2 = value;
  acc3 = value;
  acc4 = value;
  acc5 = value;
  acc6 = value;
  acc7 = value;
  acc8 = value;
  acc9 = value;
}

SCANN_SIMD_INLINE float32x4_t
FusedMultiplySubtract4Lanes(float32x4_t acc, float32x4_t a0, float32x4_t a1,
                            float32x4_t a2, float32x4_t a3, float32x4_t b) {
  acc = vfmsq_laneq_f32(acc, a0, b, 0);
  acc = vfmsq_laneq_f32(acc, a1, b, 1);
  acc = vfmsq_laneq_f32(acc, a2, b, 2);
  acc = vfmsq_laneq_f32(acc, a3, b, 3);
  return acc;
}

SCANN_SIMD_INLINE float32x4_t FusedMultiplySubtract2Lanes(float32x4_t acc,
                                                          float32x4_t a0,
                                                          float32x4_t a1,
                                                          float32x2_t b) {
  acc = vfmsq_lane_f32(acc, a0, b, 0);
  acc = vfmsq_lane_f32(acc, a1, b, 1);
  return acc;
}

SCANN_SIMD_INLINE void InvokeOptimizedNeon(
    EpsilonFilteringCallback<float>& callback, float32x4_t dists0,
    float32x4_t dists1, size_t first_dp_idx, size_t query_idx,
    size_t num_datapoints) {
  const float best_dist =
      callback.epsilons()[query_idx].load(std::memory_order_relaxed);
  if (ABSL_PREDICT_TRUE(num_datapoints == 8)) {
    const float min_dist = vminnmvq_f32(vminnmq_f32(dists0, dists1));
    if (ABSL_PREDICT_TRUE(!(min_dist < best_dist))) {
      return;
    }

    float candidates[8];
    vst1q_f32(candidates + 0, dists0);
    vst1q_f32(candidates + 4, dists1);
    callback.InvokeSlowPath(MakeMutableSpan(candidates), first_dp_idx,
                            query_idx);
    return;
  }

  const uint32x4_t lane_indices = vcombine_u32(
      vcreate_u32(0x0000000100000000ULL), vcreate_u32(0x0000000300000002ULL));
  const uint32x4_t valid_lanes = vcgtq_u32(
      vdupq_n_u32(static_cast<uint32_t>(num_datapoints - 4)), lane_indices);
  const float32x4_t masked_dists1 =
      vbslq_f32(valid_lanes, dists1, vdupq_n_f32(best_dist));
  const float min_dist = vminnmvq_f32(vminnmq_f32(dists0, masked_dists1));
  if (ABSL_PREDICT_TRUE(!(min_dist < best_dist))) {
    return;
  }

  float candidates[8];
  vst1q_f32(candidates + 0, dists0);
  vst1q_f32(candidates + 4, dists1);
  callback.InvokeSlowPath(MakeMutableSpan(candidates, num_datapoints),
                          first_dp_idx, query_idx);
}

SCANN_SIMD_INLINE void InvokeOptimizedNeon(
    EpsilonFilteringCallback<float>& callback, float32x4_t dists,
    size_t first_dp_idx, size_t query_idx, size_t num_datapoints) {
  const float best_dist =
      callback.epsilons()[query_idx].load(std::memory_order_relaxed);
  if (ABSL_PREDICT_TRUE(num_datapoints == 4)) {
    const float min_dist = vminnmvq_f32(dists);
    if (ABSL_PREDICT_TRUE(!(min_dist < best_dist))) {
      return;
    }

    float candidates[4];
    vst1q_f32(candidates, dists);
    callback.InvokeSlowPath(MakeMutableSpan(candidates), first_dp_idx,
                            query_idx);
    return;
  }

  const uint32x4_t lane_indices = vcombine_u32(
      vcreate_u32(0x0000000100000000ULL), vcreate_u32(0x0000000300000002ULL));
  const uint32x4_t valid_lanes = vcgtq_u32(
      vdupq_n_u32(static_cast<uint32_t>(num_datapoints)), lane_indices);
  const float32x4_t masked_dists =
      vbslq_f32(valid_lanes, dists, vdupq_n_f32(best_dist));
  const float min_dist = vminnmvq_f32(masked_dists);
  if (ABSL_PREDICT_TRUE(!(min_dist < best_dist))) {
    return;
  }

  float candidates[4];
  vst1q_f32(candidates, dists);
  callback.InvokeSlowPath(MakeMutableSpan(candidates, num_datapoints),
                          first_dp_idx, query_idx);
}

template <typename CallbackT>
SCANN_SIMD_INLINE void PassResultToCallback8(CallbackT& callback,
                                             float32x4_t acc0, float32x4_t acc1,
                                             size_t first_dp_idx,
                                             size_t query_idx,
                                             size_t num_datapoints) {
  if constexpr (std::is_same_v<CallbackT, EpsilonFilteringCallback<float>>) {
    InvokeOptimizedNeon(callback, acc0, acc1, first_dp_idx, query_idx,
                        num_datapoints);
    return;
  }

  float candidates[8];
  vst1q_f32(candidates + 0, acc0);
  vst1q_f32(candidates + 4, acc1);
  callback(MakeMutableSpan(candidates, num_datapoints), first_dp_idx,
           query_idx);
}

template <typename CallbackT>
SCANN_SIMD_INLINE void PassResultToCallback4(CallbackT& callback,
                                             float32x4_t acc,
                                             size_t first_dp_idx,
                                             size_t query_idx,
                                             size_t num_datapoints) {
  if constexpr (std::is_same_v<CallbackT, EpsilonFilteringCallback<float>>) {
    InvokeOptimizedNeon(callback, acc, first_dp_idx, query_idx, num_datapoints);
    return;
  }

  float candidates[4];
  vst1q_f32(candidates, acc);
  callback(MakeMutableSpan(candidates, num_datapoints), first_dp_idx,
           query_idx);
}

template <size_t kNumQueries, bool kIsSquaredL2, typename Args>
SCANN_SIMD_INLINE void DoAccumulationAndPassResults8(Args args) {
  const size_t dimensionality = args.dimensionality;
  const float* queries = args.queries;
  const float* query_norms = args.query_norms;
  const size_t first_q_idx = args.first_q_idx;
  const size_t first_dp_idx = args.first_dp_idx;
  const size_t num_datapoints = args.num_datapoints;
  const float* transposed_ptr0 =
      args.transposer->GetTransposedPtr0(dimensionality);
  const float* transposed_ptr1 =
      args.transposer->GetTransposedPtr1(dimensionality);
  auto& callback = *args.callback;
  using m2m_neon_internal::FusedMultiplySubtract2Lanes;
  using m2m_neon_internal::FusedMultiplySubtract4Lanes;
  using m2m_neon_internal::InitAccumulators;
  using m2m_neon_internal::InitAccumulatorsWithQueryNorms;
  using m2m_neon_internal::PassResultToCallback8;

  float32x4_t acc0_0, acc1_0;
  float32x4_t acc0_1, acc1_1;
  float32x4_t acc0_2, acc1_2;
  float32x4_t acc0_3, acc1_3;
  float32x4_t acc0_4, acc1_4;
  float32x4_t acc0_5, acc1_5;
  float32x4_t acc0_6, acc1_6;
  float32x4_t acc0_7, acc1_7;
  float32x4_t acc0_8, acc1_8;
  float32x4_t acc0_9, acc1_9;

  if constexpr (kIsSquaredL2) {
    const float32x4_t db_norm0 = vld1q_f32(transposed_ptr0 - 4);
    const float32x4_t db_norm1 = vld1q_f32(transposed_ptr1 - 4);
    InitAccumulatorsWithQueryNorms(db_norm0, query_norms, acc0_0, acc0_1,
                                   acc0_2, acc0_3, acc0_4, acc0_5, acc0_6,
                                   acc0_7, acc0_8, acc0_9);
    InitAccumulatorsWithQueryNorms(db_norm1, query_norms, acc1_0, acc1_1,
                                   acc1_2, acc1_3, acc1_4, acc1_5, acc1_6,
                                   acc1_7, acc1_8, acc1_9);
  } else {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    InitAccumulators(zero, acc0_0, acc0_1, acc0_2, acc0_3, acc0_4, acc0_5,
                     acc0_6, acc0_7, acc0_8, acc0_9);
    InitAccumulators(zero, acc1_0, acc1_1, acc1_2, acc1_3, acc1_4, acc1_5,
                     acc1_6, acc1_7, acc1_8, acc1_9);
  }

  size_t i = 0;
  for (; i + 4 <= dimensionality; i += 4) {
    float32x4_t tr0_0, tr0_1, tr0_2, tr0_3;
    float32x4_t tr1_0, tr1_1, tr1_2, tr1_3;
    ::research_scann::neon::load_f32_4x4(transposed_ptr0, 4, tr0_0, tr0_1,
                                         tr0_2, tr0_3);
    ::research_scann::neon::load_f32_4x4(transposed_ptr1, 4, tr1_0, tr1_1,
                                         tr1_2, tr1_3);

    float32x4_t q0, q1, q2, q3;
    float32x4_t q4, q5, q6, q7;
    ::research_scann::neon::load_f32_4x8(queries, dimensionality, q0, q1,
                                         q2, q3, q4, q5, q6, q7);
    const float32x4_t q8 = vld1q_f32(queries + 8 * dimensionality);
    const float32x4_t q9 = vld1q_f32(queries + 9 * dimensionality);

    acc0_0 =
        FusedMultiplySubtract4Lanes(acc0_0, tr0_0, tr0_1, tr0_2, tr0_3, q0);
    acc1_0 =
        FusedMultiplySubtract4Lanes(acc1_0, tr1_0, tr1_1, tr1_2, tr1_3, q0);
    acc0_1 =
        FusedMultiplySubtract4Lanes(acc0_1, tr0_0, tr0_1, tr0_2, tr0_3, q1);
    acc1_1 =
        FusedMultiplySubtract4Lanes(acc1_1, tr1_0, tr1_1, tr1_2, tr1_3, q1);
    acc0_2 =
        FusedMultiplySubtract4Lanes(acc0_2, tr0_0, tr0_1, tr0_2, tr0_3, q2);
    acc1_2 =
        FusedMultiplySubtract4Lanes(acc1_2, tr1_0, tr1_1, tr1_2, tr1_3, q2);
    acc0_3 =
        FusedMultiplySubtract4Lanes(acc0_3, tr0_0, tr0_1, tr0_2, tr0_3, q3);
    acc1_3 =
        FusedMultiplySubtract4Lanes(acc1_3, tr1_0, tr1_1, tr1_2, tr1_3, q3);
    acc0_4 =
        FusedMultiplySubtract4Lanes(acc0_4, tr0_0, tr0_1, tr0_2, tr0_3, q4);
    acc1_4 =
        FusedMultiplySubtract4Lanes(acc1_4, tr1_0, tr1_1, tr1_2, tr1_3, q4);
    acc0_5 =
        FusedMultiplySubtract4Lanes(acc0_5, tr0_0, tr0_1, tr0_2, tr0_3, q5);
    acc1_5 =
        FusedMultiplySubtract4Lanes(acc1_5, tr1_0, tr1_1, tr1_2, tr1_3, q5);
    acc0_6 =
        FusedMultiplySubtract4Lanes(acc0_6, tr0_0, tr0_1, tr0_2, tr0_3, q6);
    acc1_6 =
        FusedMultiplySubtract4Lanes(acc1_6, tr1_0, tr1_1, tr1_2, tr1_3, q6);
    acc0_7 =
        FusedMultiplySubtract4Lanes(acc0_7, tr0_0, tr0_1, tr0_2, tr0_3, q7);
    acc1_7 =
        FusedMultiplySubtract4Lanes(acc1_7, tr1_0, tr1_1, tr1_2, tr1_3, q7);
    acc0_8 =
        FusedMultiplySubtract4Lanes(acc0_8, tr0_0, tr0_1, tr0_2, tr0_3, q8);
    acc1_8 =
        FusedMultiplySubtract4Lanes(acc1_8, tr1_0, tr1_1, tr1_2, tr1_3, q8);
    acc0_9 =
        FusedMultiplySubtract4Lanes(acc0_9, tr0_0, tr0_1, tr0_2, tr0_3, q9);
    acc1_9 =
        FusedMultiplySubtract4Lanes(acc1_9, tr1_0, tr1_1, tr1_2, tr1_3, q9);

    queries += 4;
    transposed_ptr0 += 16;
    transposed_ptr1 += 16;
  }

  if (dimensionality & 2) {
    float32x4_t tr0_0, tr0_1;
    float32x4_t tr1_0, tr1_1;
    ::research_scann::neon::load_f32_4x2(transposed_ptr0, 4, tr0_0, tr0_1);
    ::research_scann::neon::load_f32_4x2(transposed_ptr1, 4, tr1_0, tr1_1);

    float32x2_t q0, q1, q2, q3, q4, q5, q6, q7, q8, q9;
    ::research_scann::neon::load_f32_2x10(queries, dimensionality, q0, q1,
                                          q2, q3, q4, q5, q6, q7, q8, q9);

    acc0_0 = FusedMultiplySubtract2Lanes(acc0_0, tr0_0, tr0_1, q0);
    acc1_0 = FusedMultiplySubtract2Lanes(acc1_0, tr1_0, tr1_1, q0);
    acc0_1 = FusedMultiplySubtract2Lanes(acc0_1, tr0_0, tr0_1, q1);
    acc1_1 = FusedMultiplySubtract2Lanes(acc1_1, tr1_0, tr1_1, q1);
    acc0_2 = FusedMultiplySubtract2Lanes(acc0_2, tr0_0, tr0_1, q2);
    acc1_2 = FusedMultiplySubtract2Lanes(acc1_2, tr1_0, tr1_1, q2);
    acc0_3 = FusedMultiplySubtract2Lanes(acc0_3, tr0_0, tr0_1, q3);
    acc1_3 = FusedMultiplySubtract2Lanes(acc1_3, tr1_0, tr1_1, q3);
    acc0_4 = FusedMultiplySubtract2Lanes(acc0_4, tr0_0, tr0_1, q4);
    acc1_4 = FusedMultiplySubtract2Lanes(acc1_4, tr1_0, tr1_1, q4);
    acc0_5 = FusedMultiplySubtract2Lanes(acc0_5, tr0_0, tr0_1, q5);
    acc1_5 = FusedMultiplySubtract2Lanes(acc1_5, tr1_0, tr1_1, q5);
    acc0_6 = FusedMultiplySubtract2Lanes(acc0_6, tr0_0, tr0_1, q6);
    acc1_6 = FusedMultiplySubtract2Lanes(acc1_6, tr1_0, tr1_1, q6);
    acc0_7 = FusedMultiplySubtract2Lanes(acc0_7, tr0_0, tr0_1, q7);
    acc1_7 = FusedMultiplySubtract2Lanes(acc1_7, tr1_0, tr1_1, q7);
    acc0_8 = FusedMultiplySubtract2Lanes(acc0_8, tr0_0, tr0_1, q8);
    acc1_8 = FusedMultiplySubtract2Lanes(acc1_8, tr1_0, tr1_1, q8);
    acc0_9 = FusedMultiplySubtract2Lanes(acc0_9, tr0_0, tr0_1, q9);
    acc1_9 = FusedMultiplySubtract2Lanes(acc1_9, tr1_0, tr1_1, q9);

    queries += 2;
    transposed_ptr0 += 8;
    transposed_ptr1 += 8;
  }

  if (dimensionality & 1) {
    const float32x4_t tr0_0 = vld1q_f32(transposed_ptr0);
    const float32x4_t tr1_0 = vld1q_f32(transposed_ptr1);

    acc0_0 = vfmsq_n_f32(acc0_0, tr0_0, queries[0 * dimensionality]);
    acc1_0 = vfmsq_n_f32(acc1_0, tr1_0, queries[0 * dimensionality]);
    acc0_1 = vfmsq_n_f32(acc0_1, tr0_0, queries[1 * dimensionality]);
    acc1_1 = vfmsq_n_f32(acc1_1, tr1_0, queries[1 * dimensionality]);
    acc0_2 = vfmsq_n_f32(acc0_2, tr0_0, queries[2 * dimensionality]);
    acc1_2 = vfmsq_n_f32(acc1_2, tr1_0, queries[2 * dimensionality]);
    acc0_3 = vfmsq_n_f32(acc0_3, tr0_0, queries[3 * dimensionality]);
    acc1_3 = vfmsq_n_f32(acc1_3, tr1_0, queries[3 * dimensionality]);
    acc0_4 = vfmsq_n_f32(acc0_4, tr0_0, queries[4 * dimensionality]);
    acc1_4 = vfmsq_n_f32(acc1_4, tr1_0, queries[4 * dimensionality]);
    acc0_5 = vfmsq_n_f32(acc0_5, tr0_0, queries[5 * dimensionality]);
    acc1_5 = vfmsq_n_f32(acc1_5, tr1_0, queries[5 * dimensionality]);
    acc0_6 = vfmsq_n_f32(acc0_6, tr0_0, queries[6 * dimensionality]);
    acc1_6 = vfmsq_n_f32(acc1_6, tr1_0, queries[6 * dimensionality]);
    acc0_7 = vfmsq_n_f32(acc0_7, tr0_0, queries[7 * dimensionality]);
    acc1_7 = vfmsq_n_f32(acc1_7, tr1_0, queries[7 * dimensionality]);
    acc0_8 = vfmsq_n_f32(acc0_8, tr0_0, queries[8 * dimensionality]);
    acc1_8 = vfmsq_n_f32(acc1_8, tr1_0, queries[8 * dimensionality]);
    acc0_9 = vfmsq_n_f32(acc0_9, tr0_0, queries[9 * dimensionality]);
    acc1_9 = vfmsq_n_f32(acc1_9, tr1_0, queries[9 * dimensionality]);
  }

  PassResultToCallback8(callback, acc0_0, acc1_0, first_dp_idx, first_q_idx + 0,
                        num_datapoints);
  PassResultToCallback8(callback, acc0_1, acc1_1, first_dp_idx, first_q_idx + 1,
                        num_datapoints);
  PassResultToCallback8(callback, acc0_2, acc1_2, first_dp_idx, first_q_idx + 2,
                        num_datapoints);
  PassResultToCallback8(callback, acc0_3, acc1_3, first_dp_idx, first_q_idx + 3,
                        num_datapoints);
  PassResultToCallback8(callback, acc0_4, acc1_4, first_dp_idx, first_q_idx + 4,
                        num_datapoints);
  PassResultToCallback8(callback, acc0_5, acc1_5, first_dp_idx, first_q_idx + 5,
                        num_datapoints);
  PassResultToCallback8(callback, acc0_6, acc1_6, first_dp_idx, first_q_idx + 6,
                        num_datapoints);
  PassResultToCallback8(callback, acc0_7, acc1_7, first_dp_idx, first_q_idx + 7,
                        num_datapoints);
  PassResultToCallback8(callback, acc0_8, acc1_8, first_dp_idx, first_q_idx + 8,
                        num_datapoints);
  PassResultToCallback8(callback, acc0_9, acc1_9, first_dp_idx, first_q_idx + 9,
                        num_datapoints);
}

template <size_t kNumQueries, bool kIsSquaredL2, typename Args>
SCANN_SIMD_INLINE void DoAccumulationAndPassResults4(Args args) {
  const size_t dimensionality = args.dimensionality;
  const float* queries = args.queries;
  const float* query_norms = args.query_norms;
  const size_t first_q_idx = args.first_q_idx;
  const size_t first_dp_idx = args.first_dp_idx;
  const size_t num_datapoints = args.num_datapoints;
  const float* transposed_ptr0 =
      args.transposer->GetTransposedPtr0(dimensionality);
  auto& callback = *args.callback;
  using m2m_neon_internal::FusedMultiplySubtract2Lanes;
  using m2m_neon_internal::FusedMultiplySubtract4Lanes;
  using m2m_neon_internal::InitAccumulators;
  using m2m_neon_internal::InitAccumulatorsWithQueryNorms;
  using m2m_neon_internal::PassResultToCallback4;

  float32x4_t acc0_0;
  float32x4_t acc0_1;
  float32x4_t acc0_2;
  float32x4_t acc0_3;
  float32x4_t acc0_4;
  float32x4_t acc0_5;
  float32x4_t acc0_6;
  float32x4_t acc0_7;
  float32x4_t acc0_8;
  float32x4_t acc0_9;

  if constexpr (kIsSquaredL2) {
    const float32x4_t db_norm0 = vld1q_f32(transposed_ptr0 - 4);
    InitAccumulatorsWithQueryNorms(db_norm0, query_norms, acc0_0, acc0_1,
                                   acc0_2, acc0_3, acc0_4, acc0_5, acc0_6,
                                   acc0_7, acc0_8, acc0_9);
  } else {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    InitAccumulators(zero, acc0_0, acc0_1, acc0_2, acc0_3, acc0_4, acc0_5,
                     acc0_6, acc0_7, acc0_8, acc0_9);
  }

  size_t i = 0;
  for (; i + 4 <= dimensionality; i += 4) {
    float32x4_t tr0_0, tr0_1, tr0_2, tr0_3;
    ::research_scann::neon::load_f32_4x4(transposed_ptr0, 4, tr0_0, tr0_1,
                                         tr0_2, tr0_3);

    float32x4_t q0, q1, q2, q3;
    float32x4_t q4, q5, q6, q7;
    ::research_scann::neon::load_f32_4x8(queries, dimensionality, q0, q1,
                                         q2, q3, q4, q5, q6, q7);
    const float32x4_t q8 = vld1q_f32(queries + 8 * dimensionality);
    const float32x4_t q9 = vld1q_f32(queries + 9 * dimensionality);

    acc0_0 =
        FusedMultiplySubtract4Lanes(acc0_0, tr0_0, tr0_1, tr0_2, tr0_3, q0);
    acc0_1 =
        FusedMultiplySubtract4Lanes(acc0_1, tr0_0, tr0_1, tr0_2, tr0_3, q1);
    acc0_2 =
        FusedMultiplySubtract4Lanes(acc0_2, tr0_0, tr0_1, tr0_2, tr0_3, q2);
    acc0_3 =
        FusedMultiplySubtract4Lanes(acc0_3, tr0_0, tr0_1, tr0_2, tr0_3, q3);
    acc0_4 =
        FusedMultiplySubtract4Lanes(acc0_4, tr0_0, tr0_1, tr0_2, tr0_3, q4);
    acc0_5 =
        FusedMultiplySubtract4Lanes(acc0_5, tr0_0, tr0_1, tr0_2, tr0_3, q5);
    acc0_6 =
        FusedMultiplySubtract4Lanes(acc0_6, tr0_0, tr0_1, tr0_2, tr0_3, q6);
    acc0_7 =
        FusedMultiplySubtract4Lanes(acc0_7, tr0_0, tr0_1, tr0_2, tr0_3, q7);
    acc0_8 =
        FusedMultiplySubtract4Lanes(acc0_8, tr0_0, tr0_1, tr0_2, tr0_3, q8);
    acc0_9 =
        FusedMultiplySubtract4Lanes(acc0_9, tr0_0, tr0_1, tr0_2, tr0_3, q9);

    queries += 4;
    transposed_ptr0 += 16;
  }

  if (dimensionality & 2) {
    float32x4_t tr0_0, tr0_1;
    ::research_scann::neon::load_f32_4x2(transposed_ptr0, 4, tr0_0, tr0_1);

    float32x2_t q0, q1, q2, q3, q4, q5, q6, q7, q8, q9;
    ::research_scann::neon::load_f32_2x10(queries, dimensionality, q0, q1,
                                          q2, q3, q4, q5, q6, q7, q8, q9);

    acc0_0 = FusedMultiplySubtract2Lanes(acc0_0, tr0_0, tr0_1, q0);
    acc0_1 = FusedMultiplySubtract2Lanes(acc0_1, tr0_0, tr0_1, q1);
    acc0_2 = FusedMultiplySubtract2Lanes(acc0_2, tr0_0, tr0_1, q2);
    acc0_3 = FusedMultiplySubtract2Lanes(acc0_3, tr0_0, tr0_1, q3);
    acc0_4 = FusedMultiplySubtract2Lanes(acc0_4, tr0_0, tr0_1, q4);
    acc0_5 = FusedMultiplySubtract2Lanes(acc0_5, tr0_0, tr0_1, q5);
    acc0_6 = FusedMultiplySubtract2Lanes(acc0_6, tr0_0, tr0_1, q6);
    acc0_7 = FusedMultiplySubtract2Lanes(acc0_7, tr0_0, tr0_1, q7);
    acc0_8 = FusedMultiplySubtract2Lanes(acc0_8, tr0_0, tr0_1, q8);
    acc0_9 = FusedMultiplySubtract2Lanes(acc0_9, tr0_0, tr0_1, q9);

    queries += 2;
    transposed_ptr0 += 8;
  }

  if (dimensionality & 1) {
    const float32x4_t tr0_0 = vld1q_f32(transposed_ptr0);

    acc0_0 = vfmsq_n_f32(acc0_0, tr0_0, queries[0 * dimensionality]);
    acc0_1 = vfmsq_n_f32(acc0_1, tr0_0, queries[1 * dimensionality]);
    acc0_2 = vfmsq_n_f32(acc0_2, tr0_0, queries[2 * dimensionality]);
    acc0_3 = vfmsq_n_f32(acc0_3, tr0_0, queries[3 * dimensionality]);
    acc0_4 = vfmsq_n_f32(acc0_4, tr0_0, queries[4 * dimensionality]);
    acc0_5 = vfmsq_n_f32(acc0_5, tr0_0, queries[5 * dimensionality]);
    acc0_6 = vfmsq_n_f32(acc0_6, tr0_0, queries[6 * dimensionality]);
    acc0_7 = vfmsq_n_f32(acc0_7, tr0_0, queries[7 * dimensionality]);
    acc0_8 = vfmsq_n_f32(acc0_8, tr0_0, queries[8 * dimensionality]);
    acc0_9 = vfmsq_n_f32(acc0_9, tr0_0, queries[9 * dimensionality]);
  }

  PassResultToCallback4(callback, acc0_0, first_dp_idx, first_q_idx + 0,
                        num_datapoints);
  PassResultToCallback4(callback, acc0_1, first_dp_idx, first_q_idx + 1,
                        num_datapoints);
  PassResultToCallback4(callback, acc0_2, first_dp_idx, first_q_idx + 2,
                        num_datapoints);
  PassResultToCallback4(callback, acc0_3, first_dp_idx, first_q_idx + 3,
                        num_datapoints);
  PassResultToCallback4(callback, acc0_4, first_dp_idx, first_q_idx + 4,
                        num_datapoints);
  PassResultToCallback4(callback, acc0_5, first_dp_idx, first_q_idx + 5,
                        num_datapoints);
  PassResultToCallback4(callback, acc0_6, first_dp_idx, first_q_idx + 6,
                        num_datapoints);
  PassResultToCallback4(callback, acc0_7, first_dp_idx, first_q_idx + 7,
                        num_datapoints);
  PassResultToCallback4(callback, acc0_8, first_dp_idx, first_q_idx + 8,
                        num_datapoints);
  PassResultToCallback4(callback, acc0_9, first_dp_idx, first_q_idx + 9,
                        num_datapoints);
}

}  // namespace m2m_neon_internal

template <bool kIsSquaredL2, bool kIsPretransposedFixed8, typename CallbackT,
          typename FloatT>
template <size_t kNumQueries>
SCANN_SIMD_INLINE void DenseManyToManyTransposed<
    kIsSquaredL2, kIsPretransposedFixed8, CallbackT,
    FloatT>::BottomLevelBatchNeon(BottomLevelBatchArgs args) {
  static_assert(kNumQueries == 10,
                "m2m_neon_internal::DoAccumulationAndPassResults only supports "
                "kNumQueries == 10.");
  if (args.num_datapoints <= 4) {
    m2m_neon_internal::DoAccumulationAndPassResults4<kNumQueries, kIsSquaredL2>(
        args);
    return;
  } else if (args.num_datapoints <= 8) {
    m2m_neon_internal::DoAccumulationAndPassResults8<kNumQueries, kIsSquaredL2>(
        args);
    return;
  }
  BottomLevelBatchHighway<kNumQueries>(args);
}

template <bool kIsSquaredL2, bool kIsPretransposedFixed8, typename CallbackT,
          typename FloatT>
SCANN_SIMD_OUTLINE void DenseManyToManyTransposed<
    kIsSquaredL2, kIsPretransposedFixed8, CallbackT,
    FloatT>::MidLevelBatchNeon(size_t first_q_idx, size_t num_queries,
                               size_t first_dp_idx, size_t num_datapoints) {
  static_assert(std::is_same_v<FloatT, float>);
  const size_t dimensionality = this->dimensionality_;
  const size_t q_idx_end = first_q_idx + num_queries;

  thread_local size_t allocated_dimensionality = 0;
  thread_local unique_ptr<Transposer> transposer_storage;
  if (allocated_dimensionality < dimensionality) {
    transposer_storage = Transposer::New(dimensionality, kResultsSize);
    allocated_dimensionality = dimensionality;
  }

  Transposer* transposer = transposer_storage.get();
  if constexpr (kIsPretransposedFixed8) {
    transposer->ExpandPretransposedFP8BlockNeon(*this->database_, first_dp_idx,
                                                num_datapoints);
  } else {
    transposer->TransposeDatabaseBlockNeon(dimensionality, this->database_,
                                           first_dp_idx, num_datapoints);
  }

  BottomLevelBatchArgs args;
  args.dimensionality = dimensionality;
  args.queries = this->queries_ + first_q_idx * dimensionality;
  if constexpr (kIsSquaredL2) {
    args.query_norms = this->query_norms_.get() + first_q_idx;
  } else {
    args.query_norms = nullptr;
  }
  args.first_q_idx = first_q_idx;
  args.first_dp_idx = first_dp_idx;
  args.num_datapoints = num_datapoints;
  args.transposer = transposer;
  args.callback = &this->callback_;

  constexpr size_t kNumQueries = 10;
  while (args.first_q_idx + kNumQueries <= q_idx_end) {
    BottomLevelBatchNeon<kNumQueries>(args);
    args.first_q_idx += kNumQueries;
    if constexpr (kIsSquaredL2) {
      args.query_norms += kNumQueries;
    }
    args.queries += kNumQueries * dimensionality;
  }

  const size_t final_batch_size = q_idx_end - args.first_q_idx;
  SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE(kNumQueries, final_batch_size,
                                       BottomLevelBatchHighway, args);
}
