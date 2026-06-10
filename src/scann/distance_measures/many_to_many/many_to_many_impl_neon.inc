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

template <bool kIsSquaredL2, bool kIsPretransposedFixed8, typename CallbackT,
          typename FloatT>
template <size_t kNumQueries>
SCANN_SIMD_INLINE Simd<FloatT, kNumQueries, 2> DenseManyToManyTransposed<
    kIsSquaredL2, kIsPretransposedFixed8, CallbackT, FloatT>::
    DoAccumulationTransposedTemplateNeon(const FloatT* transposed_block0,
                                         const FloatT* transposed_block1,
                                         const FloatT** query_ptrs,
                                         const FloatT* query_norms,
                                         size_t dimensionality) {
  static_assert(std::is_same_v<FloatT, float>);
  Simd<FloatT, kNumQueries, 2> accumulators;
  for (size_t j : Seq(kNumQueries)) {
    if constexpr (kIsSquaredL2) {
      auto query_norm = Simd<FloatT>::Broadcast(query_norms[j]);
      auto db_norms0 =
          Simd<FloatT>::Load(transposed_block0 - kElementsPerRegister);
      auto db_norms1 =
          Simd<FloatT>::Load(transposed_block1 - kElementsPerRegister);
      accumulators[j][0] = db_norms0 + query_norm;
      accumulators[j][1] = db_norms1 + query_norm;
    } else {
      accumulators[j] = Zeros();
    }
  }

  for (size_t dim : Seq(dimensionality)) {
    auto transposed_simd0 =
        Simd<FloatT>::Load(transposed_block0 + dim * kElementsPerRegister);
    auto transposed_simd1 =
        Simd<FloatT>::Load(transposed_block1 + dim * kElementsPerRegister);

    for (size_t j : Seq(kNumQueries)) {
      auto query_simd = Simd<FloatT>::Broadcast(query_ptrs[j][dim]);
      FusedMultiplySubtract(query_simd, transposed_simd0,
                            &accumulators[j][0]);
      FusedMultiplySubtract(query_simd, transposed_simd1,
                            &accumulators[j][1]);
    }
  }

  return accumulators;
}

template <bool kIsSquaredL2, bool kIsPretransposedFixed8, typename CallbackT,
          typename FloatT>
template <size_t kNumQueries>
SCANN_SIMD_INLINE void DenseManyToManyTransposed<
    kIsSquaredL2, kIsPretransposedFixed8, CallbackT,
    FloatT>::BottomLevelBatchNeon(BottomLevelBatchArgs args) {
  const size_t dimensionality = args.dimensionality;
  const FloatT* queries = args.queries;
  const FloatT* query_norms = args.query_norms;
  const size_t first_q_idx = args.first_q_idx;
  const size_t first_dp_idx = args.first_dp_idx;
  const size_t num_datapoints = args.num_datapoints;
  const FloatT* transposed_ptr0 =
      args.transposer->GetTransposedPtr0(dimensionality);
  const FloatT* transposed_ptr1 =
      args.transposer->GetTransposedPtr1(dimensionality);
  CallbackT& callback = *args.callback;

  const FloatT* volatile query_ptrs_vol[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    query_ptrs_vol[j] = queries + j * dimensionality;
  }
  const FloatT* query_ptrs[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    query_ptrs[j] = query_ptrs_vol[j];
  }

  auto accumulators = DoAccumulationTransposedTemplateNeon<kNumQueries>(
      transposed_ptr0, transposed_ptr1, query_ptrs, query_norms,
      dimensionality);
  Base::template PassResultsToCallback<kNumQueries>(
      accumulators, callback, first_q_idx, first_dp_idx, num_datapoints);
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

  while (args.first_q_idx + kSmallQueryStride <= q_idx_end) {
    BottomLevelBatchNeon<kSmallQueryStride>(args);
    args.first_q_idx += kSmallQueryStride;
    if constexpr (kIsSquaredL2) {
      args.query_norms += kSmallQueryStride;
    }
    args.queries += kSmallQueryStride * dimensionality;
  }

  const size_t final_batch_size = q_idx_end - args.first_q_idx;
  SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE(kSmallQueryStride, final_batch_size,
                                       BottomLevelBatchNeon, args);
}
