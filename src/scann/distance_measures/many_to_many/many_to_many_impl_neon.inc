template <bool kIsSquaredL2, typename FloatT>
SCANN_SIMD_INLINE void M2MTransposer<kIsSquaredL2, FloatT>::
    TransposeDatabaseBlockNeon(const size_t dimensionality,
                               const FloatT* database, size_t first_dp_idx,
                               size_t n_to_transpose) {
  static_assert(std::is_same_v<FloatT, float>);
  FloatT* __restrict__ transposed_ptr0 = GetTransposedPtr0(dimensionality);
  FloatT* __restrict__ transposed_ptr1 = GetTransposedPtr1(dimensionality);

  if (ABSL_PREDICT_FALSE(n_to_transpose <= kElementsPerRegister)) {
    TransposeDatabaseBlockImplNeon(dimensionality, database, first_dp_idx,
                                   n_to_transpose, transposed_ptr0);
  } else {
    TransposeDatabaseBlockImplNeon(dimensionality, database, first_dp_idx,
                                   kElementsPerRegister, transposed_ptr0);
    TransposeDatabaseBlockImplNeon(
        dimensionality, database, first_dp_idx + kElementsPerRegister,
        n_to_transpose - kElementsPerRegister, transposed_ptr1);
  }
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
  DCHECK_LE(n_to_transpose, kElementsPerRegister);
  size_t j = 0;
  for (; j + 4 <= n_to_transpose; j += 4) {
    const float* database_ptr = database + (first_dp_idx + j) * dimensionality;
    float* __restrict__ dest = transposed_storage + j;

    constexpr size_t kCacheLineElements = 64 / sizeof(float);
    const float* prefetch = database_ptr + 0 * dimensionality;
    const float* prefetch_end = database_ptr + 4 * dimensionality;
    for (; prefetch < prefetch_end; prefetch += kCacheLineElements) {
      absl::PrefetchToLocalCache(prefetch);
    }

    const float* ptr_begin = database_ptr + 1 * dimensionality;
    const float* ptr_end = database_ptr + 2 * dimensionality;
    for (const float* ptr = ptr_begin; ptr != ptr_end;
         ++ptr, dest += kElementsPerRegister) {
      dest[0] = *(ptr - dimensionality);
      dest[1] = ptr[0 * dimensionality];
      dest[2] = ptr[1 * dimensionality];
      dest[3] = ptr[2 * dimensionality];
    }
  }
  for (; j < n_to_transpose; ++j) {
    const float* untransposed0 = database + (first_dp_idx + j) * dimensionality;
    float* dest = transposed_storage + j;
    for (size_t dim_idx = 0; dim_idx < dimensionality;
         ++dim_idx, dest += kElementsPerRegister) {
      dest[0] = untransposed0[dim_idx];
    }
  }
  DCHECK_EQ(j, n_to_transpose);
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

template <bool kIsSquaredL2, typename FloatT>
SCANN_SIMD_INLINE void M2MTransposer<kIsSquaredL2, FloatT>::
    ExpandPretransposedFP8BlockImplNeon(
        ConstSpan<int8_t> block, size_t dimensionality, size_t n_to_transpose,
        const float* __restrict__ inverse_multipliers_or_null,
        float* __restrict__ transposed_storage) {
  DCHECK_EQ(n_to_transpose * dimensionality, block.size());
  if (n_to_transpose == kElementsPerRegister && !inverse_multipliers_or_null) {
#pragma clang loop vectorize_width(4)
    for (size_t i : Seq(kElementsPerRegister * dimensionality)) {
      transposed_storage[i] = static_cast<float>(block[i]);
    }
    return;
  }

  if (n_to_transpose == kElementsPerRegister) {
    const int8_t* __restrict__ src = block.data();
    for (size_t dim_idx : Seq(dimensionality)) {
      const float inv_multiplier = inverse_multipliers_or_null[dim_idx];
      for (size_t dp_idx : Seq(kElementsPerRegister)) {
        transposed_storage[dp_idx] = src[dp_idx] * inv_multiplier;
      }
      transposed_storage += kElementsPerRegister;
      src += kElementsPerRegister;
    }
    return;
  }

  const int8_t* __restrict__ src = block.data();
  for (size_t dim_idx : Seq(dimensionality)) {
    const float inv_multiplier = inverse_multipliers_or_null
                                     ? inverse_multipliers_or_null[dim_idx]
                                     : 1.0f;
    for (size_t dp_idx : Seq(n_to_transpose)) {
      transposed_storage[dp_idx] = src[dp_idx] * inv_multiplier;
    }
    transposed_storage += kElementsPerRegister;
    src += n_to_transpose;
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
