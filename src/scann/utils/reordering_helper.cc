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

#include "scann/utils/reordering_helper.h"

#include <stddef.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/base/prefetch.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "scann/brute_force/bfloat16_brute_force.h"
#include "scann/brute_force/brute_force.h"
#include "scann/brute_force/scalar_quantized_brute_force.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/bfloat16_helpers.h"
#include "scann/utils/common.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/scalar_quantization_helpers.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace one_to_many_low_level {

using NeighborResult = std::pair<DatapointIndex, float>;

class SetCosineDistanceFunctor {
 public:
  explicit SetCosineDistanceFunctor(MutableSpan<NeighborResult> result_span)
      : result_(result_span) {}

  template <typename ValueT>
  SCANN_INLINE void invoke(size_t index, ValueT val) {
    SetDistance(result_, index, val + 1);
  }

  SCANN_INLINE void prefetch(size_t index) {}

 private:
  MutableSpan<NeighborResult> result_;
};

class SetCosineTop1Functor {
 public:
  SCANN_INLINE void invoke(size_t index, float val) {
    val += 1;

    if (val > smallest_.load(std::memory_order_relaxed)) return;
    absl::MutexLock lock(&mutex_);
    if (!is_smaller(index, val)) return;
    smallest_.store(val, std::memory_order_relaxed);
    nn_result_index_ = index;
  }

  SCANN_INLINE void prefetch(size_t index) {}

  NeighborResult Top1Pair(const NNResultsVector& nn_result) {
    return std::make_pair(nn_result[nn_result_index_].first,
                          smallest_.load(std::memory_order_relaxed));
  }

 private:
  SCANN_INLINE bool is_smaller(size_t index, float val) {
    float smallest = smallest_.load(std::memory_order_relaxed);
    const bool is_eq_or_nan =
        smallest == val || std::isunordered(smallest, val);
    if (ABSL_PREDICT_FALSE(is_eq_or_nan)) {
      return index < nn_result_index_;
    }
    return val < smallest;
  }

  mutable absl::Mutex mutex_;
  std::atomic<float> smallest_{std::numeric_limits<float>::max()};
  DatapointIndex nn_result_index_ = kInvalidDatapointIndex;
};

class SetSquaredL2DistanceFunctor {
 public:
  explicit SetSquaredL2DistanceFunctor(MutableSpan<NeighborResult> result_span,
                                       ConstSpan<float> dp_squared_l2_norms,
                                       float query_norm)
      : result_(result_span),
        dp_squared_l2_norms_(dp_squared_l2_norms),
        query_norm_(query_norm) {}

  template <typename ValueT>
  SCANN_INLINE void invoke(size_t index, ValueT val) {
    SetDistance(
        result_, index,
        query_norm_ + dp_squared_l2_norms_[result_[index].first] + 2.0f * val);
  }

  SCANN_INLINE void prefetch(size_t index) {
    absl::PrefetchToLocalCache(&dp_squared_l2_norms_[index]);
  }

 private:
  MutableSpan<NeighborResult> result_;
  ConstSpan<float> dp_squared_l2_norms_;
  const float query_norm_;
};

class SetSquaredL2Top1Functor {
 public:
  explicit SetSquaredL2Top1Functor(ConstSpan<NeighborResult> result_span,
                                   ConstSpan<float> dp_squared_l2_norms,
                                   float query_norm)
      : result_(result_span),
        dp_squared_l2_norms_(dp_squared_l2_norms),
        query_norm_(query_norm) {}

  SCANN_INLINE void invoke(size_t index, float val) {
    val = query_norm_ + dp_squared_l2_norms_[result_[index].first] + 2.0 * val;

    if (val > smallest_.load(std::memory_order_relaxed)) return;
    absl::MutexLock lock(&mutex_);
    if (!is_smaller(index, val)) return;
    smallest_.store(val, std::memory_order_relaxed);
    dp_index_ = result_[index].first;
  }

  SCANN_INLINE void prefetch(size_t index) {
    absl::PrefetchToLocalCache(&dp_squared_l2_norms_[index]);
  }

  std::pair<DatapointIndex, float> Top1Pair() {
    return std::make_pair(dp_index_, smallest_.load(std::memory_order_relaxed));
  }

 private:
  SCANN_INLINE bool is_smaller(size_t index, float val) {
    float smallest = smallest_.load(std::memory_order_relaxed);
    const bool is_eq_or_nan =
        smallest == val || std::isunordered(smallest, val);
    if (ABSL_PREDICT_FALSE(is_eq_or_nan)) {
      return index < dp_index_;
    }
    return val < smallest;
  }

  mutable absl::Mutex mutex_;
  std::atomic<float> smallest_{std::numeric_limits<float>::max()};
  DatapointIndex dp_index_ = kInvalidDatapointIndex;
  ConstSpan<NeighborResult> result_;
  ConstSpan<float> dp_squared_l2_norms_;
  const float query_norm_;
};

class SetLimitedInnerDistanceFunctor {
 public:
  explicit SetLimitedInnerDistanceFunctor(
      MutableSpan<NeighborResult> result_span,
      ConstSpan<float> inverse_database_l2_norms, float inverse_query_norm)
      : result_(result_span),
        inverse_database_l2_norms_(inverse_database_l2_norms),
        inverse_query_norm_(inverse_query_norm) {}

  template <typename ValueT>
  SCANN_INLINE void invoke(size_t index, ValueT val) {
    val *= inverse_query_norm_ *
           std::min(inverse_database_l2_norms_[result_[index].first],
                    inverse_query_norm_);
    SetDistance(result_, index, val);
  }

  SCANN_INLINE void prefetch(size_t index) {
    absl::PrefetchToLocalCache(&inverse_database_l2_norms_[index]);
  }

 private:
  MutableSpan<NeighborResult> result_;
  ConstSpan<float> inverse_database_l2_norms_;
  const float inverse_query_norm_;
};

class SetLimitedInnerTop1Functor {
 public:
  explicit SetLimitedInnerTop1Functor(
      ConstSpan<NeighborResult> result_span,
      ConstSpan<float> inverse_database_l2_norms, float inverse_query_norm)
      : result_(result_span),
        inverse_database_l2_norms_(inverse_database_l2_norms),
        inverse_query_norm_(inverse_query_norm) {}

  SCANN_INLINE void invoke(size_t index, float val) {
    val *= inverse_query_norm_ *
           std::min(inverse_database_l2_norms_[result_[index].first],
                    inverse_query_norm_);

    if (val > smallest_.load(std::memory_order_relaxed)) return;
    absl::MutexLock lock(&mutex_);
    if (!is_smaller(index, val)) return;
    smallest_.store(val, std::memory_order_relaxed);
    dp_index_ = result_[index].first;
  }

  SCANN_INLINE void prefetch(size_t index) {
    absl::PrefetchToLocalCache(&inverse_database_l2_norms_[index]);
  }

  std::pair<DatapointIndex, float> Top1Pair() {
    return std::make_pair(dp_index_, smallest_.load(std::memory_order_relaxed));
  }

 private:
  SCANN_INLINE bool is_smaller(size_t index, float val) {
    float smallest = smallest_.load(std::memory_order_relaxed);
    const bool is_eq_or_nan =
        smallest == val || std::isunordered(smallest, val);
    if (ABSL_PREDICT_FALSE(is_eq_or_nan)) {
      return index < dp_index_;
    }
    return val < smallest;
  }

  mutable absl::Mutex mutex_;
  std::atomic<float> smallest_{std::numeric_limits<float>::max()};
  DatapointIndex dp_index_ = kInvalidDatapointIndex;
  ConstSpan<NeighborResult> result_;
  ConstSpan<float> inverse_database_l2_norms_;
  const float inverse_query_norm_;
};

}  // namespace one_to_many_low_level

namespace {

// scann-core: software prefetching of the candidates' rows for reordering.
// Reordering reads one row per candidate at a random place in the dataset,
// so it mostly waits on memory. The one-to-many kernels score the
// candidates in groups of three, one from each third of the result list
// ((i, i + m, i + 2m) with m = n / 3; the n % 3 left over last or, in the
// int8 and bfloat16 kernels, first). Their own prefetches reach only the
// next group or two (to L1), and the float32 kernels' none at all past 512
// dimensions. CandidateRowPrefetcher primes the first `ahead` rows of each
// third and, each time a candidate has been scored, prefetches every cache
// line of the row `ahead` positions further on in its third, into L2
// (prefetcht1 on x86: on Zen 4, prefetches to L1 were slower, as they take
// the L1 miss buffers the kernel's own loads need).
//
// The distance keeps a fixed number of bytes in flight, tuned on Zen 4:
//  - float32 kernels: about 9 KiB (8 rows ahead per third for 100
//    dimensions, 3 for 256; fewer and more were slower). Rows over 2 KiB
//    (above 512 dimensions) aren't prefetched: there the kernel streams at
//    the core's memory bandwidth, and every distance tried was as fast or
//    slower;
//  - int8 and bfloat16 kernels, which prefetch the next rows themselves:
//    about 6 KiB (10 rows ahead for 100-dimensional bfloat16, 2 for 512;
//    3 for 512 gained nothing). Rows over 1 KiB (bfloat16 above 512
//    dimensions, int8 above 1024) aren't prefetched: at 768 bfloat16
//    dimensions no distance was faster.
//
// Prefetching doesn't change what the kernels compute: the results are
// bit-identical, and a wrong guess of the kernel's order costs only time.
class CandidateRowPrefetcher {
 public:
  // How far ahead to prefetch, for one kind of kernel.
  struct Tuning {
    size_t bytes_in_flight;
    size_t max_row_bytes;
  };
  static constexpr Tuning kFloatKernel = {9 * 1024, 2048};
  static constexpr Tuning kQuantizedKernel = {6 * 1024, 1024};

  CandidateRowPrefetcher(const void* rows, size_t row_bytes,
                         ConstSpan<pair<DatapointIndex, float>> result,
                         Tuning tuning)
      : rows_(static_cast<const char*>(rows)),
        row_bytes_(row_bytes),
        result_(result.data()),
        size_(result.size()),
        third_(result.size() / 3),
        ahead_(RowsAhead(row_bytes, tuning)) {}

  bool enabled() const { return ahead_ != 0 && size_ != 0; }

  void Prime() const {
    const size_t per_third = std::min(ahead_, third_);
    for (size_t start : {size_t{0}, third_, 2 * third_}) {
      for (size_t pos = start; pos < start + per_third; ++pos) Row(pos);
    }
    for (size_t pos = 3 * third_; pos < size_; ++pos) Row(pos);
  }

  SCANN_INLINE void AfterScored(size_t pos) const {
    const size_t next = pos + ahead_;
    const size_t limit =
        pos < third_ ? third_ : (pos < 2 * third_ ? 2 * third_ : size_);
    if (next < limit) Row(next);
  }

 private:
  static size_t RowsAhead(size_t row_bytes, Tuning tuning) {
    if (row_bytes == 0 || row_bytes > tuning.max_row_bytes) return 0;
    const size_t group_bytes = 3 * row_bytes;
    return std::clamp<size_t>(
        (tuning.bytes_in_flight + group_bytes / 2) / group_bytes, 1, 16);
  }

  SCANN_INLINE void Row(size_t pos) const {
    const char* row = rows_ + size_t{result_[pos].first} * row_bytes_;
    const uintptr_t end = reinterpret_cast<uintptr_t>(row) + row_bytes_;
    for (uintptr_t line = reinterpret_cast<uintptr_t>(row) & ~uintptr_t{63};
         line < end; line += 64) {
#if defined(__GNUC__) || defined(__clang__)
      __builtin_prefetch(reinterpret_cast<const void*>(line), 0, 2);
#else
      absl::PrefetchToLocalCache(reinterpret_cast<const void*>(line));
#endif
    }
  }

  const char* rows_;
  size_t row_bytes_;
  const pair<DatapointIndex, float>* result_;
  size_t size_;
  size_t third_;
  size_t ahead_;
};

template <typename Inner>
class PrefetchingCallback {
 public:
  PrefetchingCallback(Inner* inner, const CandidateRowPrefetcher& prefetcher)
      : inner_(inner), prefetcher_(prefetcher) {}

  template <typename ValueT>
  SCANN_INLINE void invoke(size_t index, ValueT val) const {
    prefetcher_.AfterScored(index);
    inner_->invoke(index, val);
  }

  SCANN_INLINE void prefetch(size_t index) const { inner_->prefetch(index); }

 private:
  Inner* inner_;
  CandidateRowPrefetcher prefetcher_;
};

template <typename T>
CandidateRowPrefetcher MakeRowPrefetcher(const DefaultDenseDatasetView<T>& view,
                                         const NNResultsVector& result) {
  return CandidateRowPrefetcher(
      view.GetPtr(0), view.dimensionality() * sizeof(T), MakeConstSpan(result),
      std::is_same_v<T, float> ? CandidateRowPrefetcher::kFloatKernel
                               : CandidateRowPrefetcher::kQuantizedKernel);
}

}  // namespace

template <typename T>
Status ExactReorderingHelper<T>::ComputeDistancesForReordering(
    const DatapointPtr<T>& query, NNResultsVector* result) const {
  DCHECK(exact_reordering_dataset_);

  if (query.IsDense() && exact_reordering_dataset_->IsDense()) {
    const auto& dense_dataset =
        *down_cast<const DenseDataset<T>*>(exact_reordering_dataset_.get());
    if constexpr (std::is_same_v<T, float>) {
      const DefaultDenseDatasetView<T> view(dense_dataset);
      const CandidateRowPrefetcher prefetcher =
          MakeRowPrefetcher(view, *result);
      if (prefetcher.enabled()) {
        prefetcher.Prime();
        one_to_many_low_level::SetDistanceFunctor<pair<DatapointIndex, float>>
            set_distance(MakeMutableSpan(*result));
        PrefetchingCallback callback(&set_distance, prefetcher);
        DenseDistanceOneToMany(*exact_reordering_distance_, query, &view,
                               MakeMutableSpan(*result), &callback, nullptr);
        return OkStatus();
      }
    }
    DenseDistanceOneToMany<T, pair<DatapointIndex, float>>(
        *exact_reordering_distance_, query, dense_dataset,
        MakeMutableSpan(*result));
  } else if (query.IsSparse() && exact_reordering_dataset_->IsSparse()) {
    const auto& sparse_dataset =
        *down_cast<const SparseDataset<T>*>(exact_reordering_dataset_.get());
    for (auto& elem : *result) {
      elem.second = exact_reordering_distance_->GetDistanceSparse(
          query, sparse_dataset[elem.first]);
    }
  } else {
    for (auto& elem : *result) {
      elem.second = exact_reordering_distance_->GetDistanceHybrid(
          query, (*exact_reordering_dataset_)[elem.first]);
    }
  }

  return OkStatus();
}

template <typename T>
StatusOrPtr<SingleMachineSearcherBase<T>>
ExactReorderingHelper<T>::CreateBruteForceSearcher(int32_t num_neighbors,
                                                   float epsilon) const {
  return std::make_unique<BruteForceSearcher<T>>(exact_reordering_distance_,
                                                 exact_reordering_dataset_,
                                                 num_neighbors, epsilon);
}

template <typename T>
absl::StatusOr<std::pair<DatapointIndex, float>>
ExactReorderingHelper<T>::ComputeTop1ReorderingDistance(
    const DatapointPtr<T>& query, NNResultsVector* result) const {
  if (query.IsDense() && exact_reordering_dataset_->IsDense()) {
    const auto& dense_dataset =
        *down_cast<const DenseDataset<T>*>(exact_reordering_dataset_.get());
    if constexpr (std::is_same_v<T, float>) {
      const DefaultDenseDatasetView<T> view(dense_dataset);
      const CandidateRowPrefetcher prefetcher =
          MakeRowPrefetcher(view, *result);
      if (prefetcher.enabled()) {
        prefetcher.Prime();
        one_to_many_low_level::SetTop1Functor<pair<DatapointIndex, float>,
                                              float>
            set_top1;
        PrefetchingCallback callback(&set_top1, prefetcher);
        DenseDistanceOneToMany(*exact_reordering_distance_, query, &view,
                               MakeMutableSpan(*result), &callback, nullptr);
        return set_top1.Top1Pair(MakeMutableSpan(*result));
      }
    }
    return DenseDistanceOneToManyTop1<T, float, pair<DatapointIndex, float>>(
        *exact_reordering_distance_, query, dense_dataset,
        MakeMutableSpan(*result));
  }
  if (query.IsSparse() && exact_reordering_dataset_->IsSparse()) {
    const auto& sparse_dataset =
        *down_cast<const SparseDataset<T>*>(exact_reordering_dataset_.get());
    float smallest = std::numeric_limits<float>::max();
    DatapointIndex idx = kInvalidDatapointIndex;
    for (auto& elem : *result) {
      float dist = exact_reordering_distance_->GetDistanceSparse(
          query, sparse_dataset[elem.first]);
      idx = dist < smallest ? elem.first : idx;
      smallest = std::min(smallest, dist);
    }
    return std::make_pair(idx, smallest);
  }
  float smallest = std::numeric_limits<float>::max();
  DatapointIndex idx = kInvalidDatapointIndex;
  for (auto& elem : *result) {
    float dist = exact_reordering_distance_->GetDistanceHybrid(
        query, (*exact_reordering_dataset_)[elem.first]);
    idx = dist < smallest ? elem.first : idx;
    smallest = std::min(smallest, dist);
  }
  return std::make_pair(idx, smallest);
}

class FixedPointFloatDenseDotProductReorderingHelper::Mutator final
    : public ReorderingInterface<float>::Mutator {
 public:
  explicit Mutator(FixedPointFloatDenseDotProductReorderingHelper* helper)
      : helper_(helper),
        multipliers_(helper->inverse_multipliers_->size()),
        dataset_mutator_(helper_->fixed_point_dataset_->GetMutator().value()) {
    ConstSpan<float> inv_multipliers = *helper->inverse_multipliers_;
    for (size_t i : IndicesOf(inv_multipliers)) {
      multipliers_[i] = 1.0 / inv_multipliers[i];
    }
  }

  StatusOr<DatapointIndex> AddDatapoint(const DatapointPtr<float>& dptr) final {
    vector<int8_t> storage(dptr.dimensionality());
    DatapointPtr<int8_t> quantized =
        std::isfinite(helper_->noise_shaping_threshold_)
            ? ScalarQuantizeFloatDatapointWithNoiseShaping(
                  dptr, multipliers_, helper_->noise_shaping_threshold_,
                  &storage)
            : ScalarQuantizeFloatDatapoint(dptr, multipliers_, &storage);
    SCANN_RETURN_IF_ERROR(dataset_mutator_->AddDatapoint(quantized, ""));
    return helper_->fixed_point_dataset_->size() - 1;
  }
  StatusOr<DatapointIndex> RemoveDatapoint(DatapointIndex idx) final {
    SCANN_RETURN_IF_ERROR(dataset_mutator_->RemoveDatapoint(idx));
    return helper_->fixed_point_dataset_->size();
  }
  Status UpdateDatapoint(const DatapointPtr<float>& dptr,
                         DatapointIndex idx) final {
    vector<int8_t> storage(dptr.dimensionality());
    DatapointPtr<int8_t> quantized =
        std::isfinite(helper_->noise_shaping_threshold_)
            ? ScalarQuantizeFloatDatapointWithNoiseShaping(
                  dptr, multipliers_, helper_->noise_shaping_threshold_,
                  &storage)
            : ScalarQuantizeFloatDatapoint(dptr, multipliers_, &storage);
    return dataset_mutator_->UpdateDatapoint(quantized, idx);
  }
  void Reserve(DatapointIndex num_datapoints) final {
    dataset_mutator_->Reserve(num_datapoints);
  }

  Status Reconstruct(DatapointIndex idx, MutableSpan<float> output) const {
    return helper_->Reconstruct(idx, output);
  }

  shared_ptr<const Dataset> dataset() const { return helper_->dataset(); }

 private:
  FixedPointFloatDenseDotProductReorderingHelper* helper_ = nullptr;
  vector<float> multipliers_;
  TypedDataset<int8_t>::Mutator* dataset_mutator_;
};

FixedPointFloatDenseDotProductReorderingHelper::
    FixedPointFloatDenseDotProductReorderingHelper(
        const DenseDataset<float>& exact_reordering_dataset,
        float fixed_point_multiplier_quantile, float noise_shaping_threshold,
        ThreadPool* pool)
    : noise_shaping_threshold_(noise_shaping_threshold) {
  ScalarQuantizationResults quantization_results = ScalarQuantizeFloatDataset(
      exact_reordering_dataset, fixed_point_multiplier_quantile,
      noise_shaping_threshold_, pool);
  fixed_point_dataset_ = std::make_shared<DenseDataset<int8_t>>(
      std::move(quantization_results.quantized_dataset));
  inverse_multipliers_ = make_shared<vector<float>>(
      std::move(quantization_results.inverse_multiplier_by_dimension));
}

FixedPointFloatDenseDotProductReorderingHelper::
    FixedPointFloatDenseDotProductReorderingHelper(
        shared_ptr<DenseDataset<int8_t>> fixed_point_dataset,
        absl::Span<const float> multiplier_by_dimension,
        float noise_shaping_threshold)
    : fixed_point_dataset_(std::move(fixed_point_dataset)),
      noise_shaping_threshold_(noise_shaping_threshold) {
  DCHECK(fixed_point_dataset_ != nullptr);
  DCHECK_EQ(multiplier_by_dimension.size(),
            fixed_point_dataset_->dimensionality());
  vector<float> inverse_multipliers(multiplier_by_dimension.size());
  for (size_t i = 0; i < multiplier_by_dimension.size(); ++i) {
    inverse_multipliers[i] = 1.0f / multiplier_by_dimension[i];
  }
  inverse_multipliers_ =
      make_shared<vector<float>>(std::move(inverse_multipliers));
}

FixedPointFloatDenseDotProductReorderingHelper::
    ~FixedPointFloatDenseDotProductReorderingHelper() = default;

StatusOrPtr<SingleMachineSearcherBase<float>>
FixedPointFloatDenseDotProductReorderingHelper::CreateBruteForceSearcher(
    int32_t num_neighbors, float epsilon) const {
  return ScalarQuantizedBruteForceSearcher::
      CreateFromQuantizedDatasetAndInverseMultipliers(
          std::make_unique<DotProductDistance>(), fixed_point_dataset_,
          inverse_multipliers_, nullptr, num_neighbors, epsilon);
}

Status
FixedPointFloatDenseDotProductReorderingHelper::ComputeDistancesForReordering(
    const DatapointPtr<float>& query, NNResultsVector* result) const {
  one_to_many_low_level::SetDistanceFunctor<pair<DatapointIndex, float>>
      set_distance(MakeMutableSpan(*result));
  return ComputeDistancesForReordering(query, result, &set_distance);
}

template <typename CallbackFunctor>
Status
FixedPointFloatDenseDotProductReorderingHelper::ComputeDistancesForReordering(
    const DatapointPtr<float>& query, NNResultsVector* result,
    CallbackFunctor* __restrict__ callback) const {
  auto preprocessed = PrepareForAsymmetricScalarQuantizedDotProduct(
      query, *inverse_multipliers_);
  const DefaultDenseDatasetView<int8_t> view(*fixed_point_dataset_);
  const CandidateRowPrefetcher prefetcher = MakeRowPrefetcher(view, *result);
  constexpr const float* kNoMultipliersForDotProductDistance = nullptr;
  constexpr const DatapointIndex* kNoIndices = nullptr;
  if (prefetcher.enabled()) {
    prefetcher.Prime();
    one_to_many_low_level::OneToManyInt8FloatDispatch<false, false>(
        preprocessed.get(), view, kNoMultipliersForDotProductDistance,
        kNoIndices, MakeMutableSpan(*result),
        PrefetchingCallback(callback, prefetcher));
  } else {
    one_to_many_low_level::OneToManyInt8FloatDispatch<false, false>(
        preprocessed.get(), view, kNoMultipliersForDotProductDistance,
        kNoIndices, MakeMutableSpan(*result),
        one_to_many_low_level::GetCopyableCallback(callback));
  }
  return OkStatus();
}

absl::StatusOr<std::pair<DatapointIndex, float>>
FixedPointFloatDenseDotProductReorderingHelper::ComputeTop1ReorderingDistance(
    const DatapointPtr<float>& query, NNResultsVector* result) const {
  one_to_many_low_level::SetTop1Functor<std::pair<DatapointIndex, float>, float>
      set_top1_functor;
  SCANN_RETURN_IF_ERROR(
      ComputeDistancesForReordering(query, result, &set_top1_functor));
  return set_top1_functor.Top1Pair(MakeMutableSpan(*result));
}

Status FixedPointFloatDenseDotProductReorderingHelper::Reconstruct(
    DatapointIndex i, MutableSpan<float> output) const {
  if (i >= fixed_point_dataset_->size())
    return InvalidArgumentError(
        "The datapoint index %d is >= the dataset size %d", i,
        fixed_point_dataset_->size());

  const auto* dp_start = (*fixed_point_dataset_)[i].values();
  std::transform(dp_start, dp_start + dimensionality(),
                 inverse_multipliers_->begin(), output.begin(),
                 std::multiplies<float>());
  return OkStatus();
}

StatusOr<ReorderingInterface<float>::Mutator*>
FixedPointFloatDenseDotProductReorderingHelper::GetMutator() const {
  if (!mutator_) {
    mutator_ = std::make_unique<Mutator>(
        const_cast<FixedPointFloatDenseDotProductReorderingHelper*>(this));
  }
  return mutator_.get();
}

class FixedPointFloatDenseCosineReorderingHelper::Mutator final
    : public ReorderingInterface<float>::Mutator {
 public:
  explicit Mutator(FixedPointFloatDenseCosineReorderingHelper* helper)
      : dot_mut_(helper->dot_product_helper_.GetMutator().value()) {}

  StatusOr<DatapointIndex> AddDatapoint(const DatapointPtr<float>& dptr) final {
    auto normalized = Normalize(dptr);
    return dot_mut_->AddDatapoint(normalized.ToPtr());
  }
  StatusOr<DatapointIndex> RemoveDatapoint(DatapointIndex idx) final {
    return dot_mut_->RemoveDatapoint(idx);
  }
  Status UpdateDatapoint(const DatapointPtr<float>& dptr,
                         DatapointIndex idx) final {
    auto normalized = Normalize(dptr);
    return dot_mut_->UpdateDatapoint(normalized.ToPtr(), idx);
  }
  void Reserve(DatapointIndex num_datapoints) final {
    dot_mut_->Reserve(num_datapoints);
  }

 private:
  Datapoint<float> Normalize(const DatapointPtr<float>& dptr) {
    Datapoint<float> result;
    CopyToDatapoint(dptr, &result);
    NormalizeUnitL2(&result);
    return result;
  }

  ReorderingInterface<float>::Mutator* dot_mut_ = nullptr;
};

FixedPointFloatDenseCosineReorderingHelper::
    FixedPointFloatDenseCosineReorderingHelper(
        const DenseDataset<float>& exact_reordering_dataset,
        float fixed_point_multiplier_quantile, float noise_shaping_threshold,
        ThreadPool* pool)
    : dot_product_helper_(exact_reordering_dataset,
                          fixed_point_multiplier_quantile,
                          noise_shaping_threshold, pool) {
  DCHECK_EQ(exact_reordering_dataset.normalization(), UNITL2NORM);
}

FixedPointFloatDenseCosineReorderingHelper::
    FixedPointFloatDenseCosineReorderingHelper(
        shared_ptr<DenseDataset<int8_t>> fixed_point_dataset,
        absl::Span<const float> multiplier_by_dimension,
        float noise_shaping_threshold)
    : dot_product_helper_(std::move(fixed_point_dataset),
                          multiplier_by_dimension, noise_shaping_threshold) {}

FixedPointFloatDenseCosineReorderingHelper::
    ~FixedPointFloatDenseCosineReorderingHelper() = default;

Status
FixedPointFloatDenseCosineReorderingHelper::ComputeDistancesForReordering(
    const DatapointPtr<float>& query, NNResultsVector* result) const {
  one_to_many_low_level::SetCosineDistanceFunctor set_cosine_dist_functor(
      MakeMutableSpan(*result));
  return dot_product_helper_.ComputeDistancesForReordering(
      query, result, &set_cosine_dist_functor);
}

absl::StatusOr<std::pair<DatapointIndex, float>>
FixedPointFloatDenseCosineReorderingHelper::ComputeTop1ReorderingDistance(
    const DatapointPtr<float>& query, NNResultsVector* result) const {
  one_to_many_low_level::SetCosineTop1Functor set_cosine_top1_functor;
  SCANN_RETURN_IF_ERROR(dot_product_helper_.ComputeDistancesForReordering(
      query, result, &set_cosine_top1_functor));
  return set_cosine_top1_functor.Top1Pair(*result);
}

StatusOrPtr<SingleMachineSearcherBase<float>>
FixedPointFloatDenseCosineReorderingHelper::CreateBruteForceSearcher(
    int32_t num_neighbors, float epsilon) const {
  return ScalarQuantizedBruteForceSearcher::
      CreateFromQuantizedDatasetAndInverseMultipliers(
          std::make_unique<CosineDistance>(),
          dot_product_helper_.fixed_point_dataset_,
          dot_product_helper_.inverse_multipliers_, nullptr, num_neighbors,
          epsilon);
}

StatusOr<ReorderingInterface<float>::Mutator*>
FixedPointFloatDenseCosineReorderingHelper::GetMutator() const {
  if (!mutator_) {
    mutator_ = std::make_unique<Mutator>(
        const_cast<FixedPointFloatDenseCosineReorderingHelper*>(this));
  }
  return mutator_.get();
}

FixedPointFloatDenseSquaredL2ReorderingHelper::
    FixedPointFloatDenseSquaredL2ReorderingHelper(
        const DenseDataset<float>& exact_reordering_dataset,
        float fixed_point_multiplier_quantile)
    : dot_product_helper_(exact_reordering_dataset,
                          fixed_point_multiplier_quantile) {
  vector<float> database_squared_l2_norms;
  database_squared_l2_norms.reserve(exact_reordering_dataset.size());
  for (DatapointIndex i = 0; i < exact_reordering_dataset.size(); ++i) {
    database_squared_l2_norms.push_back(
        SquaredL2Norm(exact_reordering_dataset[i]));
  }
  database_squared_l2_norms_ =
      std::make_shared<vector<float>>(std::move(database_squared_l2_norms));
}

FixedPointFloatDenseSquaredL2ReorderingHelper::
    FixedPointFloatDenseSquaredL2ReorderingHelper(
        shared_ptr<DenseDataset<int8_t>> fixed_point_dataset,
        absl::Span<const float> multiplier_by_dimension,
        shared_ptr<const vector<float>> squared_l2_norm_by_datapoint)
    : dot_product_helper_(std::move(fixed_point_dataset),
                          multiplier_by_dimension),
      database_squared_l2_norms_(std::move(squared_l2_norm_by_datapoint)) {
  DCHECK_EQ(database_squared_l2_norms_->size(),
            dot_product_helper_.fixed_point_dataset_->size());
}

Status
FixedPointFloatDenseSquaredL2ReorderingHelper::ComputeDistancesForReordering(
    const DatapointPtr<float>& query, NNResultsVector* result) const {
  const float query_norm = SquaredL2Norm(query);
  one_to_many_low_level::SetSquaredL2DistanceFunctor set_sql2_dist(
      MakeMutableSpan(*result), *database_squared_l2_norms_, query_norm);
  return dot_product_helper_.ComputeDistancesForReordering(query, result,
                                                           &set_sql2_dist);
}

absl::StatusOr<std::pair<DatapointIndex, float>>
FixedPointFloatDenseSquaredL2ReorderingHelper::ComputeTop1ReorderingDistance(
    const DatapointPtr<float>& query, NNResultsVector* result) const {
  const float query_norm = SquaredL2Norm(query);
  one_to_many_low_level::SetSquaredL2Top1Functor set_sql2_top1(
      *result, *database_squared_l2_norms_, query_norm);
  SCANN_RETURN_IF_ERROR(dot_product_helper_.ComputeDistancesForReordering(
      query, result, &set_sql2_top1));
  return set_sql2_top1.Top1Pair();
}

StatusOrPtr<SingleMachineSearcherBase<float>>
FixedPointFloatDenseSquaredL2ReorderingHelper::CreateBruteForceSearcher(
    int32_t num_neighbors, float epsilon) const {
  return ScalarQuantizedBruteForceSearcher::
      CreateFromQuantizedDatasetAndInverseMultipliers(
          std::make_unique<SquaredL2Distance>(),
          dot_product_helper_.fixed_point_dataset_,
          dot_product_helper_.inverse_multipliers_,
          std::const_pointer_cast<vector<float>>(database_squared_l2_norms_),
          num_neighbors, epsilon);
}

FixedPointFloatDenseLimitedInnerReorderingHelper::
    FixedPointFloatDenseLimitedInnerReorderingHelper(
        const DenseDataset<float>& exact_reordering_dataset,
        float fixed_point_multiplier_quantile)
    : dot_product_helper_(exact_reordering_dataset,
                          fixed_point_multiplier_quantile) {
  vector<float> inverse_database_l2_norms;
  inverse_database_l2_norms.reserve(exact_reordering_dataset.size());
  for (DatapointIndex i = 0; i < exact_reordering_dataset.size(); ++i) {
    inverse_database_l2_norms.push_back(
        1.0 / std::sqrt(SquaredL2Norm(exact_reordering_dataset[i])));
  }
  inverse_database_l2_norms_ = std::move(inverse_database_l2_norms);
}

Status
FixedPointFloatDenseLimitedInnerReorderingHelper::ComputeDistancesForReordering(
    const DatapointPtr<float>& query, NNResultsVector* result) const {
  const float inverse_query_norm = 1.0 / std::sqrt(SquaredL2Norm(query));
  one_to_many_low_level::SetLimitedInnerDistanceFunctor set_limited_dist(
      MakeMutableSpan(*result), inverse_database_l2_norms_, inverse_query_norm);
  return dot_product_helper_.ComputeDistancesForReordering(query, result,
                                                           &set_limited_dist);
}

absl::StatusOr<std::pair<DatapointIndex, float>>
FixedPointFloatDenseLimitedInnerReorderingHelper::ComputeTop1ReorderingDistance(
    const DatapointPtr<float>& query, NNResultsVector* result) const {
  const float inverse_query_norm = 1.0 / std::sqrt(SquaredL2Norm(query));
  one_to_many_low_level::SetLimitedInnerTop1Functor top1_functor(
      *result, inverse_database_l2_norms_, inverse_query_norm);
  SCANN_RETURN_IF_ERROR(dot_product_helper_.ComputeDistancesForReordering(
      query, result, &top1_functor));
  return top1_functor.Top1Pair();
}

template <bool kIsDotProduct>
class Bfloat16ReorderingHelper<kIsDotProduct>::Mutator final
    : public ReorderingInterface<float>::Mutator {
 public:
  explicit Mutator(Bfloat16ReorderingHelper<kIsDotProduct>* helper)
      : helper_(helper),
        dataset_mutator_(helper_->bfloat16_dataset_->GetMutator().value()) {}

  StatusOr<DatapointIndex> AddDatapoint(const DatapointPtr<float>& dptr) final {
    vector<int16_t> storage(dptr.dimensionality());
    DatapointPtr<int16_t> quantized =
        std::isfinite(helper_->noise_shaping_threshold_)
            ? Bfloat16QuantizeFloatDatapointWithNoiseShaping(
                  dptr, helper_->noise_shaping_threshold_, &storage)
            : Bfloat16QuantizeFloatDatapoint(dptr, &storage);
    SCANN_RETURN_IF_ERROR(dataset_mutator_->AddDatapoint(quantized, ""));
    return helper_->bfloat16_dataset_->size() - 1;
  }
  StatusOr<DatapointIndex> RemoveDatapoint(DatapointIndex idx) final {
    SCANN_RETURN_IF_ERROR(dataset_mutator_->RemoveDatapoint(idx));
    return helper_->bfloat16_dataset_->size();
  }
  Status UpdateDatapoint(const DatapointPtr<float>& dptr,
                         DatapointIndex idx) final {
    vector<int16_t> storage(dptr.dimensionality());
    DatapointPtr<int16_t> quantized =
        std::isfinite(helper_->noise_shaping_threshold_)
            ? Bfloat16QuantizeFloatDatapointWithNoiseShaping(
                  dptr, helper_->noise_shaping_threshold_, &storage)
            : Bfloat16QuantizeFloatDatapoint(dptr, &storage);
    return dataset_mutator_->UpdateDatapoint(quantized, idx);
  }
  void Reserve(DatapointIndex num_datapoints) final {
    dataset_mutator_->Reserve(num_datapoints);
  }

 private:
  Bfloat16ReorderingHelper<kIsDotProduct>* helper_ = nullptr;
  TypedDataset<int16_t>::Mutator* dataset_mutator_;
};

template <bool kIsDotProduct>
Bfloat16ReorderingHelper<kIsDotProduct>::Bfloat16ReorderingHelper(
    const DenseDataset<float>& exact_reordering_dataset,
    float noise_shaping_threshold, ThreadPool* pool)
    : noise_shaping_threshold_(noise_shaping_threshold) {
  if (std::isfinite(noise_shaping_threshold)) {
    bfloat16_dataset_ = std::make_shared<DenseDataset<int16_t>>(
        Bfloat16QuantizeFloatDatasetWithNoiseShaping(
            exact_reordering_dataset, noise_shaping_threshold, pool));
  } else {
    bfloat16_dataset_ = std::make_shared<DenseDataset<int16_t>>(
        Bfloat16QuantizeFloatDataset(exact_reordering_dataset));
  }
}

template <bool kIsDotProduct>
Bfloat16ReorderingHelper<kIsDotProduct>::Bfloat16ReorderingHelper(
    shared_ptr<DenseDataset<int16_t>> bfloat16_dataset,
    float noise_shaping_threshold)
    : bfloat16_dataset_(bfloat16_dataset),
      noise_shaping_threshold_(noise_shaping_threshold) {}

template <bool kIsDotProduct>
Bfloat16ReorderingHelper<kIsDotProduct>::~Bfloat16ReorderingHelper() = default;

template <bool kIsDotProduct>
Status Bfloat16ReorderingHelper<kIsDotProduct>::ComputeDistancesForReordering(
    const DatapointPtr<float>& query, NNResultsVector* result) const {
  const DefaultDenseDatasetView<int16_t> view(*bfloat16_dataset_);
  const CandidateRowPrefetcher prefetcher = MakeRowPrefetcher(view, *result);
  one_to_many_low_level::SetDistanceFunctor<pair<DatapointIndex, float>>
      set_distance(MakeMutableSpan(*result));
  constexpr const DatapointIndex* kNoIndices = nullptr;
  if (prefetcher.enabled()) {
    prefetcher.Prime();
    one_to_many_low_level::OneToManyBf16FloatDispatch<false, !kIsDotProduct>(
        query.values(), view, kNoIndices, MakeMutableSpan(*result),
        PrefetchingCallback(&set_distance, prefetcher));
  } else {
    one_to_many_low_level::OneToManyBf16FloatDispatch<false, !kIsDotProduct>(
        query.values(), view, kNoIndices, MakeMutableSpan(*result),
        set_distance);
  }
  return OkStatus();
}

template <bool kIsDotProduct>
StatusOrPtr<SingleMachineSearcherBase<float>>
Bfloat16ReorderingHelper<kIsDotProduct>::CreateBruteForceSearcher(
    int32_t num_neighbors, float epsilon) const {
  unique_ptr<const DistanceMeasure> distance;
  if constexpr (kIsDotProduct) {
    distance = std::make_unique<DotProductDistance>();
  } else {
    distance = std::make_unique<SquaredL2Distance>();
  }
  return std::make_unique<Bfloat16BruteForceSearcher>(
      std::move(distance), bfloat16_dataset_, num_neighbors, epsilon,
      noise_shaping_threshold_);
}

template <bool kIsDotProduct>
StatusOr<ReorderingInterface<float>::Mutator*>
Bfloat16ReorderingHelper<kIsDotProduct>::GetMutator() const {
  if (!mutator_) {
    mutator_ = std::make_unique<Mutator>(
        const_cast<Bfloat16ReorderingHelper<kIsDotProduct>*>(this));
  }
  return mutator_.get();
}

template <bool kIsDotProduct>
Status Bfloat16ReorderingHelper<kIsDotProduct>::Reconstruct(
    DatapointIndex i, MutableSpan<float> output) const {
  DatapointPtr<int16_t> db_dptr = bfloat16_dataset_->at(i);
  for (int j : Seq(db_dptr.dimensionality()))
    output[j] = Bfloat16Decompress(db_dptr.values()[j]);
  return OkStatus();
}

template <bool kIsDotProduct>
shared_ptr<const Dataset> Bfloat16ReorderingHelper<kIsDotProduct>::dataset()
    const {
  return bfloat16_dataset_;
}

template class Bfloat16ReorderingHelper<true>;
template class Bfloat16ReorderingHelper<false>;

SCANN_INSTANTIATE_TYPED_CLASS(, ExactReorderingHelper);

template <typename T>
StatusOrPtr<SingleMachineSearcherBase<T>>
ReorderingHelper<T>::CreateBruteForceSearcher(int32_t num_neighbors,
                                              float epsilon) const {
  return UnimplementedError(
      "CreateBruteForceSearcher not implemented for reordering helper of type "
      "%s",
      typeid(*this).name());
}

SCANN_INSTANTIATE_TYPED_CLASS(, ReorderingHelper);

}  // namespace research_scann
