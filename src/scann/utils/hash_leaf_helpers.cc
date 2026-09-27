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

#include "scann/utils/hash_leaf_helpers.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "scann/distance_measures/distance_measure_factory.h"
#include "scann/hashes/asymmetric_hashing2/searcher.h"
#include "scann/hashes/asymmetric_hashing2/training.h"
#include "scann/hashes/asymmetric_hashing2/training_options.h"
#include "scann/projection/projection_factory.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace internal {
namespace {

template <typename T>
using StatusOrSearcher = StatusOr<unique_ptr<SingleMachineSearcherBase<T>>>;

StatusOr<shared_ptr<const DistanceMeasure>>
CreateOrGetAymmetricHashingQuantizationDistance(
    const AsymmetricHasherConfig& ah_config,
    const GenericSearchParameters& params) {
  if (ah_config.has_quantization_distance()) {
    return GetDistanceMeasure(ah_config.quantization_distance());
  } else {
    return params.pre_reordering_dist;
  }
}

template <typename T, typename Lambda>
shared_ptr<DenseDataset<uint8_t>> IndexDatabase(const TypedDataset<T>& dataset,
                                                Lambda index_datapoint_fn,
                                                shared_ptr<ThreadPool> pool) {
  constexpr size_t kIndexingBlockSize = 128;
  vector<Datapoint<uint8_t>> hashed_vec(dataset.size());
  Status status = OkStatus();
  absl::Mutex status_mutex;
  ParallelFor<kIndexingBlockSize>(
      Seq(dataset.size()), pool.get(), [&](size_t i) {
        Status iter_status = index_datapoint_fn(dataset[i], &hashed_vec[i]);
        if (!iter_status.ok()) {
          absl::MutexLock lock(&status_mutex);
          status = iter_status;
          return;
        }
      });
  if (!status.ok()) {
    LOG(WARNING) << status;
    return nullptr;
  }

  auto mutable_hashed = std::make_shared<DenseDataset<uint8_t>>();

  if (!hashed_vec.empty() &&
      hashed_vec[0].dimensionality() > hashed_vec[0].nonzero_entries()) {
    mutable_hashed->set_packing_strategy(HashedItem::NIBBLE);
    mutable_hashed->set_dimensionality(hashed_vec[0].dimensionality());
  }

  mutable_hashed->Reserve(dataset.size());
  for (size_t i = 0; i < dataset.size(); ++i) {
    mutable_hashed->AppendOrDie(hashed_vec[i].ToPtr(), dataset.GetDocid(i));
    FreeBackingStorage(&hashed_vec[i]);
  }

  return mutable_hashed;
}

}  // namespace

template <typename T>
StatusOr<TrainedAsymmetricHashingResults<T>>
HashLeafHelpers<T>::TrainAsymmetricHashingModel(
    shared_ptr<TypedDataset<T>> dataset, const AsymmetricHasherConfig& config,
    const GenericSearchParameters& params, shared_ptr<ThreadPool> pool) {
  if (params.pre_reordering_dist == nullptr) {
    return InvalidArgumentError(
        "pre_reordering_dist in GenericSearchParameters is not "
        "set.");
  }
  SCANN_ASSIGN_OR_RETURN(
      auto quantization_distance,
      CreateOrGetAymmetricHashingQuantizationDistance(config, params));
  asymmetric_hashing2::TrainingOptions<T> opts(config, quantization_distance,
                                               *dataset);
  // scann-core: upstream skipped Validate() here (the tree-AH residual
  // factory calls it), so an invalid projection config, whose error the
  // TrainingOptions constructor only records, surfaced as a bare
  // "SCANN_RET_CHECK failure" (null projector), and invalid AH settings
  // weren't checked at all.
  SCANN_RETURN_IF_ERROR(opts.Validate());
  SCANN_ASSIGN_OR_RETURN(
      shared_ptr<const asymmetric_hashing2::Model<T>> model,
      asymmetric_hashing2::TrainSingleMachine<T>(*dataset, opts, pool));
  internal::TrainedAsymmetricHashingResults<T> result;
  result.indexer = std::make_shared<asymmetric_hashing2::Indexer<T>>(
      opts.projector(), quantization_distance, model);
  result.queryer = std::make_shared<asymmetric_hashing2::AsymmetricQueryer<T>>(
      opts.projector(), params.pre_reordering_dist, model);
  result.lookup_type = config.lookup_type();
  result.fixed_point_lut_conversion_options =
      config.fixed_point_lut_conversion_options();
  result.noise_shaping_threshold = config.noise_shaping_threshold();
  if (config.has_centers_filename()) {
    return InvalidArgumentError("Centers file not supported.");
  }
  return result;
}

template <typename T>
StatusOrSearcher<T> HashLeafHelpers<T>::AsymmetricHasherFactory(
    shared_ptr<TypedDataset<T>> dataset,
    shared_ptr<DenseDataset<uint8_t>> hashed_dataset,
    const TrainedAsymmetricHashingResults<T>& training_results,
    const GenericSearchParameters& params, shared_ptr<ThreadPool> pool) {
  if (!hashed_dataset) {
    if (std::isnan(training_results.noise_shaping_threshold)) {
      hashed_dataset = IndexDatabase<T>(
          *dataset,
          [&](const DatapointPtr<T>& dptr, Datapoint<uint8_t>* dp) {
            return training_results.indexer->Hash(dptr, dp);
          },
          pool);
    } else {
      hashed_dataset = IndexDatabase<T>(
          *dataset,
          [&](const DatapointPtr<T>& dptr, Datapoint<uint8_t>* dp) {
            return training_results.indexer->HashWithNoiseShaping(
                dptr, dp,
                {.threshold = training_results.noise_shaping_threshold});
          },
          pool);
    }
    if (!hashed_dataset) {
      return UnknownError("Could not index database.");
    }
  }

  // scann-core: the LUT16 kernels read 16 lookup-table entries per block;
  // with fewer clusters they read past the table (heap overflow).
  if (training_results.lookup_type == AsymmetricHasherConfig::INT8_LUT16 &&
      training_results.queryer &&
      training_results.queryer->num_clusters_per_block() !=
          asymmetric_hashing2::kNumClustersPerBlockForLUT16) {
    return InvalidArgumentError(absl::StrCat(
        "lookup_type INT8_LUT16 requires num_clusters_per_block = ",
        asymmetric_hashing2::kNumClustersPerBlockForLUT16, ", not ",
        training_results.queryer->num_clusters_per_block(), "."));
  }
  asymmetric_hashing2::SearcherOptions<T> opts(training_results.queryer,
                                               training_results.indexer);
  opts.set_asymmetric_lookup_type(training_results.lookup_type);
  opts.set_noise_shaping_threshold(training_results.noise_shaping_threshold);
  opts.set_fixed_point_lut_conversion_options(
      training_results.fixed_point_lut_conversion_options);
  return StatusOrSearcher<T>(make_unique<asymmetric_hashing2::Searcher<T>>(
      std::move(dataset), std::move(hashed_dataset), std::move(opts),
      params.pre_reordering_num_neighbors, params.pre_reordering_epsilon));
}

template <typename T>
StatusOr<TrainedAsymmetricHashingResults<T>>
HashLeafHelpers<T>::LoadAsymmetricHashingModel(
    const AsymmetricHasherConfig& config, const GenericSearchParameters& params,
    shared_ptr<ThreadPool> pool, CentersForAllSubspaces* preloaded_codebook) {
  shared_ptr<const asymmetric_hashing2::Model<T>> model;
  if (preloaded_codebook) {
    SCANN_ASSIGN_OR_RETURN(
        model, asymmetric_hashing2::Model<T>::FromProto(*preloaded_codebook,
                                                        config.projection()));
  } else {
    return InvalidArgumentError("Centers files are not supported.");
  }

  return LoadAsymmetricHashingModel(config, params, model);
}

template <typename T>
StatusOr<TrainedAsymmetricHashingResults<T>>
HashLeafHelpers<T>::LoadAsymmetricHashingModel(
    const AsymmetricHasherConfig& config, const GenericSearchParameters& params,
    shared_ptr<const asymmetric_hashing2::Model<T>> model) {
  SCANN_ASSIGN_OR_RETURN(
      shared_ptr<const DistanceMeasure> quantization_distance,
      CreateOrGetAymmetricHashingQuantizationDistance(config, params));

  SCANN_ASSIGN_OR_RETURN(shared_ptr<const ChunkingProjection<T>> projector,
                         model->GetProjection(config.projection()));
  internal::TrainedAsymmetricHashingResults<T> result;
  result.indexer = std::make_shared<asymmetric_hashing2::Indexer<T>>(
      projector, quantization_distance, model);
  result.queryer = std::make_shared<asymmetric_hashing2::AsymmetricQueryer<T>>(
      projector, params.pre_reordering_dist, model);
  result.lookup_type = config.lookup_type();
  result.fixed_point_lut_conversion_options =
      config.fixed_point_lut_conversion_options();
  result.noise_shaping_threshold = config.noise_shaping_threshold();
  return result;
}

SCANN_INSTANTIATE_TYPED_CLASS(, TrainedAsymmetricHashingResults);
SCANN_INSTANTIATE_TYPED_CLASS(, HashLeafHelpers);

}  // namespace internal
}  // namespace research_scann
