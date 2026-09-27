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

#include "scann/utils/single_machine_retraining.h"

#include <cmath>
#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "scann/base/internal/single_machine_factory_impl.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/base/single_machine_factory_scann.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/proto/hash.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/scann_config_utils.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
StatusOrSearcherUntyped RetrainAndReindexSearcherImpl(
    UntypedSingleMachineSearcherBase* untyped_searcher,
    absl::Mutex* searcher_pointer_mutex, ScannConfig config,
    shared_ptr<ThreadPool> parallelization_pool) {
  if (searcher_pointer_mutex) searcher_pointer_mutex->AssertNotHeld();
  SingleMachineSearcherBase<T>* searcher =
      down_cast<SingleMachineSearcherBase<T>*>(untyped_searcher);

  SCANN_ASSIGN_OR_RETURN(auto dataset, searcher->ReconstructFloatDataset());
  if (!dataset) {
    return FailedPreconditionError(
        "Searchers passed to RetrainAndReindexSearcher must contain the "
        "original, uncompressed dataset, i.e. dataset() must not return null.");
  }
  // scann-core: upstream ran RetrainAndReindexFixup on the live searcher here,
  // replacing its dataset_ and docids_ with the reconstructed dataset before
  // the factory ran (and without holding searcher_pointer_mutex). When the
  // factory then failed (e.g. fewer points than leaves after deletions, or a
  // config with more children than points), the caller kept that searcher,
  // whose cached mutator still pointed into the docid collection the Fixup
  // had freed: the next mutation was a heap-use-after-free. Build the new
  // searcher from the reconstructed dataset directly and leave the old one
  // untouched, so a failed retrain changes nothing. The new searcher gets
  // exactly what upstream gave it on success: the reconstructed dataset's
  // docids, and retraining_requires_dataset_ == false (the Fixup's default,
  // which upstream then copied from the old searcher).
  auto new_dataset = std::dynamic_pointer_cast<TypedDataset<T>>(
      std::const_pointer_cast<DenseDataset<float>>(dataset));
  if (!new_dataset) {
    return UnimplementedError(
        "RetrainAndReindexSearcher only supports float searchers.");
  }

  // scann-core: spherical partitioning requires a dataset tagged unit-norm.
  // Only some searchers' reconstructed datasets carry the tag (e.g. not
  // float brute-force leaves, or bfloat16/int8 reordering), and the others
  // failed here with "Input vectors must be unit L2-norm". Normalize a copy
  // (a no-op for rows that already are unit-norm) and tag it.
  if (IsSphericalPartitioning(config) && dataset->normalization() != UNITL2NORM) {
    auto normalized = std::make_shared<DenseDataset<float>>(dataset->Copy());
    for (size_t i = 0; i < normalized->size(); ++i)
      NormalizeForSphericalPartitioning(normalized->mutable_data(i));
    normalized->set_normalization_tag(UNITL2NORM);
    new_dataset = std::dynamic_pointer_cast<TypedDataset<T>>(
        std::shared_ptr<DenseDataset<float>>(normalized));
    dataset = std::move(normalized);
  }

  StripPreprocessedArtifacts(&config);
  SingleMachineFactoryOptions opts;
  opts.parallelization_pool = std::move(parallelization_pool);
  SCANN_ASSIGN_OR_RETURN(auto result,
                         SingleMachineFactoryScann<T>(config, new_dataset, opts));

  auto lock_mutex = [&searcher_pointer_mutex]() ABSL_NO_THREAD_SAFETY_ANALYSIS {
    if (searcher_pointer_mutex) searcher_pointer_mutex->WriterLock();
  };
  lock_mutex();

  result->docids_ = dataset->docids();
  result->retraining_requires_dataset_ = false;
  return result;
}

bool IsSphericalPartitioning(const ScannConfig& config) {
  return config.has_partitioning() &&
         config.partitioning().partitioning_type() ==
             PartitioningConfig::SPHERICAL;
}

void NormalizeForSphericalPartitioning(MutableSpan<float> v) {
  const double squared_norm = SquaredL2Norm(MakeDatapointPtr(v.data(), v.size()));
  if (squared_norm == 0 || !std::isfinite(squared_norm) ||
      std::abs(squared_norm - 1) <= 1e-5)
    return;
  const double multiplier = 1.0 / sqrt(squared_norm);
  for (float& x : v) x *= multiplier;
}

StatusOrSearcherUntyped RetrainAndReindexSearcher(
    UntypedSingleMachineSearcherBase* searcher,
    absl::Mutex* searcher_pointer_mutex, const ScannConfig& config,
    shared_ptr<ThreadPool> parallelization_pool)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (searcher_pointer_mutex) searcher_pointer_mutex->AssertNotHeld();
  SCANN_RET_CHECK(searcher);
  return SCANN_CALL_FUNCTION_BY_TAG(
      searcher->TypeTag(), RetrainAndReindexSearcherImpl, searcher,
      searcher_pointer_mutex, config, std::move(parallelization_pool));
}

}  // namespace research_scann
