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

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "absl/strings/str_cat.h"
#include "scann/brute_force/bfloat16_brute_force.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/docid_collection.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/bfloat16_helpers.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

StatusOr<unique_ptr<Bfloat16BruteForceSearcher::Mutator>>
Bfloat16BruteForceSearcher::Mutator::Create(
    Bfloat16BruteForceSearcher* searcher) {
  const_cast<DenseDataset<int16_t>*>(searcher->bfloat16_dataset_.get())
      ->ReleaseDocids();

  SCANN_ASSIGN_OR_RETURN(auto quantized_dataset_mutator,
                         searcher->bfloat16_dataset_->GetMutator());
  // scann-core: give a searcher without docids an (empty) docid collection,
  // as ScalarQuantizedBruteForceSearcher::Mutator::Create does. The leaves of
  // a tree with bfloat16 brute force have neither a float dataset nor docids
  // (the tree releases them), so upstream's AddDatapointToBase and
  // RemoveDatapointFromBase had nothing to track the size with and returned
  // kInvalidDatapointIndex, failing the RET_CHECKs in AddDatapoint and
  // RemoveDatapoint below after the bfloat16 dataset was already changed:
  // every add, update or delete on such a tree failed and corrupted it.
  if (!searcher->docids()) {
    SCANN_RETURN_IF_ERROR(
        searcher->set_docids(make_unique<VariableLengthDocidCollection>(
            VariableLengthDocidCollection::CreateWithEmptyDocids(
                searcher->bfloat16_dataset_->size()))));
  }

  return absl::WrapUnique<Bfloat16BruteForceSearcher::Mutator>(
      new Bfloat16BruteForceSearcher::Mutator(searcher,
                                              quantized_dataset_mutator));
}

void Bfloat16BruteForceSearcher::Mutator::Reserve(size_t size) {
  quantized_dataset_mutator_->Reserve(size);
}

absl::StatusOr<Datapoint<float>>
Bfloat16BruteForceSearcher::Mutator::GetDatapoint(DatapointIndex i) const {
  SCANN_ASSIGN_OR_RETURN(Datapoint<int16_t> dp_int16,
                         quantized_dataset_mutator_->GetDatapoint(i));

  Datapoint<float> dp_fp32;
  dp_fp32.mutable_values()->reserve(dp_int16.values().size());
  for (auto value_int16 : dp_int16.values()) {
    dp_fp32.mutable_values()->push_back(Bfloat16Decompress(value_int16));
  }
  return dp_fp32;
}

StatusOr<DatapointIndex> Bfloat16BruteForceSearcher::Mutator::AddDatapoint(
    const DatapointPtr<float>& dptr, string_view docid,
    const MutationOptions& mo) {
  SCANN_RETURN_IF_ERROR(this->ValidateForAdd(dptr, docid, mo));
  const DatapointIndex result = searcher_->bfloat16_dataset_->size();
  vector<int16_t> storage(dptr.dimensionality());
  DatapointPtr<int16_t> quantized =
      std::isfinite(searcher_->noise_shaping_threshold_)
          ? Bfloat16QuantizeFloatDatapointWithNoiseShaping(
                dptr, searcher_->noise_shaping_threshold_, &storage)
          : Bfloat16QuantizeFloatDatapoint(dptr, &storage);
  SCANN_RETURN_IF_ERROR(
      quantized_dataset_mutator_->AddDatapoint(quantized, ""));
  SCANN_ASSIGN_OR_RETURN(
      auto result2, this->AddDatapointToBase(dptr, docid, MutateBaseOptions{}));
  SCANN_RET_CHECK_EQ(result, result2);
  return result;
}

Status Bfloat16BruteForceSearcher::Mutator::RemoveDatapoint(
    DatapointIndex index) {
  SCANN_RETURN_IF_ERROR(this->ValidateForRemove(index));
  SCANN_RETURN_IF_ERROR(quantized_dataset_mutator_->RemoveDatapoint(index));
  SCANN_ASSIGN_OR_RETURN(auto swapped_from,
                         this->RemoveDatapointFromBase(index));
  SCANN_RET_CHECK_EQ(swapped_from, searcher_->bfloat16_dataset_->size());
  OnDatapointIndexRename(swapped_from, index);
  return OkStatus();
}

Status Bfloat16BruteForceSearcher::Mutator::RemoveDatapoint(string_view docid) {
  SCANN_ASSIGN_OR_RETURN(DatapointIndex index,
                         LookupDatapointIndexOrError(docid));
  return RemoveDatapoint(index);
}

StatusOr<DatapointIndex> Bfloat16BruteForceSearcher::Mutator::UpdateDatapoint(
    const DatapointPtr<float>& dptr, string_view docid,
    const MutationOptions& mo) {
  SCANN_ASSIGN_OR_RETURN(DatapointIndex index,
                         LookupDatapointIndexOrError(docid));
  return UpdateDatapoint(dptr, index, mo);
}

StatusOr<DatapointIndex> Bfloat16BruteForceSearcher::Mutator::UpdateDatapoint(
    const DatapointPtr<float>& dptr, DatapointIndex index,
    const MutationOptions& mo) {
  SCANN_RETURN_IF_ERROR(this->ValidateForUpdate(dptr, index, mo));

  const bool mutate_values_vector = true;
  if (mutate_values_vector) {
    vector<int16_t> storage(dptr.dimensionality());
    DatapointPtr<int16_t> quantized =
        Bfloat16QuantizeFloatDatapoint(dptr, &storage);
    SCANN_RETURN_IF_ERROR(
        quantized_dataset_mutator_->UpdateDatapoint(quantized, index));
  }
  SCANN_RETURN_IF_ERROR(
      this->UpdateDatapointInBase(dptr, index, MutateBaseOptions{}));
  return index;
}

StatusOr<DatapointIndex>
Bfloat16BruteForceSearcher::Mutator::LookupDatapointIndexOrError(
    string_view docid) const {
  DatapointIndex index;
  if (!this->LookupDatapointIndex(docid, &index)) {
    return NotFoundError(absl::StrCat("DocId: ", docid, " is not found."));
  }
  return index;
}

}  // namespace research_scann
