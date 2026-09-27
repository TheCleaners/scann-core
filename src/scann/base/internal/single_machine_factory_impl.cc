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

#include "scann/base/internal/single_machine_factory_impl.h"

#include <cstddef>
#include <vector>

#include "absl/strings/str_cat.h"
#include "scann/distance_measures/one_to_one/binary_distance_measure_base.h"
#include "scann/utils/common.h"
#include "scann/utils/fixed_point/pre_quantized_fixed_point.h"

namespace research_scann {
namespace internal {

std::vector<float> InverseMultiplier(PreQuantizedFixedPoint* fixed_point) {
  std::vector<float> inverse_multipliers;
  inverse_multipliers.resize(fixed_point->multiplier_by_dimension->size());

  for (size_t i : Seq(inverse_multipliers.size())) {
    inverse_multipliers[i] = 1.0f / fixed_point->multiplier_by_dimension->at(i);
  }
  return inverse_multipliers;
}

namespace {

Status CheckNotBinaryDistance(const DistanceMeasureConfig& distance_config,
                              absl::string_view field) {
  SCANN_ASSIGN_OR_RETURN(auto distance, GetDistanceMeasure(distance_config));
  if (dynamic_cast<const BinaryDistanceMeasureBase*>(distance.get())) {
    return InvalidArgumentError(absl::StrCat(
        field, ": ", distance->name(),
        " only works on binary (bit-packed uint8) data."));
  }
  return OkStatus();
}

}  // namespace

Status ValidateFactoryConfig(const ScannConfig& config,
                             const GenericSearchParameters& params,
                             TypeTag type_tag) {
  // Binary distances (Hamming, binary dot product, ...) LOG(FATAL) the first
  // time they see any other data type.
  if (type_tag != InputOutputConfig::UINT8) {
    SCANN_RETURN_IF_ERROR(
        CheckNotBinaryDistance(config.distance_measure(), "distance_measure"));
    const auto& er = config.exact_reordering();
    if (er.has_approx_distance_measure())
      SCANN_RETURN_IF_ERROR(CheckNotBinaryDistance(
          er.approx_distance_measure(),
          "exact_reordering.approx_distance_measure"));
    const auto& pc = config.partitioning();
    if (pc.has_partitioning_distance())
      SCANN_RETURN_IF_ERROR(CheckNotBinaryDistance(
          pc.partitioning_distance(), "partitioning.partitioning_distance"));
    if (pc.has_query_tokenization_distance_override())
      SCANN_RETURN_IF_ERROR(CheckNotBinaryDistance(
          pc.query_tokenization_distance_override(),
          "partitioning.query_tokenization_distance_override"));
    if (pc.has_database_tokenization_distance_override())
      SCANN_RETURN_IF_ERROR(CheckNotBinaryDistance(
          pc.database_tokenization_distance_override(),
          "partitioning.database_tokenization_distance_override"));
    const auto& ah = config.hash().asymmetric_hash();
    if (ah.has_quantization_distance())
      SCANN_RETURN_IF_ERROR(CheckNotBinaryDistance(
          ah.quantization_distance(),
          "hash.asymmetric_hash.quantization_distance"));
  }

  // Bfloat16BruteForceSearcher LOG(FATAL)s on other distances.
  if (config.brute_force().bfloat16().enabled()) {
    const auto tag = params.pre_reordering_dist->specially_optimized_distance_tag();
    if (tag != DistanceMeasure::DOT_PRODUCT &&
        tag != DistanceMeasure::SQUARED_L2) {
      return InvalidArgumentError(absl::StrCat(
          "bfloat16 brute force only supports DotProductDistance and "
          "SquaredL2Distance, not ",
          params.pre_reordering_dist->name(), "."));
    }
  }

  // Tree leaves quantized with this multiplier weren't range-checked
  // (undefined float -> integer conversion); NaN passed every check.
  if (config.brute_force().fixed_point().enabled()) {
    const float q = config.brute_force().fixed_point().fixed_point_multiplier_quantile();
    if (!(q > 0.0f && q <= 1.0f)) {
      return InvalidArgumentError(absl::StrCat(
          "brute_force.fixed_point.fixed_point_multiplier_quantile must be in "
          "(0, 1], not ", q, "."));
    }
  }
  return OkStatus();
}

void MaybeSetAhProjectionInputDim(ScannConfig* config, const Dataset* dataset) {
  if (!dataset || dataset->dimensionality() == 0 ||
      !config->hash().has_asymmetric_hash())
    return;
  if (config->has_partitioning() &&
      config->hash().asymmetric_hash().use_residual_quantization())
    return;
  ProjectionConfig* proj =
      config->mutable_hash()->mutable_asymmetric_hash()->mutable_projection();
  if (proj->has_input_dim() || !proj->variable_blocks().empty() ||
      proj->projection_type() == ProjectionConfig::VARIABLE_CHUNK)
    return;
  proj->set_input_dim(dataset->dimensionality());
}

}  // namespace internal
}  // namespace research_scann
