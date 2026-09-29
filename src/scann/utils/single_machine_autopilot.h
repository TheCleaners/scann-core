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

#ifndef SCANN_UTILS_SINGLE_MACHINE_AUTOPILOT_H_
#define SCANN_UTILS_SINGLE_MACHINE_AUTOPILOT_H_

#include <cmath>
#include <memory>

#include "scann/data_format/dataset.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/types.h"

namespace research_scann {

StatusOr<ScannConfig> Autopilot(const ScannConfig& config,
                                shared_ptr<const Dataset> dataset,
                                DatapointIndex n = kInvalidDatapointIndex,
                                DimensionIndex dim = kInvalidDimension);

// scann-core: what the TUNED_V1 autopilot rules measure in the data (a
// deterministic sample of at most kAutopilotStatsSampleSize rows).
inline constexpr size_t kAutopilotStatsSampleSize = 16384;
struct AutopilotDataStats {
  // The 5th percentile of the datapoints' L2 norms.
  double norm_quantile = NAN;
  // |mean|^2 / mean |x - mean|^2: how far the data sits from the origin
  // relative to its spread (infinity if every sampled point is the same).
  double offset_ratio = NAN;
  // The coefficient of variation of |x|^2 (std / mean; NaN if every
  // sampled point is 0).
  double squared_norm_cv = NAN;
};
AutopilotDataStats ComputeAutopilotDataStats(ConstSpan<float> data,
                                             DatapointIndex n,
                                             DimensionIndex dim);

// scann-core: whether the TUNED_V1 rules build this config's index as
// l2_as_dot_product: an autopilot config with SquaredL2Distance whose
// dataset (n rows of `data`) is large enough for a tree, has nearly
// constant norms and isn't far from the origin (see the .cc).
// ScannInterface::Initialize asks before building.
bool AutopilotChoosesL2AsDotProduct(const ScannConfig& config,
                                    ConstSpan<float> data, DatapointIndex n);

// scann-core: if AutopilotChoosesL2AsDotProduct, adds l2_as_dot_product to
// *config, drops the parts autopilot fills in (a config preview's, for a
// plain squared L2 index; the searcher factory's autopilot redoes them), and
// returns true.
bool ApplyAutopilotL2AsDotProduct(ScannConfig* config, ConstSpan<float> data,
                                  DatapointIndex n);

}

#endif
