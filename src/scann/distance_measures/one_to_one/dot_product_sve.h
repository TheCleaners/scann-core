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

#ifndef SCANN_DISTANCE_MEASURES_ONE_TO_ONE_DOT_PRODUCT_SVE_H_
#define SCANN_DISTANCE_MEASURES_ONE_TO_ONE_DOT_PRODUCT_SVE_H_

#if defined(__aarch64__)

#include <cstdint>

#include "scann/data_format/datapoint.h"

namespace research_scann {
namespace dp_internal {

double DenseDotProductSve(const DatapointPtr<int8_t>& a,
                          const DatapointPtr<float>& b);

}  // namespace dp_internal
}  // namespace research_scann

#endif  // defined(__aarch64__)

#endif  // SCANN_DISTANCE_MEASURES_ONE_TO_ONE_DOT_PRODUCT_SVE_H_
