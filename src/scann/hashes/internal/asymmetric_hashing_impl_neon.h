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

#ifndef SCANN_HASHES_INTERNAL_ASYMMETRIC_HASHING_IMPL_NEON_H_
#define SCANN_HASHES_INTERNAL_ASYMMETRIC_HASHING_IMPL_NEON_H_

#include <cstdint>

#include "scann/hashes/internal/asymmetric_hashing_impl.h"

#if defined(__aarch64__)

namespace research_scann {
namespace asymmetric_hashing_internal {
namespace neon {

template <typename T>
Status IndexDatapointNoiseShaped(
    const DatapointPtr<T>& maybe_residual_dptr,
    const DatapointPtr<T>& original_dptr,
    const ChunkingProjection<T>& projection,
    ConstSpan<DenseDataset<FloatingTypeFor<T>>> centers, double threshold,
    double eta, MutableSpan<uint8_t> result);

}  // namespace neon
}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#endif  // defined(__aarch64__)

#endif
