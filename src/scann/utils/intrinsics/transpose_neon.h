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

#ifndef SCANN_UTILS_INTRINSICS_TRANSPOSE_NEON_H_
#define SCANN_UTILS_INTRINSICS_TRANSPOSE_NEON_H_

#ifdef __aarch64__

#include <arm_neon.h>

namespace research_scann {
namespace neon {

static inline void transpose_f32_4x4(const float32x4_t s0, const float32x4_t s1,
                                     const float32x4_t s2, const float32x4_t s3,
                                     float32x4_t* d0, float32x4_t* d1,
                                     float32x4_t* d2, float32x4_t* d3) {
  const float32x4_t a0 = vzip1q_f32(s0, s2);
  const float32x4_t a1 = vzip2q_f32(s0, s2);
  const float32x4_t a2 = vzip1q_f32(s1, s3);
  const float32x4_t a3 = vzip2q_f32(s1, s3);

  *d0 = vzip1q_f32(a0, a2);
  *d1 = vzip2q_f32(a0, a2);
  *d2 = vzip1q_f32(a1, a3);
  *d3 = vzip2q_f32(a1, a3);
}

}  // namespace neon
}  // namespace research_scann

#endif  // __aarch64__

#endif  // SCANN_UTILS_INTRINSICS_TRANSPOSE_NEON_H_
