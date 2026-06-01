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

#if defined(__aarch64__)

#include "scann/distance_measures/one_to_one/dot_product_sve.h"

#include <arm_sve.h>

#include <cstddef>
#include <cstdint>

#include "absl/log/check.h"
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/attributes.h"

namespace research_scann {
namespace dp_internal {

SCANN_SVE_OUTLINE double DenseDotProductSve(const DatapointPtr<int8_t>& a,
                                            const DatapointPtr<float>& b) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());

  const int8_t* a_ptr = a.values();
  const float* b_ptr = b.values();
  const size_t len = a.nonzero_entries();
  const size_t steps = svcntw();
  const size_t two_steps = 2 * steps;
  const svbool_t ptrue = svptrue_b32();

  svfloat32_t acc0 = svdup_f32(0.0f);
  svfloat32_t acc1 = svdup_f32(0.0f);

  size_t i = 0;
  for (; i + two_steps <= len; i += two_steps) {
    const svint32_t a0_s32 = svld1sb_s32(ptrue, a_ptr + i);
    const svint32_t a1_s32 = svld1sb_s32(ptrue, a_ptr + i + steps);
    const svfloat32_t b0_f32 = svld1_f32(ptrue, b_ptr + i);
    const svfloat32_t b1_f32 = svld1_f32(ptrue, b_ptr + i + steps);

    const svfloat32_t a0_f32 = svcvt_f32_s32_x(ptrue, a0_s32);
    const svfloat32_t a1_f32 = svcvt_f32_s32_x(ptrue, a1_s32);

    acc0 = svmla_f32_x(ptrue, acc0, a0_f32, b0_f32);
    acc1 = svmla_f32_x(ptrue, acc1, a1_f32, b1_f32);
  }

  if (i + steps <= len) {
    const svint32_t a0_s32 = svld1sb_s32(ptrue, a_ptr + i);
    const svfloat32_t b0_f32 = svld1_f32(ptrue, b_ptr + i);

    const svfloat32_t a0_f32 = svcvt_f32_s32_x(ptrue, a0_s32);

    acc0 = svmla_f32_x(ptrue, acc0, a0_f32, b0_f32);

    i += steps;
  }

  float acc_scalar = svaddv_f32(ptrue, svadd_f32_x(ptrue, acc0, acc1));

  for (; i < len; ++i) {
    acc_scalar += static_cast<float>(a_ptr[i]) * b_ptr[i];
  }

  return static_cast<double>(acc_scalar);
}

}  // namespace dp_internal
}  // namespace research_scann

#endif  // defined(__aarch64__)
