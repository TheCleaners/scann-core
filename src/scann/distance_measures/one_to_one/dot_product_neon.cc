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

#include "scann/distance_measures/one_to_one/dot_product_neon.h"

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

#include "absl/log/check.h"
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/mem_neon.h"

namespace research_scann {
namespace dp_internal {

// clang-format off
alignas(16) constexpr uint8_t kInt8ToInt32TblIndices[4][16] = {
    {255, 255, 255, 0,  255, 255, 255, 1,  255, 255, 255, 2,  255, 255, 255, 3 },
    {255, 255, 255, 4,  255, 255, 255, 5,  255, 255, 255, 6,  255, 255, 255, 7 },
    {255, 255, 255, 8,  255, 255, 255, 9,  255, 255, 255, 10, 255, 255, 255, 11},
    {255, 255, 255, 12, 255, 255, 255, 13, 255, 255, 255, 14, 255, 255, 255, 15},
};
// clang-format on

SCANN_OUTLINE double DenseDotProductNeon(const DatapointPtr<int8_t>& a,
                                         const DatapointPtr<float>& b) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());

  const int8_t* a_ptr = a.values();
  const int8_t* aend = a_ptr + a.nonzero_entries();
  const float* b_ptr = b.values();

  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);

  // Indices used with the TBL instruction to widen from 8-bit to 32-bit in a
  // single instruction.
  const uint8x16_t index0 = vld1q_u8(kInt8ToInt32TblIndices[0]);
  const uint8x16_t index1 = vld1q_u8(kInt8ToInt32TblIndices[1]);
  const uint8x16_t index2 = vld1q_u8(kInt8ToInt32TblIndices[2]);
  const uint8x16_t index3 = vld1q_u8(kInt8ToInt32TblIndices[3]);

  for (; a_ptr + 16 <= aend; a_ptr += 16, b_ptr += 16) {
    int8x16_t a0_s8 = vld1q_s8(a_ptr);
    float32x4_t b0_f32 = vld1q_f32(b_ptr + 0);
    float32x4_t b1_f32 = vld1q_f32(b_ptr + 4);
    float32x4_t b2_f32 = vld1q_f32(b_ptr + 8);
    float32x4_t b3_f32 = vld1q_f32(b_ptr + 12);

    int32x4_t a0_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(a0_s8, index0));
    int32x4_t a1_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(a0_s8, index1));
    int32x4_t a2_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(a0_s8, index2));
    int32x4_t a3_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(a0_s8, index3));

    float32x4_t a0_f32 = vcvtq_n_f32_s32(a0_s32, 24);
    float32x4_t a1_f32 = vcvtq_n_f32_s32(a1_s32, 24);
    float32x4_t a2_f32 = vcvtq_n_f32_s32(a2_s32, 24);
    float32x4_t a3_f32 = vcvtq_n_f32_s32(a3_s32, 24);

    acc0 = vfmaq_f32(acc0, a0_f32, b0_f32);
    acc1 = vfmaq_f32(acc1, a1_f32, b1_f32);
    acc0 = vfmaq_f32(acc0, a2_f32, b2_f32);
    acc1 = vfmaq_f32(acc1, a3_f32, b3_f32);
  }

  if (a_ptr + 8 <= aend) {
    int8x16_t a0_s8 = vcombine_s8(vld1_s8(a_ptr), vdup_n_s8(0));
    float32x4_t b0_f32 = vld1q_f32(b_ptr + 0);
    float32x4_t b1_f32 = vld1q_f32(b_ptr + 4);

    int32x4_t a0_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(a0_s8, index0));
    int32x4_t a1_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(a0_s8, index1));

    float32x4_t a0_f32 = vcvtq_n_f32_s32(a0_s32, 24);
    float32x4_t a1_f32 = vcvtq_n_f32_s32(a1_s32, 24);

    acc0 = vfmaq_f32(acc0, a0_f32, b0_f32);
    acc1 = vfmaq_f32(acc1, a1_f32, b1_f32);

    a_ptr += 8;
    b_ptr += 8;
  }

  if (a_ptr + 4 <= aend) {
    int8x16_t a0_s8 = vcombine_s8(neon::load_s8_4x1(a_ptr), vdup_n_s8(0));
    float32x4_t b0_f32 = vld1q_f32(b_ptr);

    int32x4_t a0_s32 = vreinterpretq_s32_s8(vqtbl1q_s8(a0_s8, index0));
    float32x4_t a0_f32 = vcvtq_n_f32_s32(a0_s32, 24);

    acc0 = vfmaq_f32(acc0, a0_f32, b0_f32);

    a_ptr += 4;
    b_ptr += 4;
  }

  float acc_scalar = vaddvq_f32(vaddq_f32(acc0, acc1));

  for (; a_ptr < aend; ++a_ptr, ++b_ptr) {
    acc_scalar += static_cast<float>(*a_ptr) * *b_ptr;
  }

  return static_cast<double>(acc_scalar);
}

}  // namespace dp_internal
}  // namespace research_scann

#endif
