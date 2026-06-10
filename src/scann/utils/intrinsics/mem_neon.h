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

#ifndef SCANN_UTILS_INTRINSICS_MEM_NEON_H_
#define SCANN_UTILS_INTRINSICS_MEM_NEON_H_

#ifdef __aarch64__

#include <arm_neon.h>
#include <stddef.h>
#include <string.h>

namespace research_scann {
namespace neon {

static inline int8x8_t load_s8_4x1(const int8_t* p) {
  int8x8_t ret = vdup_n_s8(0);
  uint32_t val;
  memcpy(&val, p, sizeof(uint32_t));
  ret = vreinterpret_s8_u32(vset_lane_u32(val, vreinterpret_u32_s8(ret), 0));
  return ret;
}

static inline void load_f32_2x10(const float* s, ptrdiff_t src_stride,
                                 float32x2_t& s0, float32x2_t& s1,
                                 float32x2_t& s2, float32x2_t& s3,
                                 float32x2_t& s4, float32x2_t& s5,
                                 float32x2_t& s6, float32x2_t& s7,
                                 float32x2_t& s8, float32x2_t& s9) {
  s0 = vld1_f32(s);
  s += src_stride;
  s1 = vld1_f32(s);
  s += src_stride;
  s2 = vld1_f32(s);
  s += src_stride;
  s3 = vld1_f32(s);
  s += src_stride;
  s4 = vld1_f32(s);
  s += src_stride;
  s5 = vld1_f32(s);
  s += src_stride;
  s6 = vld1_f32(s);
  s += src_stride;
  s7 = vld1_f32(s);
  s += src_stride;
  s8 = vld1_f32(s);
  s += src_stride;
  s9 = vld1_f32(s);
}

static inline void load_f32_4x2(const float* s, ptrdiff_t src_stride,
                                float32x4_t& s0, float32x4_t& s1) {
  s0 = vld1q_f32(s);
  s += src_stride;
  s1 = vld1q_f32(s);
}

static inline void load_f32_4x4(const float* s, ptrdiff_t src_stride,
                                float32x4_t& s0, float32x4_t& s1,
                                float32x4_t& s2, float32x4_t& s3) {
  s0 = vld1q_f32(s);
  s += src_stride;
  s1 = vld1q_f32(s);
  s += src_stride;
  s2 = vld1q_f32(s);
  s += src_stride;
  s3 = vld1q_f32(s);
}

static inline void load_f32_4x8(const float* s, ptrdiff_t src_stride,
                                float32x4_t& s0, float32x4_t& s1,
                                float32x4_t& s2, float32x4_t& s3,
                                float32x4_t& s4, float32x4_t& s5,
                                float32x4_t& s6, float32x4_t& s7) {
  s0 = vld1q_f32(s);
  s += src_stride;
  s1 = vld1q_f32(s);
  s += src_stride;
  s2 = vld1q_f32(s);
  s += src_stride;
  s3 = vld1q_f32(s);
  s += src_stride;
  s4 = vld1q_f32(s);
  s += src_stride;
  s5 = vld1q_f32(s);
  s += src_stride;
  s6 = vld1q_f32(s);
  s += src_stride;
  s7 = vld1q_f32(s);
}

static inline void load_dup_f32_1x10(const float* s, ptrdiff_t src_stride,
                                     float32x4_t& s0, float32x4_t& s1,
                                     float32x4_t& s2, float32x4_t& s3,
                                     float32x4_t& s4, float32x4_t& s5,
                                     float32x4_t& s6, float32x4_t& s7,
                                     float32x4_t& s8, float32x4_t& s9) {
  s0 = vdupq_n_f32(*s);
  s += src_stride;
  s1 = vdupq_n_f32(*s);
  s += src_stride;
  s2 = vdupq_n_f32(*s);
  s += src_stride;
  s3 = vdupq_n_f32(*s);
  s += src_stride;
  s4 = vdupq_n_f32(*s);
  s += src_stride;
  s5 = vdupq_n_f32(*s);
  s += src_stride;
  s6 = vdupq_n_f32(*s);
  s += src_stride;
  s7 = vdupq_n_f32(*s);
  s += src_stride;
  s8 = vdupq_n_f32(*s);
  s += src_stride;
  s9 = vdupq_n_f32(*s);
}

static inline void store_f32_4x4(float* s, ptrdiff_t dst_stride,
                                 const float32x4_t s0, const float32x4_t s1,
                                 const float32x4_t s2, const float32x4_t s3) {
  vst1q_f32(s, s0);
  s += dst_stride;
  vst1q_f32(s, s1);
  s += dst_stride;
  vst1q_f32(s, s2);
  s += dst_stride;
  vst1q_f32(s, s3);
}

}  // namespace neon
}  // namespace research_scann

#endif  // __aarch64__

#endif  // SCANN_UTILS_INTRINSICS_MEM_NEON_H_
