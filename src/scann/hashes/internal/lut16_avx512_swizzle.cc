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

#include "scann/hashes/internal/lut16_avx512_swizzle.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/log/check.h"

namespace research_scann {
namespace asymmetric_hashing_internal {

// scann-core: portable layout helpers (see the header).
namespace {

// Groups in the super-group starting at group `start`.
size_t SuperGroupSize(size_t n32, size_t start) {
  const size_t full = 8 * (n32 / 8);  // groups in 8-group super-groups
  if (start < full) return 8;
  if (n32 % 8 >= 4 && start == full) return 4;  // the one 4-group one
  return 1;
}

inline uint8_t Nibble(uint8_t byte, bool high) {
  return high ? (byte >> 4) : (byte & 0x0F);
}

// A super-group of s = 4 or 8 groups, canonical src -> AVX-512 dst.
// Block-major: per block, 64 bytes for each 4 groups (128 datapoints),
// byte 2e + h holding datapoints 32h + e (low nibble) and 32(h + 2) + e.
void SwizzleWide(const uint8_t* src, uint8_t* dst, size_t s, size_t nb) {
  for (size_t blk = 0; blk < nb; ++blk) {
    for (size_t hh = 0; hh < s / 4; ++hh) {
      uint8_t* out = dst + 16 * s * blk + 64 * hh;  // 64 bytes: 4 groups
      const uint8_t* c[4];  // canonical groups 4hh..4hh+3, this block
      for (size_t q = 0; q < 4; ++q) {
        c[q] = src + (4 * hh + q) * 16 * nb + 16 * blk;
      }
      for (size_t e = 0; e < 32; ++e) {
        const bool high = e >= 16;  // datapoint e of a group: byte e % 16
        const size_t m = e % 16;
        out[2 * e] = Nibble(c[0][m], high) | (Nibble(c[2][m], high) << 4);
        out[2 * e + 1] = Nibble(c[1][m], high) | (Nibble(c[3][m], high) << 4);
      }
    }
  }
}

// The inverse: AVX-512 src -> canonical dst.
void UnswizzleWide(const uint8_t* src, uint8_t* dst, size_t s, size_t nb) {
  for (size_t blk = 0; blk < nb; ++blk) {
    for (size_t hh = 0; hh < s / 4; ++hh) {
      const uint8_t* in = src + 16 * s * blk + 64 * hh;
      for (size_t q = 0; q < 4; ++q) {
        uint8_t* c = dst + (4 * hh + q) * 16 * nb + 16 * blk;
        const size_t odd = q & 1;  // groups 1, 3: odd bytes
        const bool high = q >= 2;  // groups 2, 3: high nibbles
        for (size_t m = 0; m < 16; ++m) {
          c[m] = Nibble(in[2 * m + odd], high) |              // dp m
                 (Nibble(in[2 * m + 32 + odd], high) << 4);  // dp m + 16
        }
      }
    }
  }
}

// A single group, in place, block by block (as Avx512Swizzle32).
void SwizzleSingle(uint8_t* p, size_t nb) {
  for (size_t blk = 0; blk < nb; ++blk, p += 16) {
    uint8_t c[16];
    std::memcpy(c, p, 16);
    for (size_t m = 0; m < 8; ++m) {
      p[2 * m] = (c[m] & 0x0F) | ((c[m + 8] & 0x0F) << 4);  // dps m, m + 8
      p[2 * m + 1] = (c[m] >> 4) | (c[m + 8] & 0xF0);  // m + 16, m + 24
    }
  }
}

void UnswizzleSingle(uint8_t* p, size_t nb) {
  for (size_t blk = 0; blk < nb; ++blk, p += 16) {
    uint8_t s[16];
    std::memcpy(s, p, 16);
    for (size_t m = 0; m < 8; ++m) {
      p[m] = (s[2 * m] & 0x0F) | ((s[2 * m + 1] & 0x0F) << 4);  // m, m + 16
      p[m + 8] = (s[2 * m] >> 4) | (s[2 * m + 1] & 0xF0);  // m + 8, m + 24
    }
  }
}

template <bool kToAvx512>
void Convert(uint8_t* packed, size_t n32, size_t nb, size_t first_group) {
  DCHECK_EQ(Lut16Avx512SuperGroupStart(n32, first_group), first_group);
  if (nb == 0 || first_group >= n32) return;
  std::vector<uint8_t> tmp;  // one super-group (wide ones aren't in place)
  for (size_t g = first_group; g < n32;) {
    const size_t s = SuperGroupSize(n32, g);
    uint8_t* p = packed + g * 16 * nb;  // the super-group's bytes
    if (s == 1) {
      kToAvx512 ? SwizzleSingle(p, nb) : UnswizzleSingle(p, nb);
    } else {
      tmp.assign(p, p + s * 16 * nb);
      kToAvx512 ? SwizzleWide(tmp.data(), p, s, nb)
                : UnswizzleWide(tmp.data(), p, s, nb);
    }
    g += s;
  }
}

}  // namespace

size_t Lut16Avx512SuperGroupStart(size_t n32, size_t g) {
  const size_t full = 8 * (n32 / 8);
  if (g < full) return g - g % 8;
  if (n32 % 8 >= 4 && g < full + 4) return full;
  return g;
}

Lut16NibbleAddress Lut16Avx512NibbleAddress(size_t n32, size_t num_blocks,
                                            size_t dp, size_t block) {
  const size_t start = Lut16Avx512SuperGroupStart(n32, dp / 32);
  const size_t s = SuperGroupSize(n32, start);
  const size_t base = start * 16 * num_blocks;  // super-group's first byte
  const size_t d = dp - 32 * start;             // datapoint within it
  if (s == 1) {  // see SwizzleSingle
    return {base + 16 * block + 2 * (d % 8) + ((d >> 4) & 1),
            ((d >> 3) & 1) != 0};
  }
  const size_t hh = d / 128, dd = d % 128;  // see SwizzleWide
  return {base + 16 * s * block + 64 * hh + 2 * (dd % 32) + ((dd >> 5) & 1),
          dd >= 64};
}

void Lut16Avx512Swizzle(uint8_t* packed, size_t n32, size_t num_blocks,
                        size_t first_group) {
  Convert<true>(packed, n32, num_blocks, first_group);
}

void Lut16Avx512Unswizzle(uint8_t* packed, size_t n32, size_t num_blocks,
                          size_t first_group) {
  Convert<false>(packed, n32, num_blocks, first_group);
}

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#ifdef __x86_64__
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/avx512.h"

namespace research_scann {
namespace asymmetric_hashing_internal {

SCANN_AVX512_OUTLINE void Avx512Swizzle128(const uint8_t* src, uint8_t* dst) {
  Avx512<uint8_t> orig = Avx512<uint8_t>::Load(src);

  Avx512<uint8_t> low_nib_mask = Avx512<uint8_t>::Broadcast(0x0F);

  Avx512<uint8_t> even_nibs = orig & low_nib_mask;
  Avx512<uint8_t> odd_nibs =
      Avx512<uint8_t>(Avx512<uint16_t>(orig) >> 4) & low_nib_mask;

  constexpr uint64_t kAA = 0;
  constexpr uint64_t kBB = 8;

  Avx512<uint64_t> hi256_idxs = _mm512_set_epi64(
      kBB + 7, kBB + 6, kBB + 5, kBB + 4, kAA + 7, kAA + 6, kAA + 5, kAA + 4);
  Avx512<uint8_t> hi256_nibs =
      _mm512_permutex2var_epi64(*even_nibs, *hi256_idxs, *odd_nibs);
  Avx512<uint64_t> lo256_idxs =

      _mm512_set_epi64(kBB + 3, kBB + 2, kBB + 1, kBB + 0, kAA + 3, kAA + 2,
                       kAA + 1, kAA + 0);
  Avx512<uint8_t> lo256_nibs =
      _mm512_permutex2var_epi64(*even_nibs, *lo256_idxs, *odd_nibs);

  hi256_nibs = Avx512<uint8_t>(Avx512<uint16_t>(hi256_nibs) << 4);

  Avx512<uint8_t> new_bytes = hi256_nibs + lo256_nibs;

  constexpr int kDDBB = 0b11'11'10'10;
  constexpr int kCCAA = 0b01'01'00'00;
  Avx512<uint8_t> new_bytes_ddbb = _mm512_permutex_epi64(*new_bytes, kDDBB);
  Avx512<uint8_t> new_bytes_ccaa = _mm512_permutex_epi64(*new_bytes, kCCAA);

  Avx512<uint8_t> interleaved =
      _mm512_unpackhi_epi8(*new_bytes_ccaa, *new_bytes_ddbb);

  interleaved.Store(dst);
}

SCANN_AVX512_OUTLINE void Avx512Swizzle32(const uint8_t* src, uint8_t* dst) {
  array<uint8_t, 32> nibbles;
  for (size_t j : Seq(16)) {
    nibbles[2 * j + 0] = (src[j] >> 0) & 0x0F;
    nibbles[2 * j + 1] = (src[j] >> 4) & 0x0F;
  }
  for (size_t j : Seq(16)) {
    dst[j] = (nibbles[j + 0] << 0) + (nibbles[j + 16] << 4);
  }
}

SCANN_AVX512_OUTLINE void Avx512PlatformSpecificSwizzle(uint8_t* packed_dataset,
                                                        int num_datapoints,
                                                        int num_codes_per_dp) {
  size_t num_32dp_simd_iters = DivRoundUp(num_datapoints, 32);

  const size_t num_256dp_simd_iters = num_32dp_simd_iters / 8;
  num_32dp_simd_iters %= 8;

  const size_t num_128dp_simd_iters = num_32dp_simd_iters / 4;
  num_32dp_simd_iters %= 4;

  using Block = array<uint8_t, 16>;
  Block* blocks = reinterpret_cast<Block*>(packed_dataset);
  vector<Block> transposed(8 * num_codes_per_dp);

  for (auto _ : Seq(num_256dp_simd_iters)) {
    for (size_t jj : Seq(num_codes_per_dp)) {
      transposed[8 * jj + 0] = blocks[0 * num_codes_per_dp + jj];
      transposed[8 * jj + 1] = blocks[1 * num_codes_per_dp + jj];
      transposed[8 * jj + 2] = blocks[2 * num_codes_per_dp + jj];
      transposed[8 * jj + 3] = blocks[3 * num_codes_per_dp + jj];
      transposed[8 * jj + 4] = blocks[4 * num_codes_per_dp + jj];
      transposed[8 * jj + 5] = blocks[5 * num_codes_per_dp + jj];
      transposed[8 * jj + 6] = blocks[6 * num_codes_per_dp + jj];
      transposed[8 * jj + 7] = blocks[7 * num_codes_per_dp + jj];
    }

    for (size_t jj : Seq(2 * num_codes_per_dp)) {
      const uint8_t* src =
          reinterpret_cast<const uint8_t*>(&transposed[4 * jj]);
      uint8_t* dst = reinterpret_cast<uint8_t*>(&blocks[4 * jj]);
      Avx512Swizzle128(src, dst);
    }
    blocks += 8 * num_codes_per_dp;
  }

  for (auto _ : Seq(num_128dp_simd_iters)) {
    for (size_t jj : Seq(num_codes_per_dp)) {
      transposed[4 * jj + 0] = blocks[0 * num_codes_per_dp + jj];
      transposed[4 * jj + 1] = blocks[1 * num_codes_per_dp + jj];
      transposed[4 * jj + 2] = blocks[2 * num_codes_per_dp + jj];
      transposed[4 * jj + 3] = blocks[3 * num_codes_per_dp + jj];
    }

    for (size_t jj : Seq(num_codes_per_dp)) {
      const uint8_t* src =
          reinterpret_cast<const uint8_t*>(&transposed[4 * jj]);
      uint8_t* dst = reinterpret_cast<uint8_t*>(&blocks[4 * jj]);
      Avx512Swizzle128(src, dst);
    }
    blocks += 4 * num_codes_per_dp;
  }

  for (auto _ : Seq(num_32dp_simd_iters)) {
    for (size_t jj : Seq(num_codes_per_dp)) {
      uint8_t* ptr = reinterpret_cast<uint8_t*>(&blocks[1 * jj]);
      Avx512Swizzle32(ptr, ptr);
    }
    blocks += 1 * num_codes_per_dp;
  }
}

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#endif
