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

#ifndef SCANN_HASHES_INTERNAL_LUT16_AVX512_SWIZZLE_H_
#define SCANN_HASHES_INTERNAL_LUT16_AVX512_SWIZZLE_H_
#include <cstddef>
#include <cstdint>

namespace research_scann {
namespace asymmetric_hashing_internal {

// scann-core: the layout LUT16Avx512 reads (upstream's
// Avx512PlatformSpecificSwizzle output, which nothing called), and in-place
// conversion from/to the canonical layout the other kernels read.
//
// Both split the datapoints into groups of 32 (n32 groups); group g owns
// bytes [16 * num_blocks * g, 16 * num_blocks * (g + 1)).
// The AVX-512 layout then joins groups into super-groups, in this order:
//   8 groups (256 datapoints), as many as fit;
//   4 groups (128), if at least 4 are left;
//   single groups for the rest (0-3).
// A super-group keeps its groups' bytes; only the nibbles move within it.
// So which super-groups exist depends on n32 alone, and resizing reshapes
// only what follows the last full 8-group one.

// First group of the super-group containing group g.
size_t Lut16Avx512SuperGroupStart(size_t n32, size_t g);

// Where the code of (dp, block) is: byte offset and nibble.
struct Lut16NibbleAddress {
  size_t byte;
  bool high;  // high nibble
};
Lut16NibbleAddress Lut16Avx512NibbleAddress(size_t n32, size_t num_blocks,
                                            size_t dp, size_t block);
// Canonical: 16 bytes per block; byte m holds datapoints m (low) and m + 16.
inline Lut16NibbleAddress Lut16CanonicalNibbleAddress(size_t num_blocks,
                                                      size_t dp,
                                                      size_t block) {
  return {(dp / 32) * 16 * num_blocks + 16 * block + (dp % 16),
          (dp & 16) != 0};
}

// Convert groups [first_group, n32) in place; the groups before stay as
// they are. first_group must start a super-group (0 or a multiple of 8 up
// to 8 * (n32 / 8)).
void Lut16Avx512Swizzle(uint8_t* packed, size_t n32, size_t num_blocks,
                        size_t first_group = 0);
void Lut16Avx512Unswizzle(uint8_t* packed, size_t n32, size_t num_blocks,
                          size_t first_group = 0);

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#ifdef __x86_64__

#include "scann/utils/intrinsics/attributes.h"

namespace research_scann {
namespace asymmetric_hashing_internal {

void Avx512Swizzle128(const uint8_t* src, uint8_t* dst);

void Avx512Swizzle32(const uint8_t* src, uint8_t* dst);

void Avx512PlatformSpecificSwizzle(uint8_t* packed_dataset, int num_datapoints,
                                   int num_codes_per_dp);

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#endif
#endif
