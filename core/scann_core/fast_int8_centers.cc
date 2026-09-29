// Copyright 2026 Elias Benali (@ebenali) and TheCleaners.
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

#include "scann_core/fast_int8_centers.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>

#include "absl/flags/flag.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann_core/scratch.h"

#ifdef __x86_64__
#include <immintrin.h>
#endif

ABSL_FLAG(bool, exact_int8_tokenization, false,
          "Score queries against int8 k-means centroids (quantize_centroids) "
          "with ScaNN's float kernel, as upstream, instead of scann-core's "
          "faster fixed-point kernel on x86-64. Read at startup; the "
          "SCANN_EXACT_TOKENIZATION=1 environment variable does the same.");

namespace scann_core {
namespace {

bool InitialEnabled() {
  if (absl::GetFlag(FLAGS_exact_int8_tokenization)) return false;
  const char* env = std::getenv("SCANN_EXACT_TOKENIZATION");
  if (env != nullptr && *env != '\0' && std::string_view(env) != "0")
    return false;
  return true;
}

std::atomic<bool> g_enabled{InitialEnabled()};

// The largest |query element| after quantization: the exact integer dot
// product with int8 centroids (|c| <= 128) must fit in int32.
int32_t MaxQueryFor(size_t dims) {
  const int64_t by_overflow =
      int64_t{2147483647} / (int64_t{128} * static_cast<int64_t>(dims));
  return static_cast<int32_t>(std::clamp<int64_t>(by_overflow, 1, 32767));
}

}  // namespace

bool FastInt8TokenizationEnabled() {
  return g_enabled.load(std::memory_order_relaxed);
}

void SetFastInt8TokenizationEnabled(bool enabled) {
  g_enabled.store(enabled, std::memory_order_relaxed);
}

const char* FastInt8KernelName() {
#ifdef __x86_64__
  if (!FastInt8TokenizationEnabled()) return "";
  if (research_scann::RuntimeSupportsAvx512Vnni()) return "avx512_vnni";
  if (research_scann::RuntimeSupportsAvx2()) return "avx2";
#endif
  return "";
}

double FastInt8Centers::MaxRelativeError(size_t dims) {
  // Half a step of the query's quantization, relative to its largest
  // element, plus float rounding of the int -> float conversion and the
  // final multiplication (and of the exact kernel's own summation).
  const double half_step = 0.5 / MaxQueryFor(std::max<size_t>(dims, 1));
  return half_step + 1.2e-7 * static_cast<double>(dims + 8);
}

struct FastInt8Centers::QuantizedQuery {
  // All zero-padded to a multiple of 32 elements (>= groups_ * 4).
  std::vector<int16_t> q;  // the query; AVX2: 4 per group as one uint64
  // VNNI: 4 bytes per group: (q >> 8) + 128 and q & 255.
  std::vector<uint32_t> hi, lo;
  float neg_inv_scale = 0;
};

void FastInt8Centers::Clear() {
  n_ = dims_ = groups_ = blocks_ = 0;
  max_query_ = 0;
  codes_ = {};
  bias_ = {};
}

void FastInt8Centers::Build(const int8_t* rows, size_t n, size_t dims) {
  Clear();
#ifdef __x86_64__
  if (n == 0 || dims == 0) return;
  n_ = n;
  dims_ = dims;
  groups_ = (dims + kGroup - 1) / kGroup;
  blocks_ = (n + kBlock - 1) / kBlock;
  max_query_ = MaxQueryFor(dims);
  codes_.assign(blocks_ * groups_ * kBlock * kGroup, 0);
  bias_.assign(blocks_ * kBlock, 0);
  for (size_t i = 0; i < n; ++i) {
    const int8_t* row = rows + i * dims;
    const size_t b = i / kBlock, lane = i % kBlock;
    int8_t* dst = codes_.data() + b * groups_ * kBlock * kGroup + lane * kGroup;
    uint32_t sum = 0;
    for (size_t j = 0; j < dims; ++j) {
      dst[(j / kGroup) * kBlock * kGroup + j % kGroup] = row[j];
      sum += static_cast<uint32_t>(static_cast<int32_t>(row[j]));
    }
    bias_[i] = static_cast<int32_t>(sum * 32768u);
  }
#else
  (void)rows;
  (void)n;
  (void)dims;
#endif
}

namespace {

#ifdef __x86_64__
__attribute__((target("avx,avx2,fma,avx512f,avx512bw"))) float MaxAbsAvx512(
    const float* x, size_t n) {
  __m512 m = _mm512_setzero_ps();
  for (size_t i = 0; i < n; i += 16) {
    const size_t len = std::min<size_t>(16, n - i);
    const __mmask16 lm = static_cast<__mmask16>((1u << len) - 1);
    m = _mm512_max_ps(m, _mm512_abs_ps(_mm512_maskz_loadu_ps(lm, x + i)));
  }
  return _mm512_reduce_max_ps(m);
}

// Same values as the scalar loop in Quantize(): one float multiplication,
// rounding to nearest even (MXCSR's default, as std::nearbyint's), clamping.
__attribute__((target("avx,avx2,fma,avx512f,avx512bw"))) void QuantizeAvx512(
    const float* x, size_t n, float scale, int32_t max_q, int16_t* q,
    uint32_t* hi, uint32_t* lo, size_t padded) {
  const __m512 sv = _mm512_set1_ps(scale);
  const __m512i hi_v = _mm512_set1_epi32(max_q), lo_v = _mm512_set1_epi32(-max_q);
  for (size_t i = 0; i < padded; i += 16) {
    const size_t len = i < n ? std::min<size_t>(16, n - i) : 0;
    const __mmask16 lm = static_cast<__mmask16>((1u << len) - 1);
    __m512i r = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_maskz_loadu_ps(lm, x + i), sv));
    r = _mm512_min_epi32(_mm512_max_epi32(r, lo_v), hi_v);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(q + i), _mm512_cvtepi32_epi16(r));
  }
  const __m512i k128 = _mm512_set1_epi16(128), k255 = _mm512_set1_epi16(255);
  for (size_t i = 0; i < padded; i += 32) {
    const __m512i v = _mm512_loadu_si512(q + i);
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(reinterpret_cast<char*>(hi) + i),
        _mm512_cvtepi16_epi8(_mm512_add_epi16(_mm512_srai_epi16(v, 8), k128)));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(reinterpret_cast<char*>(lo) + i),
        _mm512_cvtepi16_epi8(_mm512_and_si512(v, k255)));
  }
}
#endif

}  // namespace

bool FastInt8Centers::Quantize(const float* query, QuantizedQuery* q) const {
  const size_t padded = (groups_ * kGroup + 31) / 32 * 32;
  q->q.resize(padded);
  q->hi.resize(padded / 4);
  q->lo.resize(padded / 4);
#ifdef __x86_64__
  const bool avx512 = research_scann::RuntimeSupportsAvx512();
#else
  const bool avx512 = false;
#endif
  float m = 0;
  if (avx512) {
#ifdef __x86_64__
    m = MaxAbsAvx512(query, dims_);
#endif
  } else {
    for (size_t i = 0; i < dims_; ++i) m = std::max(m, std::fabs(query[i]));
  }
  // Tiny queries: m / 32767 would be a subnormal float and lose precision;
  // infinite ones (an overflowed rescaling): the exact kernel handles those.
  if (m != 0 && !(m >= 1e-30f && m <= std::numeric_limits<float>::max()))
    return false;
  // With m == 0 every element is 0 and so is every score.
  const float scale = m > 0 ? static_cast<float>(max_query_) / m : 0.0f;
  q->neg_inv_scale = m > 0 ? -(m / static_cast<float>(max_query_)) : 0.0f;
  if (avx512) {
#ifdef __x86_64__
    QuantizeAvx512(query, dims_, scale, max_query_, q->q.data(), q->hi.data(),
                   q->lo.data(), padded);
#endif
    return true;
  }
  std::fill(q->q.begin(), q->q.end(), int16_t{0});
  for (size_t i = 0; i < dims_; ++i) {
    const float v = std::nearbyint(query[i] * scale);
    q->q[i] = static_cast<int16_t>(std::clamp<float>(
        v, static_cast<float>(-max_query_), static_cast<float>(max_query_)));
  }
  for (size_t g = 0; g < padded / 4; ++g) {
    uint32_t hi = 0, lo = 0;
    for (size_t j = 0; j < kGroup; ++j) {
      const int32_t v = q->q[g * kGroup + j];
      // v = 256 * (v >> 8) + (v & 255), with v >> 8 in [-128, 127].
      hi |= static_cast<uint32_t>((v >> 8) + 128) << (8 * j);
      lo |= static_cast<uint32_t>(v & 255) << (8 * j);
    }
    q->hi[g] = hi;
    q->lo[g] = lo;
  }
  return true;
}

void FastInt8Centers::RunScalar(const QuantizedQuery& q, float* out) const {
  for (size_t i = 0; i < n_; ++i) {
    const size_t b = i / kBlock, lane = i % kBlock;
    const int8_t* src =
        codes_.data() + b * groups_ * kBlock * kGroup + lane * kGroup;
    int64_t dot = 0;
    for (size_t j = 0; j < groups_ * kGroup; ++j)
      dot += int64_t{q.q[j]} * src[(j / kGroup) * kBlock * kGroup + j % kGroup];
    out[i] = static_cast<float>(static_cast<int32_t>(dot)) * q.neg_inv_scale;
  }
}

#ifdef __x86_64__

#define SCANN_CORE_TARGET_AVX2 __attribute__((target("avx,avx2,fma")))
#define SCANN_CORE_TARGET_VNNI \
  __attribute__((target("avx,avx2,fma,avx512f,avx512vnni")))

namespace {

SCANN_CORE_TARGET_AVX2 inline __m256i Avx2Block8(__m256i a, __m256i b) {
  // a: centroids 4k..4k+3, b: 4k+4..4k+7, two partial sums each.
  return _mm256_permute4x64_epi64(_mm256_hadd_epi32(a, b), 0xD8);
}

SCANN_CORE_TARGET_AVX2 inline void Avx2Store(__m256i dot, __m256 scale,
                                             float* out, size_t rem) {
  const __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(dot), scale);
  if (rem >= 8) {
    _mm256_storeu_ps(out, f);
  } else if (rem > 0) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, f);
    std::memcpy(out, tmp, rem * sizeof(float));
  }
}

SCANN_CORE_TARGET_VNNI inline void VnniStore(__m512i h, __m512i l,
                                             const int32_t* bias,
                                             __m512 scale, float* out,
                                             size_t rem) {
  const __m512i dot =
      _mm512_sub_epi32(_mm512_add_epi32(_mm512_slli_epi32(h, 8), l),
                       _mm512_load_si512(bias));
  const __m512 f = _mm512_mul_ps(_mm512_cvtepi32_ps(dot), scale);
  if (rem >= 16) {
    _mm512_storeu_ps(out, f);
  } else {
    _mm512_mask_storeu_ps(out, static_cast<__mmask16>((1u << rem) - 1), f);
  }
}

}  // namespace

SCANN_CORE_TARGET_AVX2 void FastInt8Centers::RunAvx2(const QuantizedQuery& q,
                                                     float* out) const {
  const __m256 scale = _mm256_set1_ps(q.neg_inv_scale);
  const size_t stride = groups_ * kBlock * kGroup;
  for (size_t b = 0; b < blocks_; ++b) {
    const int8_t* p = codes_.data() + b * stride;
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    for (size_t g = 0; g < groups_; ++g, p += kBlock * kGroup) {
      long long pair;
      std::memcpy(&pair, q.q.data() + g * kGroup, sizeof(pair));
      const __m256i qv = _mm256_set1_epi64x(pair);
      const __m128i* src = reinterpret_cast<const __m128i*>(p);
      a0 = _mm256_add_epi32(
          a0, _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_load_si128(src)), qv));
      a1 = _mm256_add_epi32(
          a1,
          _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_load_si128(src + 1)), qv));
      a2 = _mm256_add_epi32(
          a2,
          _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_load_si128(src + 2)), qv));
      a3 = _mm256_add_epi32(
          a3,
          _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_load_si128(src + 3)), qv));
    }
    const size_t first = b * kBlock;
    const size_t rem = n_ - first;
    Avx2Store(Avx2Block8(a0, a1), scale, out + first, rem);
    if (rem > 8) Avx2Store(Avx2Block8(a2, a3), scale, out + first + 8, rem - 8);
  }
}

SCANN_CORE_TARGET_VNNI void FastInt8Centers::RunAvx512Vnni(
    const QuantizedQuery& q, float* out) const {
  const __m512 scale = _mm512_set1_ps(q.neg_inv_scale);
  const size_t stride = groups_ * kBlock * kGroup;
  const uint32_t* hi = q.hi.data();
  const uint32_t* lo = q.lo.data();
  size_t b = 0;
  // Four blocks (64 centroids) at a time: eight independent VPDPBUSD chains.
  for (; b + 4 <= blocks_; b += 4) {
    const int8_t* p = codes_.data() + b * stride;
    __m512i h0 = _mm512_setzero_si512(), h1 = h0, h2 = h0, h3 = h0;
    __m512i l0 = h0, l1 = h0, l2 = h0, l3 = h0;
    for (size_t g = 0; g < groups_; ++g, p += kBlock * kGroup) {
      const __m512i qh = _mm512_set1_epi32(static_cast<int>(hi[g]));
      const __m512i ql = _mm512_set1_epi32(static_cast<int>(lo[g]));
      const __m512i c0 = _mm512_load_si512(p);
      const __m512i c1 = _mm512_load_si512(p + stride);
      const __m512i c2 = _mm512_load_si512(p + 2 * stride);
      const __m512i c3 = _mm512_load_si512(p + 3 * stride);
      h0 = _mm512_dpbusd_epi32(h0, qh, c0);
      l0 = _mm512_dpbusd_epi32(l0, ql, c0);
      h1 = _mm512_dpbusd_epi32(h1, qh, c1);
      l1 = _mm512_dpbusd_epi32(l1, ql, c1);
      h2 = _mm512_dpbusd_epi32(h2, qh, c2);
      l2 = _mm512_dpbusd_epi32(l2, ql, c2);
      h3 = _mm512_dpbusd_epi32(h3, qh, c3);
      l3 = _mm512_dpbusd_epi32(l3, ql, c3);
    }
    const size_t first = b * kBlock;
    VnniStore(h0, l0, bias_.data() + first, scale, out + first, n_ - first);
    VnniStore(h1, l1, bias_.data() + first + 16, scale, out + first + 16,
              n_ - first - 16);
    VnniStore(h2, l2, bias_.data() + first + 32, scale, out + first + 32,
              n_ - first - 32);
    VnniStore(h3, l3, bias_.data() + first + 48, scale, out + first + 48,
              n_ - first - 48);
  }
  for (; b < blocks_; ++b) {
    const int8_t* p = codes_.data() + b * stride;
    __m512i h0 = _mm512_setzero_si512(), l0 = h0, h1 = h0, l1 = h0;
    size_t g = 0;
    // Two chains per accumulator (even and odd groups) for more overlap.
    for (; g + 2 <= groups_; g += 2, p += 2 * kBlock * kGroup) {
      const __m512i c0 = _mm512_load_si512(p);
      const __m512i c1 = _mm512_load_si512(p + kBlock * kGroup);
      h0 = _mm512_dpbusd_epi32(h0, _mm512_set1_epi32(static_cast<int>(hi[g])),
                               c0);
      l0 = _mm512_dpbusd_epi32(l0, _mm512_set1_epi32(static_cast<int>(lo[g])),
                               c0);
      h1 = _mm512_dpbusd_epi32(
          h1, _mm512_set1_epi32(static_cast<int>(hi[g + 1])), c1);
      l1 = _mm512_dpbusd_epi32(
          l1, _mm512_set1_epi32(static_cast<int>(lo[g + 1])), c1);
    }
    if (g < groups_) {
      const __m512i c0 = _mm512_load_si512(p);
      h0 = _mm512_dpbusd_epi32(h0, _mm512_set1_epi32(static_cast<int>(hi[g])),
                               c0);
      l0 = _mm512_dpbusd_epi32(l0, _mm512_set1_epi32(static_cast<int>(lo[g])),
                               c0);
    }
    const size_t first = b * kBlock;
    VnniStore(_mm512_add_epi32(h0, h1), _mm512_add_epi32(l0, l1),
              bias_.data() + first, scale, out + first, n_ - first);
  }
}

#else

void FastInt8Centers::RunAvx2(const QuantizedQuery& q, float* out) const {
  RunScalar(q, out);
}
void FastInt8Centers::RunAvx512Vnni(const QuantizedQuery& q,
                                    float* out) const {
  RunScalar(q, out);
}

#endif

// --- SelectTopK ---
//
// Two passes over the distances:
//  1. the minimum of each of G groups (element i is in group i mod G), G >=
//     2k: the k-th smallest group minimum T is >= the k-th smallest
//     distance (k distinct elements are <= T), and for distances in random
//     order only about 1.4k (k = 1: one) elements are <= T;
//  2. collect the elements <= min(T, max_distance), then the k smallest of
//     those by (distance, index).
// Exact; the result depends only on the input.

namespace {

// Maps floats to integers of the same order (-0 as +0).
inline uint32_t OrderedBits(float d) {
  d += 0.0f;
  uint32_t b;
  std::memcpy(&b, &d, sizeof(b));
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

struct TopKBuffer {
  std::vector<float> group_min;
  std::vector<float> d;
  std::vector<uint32_t> i;
  std::vector<uint64_t> keys;
};

// Per-thread scratch: not kept once grown past 1 MiB.
bool ScratchIsRetainable(const TopKBuffer& b) {
  return (b.group_min.capacity() + b.d.capacity() + b.i.capacity()) * 4 +
             b.keys.capacity() * 8 <=
         kMaxRetainedBytes;
}

size_t NumGroups(size_t k) {
  return (std::max<size_t>(64, 2 * k) + 15) / 16 * 16;
}

void GroupMinScalar(const float* d, size_t n, size_t g, float* out) {
  std::fill(out, out + g, std::numeric_limits<float>::infinity());
  for (size_t i = 0; i < n; ++i) {
    float& m = out[i % g];
    if (d[i] < m) m = d[i];
  }
}

size_t CollectScalar(const float* d, size_t n, float t, float* od,
                     uint32_t* oi) {
  size_t c = 0;
  for (size_t i = 0; i < n; ++i) {
    if (d[i] <= t) {
      od[c] = d[i];
      oi[c] = static_cast<uint32_t>(i);
      ++c;
    }
  }
  return c;
}

#ifdef __x86_64__

__attribute__((target("avx,avx2,fma,avx512f"))) void GroupMinAvx512(
    const float* d, size_t n, size_t g, float* out) {
  const __m512 inf = _mm512_set1_ps(std::numeric_limits<float>::infinity());
  for (size_t j = 0; j < g; j += 16) _mm512_storeu_ps(out + j, inf);
  for (size_t base = 0; base < n; base += g) {
    const size_t len = std::min(g, n - base);
    size_t j = 0;
    for (; j + 16 <= len; j += 16) {
      // min(x, m) is m when x is NaN.
      _mm512_storeu_ps(out + j, _mm512_min_ps(_mm512_loadu_ps(d + base + j),
                                              _mm512_loadu_ps(out + j)));
    }
    if (j < len) {
      const __mmask16 lm = static_cast<__mmask16>((1u << (len - j)) - 1);
      const __m512 x = _mm512_mask_loadu_ps(inf, lm, d + base + j);
      _mm512_storeu_ps(out + j, _mm512_min_ps(x, _mm512_loadu_ps(out + j)));
    }
  }
}

// The k-th smallest group minimum for G = 16 * kRegs groups, in registers;
// the k-th smallest of those by rank counting (no data-dependent branches):
// the largest minimum with fewer than k minima below it.
template <int kRegs>
__attribute__((target("avx,avx2,fma,avx512f"))) float GroupThresholdAvx512(
    const float* d, size_t n, size_t k) {
  constexpr size_t g = 16 * kRegs;
  const __m512 inf = _mm512_set1_ps(std::numeric_limits<float>::infinity());
  __m512 acc[kRegs];
  for (int a = 0; a < kRegs; ++a) acc[a] = inf;
  size_t base = 0;
  for (; base + g <= n; base += g) {
    for (int a = 0; a < kRegs; ++a) {
      // min(x, m) is m when x is NaN.
      acc[a] = _mm512_min_ps(_mm512_loadu_ps(d + base + 16 * a), acc[a]);
    }
  }
  for (int a = 0; a < kRegs; ++a) {
    const size_t first = base + 16 * a;
    if (first >= n) break;
    const size_t len = std::min<size_t>(16, n - first);
    const __mmask16 lm = static_cast<__mmask16>((1u << len) - 1);
    acc[a] = _mm512_min_ps(_mm512_mask_loadu_ps(inf, lm, d + first), acc[a]);
  }
  if (k == 1) {
    __m512 m = acc[0];
    for (int a = 1; a < kRegs; ++a) m = _mm512_min_ps(m, acc[a]);
    return _mm512_reduce_min_ps(m);
  }
  alignas(64) float mins[g];
  for (int a = 0; a < kRegs; ++a) _mm512_store_ps(mins + 16 * a, acc[a]);
  float t = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < g; ++i) {
    const __m512 v = _mm512_set1_ps(mins[i]);
    int below = 0;
    for (int a = 0; a < kRegs; ++a)
      below += __builtin_popcount(_mm512_cmp_ps_mask(acc[a], v, _CMP_LT_OQ));
    if (static_cast<size_t>(below) < k) t = std::max(t, mins[i]);
  }
  return t;
}

// Writes up to 16 past the count.
__attribute__((target("avx,avx2,fma,avx512f"))) size_t CollectAvx512(
    const float* d, size_t n, float t, float* od, uint32_t* oi) {
  const __m512 tv = _mm512_set1_ps(t);
  __m512i idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                  13, 14, 15);
  const __m512i step = _mm512_set1_epi32(16);
  size_t c = 0;
  for (size_t i = 0; i < n; i += 16, idx = _mm512_add_epi32(idx, step)) {
    const __mmask16 lm = n - i >= 16
                             ? static_cast<__mmask16>(0xFFFF)
                             : static_cast<__mmask16>((1u << (n - i)) - 1);
    const __m512 v = _mm512_maskz_loadu_ps(lm, d + i);
    const __mmask16 m = lm & _mm512_cmp_ps_mask(v, tv, _CMP_LE_OQ);
    if (m == 0) continue;
    _mm512_storeu_ps(od + c, _mm512_maskz_compress_ps(m, v));
    _mm512_storeu_si512(oi + c, _mm512_maskz_compress_epi32(m, idx));
    c += static_cast<size_t>(__builtin_popcount(m));
  }
  return c;
}

__attribute__((target("avx,avx2,fma"))) void GroupMinAvx2(const float* d,
                                                          size_t n, size_t g,
                                                          float* out) {
  const __m256 inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
  for (size_t j = 0; j < g; j += 8) _mm256_storeu_ps(out + j, inf);
  for (size_t base = 0; base < n; base += g) {
    const size_t len = std::min(g, n - base);
    size_t j = 0;
    for (; j + 8 <= len; j += 8) {
      _mm256_storeu_ps(out + j, _mm256_min_ps(_mm256_loadu_ps(d + base + j),
                                              _mm256_loadu_ps(out + j)));
    }
    for (; j < len; ++j) {
      if (d[base + j] < out[j]) out[j] = d[base + j];
    }
  }
}

__attribute__((target("avx,avx2,fma"))) size_t CollectAvx2(
    const float* d, size_t n, float t, float* od, uint32_t* oi) {
  const __m256 tv = _mm256_set1_ps(t);
  size_t c = 0, i = 0;
  for (; i + 8 <= n; i += 8) {
    unsigned m = static_cast<unsigned>(_mm256_movemask_ps(
        _mm256_cmp_ps(_mm256_loadu_ps(d + i), tv, _CMP_LE_OQ)));
    while (m) {
      const unsigned j = static_cast<unsigned>(__builtin_ctz(m));
      m &= m - 1;
      od[c] = d[i + j];
      oi[c] = static_cast<uint32_t>(i + j);
      ++c;
    }
  }
  for (; i < n; ++i) {
    if (d[i] <= t) {
      od[c] = d[i];
      oi[c] = static_cast<uint32_t>(i);
      ++c;
    }
  }
  return c;
}

#endif

}  // namespace

void SelectTopK(const float* distances, size_t n, size_t k,
                float max_distance,
                std::vector<std::pair<uint32_t, float>>* out) {
  if (k == 0 || n == 0) return;
  k = std::min(k, n);
  ScratchLease<TopKBuffer> b;
  int isa = 0;  // 2: AVX-512, 1: AVX2, 0: scalar
#ifdef __x86_64__
  if (research_scann::RuntimeSupportsAvx512()) {
    isa = 2;
  } else if (research_scann::RuntimeSupportsAvx2()) {
    isa = 1;
  }
#endif
  float t = max_distance;
  const size_t g = NumGroups(k);
  if (g < n) {
    float kth = std::numeric_limits<float>::quiet_NaN();
#ifdef __x86_64__
    // Rank counting is quadratic in G: only for the smallest G.
    if (isa == 2 && g == 64) kth = GroupThresholdAvx512<4>(distances, n, k);
#endif
    if (std::isnan(kth)) {
      b->group_min.resize(g);
#ifdef __x86_64__
    if (isa == 2) {
      GroupMinAvx512(distances, n, g, b->group_min.data());
    } else if (isa == 1) {
      GroupMinAvx2(distances, n, g, b->group_min.data());
    } else {
      GroupMinScalar(distances, n, g, b->group_min.data());
    }
#else
    GroupMinScalar(distances, n, g, b->group_min.data());
#endif
    std::nth_element(b->group_min.begin(), b->group_min.begin() + (k - 1),
                     b->group_min.end());
      kth = b->group_min[k - 1];
    }
    // Group minima are never NaN (min() skips NaNs, groups start at inf).
    if (kth < t) t = kth;
  }
  b->d.resize(n + 16);
  b->i.resize(n + 16);
  size_t c;
#ifdef __x86_64__
  if (isa == 2) {
    c = CollectAvx512(distances, n, t, b->d.data(), b->i.data());
  } else if (isa == 1) {
    c = CollectAvx2(distances, n, t, b->d.data(), b->i.data());
  } else {
    c = CollectScalar(distances, n, t, b->d.data(), b->i.data());
  }
#else
  c = CollectScalar(distances, n, t, b->d.data(), b->i.data());
#endif
  out->reserve(out->size() + std::min(c, k));
  if (c <= k) {
    for (size_t j = 0; j < c; ++j) out->emplace_back(b->i[j], b->d[j]);
    return;
  }
  b->keys.resize(c);
  for (size_t j = 0; j < c; ++j)
    b->keys[j] = (uint64_t{OrderedBits(b->d[j])} << 32) | b->i[j];
  std::nth_element(b->keys.begin(), b->keys.begin() + (k - 1),
                   b->keys.begin() + c);
  for (size_t j = 0; j < k; ++j) {
    const uint32_t i = static_cast<uint32_t>(b->keys[j]);
    out->emplace_back(i, distances[i]);
  }
}

bool FastInt8Centers::DotProductDistancesWith(const char* kernel,
                                              const float* query,
                                              float* out) const {
  if (empty()) return false;
  const std::string_view k(kernel);
#ifdef __x86_64__
  const bool vnni = k == "avx512_vnni";
  const bool avx2 = k == "avx2";
  if (vnni && !research_scann::RuntimeSupportsAvx512Vnni()) return false;
  if (avx2 && !research_scann::RuntimeSupportsAvx2()) return false;
  if (!vnni && !avx2 && k != "scalar") return false;
#else
  const bool vnni = false, avx2 = false;
  if (k != "scalar") return false;
#endif
  ScratchLease<QuantizedQuery> q;
  if (!Quantize(query, q.get())) return false;
  if (vnni) {
    RunAvx512Vnni(*q, out);
  } else if (avx2) {
    RunAvx2(*q, out);
  } else {
    RunScalar(*q, out);
  }
  return true;
}

bool FastInt8Centers::DotProductDistances(const float* query,
                                          float* out) const {
  if (empty()) return false;
  const char* kernel = FastInt8KernelName();
  if (*kernel == '\0') return false;
  return DotProductDistancesWith(kernel, query, out);
}

}  // namespace scann_core
