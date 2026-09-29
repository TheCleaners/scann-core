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

// Fast leaf selection for k-means trees with int8 centroids
// (quantize_centroids / FIXED_POINT_INT8 query tokenization), x86-64 only.
//
// ScaNN scores a query against int8 centroids in float: every centroid
// element is converted to float and multiplied with the (per-dimension
// rescaled) float query, one centroid row at a time. The conversions bound
// that kernel. The fast kernel quantizes the rescaled query once to 16-bit
// fixed point and computes each centroid's dot product in integers:
//
//   * AVX-512 VNNI: centroids in blocks of 16, 4 dimensions per 32-bit lane
//     (64 bytes per block and dimension group), so one broadcast of 4 query
//     bytes feeds 16 centroids per VPDPBUSD. The 16-bit query is split into
//     a high and a low byte (two VPDPBUSDs; VPDPBUSD multiplies unsigned by
//     signed bytes, so the high byte is offset by 128 and 128 * sum(centroid)
//     is subtracted at the end).
//   * AVX2 (no VNNI): the same layout, widened to 16 bits and VPMADDWD.
//
// Both compute the exact integer dot product of the 16-bit query with each
// centroid (the query scale is capped so that it can't overflow 32 bits),
// then one int -> float conversion and one multiplication. So their results
// are identical to each other and to ScalarReference(), for any batch size,
// thread count or search mode; they differ from ScaNN's float kernel only by
// the query's 16-bit rounding (see kMaxRelativeError below) and float
// rounding.
//
// The exact (upstream) kernel stays available: SCANN_EXACT_TOKENIZATION=1 in
// the environment, or --exact_int8_tokenization, read at startup; or
// SetFastInt8TokenizationEnabled(false) at run time (not while searching).
// Only query-time tokenization uses the fast kernel; datapoints are assigned
// to leaves at build and mutation time with ScaNN's kernels as before.

#ifndef SCANN_CORE_FAST_INT8_CENTERS_H_
#define SCANN_CORE_FAST_INT8_CENTERS_H_

#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>
#include <vector>

namespace scann_core {

// Whether query tokenization with int8 centroids uses the fast kernel
// (default: yes, unless SCANN_EXACT_TOKENIZATION=1 or
// --exact_int8_tokenization). Only a default: the kernel also needs an
// x86-64 CPU with AVX2 (FastInt8KernelName() says which one runs).
bool FastInt8TokenizationEnabled();
void SetFastInt8TokenizationEnabled(bool enabled);

// "avx512_vnni", "avx2" or "" (none: the exact kernel runs).
const char* FastInt8KernelName();

// The k smallest of distances[0, n) that are <= max_distance (NaNs never),
// by (distance, index): ties go to the lower index, so the result depends
// only on the input. Appended to *out as (index, distance), in no particular
// order. For the fast kernel's leaf selection (it replaces ScaNN's
// FastTopNeighbors there, whose ties depend on its buffer's history);
// AVX-512 or AVX2, else scalar.
void SelectTopK(const float* distances, size_t n, size_t k,
                float max_distance,
                std::vector<std::pair<uint32_t, float>>* out);

template <typename T, size_t kAlign>
struct AlignedAllocator {
  using value_type = T;
  AlignedAllocator() = default;
  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, kAlign>&) {}
  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, kAlign>;
  };
  T* allocate(size_t n) {
    return static_cast<T*>(
        ::operator new(n * sizeof(T), std::align_val_t(kAlign)));
  }
  void deallocate(T* p, size_t) {
    ::operator delete(p, std::align_val_t(kAlign));
  }
  friend bool operator==(const AlignedAllocator&, const AlignedAllocator&) {
    return true;
  }
  friend bool operator!=(const AlignedAllocator&, const AlignedAllocator&) {
    return false;
  }
};

class FastInt8Centers {
 public:
  static constexpr size_t kBlock = 16;
  static constexpr size_t kGroup = 4;
  // Bound on |fast - exact| for one centroid's dot product, relative to
  // max_i |query_i| * sum_i |centroid_i| (query in ScaNN's rescaled units):
  // half a quantization step of the 16-bit query (scaled so that its
  // largest element is 32767, or less above 512 dimensions) plus float
  // rounding (a worst-case bound for ScaNN's float kernel's summation).
  // Tests check it.
  static double MaxRelativeError(size_t dims);

  FastInt8Centers() = default;

  // Row-major int8 centroids (n x dims). Builds nothing off x86-64.
  void Build(const int8_t* rows, size_t n, size_t dims);
  void Clear();

  bool empty() const { return n_ == 0; }
  size_t size() const { return n_; }
  size_t dimensionality() const { return dims_; }
  size_t MemoryBytes() const {
    return codes_.capacity() + bias_.capacity() * sizeof(int32_t);
  }

  // Dot-product distances (-<query, centroid>) of a dense float query of
  // dimensionality() elements to every centroid, written to out[0, size()).
  // Returns false, writing nothing, when no fast kernel is enabled, or for
  // a query whose largest |element| is below 1e-30 (not 0).
  bool DotProductDistances(const float* query, float* out) const;

  // The same with a named kernel ("avx512_vnni", "avx2", "scalar"), for
  // tests. Returns false if the CPU lacks it.
  bool DotProductDistancesWith(const char* kernel, const float* query,
                               float* out) const;

 private:
  struct QuantizedQuery;
  bool Quantize(const float* query, QuantizedQuery* q) const;
  void RunScalar(const QuantizedQuery& q, float* out) const;
  void RunAvx2(const QuantizedQuery& q, float* out) const;
  void RunAvx512Vnni(const QuantizedQuery& q, float* out) const;

  size_t n_ = 0;
  size_t dims_ = 0;
  size_t groups_ = 0;
  size_t blocks_ = 0;
  int32_t max_query_ = 0;
  // blocks_ x groups_ x 64 bytes: byte 4 * lane + j of (block b, group g) is
  // element 4 * g + j of centroid 16 * b + lane (zero-padded).
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> codes_;
  // 32768 * sum of each centroid's elements (mod 2^32), for the VNNI
  // kernel's unsigned high byte.
  std::vector<int32_t, AlignedAllocator<int32_t, 64>> bias_;
};

}  // namespace scann_core

#endif  // SCANN_CORE_FAST_INT8_CENTERS_H_
