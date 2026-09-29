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

// The fast int8 centroid kernel (scann_core/fast_int8_centers.h):
//
//  - its AVX-512 VNNI, AVX2 and scalar versions give identical floats, for
//    many centroid counts and dimensionalities (tails of 16 centroids and 4
//    dimensions), extreme query values and all-zero queries (queries below
//    1e-30 are declined: the exact kernel scores them);
//  - each score is within FastInt8Centers::MaxRelativeError(dims) *
//    max|query| * sum|centroid| of the exact dot product (computed in
//    double) and of ScaNN's float kernel;
//  - SelectTopK (the leaf selection after it): the k smallest (distance,
//    index) pairs within the threshold, ties to the lower index, with
//    AVX-512, AVX2 and scalar code, NaNs, infinities, signed zeros;
//  - end to end (tree + int8 centroids: dot product, l2_as_dot_product,
//    SOAR): the fast kernel is used by default on x86-64 with AVX2; single,
//    batched and parallel searches give identical results, with AVX-512
//    VNNI and with AVX2; exact mode (SetFastInt8TokenizationEnabled(false))
//    changes few results (checked by overlap).
//
// SCANN_TEST_FORCE_AVX2=1 runs the end-to-end part with the AVX2 kernels.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann_core/config_builder.h"
#include "scann_core/fast_int8_centers.h"

namespace {

using research_scann::DatapointPtr;
using research_scann::DenseDataset;
using research_scann::NNResultsVector;
using research_scann::ScannInterface;
using scann_core::ConfigBuilder;
using scann_core::DistanceMeasure;
using scann_core::FastInt8Centers;

int g_failures = 0;

void Fail(const std::string& what) {
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++g_failures;
}

bool Ok(const absl::Status& s, const std::string& what) {
  if (!s.ok()) Fail(absl::StrCat(what, ": ", s.ToString()));
  return s.ok();
}

bool IsX86() {
#ifdef __x86_64__
  return true;
#else
  return false;
#endif
}

// --- the kernel ---

void TestKernel(size_t n, size_t dims, uint32_t seed) {
  const std::string name = absl::StrCat("kernel n=", n, " dims=", dims);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> byte(-127, 127);
  std::vector<int8_t> rows(n * dims);
  for (auto& v : rows) v = static_cast<int8_t>(byte(rng));
  // Extremes: a row of -128 and one of 127 (int8 range limits).
  if (n > 2) {
    std::fill(rows.begin(), rows.begin() + dims, int8_t{-128});
    std::fill(rows.begin() + dims, rows.begin() + 2 * dims, int8_t{127});
  }
  FastInt8Centers fast;
  fast.Build(rows.data(), n, dims);
  if (!IsX86()) {
    if (!fast.empty()) Fail(name + ": built off x86-64");
    return;
  }
  if (fast.size() != n || fast.dimensionality() != dims) {
    Fail(name + ": wrong size");
    return;
  }
  DenseDataset<int8_t> centers(std::vector<int8_t>(rows), n);
  std::normal_distribution<float> normal;
  std::uniform_real_distribution<float> log_scale(-30.0f, 30.0f);
  std::vector<float> query(dims), scalar(n), avx2(n), vnni(n), exact(n);
  const double tol_rel = FastInt8Centers::MaxRelativeError(dims);
  int kernel_mismatch = 0, out_of_tol = 0, out_of_tol_scann = 0;
  double worst = 0;
  const bool have_vnni = research_scann::RuntimeSupportsAvx512Vnni();
  const bool have_avx2 = research_scann::RuntimeSupportsAvx2();
  for (int t = 0; t < 60; ++t) {
    const float scale = t % 3 == 0 ? 1.0f : std::pow(10.0f, log_scale(rng));
    for (float& q : query) q = normal(rng) * scale;
    if (t == 1) std::fill(query.begin(), query.end(), 0.0f);
    if (t == 2) {
      std::fill(query.begin(), query.end(), 0.0f);
      query[dims - 1] = -3.0f;
    }
    if (t == 3) {  // all at the extremes
      for (size_t i = 0; i < dims; ++i) query[i] = (i % 3) ? 1e30f : -1e30f;
    }
    if (t == 5 && dims <= 512) {
      // Scale 1 (the largest element is 32767): every other element is
      // halfway between two integers (rounded to even).
      for (size_t i = 0; i < dims; ++i)
        query[i] = static_cast<float>(static_cast<int>(i % 9) - 4) + 0.5f;
      query[0] = 32767.0f;
    }
    if (t == 4) {  // subnormal: declined (the exact kernel runs)
      std::fill(query.begin(), query.end(), 1e-40f);
      if (fast.DotProductDistancesWith("scalar", query.data(), scalar.data()))
        Fail(name + ": a subnormal query wasn't declined");
      continue;
    }
    if (!fast.DotProductDistancesWith("scalar", query.data(), scalar.data())) {
      float max_q = 0;
      for (float q : query) max_q = std::max(max_q, std::fabs(q));
      if (!(max_q < 1e-30f)) Fail(name + ": scalar kernel declined");
      continue;
    }
    {
      // The query's quantization without AVX-512 (scalar code) too.
      auto& avx512 = research_scann::flags_internal::should_use_avx512;
      const bool saved = avx512;
      avx512 = false;
      const bool ok =
          fast.DotProductDistancesWith("scalar", query.data(), avx2.data());
      avx512 = saved;
      if (!ok || std::memcmp(avx2.data(), scalar.data(), n * sizeof(float)))
        ++kernel_mismatch;
    }
    if (have_avx2 &&
        (!fast.DotProductDistancesWith("avx2", query.data(), avx2.data()) ||
         std::memcmp(avx2.data(), scalar.data(), n * sizeof(float)) != 0)) {
      ++kernel_mismatch;
    }
    if (have_vnni &&
        (!fast.DotProductDistancesWith("avx512_vnni", query.data(),
                                       vnni.data()) ||
         std::memcmp(vnni.data(), scalar.data(), n * sizeof(float)) != 0)) {
      ++kernel_mismatch;
    }
    research_scann::DenseDotProductDistanceOneToManyInt8Float(
        DatapointPtr<float>(nullptr, query.data(), dims, dims), centers,
        research_scann::MakeMutableSpan(exact));
    double max_q = 0;
    for (float q : query) max_q = std::max(max_q, std::fabs(double{q}));
    for (size_t i = 0; i < n; ++i) {
      double dot = 0, abs_c = 0;
      for (size_t j = 0; j < dims; ++j) {
        dot += double{query[j]} * rows[i * dims + j];
        abs_c += std::abs(int{rows[i * dims + j]});
      }
      const double bound = tol_rel * max_q * abs_c +
                           std::numeric_limits<float>::denorm_min();
      const double err = std::fabs(double{scalar[i]} + dot);
      if (!(err <= bound)) ++out_of_tol;
      if (!(std::fabs(double{scalar[i]} - double{exact[i]}) <= 2 * bound))
        ++out_of_tol_scann;
      if (bound > 0 && std::isfinite(err)) worst = std::max(worst, err / bound);
    }
  }
  if (kernel_mismatch) {
    Fail(absl::StrCat(name, ": ", kernel_mismatch,
                      " queries where the SIMD kernels differ from scalar"));
  }
  if (out_of_tol || out_of_tol_scann) {
    Fail(absl::StrCat(name, ": ", out_of_tol, " scores off the exact dot "
                      "product and ", out_of_tol_scann,
                      " off ScaNN's float kernel beyond the tolerance"));
  }
  if (!kernel_mismatch && !out_of_tol && !out_of_tol_scann && n >= 1000) {
    std::printf("ok: %s: kernels identical (vnni %d, avx2 %d), worst error "
                "%.3f of the bound (%.2e relative)\n",
                name.c_str(), have_vnni, have_avx2, worst, tol_rel);
  }
}

// --- SelectTopK ---

// The k smallest (distance, index) pairs with distance <= max_d, sorted.
std::vector<std::pair<uint32_t, float>> ReferenceTopK(
    const std::vector<float>& d, size_t k, float max_d) {
  std::vector<std::pair<float, uint32_t>> v;
  for (size_t i = 0; i < d.size(); ++i)
    if (d[i] <= max_d) v.emplace_back(d[i] + 0.0f, static_cast<uint32_t>(i));
  std::sort(v.begin(), v.end());
  if (v.size() > k) v.resize(k);
  std::vector<std::pair<uint32_t, float>> out;
  for (auto& [dist, i] : v) out.emplace_back(i, dist);
  return out;
}

void TestSelectTopK() {
  std::mt19937 rng(77);
  std::normal_distribution<float> normal;
  std::uniform_int_distribution<int> small(0, 20);
  auto& avx2 = research_scann::flags_internal::should_use_avx2;
  auto& avx512 = research_scann::flags_internal::should_use_avx512;
  const bool saved_avx2 = avx2, saved_avx512 = avx512;
  int bad = 0, cases = 0;
  const char* isa_names[] = {"avx512", "avx2", "scalar"};
  for (int isa = 0; isa < 3; ++isa) {
    avx512 = saved_avx512 && isa == 0;
    avx2 = saved_avx2 && isa <= 1;
    for (size_t n : {1, 7, 8, 15, 16, 17, 100, 1000, 1500, 5000}) {
      for (size_t k : {size_t{1}, size_t{2}, size_t{9}, size_t{16},
                       size_t{55}, size_t{120}, size_t{1000}, n,
                       ~size_t{0}}) {
        for (int variant = 0; variant < 4; ++variant) {
          std::vector<float> d(n);
          for (float& x : d) {
            // variant 1: many ties; 2: NaNs, infinities and signed zeros
            x = variant == 1 ? static_cast<float>(small(rng)) : normal(rng);
          }
          if (variant == 2) {
            for (size_t i = 0; i < n; i += 5)
              d[i] = (i % 3 == 0)   ? std::numeric_limits<float>::quiet_NaN()
                     : (i % 3 == 1) ? -0.0f
                                    : std::numeric_limits<float>::infinity();
            d[n / 2] = -std::numeric_limits<float>::infinity();
          }
          const float max_d = variant == 3 ? 0.25f
                                           : std::numeric_limits<float>::infinity();
          std::vector<std::pair<uint32_t, float>> got;
          scann_core::SelectTopK(d.data(), n, k, max_d, &got);
          std::sort(got.begin(), got.end(), [](auto& a, auto& b) {
            return a.second < b.second ||
                   (a.second == b.second && a.first < b.first);
          });
          const auto want = ReferenceTopK(d, k, max_d);
          ++cases;
          bool same = got.size() == want.size();
          for (size_t j = 0; same && j < got.size(); ++j)
            same = got[j].first == want[j].first &&
                   (got[j].second == want[j].second);
          if (!same) {
            if (++bad < 5)
              Fail(absl::StrCat("SelectTopK ", isa_names[isa], " n=", n,
                                " k=", k, " variant ", variant, ": got ",
                                got.size(), " want ", want.size()));
          }
        }
      }
    }
  }
  avx2 = saved_avx2;
  avx512 = saved_avx512;
  if (bad) {
    Fail(absl::StrCat("SelectTopK: ", bad, " of ", cases, " cases differ"));
  } else {
    std::printf("ok: SelectTopK: %d cases (3 ISAs) match the reference\n",
                cases);
  }
}

// --- end to end ---

std::vector<float> RandomRows(size_t n, size_t dim, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd;
  std::vector<float> v(n * dim);
  // Clustered data, so that leaf choices are not all near-ties.
  std::vector<float> means(32 * dim);
  for (float& x : means) x = 3 * nd(rng);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < dim; ++j)
      v[i * dim + j] = means[(i % 32) * dim + j] + nd(rng);
  return v;
}

bool SameResults(const NNResultsVector& a, const NNResultsVector& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].first != b[i].first ||
        std::memcmp(&a[i].second, &b[i].second, sizeof(float)) != 0)
      return false;
  }
  return true;
}

void TestEndToEnd(const std::string& name, const ConfigBuilder& b,
                  size_t dim) {
  constexpr size_t kN = 8000, kQ = 200;
  const std::vector<float> data = RandomRows(kN, dim, 21);
  const std::vector<float> queries = RandomRows(kQ, dim, 22);
  auto text = b.BuildText(kN);
  if (!Ok(text.status(), name + ": config")) return;
  ScannInterface s;
  if (!Ok(s.Initialize(data, kN, *text, 2), name + ": build")) return;
  const DenseDataset<float> qds(std::vector<float>(queries), kQ);
  struct Setting {
    int k, pre, leaves;
  };
  const Setting settings[] = {{10, 10, 1}, {10, 60, 5}, {20, 100, 20}};
  for (const Setting& st : settings) {
    const std::string tag =
        absl::StrCat(name, " (", st.k, "/", st.pre, "/", st.leaves, ")");
    // [exact, fast with the best kernel, fast with AVX2]
    std::vector<NNResultsVector> single[3];
    int exact_differ = 0;
    for (int mode = 0; mode < 3; ++mode) {
      scann_core::SetFastInt8TokenizationEnabled(mode != 0);
      const bool saved_vnni = research_scann::flags_internal::
          should_use_avx512_vnni;
      if (mode == 2) {
        if (!research_scann::RuntimeSupportsAvx2()) continue;
        research_scann::flags_internal::should_use_avx512_vnni = false;
      }
      single[mode].resize(kQ);
      for (size_t i = 0; i < kQ; ++i) {
        DatapointPtr<float> q(nullptr, queries.data() + i * dim, dim, dim);
        if (!Ok(s.Search(q, &single[mode][i], st.k, st.pre, st.leaves),
                tag + ": search"))
          return;
      }
      std::vector<NNResultsVector> batched(kQ), parallel(kQ);
      Ok(s.SearchBatched(qds, research_scann::MakeMutableSpan(batched), st.k,
                         st.pre, st.leaves),
         tag + ": batched");
      Ok(s.SearchBatchedParallel(qds, research_scann::MakeMutableSpan(parallel),
                                 st.k, st.pre, st.leaves, 7),
         tag + ": parallel");
      int differ = 0, differ_ids = 0;
      for (size_t i = 0; i < kQ; ++i) {
        differ += !SameResults(single[mode][i], batched[i]);
        differ += !SameResults(single[mode][i], parallel[i]);
        for (const auto* other : {&batched[i], &parallel[i]}) {
          bool same = other->size() == single[mode][i].size();
          for (size_t j = 0; same && j < other->size(); ++j)
            same = (*other)[j].first == single[mode][i][j].first;
          differ_ids += !same;
        }
      }
      research_scann::flags_internal::should_use_avx512_vnni = saved_vnni;
      if (mode == 0) exact_differ = differ;
      if (mode == 0 && differ) {
        // Upstream behavior, not the kernel's: squared-L2 trees search
        // batches through a per-leaf batched path whose distances can differ
        // in the last bits (so near-ties can reorder).
        std::printf("note: %s exact mode: %d batched/parallel results differ "
                    "from search() (%d in ids), before any fast kernel\n",
                    tag.c_str(), differ, differ_ids);
      } else if (differ && !exact_differ) {
        Fail(absl::StrCat(tag, " mode ", mode, ": ", differ,
                          " batched/parallel results differ from search()"));
      }
    }
    scann_core::SetFastInt8TokenizationEnabled(true);
    if (!single[2].empty()) {
      int differ = 0;
      for (size_t i = 0; i < kQ; ++i)
        differ += !SameResults(single[1][i], single[2][i]);
      if (differ)
        Fail(absl::StrCat(tag, ": ", differ,
                          " results differ between the AVX-512 VNNI and "
                          "AVX2 fast kernels"));
    }
    // Exact vs fast: nearly the same neighbors.
    double overlap = 0;
    int identical = 0;
    for (size_t i = 0; i < kQ; ++i) {
      std::set<uint64_t> a, c;
      for (auto& r : single[0][i]) a.insert(r.first);
      for (auto& r : single[1][i]) c.insert(r.first);
      size_t in = 0;
      for (auto x : a) in += c.count(x);
      overlap += a.empty() ? 1.0 : double(in) / a.size();
      identical += SameResults(single[0][i], single[1][i]);
    }
    overlap /= kQ;
    if (overlap < 0.97) {
      Fail(absl::StrCat(tag, ": exact and fast results overlap only ",
                        overlap));
    } else {
      std::printf("ok: %s: modes consistent; exact vs fast overlap %.4f, "
                  "%d/%zu identical\n",
                  tag.c_str(), overlap, identical, kQ);
    }
  }
}

scann_core::TreeOptions Tree(int leaves, int search) {
  scann_core::TreeOptions t;
  t.num_leaves = leaves;
  t.num_leaves_to_search = search;
  t.random_init = false;
  t.quantize_centroids = true;
  return t;
}

}  // namespace

int main() {
  if (const char* e = std::getenv("SCANN_TEST_FORCE_AVX2"); e && *e == '1') {
    research_scann::flags_internal::should_use_avx512 = false;
    research_scann::flags_internal::should_use_avx512_vnni = false;
    research_scann::flags_internal::should_use_amx = false;
    std::printf("SCANN_TEST_FORCE_AVX2: AVX-512 kernels disabled\n");
  }
  if (!scann_core::FastInt8TokenizationEnabled()) {
    Fail("the fast kernel is off by default (SCANN_EXACT_TOKENIZATION set?)");
  }
  const std::string kernel = scann_core::FastInt8KernelName();
  std::printf("fast kernel: %s\n", kernel.empty() ? "(none)" : kernel.c_str());
  if (IsX86() && research_scann::RuntimeSupportsAvx2() && kernel.empty())
    Fail("no fast kernel on an x86-64 CPU with AVX2");

  uint32_t seed = 1;
  for (size_t n : {1, 2, 3, 15, 16, 17, 31, 63, 64, 65, 100, 1000, 1500})
    for (size_t dims : {1, 2, 3, 4, 5, 7, 8, 9, 100, 128})
      TestKernel(n, dims, seed++);
  for (size_t dims : {384, 512, 513, 768, 1024, 4096}) TestKernel(1000, dims, seed++);
  TestKernel(2000, 100, seed++);
  {
    std::vector<int8_t> rows(3 * 4097, 1);
    FastInt8Centers too_wide;
    too_wide.Build(rows.data(), 3, 4097);
    if (!too_wide.empty()) Fail("built above kMaxDims dimensions");
  }
  TestSelectTopK();

  constexpr size_t kDim = 32;
  {
    ConfigBuilder b(10, DistanceMeasure::kDotProduct, kDim);
    auto t = Tree(80, 8);
    t.spherical = true;
    scann_core::AhOptions ah;
    ah.anisotropic_quantization_threshold = 0.2;
    b.Tree(t).ScoreAh(ah).Reorder({100, scann_core::Quantization::kBfloat16});
    TestEndToEnd("tree, int8 centroids, dot product", b, kDim);
  }
  {
    ConfigBuilder b(10, DistanceMeasure::kSquaredL2, kDim);
    scann_core::AhOptions ah;
    ah.dimensions_per_block = 3;
    b.Tree(Tree(80, 8)).ScoreAh(ah).Reorder({100}).L2AsDotProduct();
    TestEndToEnd("tree, int8 centroids, l2_as_dot_product", b, kDim);
  }
  {
    ConfigBuilder b(10, DistanceMeasure::kSquaredL2, kDim);
    b.Tree(Tree(80, 8)).ScoreAh({}).Reorder({100});
    TestEndToEnd("tree, int8 centroids, squared L2", b, kDim);
  }
  {
    ConfigBuilder b(10, DistanceMeasure::kDotProduct, kDim);
    auto t = Tree(80, 8);
    t.soar_lambda = 1.0;
    b.Tree(t).ScoreAh({}).Reorder({100});
    TestEndToEnd("tree + SOAR, int8 centroids", b, kDim);
  }
  std::printf("fast_int8_tokenization: %s (%d failure(s))\n",
              g_failures ? "FAILED" : "PASSED", g_failures);
  return g_failures ? 1 : 0;
}
