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

// The single-query fast paths (0.2.1) give the results of the code they
// replace, bit for bit, and are safe to use from many threads:
//
//  - Lut16DotProductLookupBuilder: the raw LUT16 lookup table of every block
//    shape it takes (1 to 4 dimensions, mixed) equals the generic
//    DenseDistanceOneToMany's, float bit for float bit, on random queries
//    over many magnitudes; on x86-64 it must be in use (not declined);
//  - FastTopNeighbors::InitLikeNew: one object reused across many top-N
//    problems (sizes, epsilons, ties) returns what a new object returns;
//  - ScratchLease: nested leases get distinct objects;
//  - searches from 8 threads at once, with different parameters and a
//    batched search in between, return what the same searches return one at
//    a time (tree + AH indexes: int8 centroids with bfloat16 reordering,
//    SOAR, l2_as_dot_product, an upper tree).

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/hashes/internal/asymmetric_hashing_impl.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/types.h"
#include "scann_core/config_builder.h"
#include "scann_core/scratch.h"

namespace {

using research_scann::DatapointIndex;
using research_scann::DatapointPtr;
using research_scann::DenseDataset;
using research_scann::FastTopNeighbors;
using research_scann::NNResultsVector;
using research_scann::ScannInterface;
using research_scann::asymmetric_hashing_internal::
    Lut16DotProductLookupBuilder;
using scann_core::ConfigBuilder;
using scann_core::DistanceMeasure;

int g_failures = 0;

void Fail(const std::string& what) {
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++g_failures;
}

bool Ok(const absl::Status& s, const std::string& what) {
  if (!s.ok()) Fail(absl::StrCat(what, ": ", s.ToString()));
  return s.ok();
}

// --- Lut16DotProductLookupBuilder ---

void TestLut16Builder(const std::vector<int>& block_dims, uint32_t seed) {
  const std::string name = absl::StrCat("lut16 builder, ", block_dims.size(),
                                        " blocks from ", block_dims[0],
                                        " dims, seed ", seed);
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal;
  std::uniform_real_distribution<float> log_scale(-20.0f, 20.0f);
  std::vector<DenseDataset<float>> centers;
  size_t total_dims = 0;
  for (int dims : block_dims) {
    std::vector<float> values(16 * dims);
    const float scale = std::pow(10.0f, log_scale(rng) / 4);
    for (float& v : values) v = normal(rng) * scale;
    if (seed % 2 == 0) values[3] = 0.0f;  // a zero coordinate
    centers.emplace_back(std::move(values), 16);
    total_dims += dims;
  }
  auto builder = Lut16DotProductLookupBuilder::Create(centers);
#if defined(__x86_64__)
  if (builder == nullptr) {
    Fail(name + ": declined on x86-64 (the fast path isn't used)");
    return;
  }
#endif
  if (builder == nullptr) {
    std::printf("%s: declined on this platform (generic path)\n",
                name.c_str());
    return;
  }
  const research_scann::DotProductDistance dot;
  std::vector<float> query(total_dims), fast(16 * centers.size());
  std::array<float, 16> generic;
  int mismatches = 0;
  for (int t = 0; t < 4000; ++t) {
    const float scale = t % 4 == 0 ? 1.0f : std::pow(10.0f, log_scale(rng));
    for (float& q : query) q = normal(rng) * scale;
    if (t % 7 == 0) query[t % total_dims] = (t % 2) ? 0.0f : -0.0f;
    builder->ComputeContiguous(query.data(),
                               research_scann::MakeMutableSpan(fast));
    size_t offset = 0;
    for (size_t b = 0; b < centers.size(); ++b) {
      const size_t dims = block_dims[b];
      research_scann::DenseDistanceOneToMany(
          dot, DatapointPtr<float>(nullptr, query.data() + offset, dims, dims),
          centers[b], research_scann::MakeMutableSpan(generic));
      if (std::memcmp(generic.data(), fast.data() + 16 * b,
                      sizeof(generic)) != 0) {
        ++mismatches;
      }
      offset += dims;
    }
  }
  if (mismatches) {
    Fail(absl::StrCat(name, ": ", mismatches, " blocks differ"));
  } else {
    std::printf("ok: %s: bit-identical to the generic path\n", name.c_str());
  }
}

// --- FastTopNeighbors::InitLikeNew ---

void TestInitLikeNew() {
  std::mt19937 rng(7);
  FastTopNeighbors<float> reused;
  const size_t sizes[] = {0, 1, 3, 10, 33, 100, 300, 5000, 20000};
  int mismatches = 0;
  for (int t = 0; t < 400; ++t) {
    const size_t max_results = sizes[rng() % std::size(sizes)];
    const size_t n = rng() % 30000;
    // Few distinct values: many ties.
    const int levels = (t % 3 == 0) ? 50 : 1000000;
    std::vector<float> distances(n);
    for (float& d : distances) d = static_cast<float>(rng() % levels);
    const float epsilon =
        (t % 2) ? std::numeric_limits<float>::infinity()
                : static_cast<float>(rng() % levels);
    FastTopNeighbors<float> fresh(max_results, epsilon);
    reused.InitLikeNew(max_results, epsilon);
    fresh.PushBlock(distances, 0);
    reused.PushBlock(distances, 0);
    std::vector<std::pair<DatapointIndex, float>> a, b;
    fresh.FinishUnsorted(&a);
    reused.FinishUnsorted(&b);
    if (a != b || fresh.epsilon() != reused.epsilon()) ++mismatches;
  }
  if (mismatches) {
    Fail(absl::StrCat("InitLikeNew: ", mismatches, " of 400 differ"));
  } else {
    std::printf("ok: InitLikeNew: 400 reuses match new objects\n");
  }
}

void TestScratchLease() {
  scann_core::ScratchLease<std::vector<float>> a;
  a->assign(10, 1.0f);
  float* first = a->data();
  {
    scann_core::ScratchLease<std::vector<float>> b;
    if (b.get() == a.get()) Fail("ScratchLease: nested leases share");
    b->assign(20, 2.0f);
  }
  if (a->size() != 10 || a->data() != first || (*a)[0] != 1.0f) {
    Fail("ScratchLease: a nested lease changed the outer one");
  }
  std::printf("ok: ScratchLease: nested leases are distinct\n");
}

// --- Concurrent searches ---

std::vector<float> RandomRows(size_t n, size_t dim, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd;
  std::vector<float> v(n * dim);
  for (float& x : v) x = nd(rng);
  return v;
}

struct Setting {
  int k, pre, leaves;
};

void TestConcurrentSearch(const std::string& name, const ConfigBuilder& b,
                          size_t dim) {
  constexpr size_t kN = 6000, kQ = 120;
  const std::vector<float> data = RandomRows(kN, dim, 11);
  const std::vector<float> queries = RandomRows(kQ, dim, 12);
  auto text = b.BuildText(kN);
  if (!Ok(text.status(), name + ": config")) return;
  ScannInterface s;
  if (!Ok(s.Initialize(data, kN, *text, 2), name + ": build")) return;
  const Setting settings[] = {{10, 40, 1}, {10, 100, 8}, {50, 200, 25}};
  auto q = [&](size_t i) {
    return DatapointPtr<float>(nullptr, queries.data() + i * dim, dim, dim);
  };
  std::vector<std::vector<NNResultsVector>> serial(std::size(settings));
  for (size_t si = 0; si < std::size(settings); ++si) {
    serial[si].resize(kQ);
    for (size_t i = 0; i < kQ; ++i) {
      const Setting& st = settings[si];
      if (!Ok(s.Search(q(i), &serial[si][i], st.k, st.pre, st.leaves),
              name + ": serial search")) {
        return;
      }
    }
  }
  const DenseDataset<float> query_ds(
      std::vector<float>(queries.begin(), queries.begin() + 8 * dim), 8);
  std::vector<int> mismatches(8, 0), errors(8, 0);
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&, t] {
      NNResultsVector res;
      std::vector<NNResultsVector> batched(8);
      for (int round = 0; round < 3; ++round) {
        for (size_t j = 0; j < kQ; ++j) {
          const size_t i = (j + 17 * t) % kQ;
          const size_t si = (i + t + round) % std::size(settings);
          const Setting& st = settings[si];
          if (!s.Search(q(i), &res, st.k, st.pre, st.leaves).ok()) {
            ++errors[t];
          } else if (res != serial[si][i]) {
            ++mismatches[t];
          }
          if (j % 40 == 0 &&
              !s.SearchBatched(query_ds, research_scann::MakeMutableSpan(
                                             batched),
                               10, 100, 8)
                   .ok()) {
            ++errors[t];
          }
        }
      }
    });
  }
  for (auto& th : threads) th.join();
  int total_mismatches = 0, total_errors = 0;
  for (int t = 0; t < 8; ++t) {
    total_mismatches += mismatches[t];
    total_errors += errors[t];
  }
  if (total_mismatches || total_errors) {
    Fail(absl::StrCat(name, ": concurrent searches: ", total_mismatches,
                      " differ from serial ones, ", total_errors, " errors"));
  } else {
    std::printf("ok: %s: 8 threads x 360 searches match serial searches\n",
                name.c_str());
  }
}

scann_core::TreeOptions Tree(int leaves, int search) {
  scann_core::TreeOptions t;
  t.num_leaves = leaves;
  t.num_leaves_to_search = search;
  t.random_init = false;
  return t;
}

}  // namespace

int main() {
  TestLut16Builder({2, 2, 2, 2, 2, 2, 2, 2}, 1);
  TestLut16Builder({1, 1, 1, 1}, 2);
  TestLut16Builder({3, 3, 3, 1}, 3);
  TestLut16Builder({4, 4, 4, 4, 4}, 4);
  TestLut16Builder({2, 3, 1, 4, 2, 4, 3, 1}, 5);
  TestInitLikeNew();
  TestScratchLease();

  constexpr size_t kDim = 24;
  {
    ConfigBuilder b(10, DistanceMeasure::kDotProduct, kDim);
    auto t = Tree(60, 8);
    t.quantize_centroids = true;
    t.spherical = true;
    scann_core::AhOptions ah;
    ah.anisotropic_quantization_threshold = 0.2;
    b.Tree(t).ScoreAh(ah).Reorder({100, scann_core::Quantization::kBfloat16});
    TestConcurrentSearch("tree, int8 centroids, bf16 reordering", b, kDim);
  }
  {
    ConfigBuilder b(10, DistanceMeasure::kDotProduct, kDim);
    auto t = Tree(60, 8);
    t.soar_lambda = 1.0;
    b.Tree(t).ScoreAh({}).Reorder({100});
    TestConcurrentSearch("tree + SOAR", b, kDim);
  }
  {
    ConfigBuilder b(10, DistanceMeasure::kSquaredL2, kDim);
    auto t = Tree(60, 8);
    t.quantize_centroids = true;
    scann_core::AhOptions ah;
    ah.dimensions_per_block = 3;
    b.Tree(t).ScoreAh(ah).Reorder({100}).L2AsDotProduct();
    TestConcurrentSearch("l2_as_dot_product", b, kDim);
  }
  {
    ConfigBuilder b(10, DistanceMeasure::kSquaredL2, kDim);
    scann_core::UpperTreeOptions u;
    u.num_leaves = 12;
    u.num_leaves_to_search = 5;
    b.Tree(Tree(150, 20)).UpperTree(u).ScoreAh({}).Reorder({100});
    TestConcurrentSearch("upper tree, squared L2", b, kDim);
  }
  std::printf("search_scratch: %s (%d failure(s))\n",
              g_failures ? "FAILED" : "PASSED", g_failures);
  return g_failures ? 1 : 0;
}
