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

// The reordering helpers prefetch the candidates' rows ahead of the
// one-to-many kernels (0.3). This checks that they return what the kernels
// return without it, float bit for float bit: exact float32 reordering (every
// specially optimized distance, all candidates and top-1), bfloat16 (dot
// product, squared L2) and int8 fixed point (dot product, cosine, squared
// L2), over many dimensionalities (including those whose rows are too long
// to be prefetched) and candidate counts (0 to 3 candidates, counts around
// multiples of 3, more candidates than the prefetch distance and fewer),
// with repeated candidates. Under the sanitizers it also covers
// the prefetcher's reads of the candidate list.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measures.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/distance_measures/one_to_one/cosine_distance.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/distance_measures/one_to_one/l1_distance.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/distance_measures/one_to_one/limited_inner_product.h"
#include "scann/utils/reordering_helper.h"
#include "scann/utils/scalar_quantization_helpers.h"
#include "scann/utils/types.h"

namespace {

using research_scann::DatapointIndex;
using research_scann::DatapointPtr;
using research_scann::DefaultDenseDatasetView;
using research_scann::DenseDataset;
using research_scann::DistanceMeasure;
using research_scann::MakeDatapointPtr;
using research_scann::MakeMutableSpan;
using research_scann::NNResultsVector;

int g_failures = 0;
int g_checks = 0;

void Fail(const std::string& what) {
  if (g_failures < 20) std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++g_failures;
}

void Expect(const NNResultsVector& got, const NNResultsVector& want,
            const std::string& what) {
  ++g_checks;
  if (got.size() != want.size()) {
    Fail(absl::StrCat(what, ": size ", got.size(), " vs ", want.size()));
    return;
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i].first != want[i].first ||
        std::memcmp(&got[i].second, &want[i].second, sizeof(float)) != 0) {
      Fail(absl::StrCat(what, ": candidate ", i, " (", got[i].first, ", ",
                        got[i].second, ") vs (", want[i].first, ", ",
                        want[i].second, ")"));
      return;
    }
  }
}

void ExpectTop1(const std::pair<DatapointIndex, float>& got,
                const std::pair<DatapointIndex, float>& want,
                const std::string& what) {
  ++g_checks;
  if (got.first != want.first ||
      std::memcmp(&got.second, &want.second, sizeof(float)) != 0) {
    Fail(absl::StrCat(what, ": top-1 (", got.first, ", ", got.second, ") vs (",
                      want.first, ", ", want.second, ")"));
  }
}

DenseDataset<float> RandomDataset(size_t n, size_t dims, std::mt19937* rng) {
  std::normal_distribution<float> normal;
  std::vector<float> values(n * dims);
  for (float& v : values) v = normal(*rng);
  return DenseDataset<float>(std::move(values), n);
}

NNResultsVector RandomCandidates(size_t count, size_t dataset_size,
                                 std::mt19937* rng) {
  std::uniform_int_distribution<DatapointIndex> pick(0, dataset_size - 1);
  NNResultsVector result(count);
  for (auto& r : result) r = {pick(*rng), 0.0f};
  // A few repeats, as SOAR's spilled candidates can give.
  for (size_t i = 3; i < count; i += 7) result[i].first = result[i / 2].first;
  return result;
}

void TestDims(size_t dims, std::mt19937* rng) {
  constexpr size_t kDatasetSize = 3000;
  const DenseDataset<float> dataset = RandomDataset(kDatasetSize, dims, rng);
  const DenseDataset<float> queries = RandomDataset(3, dims, rng);

  // Exact float32 reordering.
  std::vector<std::pair<std::string, std::shared_ptr<const DistanceMeasure>>>
      distances = {
          {"dot", std::make_shared<research_scann::DotProductDistance>()},
          {"sql2", std::make_shared<research_scann::SquaredL2Distance>()},
          {"cosine", std::make_shared<research_scann::CosineDistance>()},
          {"l1", std::make_shared<research_scann::L1Distance>()},
          {"limited",
           std::make_shared<research_scann::LimitedInnerProductDistance>()},
  };
  auto shared_dataset = std::make_shared<DenseDataset<float>>(dataset.Copy());

  // bfloat16 reordering, and the kernels' inputs for the references.
  research_scann::Bfloat16DenseDotProductReorderingHelper bf16_dot(dataset);
  research_scann::Bfloat16DenseSquaredL2ReorderingHelper bf16_sql2(dataset);
  const auto& bf16_dataset =
      *std::dynamic_pointer_cast<const DenseDataset<int16_t>>(
          bf16_dot.dataset());

  // int8 fixed-point reordering, from a quantized dataset given explicitly
  // so that the references can use the same inverse multipliers.
  auto quantized = research_scann::ScalarQuantizeFloatDataset(dataset);
  auto int8_dataset = std::make_shared<DenseDataset<int8_t>>(
      std::move(quantized.quantized_dataset));
  std::vector<float> multipliers(dims), inverse_multipliers(dims);
  for (size_t d = 0; d < dims; ++d) {
    multipliers[d] = 1.0f / quantized.inverse_multiplier_by_dimension[d];
    inverse_multipliers[d] = 1.0f / multipliers[d];
  }
  auto squared_norms = std::make_shared<std::vector<float>>();
  for (size_t i = 0; i < kDatasetSize; ++i) {
    squared_norms->push_back(research_scann::SquaredL2Norm(dataset[i]));
  }
  research_scann::FixedPointFloatDenseDotProductReorderingHelper int8_dot(
      int8_dataset, multipliers);
  research_scann::FixedPointFloatDenseCosineReorderingHelper int8_cosine(
      int8_dataset, multipliers);
  research_scann::FixedPointFloatDenseSquaredL2ReorderingHelper int8_sql2(
      int8_dataset, multipliers, squared_norms);

  const std::vector<size_t> counts = {0,  1,  2,  3,  4,   5,   6,   7,  8,
                                      9,  10, 11, 12, 13,  17,  31,  47, 48,
                                      49, 50, 97, 99, 100, 301, 1000};
  for (size_t qi = 0; qi < queries.size(); ++qi) {
    const DatapointPtr<float> query = queries[qi];
    for (size_t count : counts) {
      const std::string where =
          absl::StrCat("dims ", dims, ", query ", qi, ", ", count, " cands");
      const NNResultsVector candidates =
          RandomCandidates(count, kDatasetSize, rng);

      for (const auto& [name, dist] : distances) {
        research_scann::ExactReorderingHelper<float> helper(dist,
                                                            shared_dataset);
        NNResultsVector got = candidates, want = candidates;
        if (!helper.ComputeDistancesForReordering(query, &got).ok()) {
          Fail(absl::StrCat(where, " exact ", name, ": error"));
        }
        research_scann::DenseDistanceOneToMany<
            float, std::pair<DatapointIndex, float>>(*dist, query, dataset,
                                                     MakeMutableSpan(want));
        Expect(got, want, absl::StrCat(where, " exact ", name));

        NNResultsVector top1_input = candidates, top1_ref = candidates;
        auto top1 = helper.ComputeTop1ReorderingDistance(query, &top1_input);
        auto top1_want = research_scann::DenseDistanceOneToManyTop1<
            float, float, std::pair<DatapointIndex, float>>(
            *dist, query, dataset, MakeMutableSpan(top1_ref));
        if (!top1.ok()) {
          Fail(absl::StrCat(where, " exact top-1 ", name, ": error"));
        } else {
          ExpectTop1(*top1, top1_want,
                     absl::StrCat(where, " exact top-1 ", name));
        }
      }

      {
        NNResultsVector got = candidates, want = candidates;
        (void)bf16_dot.ComputeDistancesForReordering(query, &got);
        research_scann::DenseDotProductDistanceOneToManyBf16Float(
            query, DefaultDenseDatasetView<int16_t>(bf16_dataset),
            MakeMutableSpan(want));
        Expect(got, want, absl::StrCat(where, " bf16 dot"));

        got = candidates;
        want = candidates;
        (void)bf16_sql2.ComputeDistancesForReordering(query, &got);
        research_scann::OneToManyBf16FloatSquaredL2(
            query, DefaultDenseDatasetView<int16_t>(bf16_dataset),
            MakeMutableSpan(want));
        Expect(got, want, absl::StrCat(where, " bf16 sql2"));
      }

      {
        auto preprocessed =
            research_scann::PrepareForAsymmetricScalarQuantizedDotProduct(
                query, inverse_multipliers);
        NNResultsVector dot_ref = candidates;
        research_scann::DenseDotProductDistanceOneToManyInt8Float(
            MakeDatapointPtr(preprocessed.get(), dims),
            DefaultDenseDatasetView<int8_t>(*int8_dataset),
            MakeMutableSpan(dot_ref));

        NNResultsVector got = candidates;
        (void)int8_dot.ComputeDistancesForReordering(query, &got);
        Expect(got, dot_ref, absl::StrCat(where, " int8 dot"));

        NNResultsVector want = dot_ref;
        for (auto& w : want) w.second = w.second + 1;
        got = candidates;
        (void)int8_cosine.ComputeDistancesForReordering(query, &got);
        Expect(got, want, absl::StrCat(where, " int8 cosine"));

        const float query_norm = research_scann::SquaredL2Norm(query);
        want = dot_ref;
        for (auto& w : want) {
          w.second = query_norm + (*squared_norms)[w.first] + 2.0f * w.second;
        }
        got = candidates;
        (void)int8_sql2.ComputeDistancesForReordering(query, &got);
        Expect(got, want, absl::StrCat(where, " int8 sql2"));
      }
    }
  }
}

}  // namespace

int main() {
  std::mt19937 rng(20260929);
  for (size_t dims : {1, 3, 8, 16, 64, 100, 128, 257, 512, 768, 1100})
    TestDims(dims, &rng);
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
