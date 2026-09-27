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

// Regression tests for raw config values that crashed upstream ScaNN.
//
// Each config below is a builder-made config with one field changed, passed
// to ScannInterface::Initialize as text (what Python's create_searcher, the
// Rust ScannIndex::new, the C API and a loaded scann_config.pb all do).
// Upstream aborted the process (SIGFPE, CHECK, LOG(FATAL)), overflowed the
// heap, or hit undefined behavior on these; each must now be an error. Run
// under ASan+UBSan to catch the memory errors and UB directly.
//
// Also: configs that upstream rejected with a bare "SCANN_RET_CHECK failure"
// but that are valid (a tree with a PCA/TRUNCATE projection and asymmetric
// hashing without residual quantization) must build and search well, and
// distances between an empty sparse datapoint and a dense one must not do
// arithmetic on a null pointer.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <unistd.h>

#include "absl/strings/str_cat.h"
#include "scann/data_format/datapoint.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann_core/config_builder.h"

namespace {

using research_scann::DatapointPtr;
using research_scann::NNResultsVector;
using research_scann::ScannInterface;
using scann_core::ConfigBuilder;
using scann_core::DistanceMeasure;
using scann_core::Quantization;

int g_failures = 0;

void Fail(const std::string& what) {
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++g_failures;
}

constexpr size_t kN = 600;
constexpr size_t kDim = 16;

std::vector<float> Data(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd;
  std::vector<float> v(n * kDim);
  for (auto& x : v) x = nd(rng);
  return v;
}

// A builder-made config, as text.
std::string Make(DistanceMeasure distance,
                 const std::function<void(ConfigBuilder&)>& setup) {
  ConfigBuilder b(10, distance, kDim);
  setup(b);
  auto text = b.BuildText(kN);
  if (!text.ok()) {
    Fail(absl::StrCat("ConfigBuilder: ", text.status().ToString()));
    return "";
  }
  return *text;
}

// `config` with the first `from` replaced by `to`.
std::string Edit(const std::string& config, const std::string& from,
                 const std::string& to) {
  const size_t pos = config.find(from);
  if (pos == std::string::npos) {
    Fail(absl::StrCat("\"", from, "\" not in config:\n", config));
    return config;
  }
  std::string out = config;
  out.replace(pos, from.size(), to);
  return out;
}

void ExpectError(const std::string& what, const std::string& config,
                 const std::string& expected_substring = "") {
  if (config.empty()) return;
  const std::vector<float> data = Data(kN, 1);
  ScannInterface s;
  const absl::Status st = s.Initialize(data, kN, config, 1);
  if (st.ok()) {
    Fail(absl::StrCat(what, ": Initialize succeeded; expected an error"));
  } else if (!expected_substring.empty() &&
             st.message().find(expected_substring) == absl::string_view::npos) {
    Fail(absl::StrCat(what, ": error lacks \"", expected_substring,
                      "\": ", st.ToString()));
  } else {
    std::printf("ok (error as intended): %s -> %s\n", what.c_str(),
                std::string(st.message()).c_str());
  }
}

double ExactDistance(const float* q, const float* x, bool l2) {
  double d = 0;
  for (size_t k = 0; k < kDim; ++k)
    d += l2 ? (q[k] - x[k]) * (q[k] - x[k]) : -double(q[k]) * x[k];
  return d;
}

// Builds `config`, checks recall@10 against brute force on random queries,
// then serializes, reloads and checks that the reloaded index agrees.
void ExpectGoodRecall(const std::string& what, const std::string& config,
                      bool l2, double min_recall) {
  if (config.empty()) return;
  const size_t n = 3000;
  const std::vector<float> data = Data(n, 2);
  const std::vector<float> queries = Data(40, 3);
  ScannInterface s;
  absl::Status st = s.Initialize(data, n, config, 1);
  if (!st.ok()) return Fail(absl::StrCat(what, ": Initialize: ", st.ToString()));

  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() /
      absl::StrCat("scann_core_config_regressions_", getpid());
  std::filesystem::create_directories(dir);
  ScannInterface reloaded;
  auto assets = s.Serialize(dir.string());
  if (!assets.ok()) {
    std::filesystem::remove_all(dir);
    return Fail(absl::StrCat(what, ": Serialize: ", assets.status().ToString()));
  }
  auto artifacts = ScannInterface::LoadArtifacts(*s.config(), *assets);
  st = artifacts.ok() ? reloaded.Initialize(*std::move(artifacts))
                      : artifacts.status();
  std::filesystem::remove_all(dir);
  if (!st.ok()) return Fail(absl::StrCat(what, ": reload: ", st.ToString()));

  double hits = 0;
  for (size_t qi = 0; qi < 40; ++qi) {
    const float* q = queries.data() + qi * kDim;
    const DatapointPtr<float> query(nullptr, q, kDim, kDim);
    NNResultsVector res, res2;
    st = s.Search(query, &res, 10, 100, /*leaves=*/12);
    if (st.ok()) st = reloaded.Search(query, &res2, 10, 100, /*leaves=*/12);
    if (!st.ok()) return Fail(absl::StrCat(what, ": Search: ", st.ToString()));
    if (res != res2) return Fail(absl::StrCat(what, ": reloaded index differs"));
    std::vector<std::pair<double, size_t>> all;
    for (size_t i = 0; i < n; ++i)
      all.push_back({ExactDistance(q, data.data() + i * kDim, l2), i});
    std::partial_sort(all.begin(), all.begin() + 10, all.end());
    std::set<size_t> want;
    for (size_t i = 0; i < 10; ++i) want.insert(all[i].second);
    for (const auto& [idx, dist] : res) hits += want.count(idx);
  }
  const double recall = hits / (40 * 10);
  if (recall < min_recall)
    Fail(absl::StrCat(what, ": recall@10 ", recall, " < ", min_recall));
  else
    std::printf("ok: %s builds, recall@10 %.3f\n", what.c_str(), recall);
}

}  // namespace

int main() {
  const auto dot = DistanceMeasure::kDotProduct;
  const auto l2 = DistanceMeasure::kSquaredL2;
  scann_core::TreeOptions tree;
  tree.num_leaves = 12;
  tree.num_leaves_to_search = 4;
  tree.training_sample_size = kN;
  scann_core::AhOptions ah;  // LUT16, 2 dims per block: 8 blocks
  scann_core::ReorderOptions reorder;
  reorder.reordering_num_neighbors = 40;

  const std::string ah_cfg = Make(dot, [&](ConfigBuilder& b) { b.ScoreAh(ah).Reorder(reorder); });
  const std::string tree_ah_cfg = Make(dot, [&](ConfigBuilder& b) {
    b.Tree(tree).ScoreAh(ah).Reorder(reorder);  // residual quantization
  });
  const std::string tree_ah_l2_cfg = Make(l2, [&](ConfigBuilder& b) {
    b.Tree(tree).ScoreAh(ah).Reorder(reorder);  // no residual quantization
  });
  const std::string bf_cfg = Make(dot, [&](ConfigBuilder& b) { b.ScoreBruteForce(); });
  const std::string bf16_cfg =
      Make(dot, [&](ConfigBuilder& b) { b.ScoreBruteForce(Quantization::kBfloat16); });
  const std::string tree_bf_l2_cfg = Make(l2, [&](ConfigBuilder& b) { b.Tree(tree).ScoreBruteForce(); });
  const std::string tree_int8_cfg = Make(dot, [&](ConfigBuilder& b) {
    b.Tree(tree).ScoreBruteForce(Quantization::kInt8);
  });
  const std::string ah_rint8_cfg = Make(l2, [&](ConfigBuilder& b) {
    auto r = reorder;
    r.quantize = Quantization::kInt8;
    b.ScoreAh(ah).Reorder(r);
  });

  // Chunking projection sizes: SIGFPE (division by zero) and CHECK aborts.
  ExpectError("num_dims_per_block = 0",
              Edit(ah_cfg, "num_dims_per_block: 2", "num_dims_per_block: 0"),
              "num_dims_per_block");
  ExpectError("num_dims_per_block = 0 (tree)",
              Edit(tree_ah_l2_cfg, "num_dims_per_block: 2", "num_dims_per_block: 0"));
  ExpectError("num_blocks = 0", Edit(ah_cfg, "num_blocks: 8", "num_blocks: 0"),
              "num_blocks");
  ExpectError("num_blocks = 0 (tree)", Edit(tree_ah_cfg, "num_blocks: 8", "num_blocks: 0"));
  ExpectError("input_dim = 0", Edit(ah_cfg, "input_dim: 16", "input_dim: 0"));
  {
    auto a = ah;
    a.dimensions_per_block = 3;  // 16 = 5 * 3 + 1: VARIABLE_CHUNK
    const std::string var = Make(dot, [&](ConfigBuilder& b) { b.ScoreAh(a); });
    ExpectError("variable_blocks num_dims_per_block = 0",
                Edit(var, "num_blocks: 5 num_dims_per_block: 3",
                     "num_blocks: 5 num_dims_per_block: 0"));
  }

  // LUT16 with fewer than 16 clusters per block: heap overflow in search.
  for (const char* n : {"1", "2", "15"}) {
    const std::string to = absl::StrCat("num_clusters_per_block: ", n);
    ExpectError(absl::StrCat("LUT16 with ", to, " (tree, residual)"),
                Edit(tree_ah_cfg, "num_clusters_per_block: 16", to), "INT8_LUT16");
    ExpectError(absl::StrCat("LUT16 with ", to, " (tree)"),
                Edit(tree_ah_l2_cfg, "num_clusters_per_block: 16", to), "INT8_LUT16");
    ExpectError(absl::StrCat("LUT16 with ", to),
                Edit(ah_cfg, "num_clusters_per_block: 16", to), "INT8_LUT16");
  }

  // Binary distances on float data: LOG(FATAL).
  const std::string hamming = "\"BinaryHammingDistance\"";
  ExpectError("Hamming distance, brute force",
              Edit(bf_cfg, "\"DotProductDistance\"", hamming), "binary");
  ExpectError("Hamming distance, AH", Edit(ah_cfg, "\"DotProductDistance\"", hamming));
  ExpectError("Hamming partitioning distance",
              Edit(tree_bf_l2_cfg, "partitioning_distance { distance_measure: \"SquaredL2Distance\"",
                   "partitioning_distance { distance_measure: " + hamming));
  ExpectError("Hamming AH quantization distance",
              Edit(ah_cfg, "quantization_distance { distance_measure: \"SquaredL2Distance\"",
                   "quantization_distance { distance_measure: " + hamming));

  // Bfloat16 brute force with other distances: LOG(FATAL).
  for (const char* d : {"L1Distance", "LimitedInnerProductDistance", "AbsDotProductDistance"}) {
    ExpectError(absl::StrCat("bfloat16 brute force with ", d),
                Edit(bf16_cfg, "\"DotProductDistance\"", absl::StrCat("\"", d, "\"")),
                "bfloat16");
  }

  // Upper tree with quantized scoring and a distance it doesn't support.
  for (auto q : {Quantization::kInt8, Quantization::kBfloat16}) {
    scann_core::UpperTreeOptions u;
    u.num_leaves = 4;
    u.num_leaves_to_search = 2;
    u.scoring_mode = q;
    const std::string cfg = Make(l2, [&](ConfigBuilder& b) {
      b.Tree(tree).UpperTree(u).ScoreAh(ah).Reorder(reorder);
    });
    ExpectError(absl::StrCat("upper tree, quantization ", static_cast<int>(q), ", L1"),
                Edit(cfg, "query_tokenization_distance_override {distance_measure: \"SquaredL2Distance\"}",
                     "query_tokenization_distance_override {distance_measure: \"L1Distance\"}"),
                "bottom_up_top_level_partitioner");
  }

  // fixed_point_multiplier_quantile outside (0, 1]: undefined float -> int
  // conversion when quantizing the tree's leaves or the reordering data.
  for (const char* q : {"2", "nan", "-1"}) {
    ExpectError(absl::StrCat("tree int8 leaves, quantile ", q),
                Edit(tree_int8_cfg, "fixed_point { enabled: true",
                     absl::StrCat("fixed_point { fixed_point_multiplier_quantile: ", q,
                                  " enabled: true")),
                "quantile");
    ExpectError(absl::StrCat("int8 reordering, quantile ", q),
                Edit(ah_rint8_cfg, "fixed_point { enabled: true",
                     absl::StrCat("fixed_point { fixed_point_multiplier_quantile: ", q,
                                  " enabled: true")),
                "quantile");
  }

  // Incremental training with a projected tree failed with a bare
  // RET_CHECK; it is a config error.
  {
    scann_core::PcaOptions p;
    p.reduction_dim = 8;
    const std::string cfg = Make(dot, [&](ConfigBuilder& b) {
      b.Tree(tree).Pca(p).ScoreAh(ah).Reorder(reorder);
    });
    ExpectError("incremental training with PCA",
                Edit(cfg, "partitioning {", "partitioning { incremental_training_config { fraction: 0.2 }"),
                "incremental");
  }

  // A tree with a PCA/TRUNCATE projection, scored with AH without residual
  // quantization (every squared_l2 tree): upstream failed to build it with
  // "SCANN_RET_CHECK failure" because the AH projection had no input_dim.
  {
    scann_core::PcaOptions p;
    p.reduction_dim = 12;
    ExpectGoodRecall("tree + PCA + AH, squared_l2", Make(l2, [&](ConfigBuilder& b) {
      b.Tree(tree).Pca(p).ScoreAh(ah).Reorder({100});
    }), true, 0.8);
    ExpectGoodRecall("tree + TRUNCATE + AH, squared_l2", Make(l2, [&](ConfigBuilder& b) {
      b.Tree(tree).Truncate(12).ScoreAh(ah).Reorder({100});
    }), true, 0.8);
    auto a = ah;
    a.residual_quantization = false;
    ExpectGoodRecall("tree + PCA + AH without residuals, dot product",
                     Make(dot, [&](ConfigBuilder& b) {
                       b.Tree(tree).Pca(p).ScoreAh(a).Reorder({100});
                     }), false, 0.8);
  }

  // Distances between an empty sparse datapoint (null indices) and a dense
  // one did pointer arithmetic on the null indices (UBSan).
  {
    std::vector<float> dense(kDim, 0.5f);
    const DatapointPtr<float> empty(nullptr, nullptr, 0, kDim);
    const DatapointPtr<float> x(nullptr, dense.data(), kDim, kDim);
    const double d1 = research_scann::DotProductDistance().GetDistance(empty, x);
    const double d2 = research_scann::SquaredL2Distance().GetDistance(x, empty);
    if (d1 != 0.0 || std::fabs(d2 - 0.25 * kDim) > 1e-6)
      Fail(absl::StrCat("empty sparse vs dense: dot ", d1, ", squared L2 ", d2));
    else
      std::printf("ok: empty sparse vs dense distances\n");
  }

  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "PASSED", g_failures);
  return g_failures ? 1 : 0;
}
