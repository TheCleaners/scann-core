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

// Autopilot's rules (scann/utils/single_machine_autopilot.cc):
//
//  - TUNED_V1 without data (ConfigBuilder, a config preview): the configs
//    for the tuning study's shapes and a few others, value by value;
//  - UPSTREAM (and a config without `rules`): upstream's configs, unchanged;
//  - the data statistics, and what the tuned rules derive from them: the
//    anisotropic threshold scales with the norms (their 5th percentile), is
//    recorded and reused when the data isn't available (reload, a bfloat16
//    dataset);
//  - AutopilotChoosesL2AsDotProduct;
//  - end to end through ScannInterface: squared L2 data with constant norms
//    becomes an l2_as_dot_product index (with varying norms it doesn't), dot-product data gets its recorded threshold,
//    both reload (directory) and retrain to the same config, with recall
//    against brute force; an index built with upstream's rules keeps them.

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/proto/auto_tuning.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/single_machine_autopilot.h"
#include "scann/utils/types.h"
#include "scann_core/config_builder.h"

namespace {

namespace fs = std::filesystem;
using research_scann::AutopilotTreeAH;
using research_scann::DatapointPtr;
using research_scann::DenseDataset;
using research_scann::NNResultsVector;
using research_scann::ScannConfig;
using research_scann::ScannInterface;
using scann_core::AutopilotOptions;
using scann_core::AutopilotRules;
using scann_core::ConfigBuilder;
using scann_core::DistanceMeasure;
using scann_core::Quantization;

int g_failures = 0;

void Fail(const std::string& what) {
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++g_failures;
}

bool Ok(const absl::Status& s, const std::string& what) {
  if (!s.ok()) Fail(absl::StrCat(what, ": ", s.ToString()));
  return s.ok();
}

void Expect(bool cond, const std::string& what) {
  if (!cond) Fail(what);
}

bool Near(double a, double b, double rel = 1e-6) {
  return std::abs(a - b) <= rel * std::max(std::abs(a), std::abs(b));
}

ScannConfig Preview(int k, DistanceMeasure d, uint32_t dim, uint64_t n,
                    const AutopilotOptions& o = {}) {
  ConfigBuilder b(k, d, dim);
  b.Autopilot(o);
  auto c = b.Build(n);
  if (!Ok(c.status(), absl::StrCat("preview n=", n, " dim=", dim)))
    return ScannConfig();
  return *c;
}

int DimsPerBlock(const ScannConfig& c) {
  const auto& p = c.hash().asymmetric_hash().projection();
  return p.num_dims_per_block();
}

// The tuning study's shapes and a few more, without data: exact values.
void TestTunedPreviews() {
  struct Case {
    const char* name;
    int k;
    DistanceMeasure d;
    uint32_t dim;
    uint64_t n;
    int leaves, leaves_to_search, reorder, dims_per_block;
    double threshold;  // NaN: none
  };
  const double kNan = std::nan("");
  // T(d) = 0.2 min(1, 128 / d)^0.75 for unit norms.
  auto t = [](double d) { return 0.2 * std::pow(std::min(1.0, 128 / d), 0.75); };
  const Case cases[] = {
      // Upstream's leaves (at most sqrt(n)) and leaves_to_search.
      {"glove-100", 10, DistanceMeasure::kDotProduct, 100, 1183514, 903, 106,
       317, 2, 0.2},
      {"sift-128 (preview: plain L2)", 10, DistanceMeasure::kSquaredL2, 128,
       1000000, 976, 109, 317, 2, kNan},
      // sqrt(n) leaves, upstream's leaves_to_search times
      // sqrt(1160 / 5000); 4 dimensions per block; threshold 0.052.
      {"arxiv-768", 100, DistanceMeasure::kDotProduct, 768, 1344643, 1160, 86,
       1000, 4, t(768)},
      {"imagenet-512", 100, DistanceMeasure::kDotProduct, 512, 1281167, 1132,
       85, 1000, 3, t(512)},
      {"100k x 100", 10, DistanceMeasure::kDotProduct, 100, 100000, 76, 51,
       317, 2, 0.2},
      {"5M x 64 L2", 10, DistanceMeasure::kSquaredL2, 64, 5000000, 2236, 137,
       317, 2, kNan},
      {"1536-d", 10, DistanceMeasure::kDotProduct, 1536, 200000, 447, 74, 317,
       8, t(1536)},
  };
  for (const Case& t : cases) {
    const ScannConfig c = Preview(t.k, t.d, t.dim, t.n);
    const std::string what = absl::StrCat("tuned preview ", t.name, ":\n",
                                          c.DebugString());
    const auto& part = c.partitioning();
    const auto& ah = c.hash().asymmetric_hash();
    Expect(!c.has_brute_force() && c.has_partitioning() && c.has_hash(), what);
    Expect(part.num_children() == t.leaves, absl::StrCat(what, " leaves"));
    Expect(part.query_spilling().max_spill_centers() == t.leaves_to_search,
           absl::StrCat(what, " leaves_to_search"));
    Expect(c.exact_reordering().approx_num_neighbors() == t.reorder,
           absl::StrCat(what, " reorder"));
    Expect(DimsPerBlock(c) == t.dims_per_block,
           absl::StrCat(what, " dims per block"));
    // Float32 reordering (the builder's default); without data, nothing is
    // recorded.
    Expect(!c.exact_reordering().bfloat16().enabled() &&
               !c.exact_reordering().fixed_point().enabled() &&
               !c.autopilot().tree_ah().has_noise_shaping_threshold(),
           absl::StrCat(what, " float32, nothing recorded"));
    const bool dot = t.d == DistanceMeasure::kDotProduct;
    Expect(std::isnan(t.threshold)
               ? std::isnan(ah.noise_shaping_threshold())
               : Near(ah.noise_shaping_threshold(), t.threshold),
           absl::StrCat(what, " threshold"));
    Expect(dot == (part.avq() == 2.5f), absl::StrCat(what, " avq"));
    Expect(dot == ah.use_residual_quantization(), absl::StrCat(what, " residual"));
    Expect(c.autopilot().tree_ah().rules() == AutopilotTreeAH::TUNED_V1,
           absl::StrCat(what, " rules recorded"));
  }
  // Upstream's brute-force cutoff: 42 leaves of 4 * 32 KiB / dim points.
  for (uint32_t dim : {100u, 768u}) {
    const uint64_t bound = dim == 100 ? 42 * 1310 : 42 * 200;
    Expect(Preview(10, DistanceMeasure::kDotProduct, dim, bound - 1)
               .has_brute_force(),
           absl::StrCat("brute force below ", bound));
    Expect(!Preview(10, DistanceMeasure::kDotProduct, dim, bound)
                .has_brute_force(),
           absl::StrCat("tree from ", bound));
  }
  // The reordering precision asked for.
  AutopilotOptions o;
  o.quantize = Quantization::kBfloat16;
  Expect(Preview(10, DistanceMeasure::kDotProduct, 100, 1000000, o)
             .exact_reordering()
             .bfloat16()
             .enabled(),
         "bfloat16 reordering");
  o.quantize = Quantization::kInt8;
  Expect(Preview(10, DistanceMeasure::kDotProduct, 100, 1000000, o)
             .exact_reordering()
             .fixed_point()
             .enabled(),
         "explicit int8 reordering");
  // The same input, the same config.
  Expect(Preview(10, DistanceMeasure::kDotProduct, 300, 777777).DebugString() ==
             Preview(10, DistanceMeasure::kDotProduct, 300, 777777).DebugString(),
         "deterministic");
}

// kUpstream is scann-core 0.2.0's autopilot: upstream's rules, float32
// reordering by default, no `rules` field.
void TestUpstreamRules() {
  AutopilotOptions o;
  o.rules = AutopilotRules::kUpstream;
  const ScannConfig c = Preview(10, DistanceMeasure::kDotProduct, 100, 1183514, o);
  const std::string what = absl::StrCat("upstream rules:\n", c.DebugString());
  Expect(!c.autopilot().tree_ah().has_rules(), what + " (rules field)");
  Expect(c.partitioning().num_children() == 903 &&
             c.partitioning().query_spilling().max_spill_centers() == 106 &&
             c.exact_reordering().approx_num_neighbors() == 317 &&
             DimsPerBlock(c) == 2 &&
             c.hash().asymmetric_hash().noise_shaping_threshold() == 0.2 &&
             std::isnan(c.partitioning().avq()) &&
             !c.exact_reordering().bfloat16().enabled(),
         what);
  const ScannConfig v = Preview(100, DistanceMeasure::kDotProduct, 768, 1344643, o);
  Expect(v.partitioning().num_children() == 5000 &&
             v.partitioning().query_spilling().max_spill_centers() == 178 &&
             DimsPerBlock(v) == 2,
         absl::StrCat("upstream rules at 768 dims:\n", v.DebugString()));
  o.allow_l2_as_dot_product = false;
  ConfigBuilder b(10, DistanceMeasure::kSquaredL2, 16);
  b.Autopilot(o);
  Expect(!b.Build(100000).ok(), "allow_l2_as_dot_product with upstream rules");
}

std::vector<float> Gaussian(size_t n, size_t dim, uint32_t seed, float sigma,
                            float mean = 0.0f) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(mean, sigma);
  std::vector<float> v(n * dim);
  for (float& x : v) x = nd(rng);
  return v;
}

void Normalize(std::vector<float>& v, size_t dim, float to = 1.0f) {
  for (size_t i = 0; i < v.size() / dim; ++i) {
    double s = 0;
    for (size_t j = 0; j < dim; ++j) s += v[i * dim + j] * v[i * dim + j];
    const float f = to / std::sqrt(s);
    for (size_t j = 0; j < dim; ++j) v[i * dim + j] *= f;
  }
}

ScannConfig TunedConfig(int k, const char* distance) {
  ScannConfig c;
  google::protobuf::TextFormat::ParseFromString(
      absl::StrCat("num_neighbors: ", k, " distance_measure { distance_measure: \"",
                   distance, "\" } autopilot { tree_ah { rules: TUNED_V1 } }"),
      &c);
  return c;
}

// Data statistics, and what the rules derive from them.
void TestDataDependence() {
  // 45 leaves; brute force below 42 * 436 = 18312.
  constexpr size_t kDim = 300;
  constexpr size_t kN = 20000;
  std::vector<float> unit = Gaussian(kN, kDim, 1, 1.0f);
  Normalize(unit, kDim);
  auto stats = research_scann::ComputeAutopilotDataStats(unit, kN, kDim);
  Expect(Near(stats.norm_quantile, 1.0, 1e-5) && stats.offset_ratio < 0.1,
         absl::StrCat("stats of unit vectors: ", stats.norm_quantile, " ",
                      stats.offset_ratio));
  std::vector<float> offset = Gaussian(kN, kDim, 2, 1.0f, 10.0f);
  stats = research_scann::ComputeAutopilotDataStats(offset, kN, kDim);
  Expect(stats.offset_ratio > 50, absl::StrCat("offset ratio ", stats.offset_ratio));
  std::vector<float> same(kN * kDim, 1.0f);
  stats = research_scann::ComputeAutopilotDataStats(same, kN, kDim);
  Expect(std::isinf(stats.offset_ratio), "identical points: infinite offset ratio");

  const ScannConfig config = TunedConfig(10, "DotProductDistance");
  auto run = [&](const std::vector<float>& data) {
    auto ds = std::make_shared<DenseDataset<float>>(
        std::vector<float>(data), kN);
    auto r = research_scann::Autopilot(config, ds);
    Ok(r.status(), "Autopilot with data");
    return r.ok() ? *r : ScannConfig();
  };
  const double t_unit = 0.2 * std::pow(128.0 / kDim, 0.75);
  const ScannConfig u = run(unit);
  Expect(Near(u.hash().asymmetric_hash().noise_shaping_threshold(), t_unit, 1e-5) &&
             Near(u.autopilot().tree_ah().noise_shaping_threshold(), t_unit, 1e-5) &&
             DimsPerBlock(u) == 2 && u.partitioning().num_children() == 45,
         absl::StrCat("unit vectors:\n", u.DebugString()));
  // Norm 7: the threshold scales with it.
  std::vector<float> seven = unit;
  for (float& x : seven) x *= 7.0f;
  const ScannConfig s = run(seven);
  Expect(Near(s.hash().asymmetric_hash().noise_shaping_threshold(), 7 * t_unit, 1e-5),
         absl::StrCat("norm 7:\n", s.DebugString()));
  // Norms from 1 to 20: relative to the 5th percentile.
  std::vector<float> spread = unit;
  for (size_t i = 0; i < kN; ++i)
    for (size_t j = 0; j < kDim; ++j)
      spread[i * kDim + j] *= 1.0f + 19.0f * i / kN;
  const ScannConfig sp = run(spread);
  Expect(Near(sp.hash().asymmetric_hash().noise_shaping_threshold(),
              (1 + 19 * 0.05) * t_unit, 1e-2),
         absl::StrCat("norms 1-20:\n", sp.DebugString()));
  // Recorded decisions are reused: the recorded config without data (as when
  // reloading an index whose dataset is bfloat16) gives the same config.
  auto again = research_scann::Autopilot(s, nullptr, kN, kDim);
  Expect(again.ok() && again->DebugString() == s.DebugString(),
         "recorded threshold reused without data");
  // A given threshold is kept; NaN turns it off.
  ScannConfig given = config;
  given.mutable_autopilot()->mutable_tree_ah()->set_noise_shaping_threshold(0.5);
  auto g = research_scann::Autopilot(given, nullptr, kN, kDim);
  Expect(g.ok() && g->hash().asymmetric_hash().noise_shaping_threshold() == 0.5,
         "given threshold");
  given.mutable_autopilot()->mutable_tree_ah()->set_noise_shaping_threshold(std::nan(""));
  g = research_scann::Autopilot(given, nullptr, kN, kDim);
  Expect(g.ok() && std::isnan(g->hash().asymmetric_hash().noise_shaping_threshold()),
         "threshold turned off");
  // Zero vectors: no threshold.
  std::vector<float> zeros(kN * kDim, 0.0f);
  const ScannConfig z = run(zeros);
  Expect(std::isnan(z.hash().asymmetric_hash().noise_shaping_threshold()) &&
             z.autopilot().tree_ah().has_noise_shaping_threshold(),
         absl::StrCat("zero vectors:\n", z.DebugString()));

  // AutopilotChoosesL2AsDotProduct.
  // Only for nearly constant norms (squared norms' coefficient of variation
  // at most 0.05) near the origin.
  const ScannConfig l2 = TunedConfig(10, "SquaredL2Distance");
  std::vector<float> centered = Gaussian(kN, kDim, 3, 1.0f, 0.5f);
  stats = research_scann::ComputeAutopilotDataStats(centered, kN, kDim);
  // N(0.5, 1) coordinates: |x|^2 has mean 1.25 d and variance 3 d.
  Expect(Near(stats.squared_norm_cv, std::sqrt(3.0 * kDim) / (1.25 * kDim), 0.05),
         absl::StrCat("squared norms' variation ", stats.squared_norm_cv));
  Expect(!research_scann::AutopilotChoosesL2AsDotProduct(l2, centered, kN),
         "no l2_as_dot_product with varying norms");
  Normalize(centered, kDim, 10.0f);
  Expect(research_scann::AutopilotChoosesL2AsDotProduct(l2, centered, kN),
         "l2_as_dot_product with constant norms");
  Expect(!research_scann::AutopilotChoosesL2AsDotProduct(l2, offset, kN),
         "no l2_as_dot_product far from the origin");
  Expect(!research_scann::AutopilotChoosesL2AsDotProduct(
             l2, research_scann::ConstSpan<float>(centered.data(), 8000 * kDim), 8000),
         "no l2_as_dot_product below the brute-force cutoff");
  ScannConfig no = l2;
  no.mutable_autopilot()->mutable_tree_ah()->set_allow_l2_as_dot_product(false);
  Expect(!research_scann::AutopilotChoosesL2AsDotProduct(no, centered, kN),
         "allow_l2_as_dot_product: false");
  no = l2;
  no.mutable_autopilot()->mutable_tree_ah()->clear_rules();
  Expect(!research_scann::AutopilotChoosesL2AsDotProduct(no, centered, kN),
         "no l2_as_dot_product with upstream's rules");
  Expect(!research_scann::AutopilotChoosesL2AsDotProduct(
             TunedConfig(10, "DotProductDistance"), centered, kN),
         "no l2_as_dot_product for dot product");
}

DatapointPtr<float> Ptr(const float* v, size_t dim) {
  return DatapointPtr<float>(nullptr, v, dim, dim);
}

// Recall@k of `s` at its default settings against brute force.
double Recall(ScannInterface& s, const std::vector<float>& data,
              const std::vector<float>& queries, size_t dim, int k, bool l2) {
  const size_t n = data.size() / dim, nq = queries.size() / dim;
  double hits = 0;
  for (size_t i = 0; i < nq; ++i) {
    const float* q = queries.data() + i * dim;
    std::vector<std::pair<double, size_t>> all(n);
    for (size_t j = 0; j < n; ++j) {
      double v = 0;
      for (size_t t = 0; t < dim; ++t) {
        const double a = q[t], b = data[j * dim + t];
        v += l2 ? (a - b) * (a - b) : -a * b;
      }
      all[j] = {v, j};
    }
    std::partial_sort(all.begin(), all.begin() + k, all.end());
    NNResultsVector res;
    if (!Ok(s.Search(Ptr(q, dim), &res, k, -1, -1), "search")) return 0;
    for (const auto& [idx, dist] : res)
      for (int t = 0; t < k; ++t)
        if (all[t].second == idx) ++hits;
  }
  return hits / (nq * k);
}

std::vector<std::pair<research_scann::DatapointIndex, float>> Ids(
    ScannInterface& s, const std::vector<float>& queries, size_t dim, int k) {
  std::vector<std::pair<research_scann::DatapointIndex, float>> out;
  for (size_t i = 0; i < queries.size() / dim; ++i) {
    NNResultsVector res;
    Ok(s.Search(Ptr(queries.data() + i * dim, dim), &res, k, -1, -1), "search");
    for (const auto& r : res) out.push_back(r);
  }
  return out;
}

fs::path TestDir(const std::string& name) {
  const char* tmp = std::getenv("TEST_TMPDIR");
  fs::path dir = fs::path(tmp ? tmp : fs::temp_directory_path().string()) /
                 absl::StrCat("scann_core_autopilot_test_", ::getpid(), "_", name);
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

// Builds, checks, serializes, reloads and retrains one autopilot index.
void EndToEnd(const std::string& name, DistanceMeasure d,
              const AutopilotOptions& o, const std::vector<float>& data,
              const std::vector<float>& queries, size_t dim,
              void (*check)(const ScannConfig&, const std::string&)) {
  constexpr int kK = 10;
  const size_t n = data.size() / dim;
  ConfigBuilder b(kK, d, dim);
  b.Autopilot(o);
  auto text = b.BuildText(n);
  if (!Ok(text.status(), name + ": config")) return;
  ScannInterface s;
  if (!Ok(s.Initialize(data, n, *text, 4), name + ": build")) return;
  const ScannConfig built = *s.config();
  check(built, name);
  const bool l2 = d == DistanceMeasure::kSquaredL2;
  const double recall = Recall(s, data, queries, dim, kK, l2);
  std::printf("%s: recall@%d at the default settings %.4f\n", name.c_str(), kK,
              recall);
  if (recall < 0.9) Fail(absl::StrCat(name, ": recall ", recall, " < 0.9"));
  const auto before = Ids(s, queries, dim, kK);

  const fs::path dir = TestDir(name);
  if (!Ok(s.SerializeToDirectory(dir.string(), true), name + ": serialize")) return;
  auto artifacts = ScannInterface::LoadArtifacts(dir.string());
  if (!Ok(artifacts.status(), name + ": LoadArtifacts")) return;
  ScannInterface loaded;
  if (!Ok(loaded.Initialize(*artifacts), name + ": load")) return;
  Expect(loaded.config()->DebugString() == built.DebugString(),
         absl::StrCat(name, ": reloaded config\n", loaded.config()->DebugString(),
                      "\nbuilt\n", built.DebugString()));
  Expect(Ids(loaded, queries, dim, kK) == before, name + ": reloaded results");
  auto retrained = loaded.RetrainAndReindex("");
  if (Ok(retrained.status(), name + ": retrain")) {
    Expect(retrained->DebugString() == built.DebugString(),
           absl::StrCat(name, ": retrained config\n", retrained->DebugString()));
    const double r2 = Recall(loaded, data, queries, dim, kK, l2);
    if (r2 < 0.9) Fail(absl::StrCat(name, ": recall after retraining ", r2));
  }
  fs::remove_all(dir);
}

void TestEndToEnd() {
  // 200 dimensions: leaves of 655 points upstream, so a tree from 27,510.
  constexpr size_t kDim = 200, kN = 30000, kQ = 50;
  std::mt19937 rng(7);
  // Clustered data, so that a tree finds neighbours.
  std::vector<float> centers = Gaussian(200, kDim, 8, 1.0f);
  auto clustered = [&](size_t n, uint32_t seed) {
    std::vector<float> noise = Gaussian(n, kDim, seed, 0.3f);
    std::uniform_int_distribution<size_t> pick(0, 199);
    std::mt19937 r(seed + 1);
    for (size_t i = 0; i < n; ++i) {
      const size_t c = pick(r);
      for (size_t j = 0; j < kDim; ++j) noise[i * kDim + j] += centers[c * kDim + j];
    }
    return noise;
  };
  // Constant norms (like SIFT's), so that the tuned rules choose
  // l2_as_dot_product for squared L2.
  std::vector<float> data = clustered(kN, 9), queries = clustered(kQ, 10);
  Normalize(data, kDim, 10.0f);
  Normalize(queries, kDim, 10.0f);

  EndToEnd("squared_l2", DistanceMeasure::kSquaredL2, {}, data, queries, kDim,
           [](const ScannConfig& c, const std::string& what) {
             Expect(c.has_l2_as_dot_product() &&
                        c.distance_measure().distance_measure() ==
                            "SquaredL2Distance" &&
                        c.hash().asymmetric_hash().projection().input_dim() ==
                            kDim + 1 &&
                        c.partitioning().avq() == 2.5f &&
                        c.partitioning().num_children() == 30000 / 652 &&
                        c.autopilot().tree_ah().has_noise_shaping_threshold() &&
                        !c.exact_reordering().bfloat16().enabled(),
                    absl::StrCat(what, ": built config\n", c.DebugString()));
           });
  AutopilotOptions plain;
  plain.allow_l2_as_dot_product = false;
  EndToEnd("squared_l2_plain", DistanceMeasure::kSquaredL2, plain, data,
           queries, kDim, [](const ScannConfig& c, const std::string& what) {
             Expect(!c.has_l2_as_dot_product() &&
                        std::isnan(c.hash().asymmetric_hash().noise_shaping_threshold()) &&
                        std::isnan(c.partitioning().avq()),
                    absl::StrCat(what, ": built config\n", c.DebugString()));
           });
  std::vector<float> unit = data, unit_q = queries;
  Normalize(unit, kDim, 3.0f);  // norm 3: threshold 3 * 0.2 (128 / 200)^0.75
  Normalize(unit_q, kDim);
  EndToEnd("dot_product", DistanceMeasure::kDotProduct, {}, unit, unit_q, kDim,
           [](const ScannConfig& c, const std::string& what) {
             const double t = 3 * 0.2 * std::pow(128.0 / kDim, 0.75);
             Expect(!c.has_l2_as_dot_product() &&
                        Near(c.hash().asymmetric_hash().noise_shaping_threshold(), t, 1e-4) &&
                        Near(c.autopilot().tree_ah().noise_shaping_threshold(), t, 1e-4) &&
                        c.partitioning().num_children() == 30000 / 655,
                    absl::StrCat(what, ": built config\n", c.DebugString()));
           });
  // Varying norms: plain squared L2.
  EndToEnd("squared_l2_varying_norms", DistanceMeasure::kSquaredL2, {},
           clustered(kN, 11), clustered(kQ, 12), kDim,
           [](const ScannConfig& c, const std::string& what) {
             Expect(!c.has_l2_as_dot_product(),
                    absl::StrCat(what, ": built config\n", c.DebugString()));
           });
  AutopilotOptions bf16;
  bf16.quantize = Quantization::kBfloat16;
  EndToEnd("squared_l2_bfloat16", DistanceMeasure::kSquaredL2, bf16, data,
           queries, kDim, [](const ScannConfig& c, const std::string& what) {
             Expect(c.has_l2_as_dot_product() &&
                        c.exact_reordering().bfloat16().enabled(),
                    absl::StrCat(what, ": built config\n", c.DebugString()));
           });
  AutopilotOptions up;
  up.rules = AutopilotRules::kUpstream;
  EndToEnd("dot_product_upstream", DistanceMeasure::kDotProduct, up, unit,
           unit_q, kDim, [](const ScannConfig& c, const std::string& what) {
             Expect(!c.autopilot().tree_ah().has_rules() &&
                        c.hash().asymmetric_hash().noise_shaping_threshold() == 0.2 &&
                        c.partitioning().num_children() == kN / 655 &&
                        !c.exact_reordering().bfloat16().enabled(),
                    absl::StrCat(what, ": built config\n", c.DebugString()));
           });
}

}  // namespace

int main() {
  TestTunedPreviews();
  TestUpstreamRules();
  TestDataDependence();
  TestEndToEnd();
  std::printf("autopilot: %s (%d failure(s))\n", g_failures ? "FAILED" : "PASSED",
              g_failures);
  return g_failures ? 1 : 0;
}
