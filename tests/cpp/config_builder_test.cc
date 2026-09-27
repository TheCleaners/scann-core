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

// Replays the builder calls recorded by tests/config_builder/make_expected.py
// through scann_core::ConfigBuilder and requires the same ScannConfig as
// upstream's Python ScannBuilder produced. Also checks the cases where the
// C++ builder intentionally reports an error instead of silently ignoring
// options.
//
// Usage: scann_core_config_builder_test <dir written by make_expected.py>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/field_comparator.h"
#include "google/protobuf/util/message_differencer.h"
#include "scann_core/config_builder.h"

namespace {

using scann_core::ConfigBuilder;
using scann_core::DistanceMeasure;
using scann_core::HashType;
using scann_core::IncrementalMode;
using scann_core::Quantization;

int g_failures = 0;
void Fail(const std::string& msg) {
  std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
  ++g_failures;
}

using Args = std::map<std::string, std::string>;

bool IsFloat(const std::string& v) { return !v.empty() && v.back() == 'f'; }
double F(const std::string& v) { return std::strtod(v.c_str(), nullptr); }  // "0.2f" -> 0.2
int64_t I(const std::string& v) { return std::strtoll(v.c_str(), nullptr, 10); }
bool B(const std::string& v) { return v == "true"; }
Quantization Q(const std::string& v) {
  return v == "INT8" ? Quantization::kInt8
         : v == "BFLOAT16" ? Quantization::kBfloat16 : Quantization::kFloat32;
}

void Apply(ConfigBuilder& b, const std::string& method, const Args& a) {
  auto has = [&](const char* k) { return a.count(k) && a.at(k) != "None"; };
  if (method == "score_brute_force") {
    b.ScoreBruteForce(has("quantize") ? Q(a.at("quantize")) : Quantization::kFloat32);
  } else if (method == "score_ah") {
    scann_core::AhOptions o;
    o.dimensions_per_block = I(a.at("dimensions_per_block"));
    if (has("anisotropic_quantization_threshold"))
      o.anisotropic_quantization_threshold = F(a.at("anisotropic_quantization_threshold"));
    if (has("training_sample_size")) o.training_sample_size = I(a.at("training_sample_size"));
    if (has("training_iterations")) o.training_iterations = I(a.at("training_iterations"));
    if (has("hash_type"))
      o.hash_type = a.at("hash_type") == "lut256" ? HashType::kLut256 : HashType::kLut16;
    if (has("residual_quantization")) o.residual_quantization = B(a.at("residual_quantization"));
    b.ScoreAh(o);
  } else if (method == "reorder") {
    scann_core::ReorderOptions o;
    o.reordering_num_neighbors = I(a.at("reordering_num_neighbors"));
    if (has("quantize")) o.quantize = Q(a.at("quantize"));
    if (has("anisotropic_quantization_threshold"))
      o.anisotropic_quantization_threshold = F(a.at("anisotropic_quantization_threshold"));
    b.Reorder(o);
  } else if (method == "tree") {
    scann_core::TreeOptions o;
    o.num_leaves = I(a.at("num_leaves"));
    o.num_leaves_to_search = I(a.at("num_leaves_to_search"));
    if (has("training_sample_size")) o.training_sample_size = I(a.at("training_sample_size"));
    if (has("min_partition_size")) o.min_partition_size = I(a.at("min_partition_size"));
    if (has("training_iterations")) o.training_iterations = I(a.at("training_iterations"));
    if (has("spherical")) o.spherical = B(a.at("spherical"));
    if (has("quantize_centroids")) o.quantize_centroids = B(a.at("quantize_centroids"));
    if (has("random_init")) o.random_init = B(a.at("random_init"));
    if (has("incremental_threshold")) {
      const std::string& v = a.at("incremental_threshold");
      if (IsFloat(v)) o.incremental_threshold_fraction = F(v);
      else o.incremental_threshold_points = I(v);
    }
    if (has("avq")) o.avq = F(a.at("avq"));
    if (has("soar_lambda")) o.soar_lambda = F(a.at("soar_lambda"));
    if (has("overretrieve_factor")) o.overretrieve_factor = F(a.at("overretrieve_factor"));
    b.Tree(o);
  } else if (method == "upper_tree") {
    scann_core::UpperTreeOptions o;
    o.num_leaves = I(a.at("num_leaves"));
    o.num_leaves_to_search = I(a.at("num_leaves_to_search"));
    if (has("avq")) o.avq = F(a.at("avq"));
    if (has("soar_lambda")) o.soar_lambda = F(a.at("soar_lambda"));
    if (has("overretrieve_factor")) o.overretrieve_factor = F(a.at("overretrieve_factor"));
    if (has("scoring_mode")) o.scoring_mode = Q(a.at("scoring_mode"));
    if (has("anisotropic_quantization_threshold"))
      o.anisotropic_quantization_threshold = F(a.at("anisotropic_quantization_threshold"));
    b.UpperTree(o);
  } else if (method == "pca") {
    scann_core::PcaOptions o;
    if (has("reduction_dim")) o.reduction_dim = I(a.at("reduction_dim"));
    if (has("pca_significance_threshold"))
      o.pca_significance_threshold = F(a.at("pca_significance_threshold"));
    if (has("pca_truncation_threshold"))
      o.pca_truncation_threshold = F(a.at("pca_truncation_threshold"));
    b.Pca(o);
  } else if (method == "truncate") {
    b.Truncate(I(a.at("reduction_dim")));
  } else if (method == "autopilot") {
    IncrementalMode m = IncrementalMode::kNone;
    if (has("mode"))
      m = a.at("mode") == "ONLINE" ? IncrementalMode::kOnline
          : a.at("mode") == "ONLINE_INCREMENTAL" ? IncrementalMode::kOnlineIncremental
                                                 : IncrementalMode::kNone;
    b.Autopilot(m, has("quantize") ? Q(a.at("quantize")) : Quantization::kFloat32);
  } else {
    Fail("unknown method " + method);
  }
}

std::string ReadFile(const std::string& p) {
  std::ifstream f(p);
  return std::string(std::istreambuf_iterator<char>(f), {});
}

void ExpectError(const char* what, const std::function<void(ConfigBuilder&)>& setup) {
  ConfigBuilder b(10, DistanceMeasure::kDotProduct, 128);
  setup(b);
  auto r = b.Build(5000);
  if (r.ok()) Fail(absl::StrCat("expected an error: ", what));
  else std::printf("ok (error as intended): %s -> %s\n", what,
                   std::string(r.status().message()).c_str());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <expected dir>\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  std::istringstream in(ReadFile(dir + "/cases.txt"));
  google::protobuf::util::DefaultFieldComparator comparator;
  comparator.set_treat_nan_as_equal(true);
  google::protobuf::util::MessageDifferencer differencer;
  differencer.set_field_comparator(&comparator);

  int cases = 0;
  std::string line, name;
  std::unique_ptr<ConfigBuilder> builder;
  uint64_t num_points = 0;
  while (std::getline(in, line)) {
    std::vector<std::string> tok = absl::StrSplit(line, ' ', absl::SkipEmpty());
    if (tok.empty()) continue;
    if (tok[0] == "case") {
      name = tok[1];
      builder = std::make_unique<ConfigBuilder>(
          std::atoi(tok[2].c_str()),
          tok[3] == "dot_product" ? DistanceMeasure::kDotProduct : DistanceMeasure::kSquaredL2,
          std::atoi(tok[4].c_str()));
      num_points = std::strtoull(tok[5].c_str(), nullptr, 10);
    } else if (tok[0] == "call") {
      Args a;
      for (size_t i = 2; i < tok.size(); ++i) {
        std::vector<std::string> kv = absl::StrSplit(tok[i], absl::MaxSplits('=', 1));
        a[kv[0]] = kv[1];
      }
      Apply(*builder, tok[1], a);
    } else if (tok[0] == "expected") {
      ++cases;
      research_scann::ScannConfig want;
      if (!google::protobuf::TextFormat::ParseFromString(ReadFile(dir + "/" + tok[1]), &want)) {
        Fail(name + ": expected config does not parse");
        continue;
      }
      auto got = builder->Build(num_points);
      if (!got.ok()) {
        Fail(absl::StrCat(name, ": Build failed: ", got.status().ToString()));
        continue;
      }
      std::string report;
      differencer.ReportDifferencesToString(&report);
      if (!differencer.Compare(want, *got))
        Fail(absl::StrCat(name, ": differs from the Python builder:\n", report));
    }
  }

  // Where the Python builder silently ignores or drops options, the C++
  // builder reports an error.
  scann_core::TreeOptions tree;
  tree.num_leaves = 64;
  tree.num_leaves_to_search = 8;
  scann_core::AhOptions ah;
  ah.dimensions_per_block = 2;
  ExpectError("upper_tree without tree", [&](ConfigBuilder& b) {
    b.UpperTree({40, 10}).ScoreAh(ah);
  });
  ExpectError("pca without tree", [&](ConfigBuilder& b) {
    b.Pca({}).ScoreAh(ah);
  });
  ExpectError("autopilot combined with tree", [&](ConfigBuilder& b) {
    b.Autopilot().Tree(tree);
  });
  ExpectError("neither score_ah nor score_brute_force", [&](ConfigBuilder& b) { b.Tree(tree); });
  ExpectError("both score_ah and score_brute_force", [&](ConfigBuilder& b) {
    b.ScoreAh(ah).ScoreBruteForce();
  });
  ExpectError("tree configured twice", [&](ConfigBuilder& b) {
    b.Tree(tree).Tree(tree).ScoreAh(ah);
  });
  ExpectError("avq with squared_l2", [&](ConfigBuilder& b) {
    b = ConfigBuilder(10, DistanceMeasure::kSquaredL2, 128);
    auto t = tree;
    t.avq = 1.0;
    b.Tree(t).ScoreAh(ah);
  });
  ExpectError("truncate to >= dimensionality", [&](ConfigBuilder& b) {
    b.Truncate(128).Tree(tree).ScoreAh(ah);
  });

  // Values the Python builder passes through, and ScaNN then mishandles
  // (a search error on every query, an abort, fewer results than asked).
  ExpectError("num_leaves_to_search = 0", [&](ConfigBuilder& b) {
    auto t = tree;
    t.num_leaves_to_search = 0;
    b.Tree(t).ScoreAh(ah);
  });
  ExpectError("num_leaves unset (0)", [&](ConfigBuilder& b) {
    b.Tree(scann_core::TreeOptions{}).ScoreAh(ah);
  });
  ExpectError("num_neighbors = 0", [&](ConfigBuilder& b) {
    b = ConfigBuilder(0, DistanceMeasure::kDotProduct, 128);
    b.ScoreBruteForce();
  });
  ExpectError("reorder fewer than num_neighbors", [&](ConfigBuilder& b) {
    b.ScoreAh(ah).Reorder({5});
  });
  ExpectError("truncate to 0", [&](ConfigBuilder& b) {
    b.Truncate(0).Tree(tree).ScoreAh(ah);
  });
  ExpectError("pca reduction_dim = 0", [&](ConfigBuilder& b) {
    scann_core::PcaOptions p;
    p.reduction_dim = 0;
    b.Pca(p).Tree(tree).ScoreAh(ah);
  });
  ExpectError("pca reduction_dim > dimensionality", [&](ConfigBuilder& b) {
    scann_core::PcaOptions p;
    p.reduction_dim = 256;
    b.Pca(p).Tree(tree).ScoreAh(ah);
  });
  ExpectError("dimensions_per_block > dimensionality", [&](ConfigBuilder& b) {
    auto a = ah;
    a.dimensions_per_block = 129;
    b.ScoreAh(a);
  });
  ExpectError("residual quantization without tree", [&](ConfigBuilder& b) {
    auto a = ah;
    a.residual_quantization = true;
    b.ScoreAh(a);
  });
  ExpectError("upper_tree num_leaves_to_search = 0", [&](ConfigBuilder& b) {
    b.Tree(tree).UpperTree({40, 0}).ScoreAh(ah);
  });

  // soar_lambda = 0.0 on the upper tree is kept (Python turns it into 1.5).
  {
    ConfigBuilder b(10, DistanceMeasure::kDotProduct, 128);
    scann_core::UpperTreeOptions u{40, 10};
    u.soar_lambda = 0.0;
    auto c = b.Tree(tree).UpperTree(u).ScoreAh(ah).Build();
    if (!c.ok()) Fail("upper_tree soar_lambda=0: " + c.status().ToString());
    else if (c->partitioning().bottom_up_top_level_partitioner().soar().lambda() != 0.0f)
      Fail("upper_tree soar_lambda=0 was not preserved");
  }

  std::printf("%d cases compared against the Python builder; %s: %d failure(s)\n",
              cases, g_failures ? "FAILED" : "PASSED", g_failures);
  return g_failures ? 1 : 0;
}
