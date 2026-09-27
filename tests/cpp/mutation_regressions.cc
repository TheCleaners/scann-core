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

// Regression tests for mutation defects in upstream ScaNN:
//
//  - A RetrainAndReindex (rebalance) that fails must leave the index exactly
//    as it was. Upstream replaced the live searcher's dataset and docids
//    before building the new one, so after a failure the old searcher's
//    cached mutator used a freed docid collection (heap-use-after-free on the
//    next add or delete). Two triggers: deleting down to fewer points than
//    leaves, and a config with more children than points.
//  - A tree with bfloat16 brute-force leaves must support add, update and
//    delete. Upstream's leaves had no docids, so every mutation failed a
//    RET_CHECK after partially changing the index.
//  - An update of a datapoint in a tree (with SOAR, in two leaves) whose leaf
//    mutation fails must leave the index unchanged. Upstream updated the base,
//    the health stats and the datapoint's leaf assignments step by step and
//    returned at the first failing leaf, leaving the datapoint half-updated
//    (e.g. its old spill assignment removed and the new one never added).
//    After the input validation in ScannNumpy::Upsert nothing reachable from
//    Python fails there, so the test injects the failure: it hands one leaf a
//    precomputed artifact of the wrong type, which that leaf rejects.
//
// Each case keeps a shadow copy of the stored vectors (deletes move the last
// datapoint into the freed index, as the index does) and checks that
// searching for each stored vector, over all leaves, returns its index.
// Run it under ASan to catch the use-after-free directly.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "google/protobuf/text_format.h"
#include "scann/data_format/datapoint.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/tree_x_hybrid/mutator.h"
#include "scann/tree_x_hybrid/tree_ah_hybrid_residual.h"
#include "scann/utils/types.h"
#include "scann_core/config_builder.h"

namespace {

using research_scann::ConstSpan;
using research_scann::DatapointIndex;
using research_scann::DatapointPtr;
using research_scann::NNResultsVector;
using research_scann::ScannInterface;
using Mutator = research_scann::SingleMachineSearcherBase<float>::Mutator;
using MutationOptions =
    research_scann::UntypedSingleMachineSearcherBase::MutationOptions;
using PrecomputedMutationArtifacts = research_scann::
    UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts;
using TreeAhArtifacts = research_scann::TreeXHybridMutator<
    research_scann::TreeAHHybridResidual>::TreeXPrecomputedMutationArtifacts;

int g_failures = 0;

void Fail(const std::string& what) {
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++g_failures;
}

bool Ok(const absl::Status& s, const std::string& what) {
  if (!s.ok()) Fail(absl::StrCat(what, ": ", s.ToString()));
  return s.ok();
}

constexpr size_t kDim = 8;
constexpr int kNumLeaves = 12;

using Vec = std::vector<float>;

DatapointPtr<float> Ptr(const Vec& v) {
  return DatapointPtr<float>(nullptr, v.data(), v.size(), v.size());
}

struct Harness {
  std::string name;
  ScannInterface s;
  std::vector<Vec> shadow;  // shadow[i] = vector stored at datapoint index i
  std::mt19937 rng{7};
  // For dot product, store unit vectors: each is then its own nearest
  // neighbor.
  bool unit_norm = false;

  Vec Random() {
    std::normal_distribution<float> nd;
    Vec v(kDim);
    for (float& x : v) x = nd(rng);
    if (unit_norm) {
      float norm = 0;
      for (float x : v) norm += x * x;
      for (float& x : v) x /= std::sqrt(norm);
    }
    return v;
  }

  bool Build(const std::string& config, size_t n) {
    std::vector<float> flat;
    for (size_t i = 0; i < n; ++i) {
      shadow.push_back(Random());
      flat.insert(flat.end(), shadow.back().begin(), shadow.back().end());
    }
    return Ok(s.Initialize(ConstSpan<float>(flat), n, config, 1),
              name + ": Initialize");
  }

  Mutator* GetMutator(const std::string& what) {
    auto m = s.GetMutator();
    if (!Ok(m.status(), absl::StrCat(name, ": GetMutator (", what, ")")))
      return nullptr;
    return *m;
  }

  void Add(size_t count) {
    auto* m = GetMutator("add");
    if (!m) return;
    for (size_t i = 0; i < count; ++i) {
      Vec v = Random();
      auto idx = m->AddDatapoint(Ptr(v), "");
      if (!Ok(idx.status(), name + ": AddDatapoint")) return;
      if (*idx != shadow.size())
        Fail(absl::StrCat(name, ": AddDatapoint returned ", *idx, ", expected ",
                          shadow.size()));
      shadow.push_back(std::move(v));
    }
  }

  void Remove(DatapointIndex idx) {
    auto* m = GetMutator("remove");
    if (!m) return;
    if (!Ok(m->RemoveDatapoint(idx), absl::StrCat(name, ": RemoveDatapoint ", idx)))
      return;
    shadow[idx] = shadow.back();
    shadow.pop_back();
  }

  void Update(DatapointIndex idx) {
    auto* m = GetMutator("update");
    if (!m) return;
    Vec v = Random();
    auto r = m->UpdateDatapoint(Ptr(v), idx);
    if (!Ok(r.status(), absl::StrCat(name, ": UpdateDatapoint ", idx))) return;
    shadow[idx] = std::move(v);
  }

  // Every stored vector is found at its own index, and no result is out of
  // range.
  void Verify(const std::string& when) {
    const std::string what = absl::StrCat(name, " (", when, ")");
    if (s.n_points() != shadow.size()) {
      Fail(absl::StrCat(what, ": n_points ", s.n_points(), " != ",
                        shadow.size()));
      return;
    }
    size_t misses = 0;
    for (size_t i = 0; i < shadow.size(); ++i) {
      NNResultsVector res;
      if (!Ok(s.Search(Ptr(shadow[i]), &res, 3, -1, kNumLeaves),
              what + ": Search"))
        return;
      for (const auto& [idx, dist] : res) {
        if (idx >= shadow.size()) {
          Fail(absl::StrCat(what, ": result index ", idx, " >= size ",
                            shadow.size()));
          return;
        }
      }
      if (res.empty() || res[0].first != i) ++misses;
    }
    if (misses)
      Fail(absl::StrCat(what, ": ", misses, " of ", shadow.size(),
                        " stored vectors not found at their index"));
  }

  // Mixed adds, deletes (first, middle, last) and updates, then Verify.
  void Churn(const std::string& when) {
    Add(40);
    Remove(0);
    Remove(static_cast<DatapointIndex>(shadow.size() / 2));
    Remove(static_cast<DatapointIndex>(shadow.size() - 1));
    for (DatapointIndex i = 0; i < shadow.size(); i += 7) Update(i);
    for (int i = 0; i < 10; ++i)
      Remove(static_cast<DatapointIndex>((i * 13) % shadow.size()));
    Add(5);
    Verify(when);
  }
};

using Setup = std::function<void(scann_core::ConfigBuilder&)>;

std::string Config(const Setup& setup, size_t n,
                   scann_core::DistanceMeasure distance =
                       scann_core::DistanceMeasure::kSquaredL2,
                   std::optional<double> soar_lambda = std::nullopt) {
  scann_core::ConfigBuilder b(10, distance, kDim);
  scann_core::TreeOptions tree;
  tree.num_leaves = kNumLeaves;
  tree.num_leaves_to_search = 4;
  tree.training_sample_size = static_cast<int64_t>(n);
  tree.random_init = false;  // reproducible
  tree.soar_lambda = soar_lambda;
  b.Tree(tree);
  setup(b);
  auto text = b.BuildText(n);
  if (!Ok(text.status(), "ConfigBuilder")) return "";
  return *text;
}

void BruteForce(scann_core::ConfigBuilder& b) { b.ScoreBruteForce(); }

void AhReorder(scann_core::ConfigBuilder& b) {
  scann_core::AhOptions ah;
  scann_core::ReorderOptions reorder;
  reorder.reordering_num_neighbors = 40;
  b.ScoreAh(ah).Reorder(reorder);
}

void Bfloat16(scann_core::ConfigBuilder& b) {
  b.ScoreBruteForce(scann_core::Quantization::kBfloat16);
}

// Trigger 1: delete down to fewer points than leaves, then rebalance.
void FailedRetrainTooFewPoints(
    const std::string& name,
    const Setup& setup) {
  Harness h;
  h.name = name + "/too_few_points";
  std::printf("== %s\n", h.name.c_str());
  const size_t n = 600;
  if (!h.Build(Config(setup, n), n)) return;
  while (h.shadow.size() > 3)
    h.Remove(static_cast<DatapointIndex>(h.shadow.size() - 1));
  h.Verify("after deletes");
  if (h.s.RetrainAndReindex("").ok())
    Fail(h.name + ": RetrainAndReindex with 3 points and 12 leaves succeeded");
  h.Verify("after failed retrain");
  h.Remove(0);
  h.Add(100);
  h.Update(1);
  h.Remove(5);
  h.Verify("after mutating the kept searcher");
  if (Ok(h.s.RetrainAndReindex("").status(), h.name + ": RetrainAndReindex"))
    h.Churn("after a successful retrain");
}

// Trigger 2: a config with more children than points.
void FailedRetrainBadConfig(
    const std::string& name,
    const Setup& setup) {
  Harness h;
  h.name = name + "/bad_config";
  std::printf("== %s\n", h.name.c_str());
  const size_t n = 600;
  if (!h.Build(Config(setup, n), n)) return;
  h.Churn("before retrain");
  research_scann::ScannConfig bad = *h.s.config();
  bad.mutable_partitioning()->set_num_children(5000);
  std::string bad_text;
  google::protobuf::TextFormat::PrintToString(bad, &bad_text);
  if (h.s.RetrainAndReindex(bad_text).ok())
    Fail(h.name + ": RetrainAndReindex with 5000 children succeeded");
  h.Verify("after failed retrain");
  h.Churn("after failed retrain, mutated");
  if (Ok(h.s.RetrainAndReindex("").status(), h.name + ": RetrainAndReindex"))
    h.Churn("after a successful retrain");
}

// Tree + bfloat16 brute-force leaves: mutate, then retrain and mutate again.
// With SOAR, a datapoint lives in two leaves, so a mutation touches several
// leaf mutators.
//
// Not covered: serialization. Upstream serializes neither the float dataset
// (released after the build, as the bfloat16 leaves don't need it) nor the
// leaves' bfloat16 data for this configuration, so the artifacts don't load;
// for the same reason RetrainAndReindex fails (cleanly) with
// FAILED_PRECONDITION, which this checks leaves the index unchanged.
void TreeBfloat16(const std::string& name, scann_core::DistanceMeasure distance,
                  std::optional<double> soar_lambda) {
  Harness h;
  h.name = name;
  h.unit_norm = distance == scann_core::DistanceMeasure::kDotProduct;
  std::printf("== %s\n", h.name.c_str());
  const size_t n = 600;
  if (!h.Build(Config(Bfloat16, n, distance, soar_lambda), n)) return;
  h.Verify("built");
  h.Churn("mutated");
  for (int round = 0; round < 2; ++round) {
    auto retrained = h.s.RetrainAndReindex("");
    if (!retrained.ok())
      std::printf("   RetrainAndReindex: %s\n",
                  retrained.status().ToString().c_str());
    h.Verify(retrained.ok() ? "retrained" : "after failed retrain");
    h.Churn(retrained.ok() ? "retrained, mutated"
                           : "after failed retrain, mutated");
  }
  // Delete almost everything, then grow again.
  while (h.shadow.size() > 2)
    h.Remove(static_cast<DatapointIndex>(h.shadow.size() / 2));
  h.Verify("after deleting all but 2");
  h.Churn("regrown");
}

// A precomputed artifact no AH leaf accepts: they require their own type and
// reject anything else with InvalidArgumentError.
struct BogusArtifacts : PrecomputedMutationArtifacts {};

// What searches and health stats can observe: for each probe query its
// neighbors over all leaves (by index), and the stats.
struct Snapshot {
  size_t n_points = 0;
  std::vector<std::vector<std::pair<DatapointIndex, float>>> results;
  double quantization_error = 0, imbalance = 0;
  uint64_t sum_partition_sizes = 0;

  // Distances may differ in the last bits: with SOAR, searches deduplicate
  // candidates through a hash map whose iteration order (absl seeds it per
  // table) decides how the reordering kernel batches them, so repeating a
  // search on an unchanged index can round differently.
  bool operator==(const Snapshot& o) const {
    if (n_points != o.n_points || results.size() != o.results.size() ||
        quantization_error != o.quantization_error ||
        imbalance != o.imbalance ||
        sum_partition_sizes != o.sum_partition_sizes)
      return false;
    for (size_t i = 0; i < results.size(); ++i) {
      if (results[i].size() != o.results[i].size()) return false;
      for (size_t j = 0; j < results[i].size(); ++j) {
        const auto [idx, dist] = results[i][j];
        const auto [o_idx, o_dist] = o.results[i][j];
        if (idx != o_idx || std::abs(dist - o_dist) > 1e-5f * (1 + std::abs(dist)))
          return false;
      }
    }
    return true;
  }
};

Snapshot Take(Harness& h, const std::vector<Vec>& probes) {
  Snapshot snap;
  snap.n_points = h.s.n_points();
  for (const Vec& q : probes) {
    NNResultsVector res;
    Ok(h.s.Search(Ptr(q), &res, 20, 100000, kNumLeaves), h.name + ": Search");
    // By index, so near-ties that swap order compare equal.
    std::sort(res.begin(), res.end());
    snap.results.emplace_back(res.begin(), res.end());
  }
  auto stats = h.s.GetHealthStats();
  if (Ok(stats.status(), h.name + ": GetHealthStats")) {
    snap.quantization_error = stats->avg_quantization_error;
    snap.imbalance = stats->partition_weighted_avg_relative_imbalance;
    snap.sum_partition_sizes = stats->sum_partition_sizes;
  }
  return snap;
}

// Tree + AH with SOAR: each update gets a new vector whose precomputed
// artifacts are valid except for the leaf at position `bad`, which then
// fails. That covers failures in a leaf the datapoint stays in and in one it
// newly joins; after each, the index must be observably unchanged.
void SoarFailedUpdateIsAtomic() {
  Harness h;
  h.name = "tree_ah_soar/failed_update";
  h.unit_norm = true;
  std::printf("== %s\n", h.name.c_str());
  const size_t n = 600;
  if (!h.Build(Config(AhReorder, n, scann_core::DistanceMeasure::kDotProduct,
                      1.5),
               n))
    return;
  h.Verify("built");
  std::vector<Vec> probes;
  for (int i = 0; i < 20; ++i) probes.push_back(h.Random());
  for (size_t i = 0; i < h.shadow.size(); i += 29) probes.push_back(h.shadow[i]);

  int injected = 0, two_leaves = 0;
  for (DatapointIndex idx = 0; idx < h.shadow.size(); idx += 13) {
    for (size_t bad = 0; bad < 2; ++bad) {
      Mutator* m = h.GetMutator("failed update");
      if (!m) return;
      const Vec v = h.Random();
      auto artifacts = m->ComputePrecomputedMutationArtifacts(Ptr(v));
      auto* tree_artifacts = dynamic_cast<TreeAhArtifacts*>(artifacts.get());
      if (!tree_artifacts) {
        Fail(h.name + ": precomputed artifacts are not TreeAHHybridResidual's");
        return;
      }
      if (tree_artifacts->tokens.size() <= bad) continue;
      if (tree_artifacts->tokens.size() > 1) ++two_leaves;
      tree_artifacts->leaf_precomputed_artifacts[bad] =
          std::make_unique<BogusArtifacts>();
      const Snapshot before = Take(h, probes);
      auto r = m->UpdateDatapoint(
          Ptr(v), idx,
          MutationOptions{.precomputed_mutation_artifacts = artifacts.get()});
      ++injected;
      if (r.ok()) {
        Fail(absl::StrCat(h.name, ": update of ", idx,
                          " with a bogus leaf artifact succeeded"));
        return;
      }
      if (!(Take(h, probes) == before)) {
        Fail(absl::StrCat(h.name, ": failed update of ", idx, " (leaf ", bad,
                          " of ", tree_artifacts->tokens.size(),
                          ") changed the index; error: ",
                          r.status().ToString()));
        return;
      }
    }
  }
  std::printf("   %d injected failures (%d on two-leaf updates)\n", injected,
              two_leaves);
  if (injected < 40 || two_leaves < 20)
    Fail(h.name + ": too few injected failures");
  h.Verify("after failed updates");
  h.Churn("after failed updates, mutated");
  for (DatapointIndex i = 0; i < h.shadow.size(); i += 5) h.Update(i);
  h.Verify("after updates");
}

}  // namespace

int main() {
  FailedRetrainTooFewPoints("tree_bf", BruteForce);
  FailedRetrainTooFewPoints("tree_ah", AhReorder);
  FailedRetrainBadConfig("tree_bf", BruteForce);
  FailedRetrainBadConfig("tree_ah", AhReorder);
  TreeBfloat16("tree_bf16_l2", scann_core::DistanceMeasure::kSquaredL2,
               std::nullopt);
  TreeBfloat16("tree_bf16_dot", scann_core::DistanceMeasure::kDotProduct,
               std::nullopt);
  TreeBfloat16("tree_bf16_soar_dot", scann_core::DistanceMeasure::kDotProduct,
               1.5);
  SoarFailedUpdateIsAtomic();
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "PASSED",
              g_failures);
  return g_failures ? 1 : 0;
}
