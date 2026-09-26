// Copyright 2026 ebenali and TheCleaners.
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

// Exercises scann-core's C++ API end to end, for running under ASan/UBSan,
// TSan and Valgrind: index construction for every fixture config, agreement
// of single / batched / parallel search, a serialize -> load round trip,
// mutation (delete / add / update / incremental maintenance / retrain),
// health stats, and error handling on bad input.
//
// Usage: scann_core_api_exercise [<fixtures dir>] [training_threads]
// With no fixtures dir (or "-"), it generates synthetic datasets and builds
// their configs with scann_core::ConfigBuilder, so it needs nothing else.
// A fixtures dir is what tests/equivalence/run.py writes (the same format
// the Rust equivalence test reads).

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "google/protobuf/text_format.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/types.h"
#include "scann_core/config_builder.h"

namespace {

using research_scann::ConstSpan;
using research_scann::DatapointIndex;
using research_scann::DatapointPtr;
using research_scann::DenseDataset;
using research_scann::MakeMutableSpan;
using research_scann::NNResultsVector;
using research_scann::ScannInterface;
namespace fs = std::filesystem;

int g_failures = 0;

void Fail(const std::string& what) {
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++g_failures;
}

bool Ok(const absl::Status& s, const std::string& what) {
  if (!s.ok()) Fail(absl::StrCat(what, ": ", s.ToString()));
  return s.ok();
}

struct Fixture {
  std::string name;
  size_t n = 0, dim = 0, nq = 0, k = 0;
  std::vector<float> db, queries;
  std::string config;
};

std::string ReadFile(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), {});
}

std::vector<float> ReadF32(const fs::path& p) {
  std::string bytes = ReadFile(p);
  std::vector<float> v(bytes.size() / sizeof(float));
  std::memcpy(v.data(), bytes.data(), v.size() * sizeof(float));
  return v;
}

Fixture LoadFixture(const fs::path& dir) {
  Fixture f;
  f.name = dir.filename().string();
  std::istringstream meta(ReadFile(dir / "meta.txt"));
  meta >> f.n >> f.dim >> f.nq >> f.k;
  f.db = ReadF32(dir / "dataset.f32");
  f.queries = ReadF32(dir / "queries.f32");
  f.config = ReadFile(dir / "config.pbtxt");
  return f;
}

DatapointPtr<float> Row(const std::vector<float>& flat, size_t dim, size_t i) {
  return DatapointPtr<float>(nullptr, flat.data() + i * dim, dim, dim);
}

std::vector<NNResultsVector> SearchEach(const ScannInterface& s,
                                        const Fixture& f) {
  std::vector<NNResultsVector> out(f.nq);
  for (size_t i = 0; i < f.nq; ++i) {
    Ok(s.Search(Row(f.queries, f.dim, i), &out[i], f.k, -1, -1),
       absl::StrCat(f.name, ": Search"));
  }
  return out;
}

std::vector<NNResultsVector> SearchBatched(const ScannInterface& s,
                                           const Fixture& f, bool parallel) {
  DenseDataset<float> qs(std::vector<float>(f.queries), f.nq);
  std::vector<NNResultsVector> out(f.nq);
  if (parallel) {
    Ok(s.SearchBatchedParallel(qs, MakeMutableSpan(out), f.k, -1, -1, 16),
       absl::StrCat(f.name, ": SearchBatchedParallel"));
  } else {
    Ok(s.SearchBatched(qs, MakeMutableSpan(out), f.k, -1, -1),
       absl::StrCat(f.name, ": SearchBatched"));
  }
  return out;
}

// Bit-exact: same search mode, same index.
void ExpectSame(const std::vector<NNResultsVector>& a,
                const std::vector<NNResultsVector>& b, const std::string& what) {
  if (a.size() != b.size()) return Fail(absl::StrCat(what, ": size differs"));
  size_t differing = 0;
  for (size_t i = 0; i < a.size(); ++i) differing += (a[i] != b[i]);
  if (differing)
    Fail(absl::StrCat(what, ": ", differing, "/", a.size(), " queries differ"));
}

// Across search modes: single-query and batched search use different
// kernels (one-to-many vs. many-to-many distances; single- vs. multi-query
// AH lookup tables), and parallel batched search splits the batch.
//   * A neighbour both results contain must have the same distance (within
//     `tol`, relative): the final scores agree.
//   * Exact configs must return the same neighbours. Approximate configs
//     may differ in which partitions and candidates the two modes select
//     (their kernels round differently, and near-ties flip), so they only
//     have to share `min_overlap` of their neighbours overall. Measured
//     legitimate overlaps go down to ~0.95 (it depends on the trained tree,
//     so on compiler and data); a broken mode lands far lower.
void ExpectEquivalent(const std::vector<NNResultsVector>& a,
                      const std::vector<NNResultsVector>& b,
                      const std::string& what, bool exact,
                      float tol = 1e-5f, double min_overlap = 0.90) {
  if (a.size() != b.size()) return Fail(absl::StrCat(what, ": size differs"));
  size_t shared = 0, total = 0, score_mismatches = 0;
  float worst = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    total += std::max(a[i].size(), b[i].size());
    for (const auto& [index, da] : a[i]) {
      auto it = std::find_if(b[i].begin(), b[i].end(),
                             [&](const auto& p) { return p.first == index; });
      if (it == b[i].end()) continue;
      ++shared;
      const float rel = std::abs(da - it->second) / std::max(1.0f, std::abs(da));
      worst = std::max(worst, rel);
      if (rel > tol) ++score_mismatches;
    }
  }
  const double overlap = total ? static_cast<double>(shared) / total : 1.0;
  if (getenv("SCANN_PRINT_OVERLAP"))
    std::printf("   overlap %.4f  %s\n", overlap, what.c_str());
  if (score_mismatches)
    Fail(absl::StrCat(what, ": ", score_mismatches, " shared neighbours have "
                      "different distances (worst rel ", worst, ")"));
  if (exact ? shared != total : overlap < min_overlap)
    Fail(absl::StrCat(what, ": neighbour overlap ", overlap,
                      exact ? " (exact config: must be 1)" : " < ", exact ? "" : absl::StrCat(min_overlap)));
}

void ExpectIndicesBelow(const std::vector<NNResultsVector>& r, size_t limit,
                        const std::string& what) {
  for (const auto& v : r)
    for (const auto& [idx, dist] : v)
      if (idx >= limit || std::isnan(dist))
        return Fail(absl::StrCat(what, ": bad result (", idx, ", ", dist,
                                 ") with ", limit, " points"));
}

void ExerciseFixture(const Fixture& f, int training_threads) {
  std::printf("== %s (n=%zu dim=%zu nq=%zu k=%zu)\n", f.name.c_str(), f.n,
              f.dim, f.nq, f.k);
  ScannInterface s;
  if (!Ok(s.Initialize(ConstSpan<float>(f.db), f.n, f.config, training_threads),
          absl::StrCat(f.name, ": Initialize")))
    return;
  if (s.n_points() != f.n) Fail(absl::StrCat(f.name, ": n_points"));

  auto single = SearchEach(s, f);
  auto batched = SearchBatched(s, f, /*parallel=*/false);
  s.SetNumThreads(4);
  auto parallel = SearchBatched(s, f, /*parallel=*/true);
  // Parallel batched search runs the same kernels on chunks of the batch,
  // but brute-force many-to-many blocks by batch size, so the last bits of
  // a distance can depend on the chunking.
  const bool exact = f.name.find("brute_force") != std::string::npos;
  ExpectEquivalent(batched, parallel, f.name + ": batched vs parallel", exact);
  ExpectEquivalent(single, batched, f.name + ": single vs batched", exact);
  ExpectIndicesBelow(single, f.n, f.name + ": search");

  // Serialize -> load -> search again.
  const fs::path dir = fs::temp_directory_path() /
                       absl::StrCat("scann-core-exercise-", getpid(), "-", f.name);
  fs::create_directories(dir);
  auto assets = s.Serialize(dir.string(), /*relative_path=*/false);
  if (Ok(assets.status(), f.name + ": Serialize")) {
    std::string assets_text;
    google::protobuf::TextFormat::PrintToString(*assets, &assets_text);
    auto artifacts = ScannInterface::LoadArtifacts(dir.string(), assets_text);
    ScannInterface loaded;
    if (Ok(artifacts.status(), f.name + ": LoadArtifacts") &&
        Ok(loaded.Initialize(std::move(*artifacts)), f.name + ": Initialize(loaded)")) {
      ExpectSame(batched, SearchBatched(loaded, f, false),
                 f.name + ": original vs reloaded (batched)");
      ExpectSame(single, SearchEach(loaded, f),
                 f.name + ": original vs reloaded (single)");
    }
  }
  fs::remove_all(dir);

  // Mutation: delete 5, add 10, update 3, maintenance, retrain.
  auto mutator_or = s.GetMutator();
  if (!mutator_or.ok()) {
    std::printf("   mutation not supported: %s\n",
                std::string(mutator_or.status().message()).c_str());
  } else {
    auto* m = *mutator_or;
    for (size_t i = 0; i < 5; ++i)
      Ok(m->RemoveDatapoint(static_cast<DatapointIndex>(f.n - 1 - i)),
         f.name + ": RemoveDatapoint");
    std::vector<float> fresh(f.queries.begin(), f.queries.begin() + 10 * f.dim);
    for (size_t i = 0; i < 10; ++i)
      Ok(m->AddDatapoint(Row(fresh, f.dim, i), "").status(),
         f.name + ": AddDatapoint");
    for (size_t i = 0; i < 3; ++i)
      Ok(m->UpdateDatapoint(Row(fresh, f.dim, 9 - i), static_cast<DatapointIndex>(i))
             .status(),
         f.name + ": UpdateDatapoint");
    auto maint = m->IncrementalMaintenance();
    Ok(maint.status(), f.name + ": IncrementalMaintenance");
    const size_t expected = f.n - 5 + 10;
    if (s.n_points() != expected)
      Fail(absl::StrCat(f.name, ": n_points after mutation ", s.n_points(),
                        " != ", expected));
    ExpectIndicesBelow(SearchBatched(s, f, false), s.n_points(),
                       f.name + ": search after mutation");
    auto retrained = s.RetrainAndReindex("");
    if (Ok(retrained.status(), f.name + ": RetrainAndReindex")) {
      if (s.n_points() != expected) Fail(f.name + ": n_points after retrain");
      ExpectIndicesBelow(SearchBatched(s, f, false), s.n_points(),
                         f.name + ": search after retrain");
    }
  }

  if (Ok(s.InitializeHealthStats(), f.name + ": InitializeHealthStats"))
    Ok(s.GetHealthStats().status(), f.name + ": GetHealthStats");
}

// Inputs that must be rejected with an error -- not accepted silently, and
// not crash or read out of bounds.
void ExerciseBadInput(const Fixture& f) {
  std::printf("== bad input (using %s)\n", f.name.c_str());
  {
    // A misspelled field at the end: everything before it is a complete,
    // valid config, so ignoring the parse error means silently accepting it.
    ScannInterface s;
    std::string config = f.config + "\nnum_neighbours: 20\n";
    if (s.Initialize(ConstSpan<float>(f.db), f.n, config, 1).ok())
      Fail("config with a misspelled trailing field was accepted");
  }
  {
    // n_points == 0 with a non-empty dataset (dimensionality = size / 0).
    ScannInterface s;
    if (s.Initialize(ConstSpan<float>(f.db), 0, f.config, 1).ok())
      Fail("n_points == 0 with a non-empty dataset was accepted");
  }
  {
    ScannInterface s;
    if (s.Initialize(ConstSpan<float>(f.db), f.n, "this is { not a config", 1).ok())
      Fail("garbage config was accepted");
  }
  {
    // Dataset shorter than n_points * dim claims.
    ScannInterface s;
    ConstSpan<float> short_db(f.db.data(), f.db.size() - f.dim / 2);
    if (s.Initialize(short_db, f.n, f.config, 1).ok())
      Fail("dataset whose size is not a multiple of n_points was accepted");
  }
  ScannInterface s;
  if (!Ok(s.Initialize(ConstSpan<float>(f.db), f.n, f.config, 1), "bad input: Initialize"))
    return;
  for (size_t dim : {f.dim - 1, f.dim + 1, size_t{1}}) {
    std::vector<float> q(dim, 0.5f);
    NNResultsVector r;
    if (s.Search(DatapointPtr<float>(nullptr, q.data(), dim, dim), &r, f.k, -1, -1).ok())
      Fail(absl::StrCat("query of dimensionality ", dim, " (index has ", f.dim,
                        ") was accepted"));
  }
  {
    NNResultsVector r;
    auto st = s.Search(Row(f.queries, f.dim, 0), &r, static_cast<int>(f.n) + 50, -1, -1);
    if (st.ok()) ExpectIndicesBelow({r}, f.n, "final_nn > n_points");
  }
  {
    DenseDataset<float> qs(std::vector<float>(f.queries), f.nq);
    std::vector<NNResultsVector> out(f.nq);
    if (s.SearchBatchedParallel(qs, MakeMutableSpan(out), f.k, -1, -1, 0).ok())
      Fail("SearchBatchedParallel with batch_size 0 was accepted");
    // No query pool (SetNumThreads(0), or a single-CPU machine, where the
    // default is GetNumCPUs() - 1 threads): runs inline instead of crashing.
    s.SetNumThreads(0);
    if (Ok(s.SearchBatchedParallel(qs, MakeMutableSpan(out), f.k, -1, -1, 16),
           "SearchBatchedParallel without a thread pool"))
      ExpectEquivalent(SearchBatched(s, f, false), out,
                       "SearchBatchedParallel without a thread pool",
                       f.name.find("brute_force") != std::string::npos);
  }
}

// Unit-norm points around `clusters` random centres, row-major n x dim.
std::vector<float> Clustered(size_t n, size_t dim, size_t clusters,
                             uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> gauss;
  std::vector<float> centers(clusters * dim);
  for (float& c : centers) c = gauss(rng);
  std::vector<float> out(n * dim);
  for (size_t i = 0; i < n; ++i) {
    float* row = &out[i * dim];
    double norm = 0;
    for (size_t d = 0; d < dim; ++d) {
      row[d] = centers[(i % clusters) * dim + d] + 0.4f * gauss(rng);
      norm += row[d] * row[d];
    }
    for (size_t d = 0; d < dim; ++d) row[d] /= std::sqrt(norm);
  }
  return out;
}

// The same kinds of configs the equivalence fixtures cover, plus SOAR and a
// bfloat16 reorder, on two synthetic datasets (one with a dimensionality
// that isn't a multiple of 4).
std::vector<Fixture> SyntheticFixtures() {
  using scann_core::ConfigBuilder;
  using scann_core::DistanceMeasure;
  using scann_core::Quantization;
  struct Data { const char* name; size_t n, dim; uint32_t seed; };
  const Data datasets[] = {{"A", 4000, 64, 1}, {"B", 3000, 98, 2}};
  scann_core::TreeOptions tree;
  tree.num_leaves = 60;
  tree.num_leaves_to_search = 8;
  tree.random_init = false;
  scann_core::AhOptions ah;
  ah.anisotropic_quantization_threshold = 0.2;
  scann_core::AhOptions ah_l2;  // AQ is for dot product
  scann_core::ReorderOptions reorder;
  reorder.reordering_num_neighbors = 100;
  scann_core::ReorderOptions reorder_int8 = reorder;
  reorder_int8.quantize = Quantization::kInt8;
  scann_core::ReorderOptions reorder_bf16 = reorder;
  reorder_bf16.quantize = Quantization::kBfloat16;
  scann_core::TreeOptions soar = tree;
  soar.soar_lambda = 1.5;

  const size_t k = 10;
  struct Config {
    const char* name;
    DistanceMeasure distance;
    std::function<void(ConfigBuilder&)> setup;
  };
  const std::vector<Config> configs = {
      {"brute_force_dot", DistanceMeasure::kDotProduct,
       [](ConfigBuilder& b) { b.ScoreBruteForce(); }},
      {"ah_int8reorder_dot", DistanceMeasure::kDotProduct,
       [&](ConfigBuilder& b) { b.ScoreAh(ah).Reorder(reorder_int8); }},
      {"autopilot_dot", DistanceMeasure::kDotProduct,
       [](ConfigBuilder& b) { b.Autopilot(); }},
      {"tree_ah_reorder_dot", DistanceMeasure::kDotProduct,
       [&](ConfigBuilder& b) { b.Tree(tree).ScoreAh(ah).Reorder(reorder); }},
      {"tree_ah_reorder_l2", DistanceMeasure::kSquaredL2,
       [&](ConfigBuilder& b) { b.Tree(tree).ScoreAh(ah_l2).Reorder(reorder); }},
      {"tree_soar_bf16reorder_dot", DistanceMeasure::kDotProduct,
       [&](ConfigBuilder& b) { b.Tree(soar).ScoreAh(ah).Reorder(reorder_bf16); }},
  };
  std::vector<Fixture> out;
  for (const Data& d : datasets) {
    std::vector<float> all = Clustered(d.n + 200, d.dim, 40, d.seed);
    for (const Config& c : configs) {
      Fixture f;
      f.name = absl::StrCat(d.name, "_", c.name);
      f.n = d.n;
      f.dim = d.dim;
      f.nq = 200;
      f.k = k;
      f.db.assign(all.begin(), all.begin() + d.n * d.dim);
      f.queries.assign(all.begin() + d.n * d.dim, all.end());
      ConfigBuilder b(k, c.distance, d.dim);
      c.setup(b);
      auto config = b.BuildText(d.n);
      if (!Ok(config.status(), f.name + ": ConfigBuilder")) continue;
      f.config = *config;
      out.push_back(std::move(f));
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const bool synthetic = argc < 2 || std::string(argv[1]) == "-";
  const int training_threads = argc > 2 ? std::atoi(argv[2]) : 1;
  std::vector<Fixture> fixtures;
  if (synthetic) {
    fixtures = SyntheticFixtures();
  } else {
    std::vector<fs::path> dirs;
    for (const auto& e : fs::directory_iterator(argv[1]))
      if (fs::exists(e.path() / "meta.txt")) dirs.push_back(e.path());
    std::sort(dirs.begin(), dirs.end());
    for (const auto& d : dirs) fixtures.push_back(LoadFixture(d));
  }
  if (fixtures.empty()) {
    std::fprintf(stderr, "no fixtures%s%s\n", synthetic ? "" : " under ",
                 synthetic ? "" : argv[1]);
    return 2;
  }
  for (const auto& f : fixtures) ExerciseFixture(f, training_threads);
  ExerciseBadInput(fixtures.front());
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "PASSED", g_failures);
  return g_failures ? 1 : 0;
}
