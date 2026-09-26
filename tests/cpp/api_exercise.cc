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
// Usage: scann_core_api_exercise <fixtures dir> [training_threads]
// Fixtures are written by tests/equivalence/run.py (same format the Rust
// equivalence test reads).

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
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

// Across search modes: single-query search and batched search use
// different distance kernels (one-to-many vs. many-to-many), so distances
// may differ in the last bits and near-ties may swap order. Require the
// same neighbour sets unless the displaced neighbours are within `tol` of
// each other, and distances within `tol` (relative).
void ExpectEquivalent(const std::vector<NNResultsVector>& a,
                      const std::vector<NNResultsVector>& b,
                      const std::string& what, float tol = 1e-5f) {
  if (a.size() != b.size()) return Fail(absl::StrCat(what, ": size differs"));
  size_t bad = 0;
  float worst = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].size() != b[i].size()) { ++bad; continue; }
    for (size_t j = 0; j < a[i].size(); ++j) {
      const float da = a[i][j].second, db = b[i][j].second;
      const float rel = std::abs(da - db) / std::max(1.0f, std::abs(da));
      worst = std::max(worst, rel);
      if (rel > tol) { ++bad; break; }
    }
  }
  if (bad)
    Fail(absl::StrCat(what, ": ", bad, "/", a.size(),
                      " queries differ beyond tolerance (worst rel ", worst, ")"));
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
  ExpectEquivalent(batched, parallel, f.name + ": batched vs parallel");
  ExpectEquivalent(single, batched, f.name + ": single vs batched");
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
                       "SearchBatchedParallel without a thread pool");
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <fixtures dir> [training_threads]\n", argv[0]);
    return 2;
  }
  const int training_threads = argc > 2 ? std::atoi(argv[2]) : 1;
  std::vector<fs::path> dirs;
  for (const auto& e : fs::directory_iterator(argv[1]))
    if (fs::exists(e.path() / "meta.txt")) dirs.push_back(e.path());
  std::sort(dirs.begin(), dirs.end());
  if (dirs.empty()) {
    std::fprintf(stderr, "no fixtures under %s\n", argv[1]);
    return 2;
  }
  std::vector<Fixture> fixtures;
  for (const auto& d : dirs) fixtures.push_back(LoadFixture(d));
  for (const auto& f : fixtures) ExerciseFixture(f, training_threads);
  ExerciseBadInput(fixtures.front());
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "PASSED", g_failures);
  return g_failures ? 1 : 0;
}
