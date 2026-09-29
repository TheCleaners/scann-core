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

// The exact L2 -> inner-product reduction (config: l2_as_dot_product;
// builders: L2AsDotProduct() / l2_as_dot_product()), through ScannInterface:
//
//  - search, batched search and parallel batched search return the ids of a
//    "manual" index (a dot-product index built on [x, (c - |x|^2) / (2s)]
//    and searched with [q, s]), and |q|^2 + c - 2 q'.x' as distances, which
//    are the exact squared L2 distances (to rounding);
//  - recall against brute force, for a tree + AH index and a brute-force
//    one;
//  - serialize + load (directory and in memory) keeps the reduction, and the
//    saved config fails to load where the reduction is unknown;
//  - upserts (including points with |x|^2 far above the center), updates and
//    deletes through GetMutator() + ToStoredDatapoints(), before and after
//    loading; retraining (RetrainAndReindex) keeps scale and center, and
//    rejects configs that would change or drop them;
//  - an empty index built with an explicit scale, then grown;
//  - config errors.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "scann/base/single_machine_factory_scann.h"
#include "scann/data_format/datapoint.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/io_oss_wrapper.h"
#include "scann/utils/types.h"
#include "scann_core/config_builder.h"

namespace {

namespace fs = std::filesystem;
using research_scann::ConstSpan;
using research_scann::DatapointIndex;
using research_scann::DatapointPtr;
using research_scann::NNResultsVector;
using research_scann::ScannConfig;
using research_scann::ScannInterface;
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

void ExpectError(const absl::Status& s, absl::string_view contains,
                 const std::string& what) {
  if (s.ok()) {
    Fail(absl::StrCat(what, ": expected an error"));
  } else if (!absl::StrContains(s.message(), contains)) {
    Fail(absl::StrCat(what, ": error doesn't mention \"", contains,
                      "\": ", s.ToString()));
  } else {
    std::printf("ok (error as intended): %s -> %s\n", what.c_str(),
                std::string(s.message()).c_str());
  }
}

constexpr size_t kDim = 12;
constexpr size_t kN = 3000;
constexpr size_t kQueries = 64;
constexpr int kK = 10;
constexpr int kLeaves = 20;
constexpr double kScale = 40.0;

using Vec = std::vector<float>;

// Multiples of 1/8 below 64 in magnitude: every |x|^2 and q.x is exact in
// double whatever the summation order, so the test's augmentation is
// bit-identical to the library's.
Vec RandomRows(size_t n, size_t dim, uint32_t seed, float sigma = 4.0f,
               float mean = 3.0f) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(mean, sigma);
  Vec v(n * dim);
  for (float& x : v) x = std::round(nd(rng) * 8.0f) / 8.0f;
  return v;
}

double SqNorm(const float* x, size_t dim) {
  double s = 0;
  for (size_t j = 0; j < dim; ++j) s += static_cast<double>(x[j]) * x[j];
  return s;
}

double SqL2(const float* a, const float* b, size_t dim) {
  double s = 0;
  for (size_t j = 0; j < dim; ++j) {
    const double d = static_cast<double>(a[j]) - b[j];
    s += d * d;
  }
  return s;
}

Vec Augment(const Vec& rows, size_t dim, double center, double scale) {
  const size_t n = rows.size() / dim;
  Vec out(n * (dim + 1));
  for (size_t i = 0; i < n; ++i) {
    std::copy(rows.begin() + i * dim, rows.begin() + (i + 1) * dim,
              out.begin() + i * (dim + 1));
    out[i * (dim + 1) + dim] = static_cast<float>(
        (center - SqNorm(rows.data() + i * dim, dim)) / (2.0 * scale));
  }
  return out;
}

DatapointPtr<float> Ptr(const float* v, size_t dim) {
  return DatapointPtr<float>(nullptr, v, dim, dim);
}

scann_core::TreeOptions Tree() {
  scann_core::TreeOptions t;
  t.num_leaves = kLeaves;
  t.num_leaves_to_search = 4;
  t.random_init = false;  // k-means++: reproducible
  t.avq = 2.5;
  t.quantize_centroids = true;
  return t;
}

scann_core::AhOptions Ah() {
  scann_core::AhOptions a;
  a.dimensions_per_block = 3;
  a.anisotropic_quantization_threshold = 8.0;
  return a;
}

std::string MustBuild(const ConfigBuilder& b, const char* what) {
  auto text = b.BuildText(kN);
  if (!text.ok()) {
    Fail(absl::StrCat(what, ": ", text.status().ToString()));
    return "";
  }
  return *text;
}

// Recall@k by distance (ties count): a returned id is a hit if its true
// distance is at most the k-th smallest true distance.
double Recall(const Vec& data, size_t dim, const float* q,
              const NNResultsVector& res, int k) {
  const size_t n = data.size() / dim;
  std::vector<double> d(n);
  for (size_t i = 0; i < n; ++i) d[i] = SqL2(data.data() + i * dim, q, dim);
  std::vector<double> sorted = d;
  std::nth_element(sorted.begin(), sorted.begin() + (k - 1), sorted.end());
  const double kth = sorted[k - 1];
  int hits = 0;
  for (int i = 0; i < k && i < static_cast<int>(res.size()); ++i)
    hits += d[res[i].first] <= kth + 1e-6;
  return static_cast<double>(hits) / k;
}

// Every result's distance is its exact squared L2 distance, to float32
// rounding of the terms (|q|^2 + center and 2 q'.x' cancel).
void CheckDistances(const Vec& data, const float* q, const NNResultsVector& res,
                    double center, const std::string& what) {
  const double qn = SqNorm(q, kDim);
  for (const auto& [i, dist] : res) {
    const float* x = data.data() + static_cast<size_t>(i) * kDim;
    const double want = SqL2(x, q, kDim);
    const double tol = 1e-5 * (qn + SqNorm(x, kDim) + std::abs(center) + 1.0);
    if (!(std::abs(dist - want) <= tol)) {
      Fail(absl::StrCat(what, ": datapoint ", i, " distance ", dist,
                        ", exact squared L2 ", want));
      return;
    }
  }
}

// The option's results against the manual index's: identical ids,
// distances |q|^2 + c + 2 * (manual distance, which is -q'.x').
void CompareWithManual(const NNResultsVector& got, const NNResultsVector& manual,
                       const float* q, double center, const std::string& what) {
  if (got.size() != manual.size()) {
    Fail(absl::StrCat(what, ": ", got.size(), " results, manual ",
                      manual.size()));
    return;
  }
  const double qn = SqNorm(q, kDim);
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i].first != manual[i].first) {
      Fail(absl::StrCat(what, ": result ", i, " is ", got[i].first,
                        ", manual ", manual[i].first));
      return;
    }
    const double want = std::max(qn + center + 2.0 * manual[i].second, 0.0);
    if (std::abs(got[i].second - want) > 1e-6 * (qn + std::abs(center) + 1)) {
      Fail(absl::StrCat(what, ": distance ", got[i].second, ", manual ",
                        manual[i].second, " -> ", want));
      return;
    }
  }
}

struct Searches {
  std::vector<NNResultsVector> single, batched, parallel;
};

Searches SearchAll(const ScannInterface& s, const Vec& queries, size_t dim,
                   int leaves, int pre_reorder, const std::string& what) {
  const size_t nq = queries.size() / dim;
  Searches r;
  r.single.resize(nq);
  r.batched.resize(nq);
  r.parallel.resize(nq);
  for (size_t i = 0; i < nq; ++i)
    Ok(s.Search(Ptr(queries.data() + i * dim, dim), &r.single[i], kK,
                pre_reorder, leaves),
       what + ": Search");
  Ok(s.SearchBatchedRows(queries, dim, research_scann::MakeMutableSpan(r.batched),
                         kK, pre_reorder, leaves, /*parallel=*/false),
     what + ": SearchBatchedRows");
  Ok(s.SearchBatchedRows(queries, dim,
                         research_scann::MakeMutableSpan(r.parallel), kK,
                         pre_reorder, leaves, /*parallel=*/true,
                         /*batch_size=*/16),
     what + ": SearchBatchedRows (parallel)");
  return r;
}

fs::path TestDir(const std::string& name) {
  fs::path dir = fs::temp_directory_path() /
                 absl::StrCat("scann_l2_as_dot_product_", ::getpid(), "_", name);
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

absl::flat_hash_map<std::string, std::string> ReadDir(const fs::path& dir) {
  absl::flat_hash_map<std::string, std::string> files;
  for (const auto& e : fs::directory_iterator(dir)) {
    if (!e.is_regular_file()) continue;
    std::ifstream in(e.path(), std::ios::binary);
    files[e.path().filename().string()] =
        std::string(std::istreambuf_iterator<char>(in), {});
  }
  return files;
}

// Adds (id < 0) or updates datapoint `id` with the user-space vector `v`, as
// the bindings do: through ToStoredDatapoints().
bool Upsert(ScannInterface& s, const float* v, int64_t id, DatapointIndex* out,
            const std::string& what) {
  std::vector<float> stored;
  s.ToStoredDatapoints(ConstSpan<float>(v, kDim), &stored);
  if (stored.size() != s.stored_dimensionality()) {
    Fail(absl::StrCat(what, ": ToStoredDatapoints gave ", stored.size(),
                      " values"));
    return false;
  }
  auto mutator = s.GetMutator();
  if (!Ok(mutator.status(), what + ": GetMutator")) return false;
  auto ptr = Ptr(stored.data(), stored.size());
  if (id < 0) {
    auto r = (*mutator)->AddDatapoint(ptr, "", {});
    if (!Ok(r.status(), what + ": AddDatapoint")) return false;
    *out = *r;
  } else {
    auto r = (*mutator)->UpdateDatapoint(ptr, id, {});
    if (!Ok(r.status(), what + ": UpdateDatapoint")) return false;
    *out = *r;
  }
  return true;
}

// Every shadow vector's nearest neighbor is itself, at distance ~0, when
// every datapoint is reordered (all leaves, every candidate): the search is
// then exact.
void CheckAgainstShadow(const ScannInterface& s, const Vec& shadow,
                        const std::string& what) {
  const size_t n = shadow.size() / kDim;
  if (s.n_points() != n) {
    Fail(absl::StrCat(what, ": ", s.n_points(), " points, shadow ", n));
    return;
  }
  for (size_t i = 0; i < n; i += 37) {
    const float* v = shadow.data() + i * kDim;
    NNResultsVector res;
    if (!Ok(s.Search(Ptr(v, kDim), &res, kK, static_cast<int>(n), kLeaves),
            what))
      return;
    if (res.empty() || SqL2(shadow.data() + res[0].first * kDim, v, kDim) != 0) {
      Fail(absl::StrCat(what, ": datapoint ", i, " not found first"));
      return;
    }
    CheckDistances(shadow, v, res, s.l2_as_dot_product()->center, what);
  }
}

void TestAgainstManualAndPersistence() {
  const Vec data = RandomRows(kN, kDim, 1);
  const Vec queries = RandomRows(kQueries, kDim, 2);
  double center = 0;
  for (size_t i = 0; i < kN; ++i) center += SqNorm(data.data() + i * kDim, kDim);
  center /= kN;

  ConfigBuilder opt_b(kK, DistanceMeasure::kSquaredL2, kDim);
  opt_b.Tree(Tree()).ScoreAh(Ah()).Reorder({200});
  opt_b.L2AsDotProduct({kScale, center});
  ConfigBuilder man_b(kK, DistanceMeasure::kDotProduct, kDim + 1);
  man_b.Tree(Tree()).ScoreAh(Ah()).Reorder({200});
  const std::string opt_config = MustBuild(opt_b, "option config");
  const std::string man_config = MustBuild(man_b, "manual config");

  ScannInterface opt, man;
  if (!Ok(opt.Initialize(data, kN, opt_config, 1), "build (option)")) return;
  const Vec aug = Augment(data, kDim, center, kScale);
  if (!Ok(man.Initialize(aug, kN, man_config, 1), "build (manual)")) return;

  if (opt.dimensionality() != kDim || opt.stored_dimensionality() != kDim + 1 ||
      opt.result_multiplier() != 1 || !opt.l2_as_dot_product() ||
      opt.l2_as_dot_product()->scale != kScale ||
      opt.l2_as_dot_product()->center != center)
    Fail("the option index's dimensionality / parameters");
  {
    const ScannConfig* c = opt.config();
    if (c->distance_measure().distance_measure() != "SquaredL2Distance" ||
        c->l2_as_dot_product().scale() != kScale ||
        c->l2_as_dot_product().center() != center)
      Fail(absl::StrCat("config() of the option index:\n", c->DebugString()));
  }

  // Queries with the manual recipe's extra coordinate.
  Vec aug_queries(kQueries * (kDim + 1));
  for (size_t i = 0; i < kQueries; ++i) {
    std::copy(queries.begin() + i * kDim, queries.begin() + (i + 1) * kDim,
              aug_queries.begin() + i * (kDim + 1));
    aug_queries[i * (kDim + 1) + kDim] = static_cast<float>(kScale);
  }
  const Searches o = SearchAll(opt, queries, kDim, 4, 50, "option");
  const Searches m = SearchAll(man, aug_queries, kDim + 1, 4, 50, "manual");
  double recall = 0;
  for (size_t i = 0; i < kQueries; ++i) {
    const float* q = queries.data() + i * kDim;
    CompareWithManual(o.single[i], m.single[i], q, center,
                      absl::StrCat("search, query ", i));
    CompareWithManual(o.batched[i], m.batched[i], q, center,
                      absl::StrCat("batched search, query ", i));
    CompareWithManual(o.parallel[i], m.parallel[i], q, center,
                      absl::StrCat("parallel batched search, query ", i));
    CheckDistances(data, q, o.single[i], center,
                   absl::StrCat("search distances, query ", i));
    CheckDistances(data, q, o.batched[i], center,
                   absl::StrCat("batched distances, query ", i));
    NNResultsVector wide, all;
    Ok(opt.Search(Ptr(q, kDim), &wide, kK, 300, kLeaves), "wide search");
    recall += Recall(data, kDim, q, wide, kK);
    // Reordering every datapoint is exact.
    Ok(opt.Search(Ptr(q, kDim), &all, kK, kN, kLeaves), "exhaustive search");
    if (Recall(data, kDim, q, all, kK) != 1.0)
      Fail(absl::StrCat("exhaustive search, query ", i, ": recall < 1"));
    CheckDistances(data, q, all, center,
                   absl::StrCat("exhaustive search distances, query ", i));
  }
  recall /= kQueries;
  std::printf("tree + AH, all leaves, 300 candidates: recall@%d %.4f\n", kK,
              recall);
  if (recall < 0.95) Fail(absl::StrCat("recall ", recall, " < 0.95"));

  // Serialize and load, from the directory and from memory.
  const fs::path dir = TestDir("persist");
  if (!Ok(opt.SerializeToDirectory(dir.string(), true), "serialize")) return;
  {
    ScannConfig saved;
    Ok(research_scann::ReadProtobufFromFile((dir / "scann_config.pb").string(),
                                            &saved),
       "read scann_config.pb");
    if (!absl::StrContains(saved.distance_measure().distance_measure(),
                           "l2_as_dot_product") ||
        saved.l2_as_dot_product().scale() != kScale)
      Fail(absl::StrCat("saved config:\n", saved.DebugString()));
  }
  auto artifacts = ScannInterface::LoadArtifacts(dir.string());
  if (!Ok(artifacts.status(), "LoadArtifacts")) return;

  // Loaders that don't know the reduction: the raw searcher factory (and
  // upstream's, which has no l2_as_dot_product field) fail on the distance
  // measure, and so does CreateSearcher.
  {
    auto [config, dataset, opts] = *artifacts;
    ExpectError(ScannInterface::CreateSearcher(*artifacts).status(),
                "l2_as_dot_product", "CreateSearcher on a saved reduction");
    config.clear_l2_as_dot_product();
    ExpectError(research_scann::SingleMachineFactoryScann<float>(
                    config, dataset, std::move(opts))
                    .status(),
                "Invalid distance_measure",
                "a loader without l2_as_dot_product");
    ScannInterface stripped;
    ExpectError(stripped.Initialize(std::make_tuple(config, dataset,
                                                    std::get<2>(*artifacts))),
                "no l2_as_dot_product", "the saved config without the field");
  }

  ScannInterface loaded, from_memory;
  Ok(loaded.Initialize(*artifacts), "Initialize (loaded)");
  const auto files = ReadDir(dir);
  absl::flat_hash_map<std::string, absl::string_view> views;
  for (const auto& [name, contents] : files) views[name] = contents;
  auto mem = ScannInterface::LoadArtifactsFromMemory(views);
  if (Ok(mem.status(), "LoadArtifactsFromMemory"))
    Ok(from_memory.Initialize(*mem), "Initialize (from memory)");
  for (ScannInterface* s : {&loaded, &from_memory}) {
    const std::string what = s == &loaded ? "loaded" : "loaded from memory";
    if (s->dimensionality() != kDim || !s->l2_as_dot_product() ||
        s->l2_as_dot_product()->scale != kScale ||
        s->l2_as_dot_product()->center != center ||
        s->config()->distance_measure().distance_measure() !=
            "SquaredL2Distance") {
      Fail(what + ": parameters or config");
      continue;
    }
    const Searches l = SearchAll(*s, queries, kDim, 4, 50, what);
    for (size_t i = 0; i < kQueries; ++i) {
      const float* q = queries.data() + i * kDim;
      CompareWithManual(l.single[i], m.single[i], q, center,
                        absl::StrCat(what, " search, query ", i));
      CompareWithManual(l.parallel[i], m.parallel[i], q, center,
                        absl::StrCat(what, " parallel search, query ", i));
    }
  }

  // Mutations after loading: a point far outside the data (|x|^2 >> center:
  // its extra coordinate is large and negative), updates and deletes.
  Vec shadow = data;
  std::mt19937 rng(5);
  Vec far = RandomRows(1, kDim, 9, 60.0f, 100.0f);
  DatapointIndex idx;
  if (Upsert(loaded, far.data(), -1, &idx, "add a far point")) {
    if (idx != kN) Fail("the far point's index");
    shadow.insert(shadow.end(), far.begin(), far.end());
    NNResultsVector res;
    Ok(loaded.Search(Ptr(far.data(), kDim), &res, kK, -1, -1), "search far");
    if (res.empty() || res[0].first != idx || res[0].second > 1e-3 * SqNorm(far.data(), kDim))
      Fail("the far point isn't its own nearest neighbor");
    std::printf("far point: |x|^2 = %.0f (center %.0f), self distance %g\n",
                SqNorm(far.data(), kDim), center, res.empty() ? -1.0 : res[0].second);
  }
  const Vec added = RandomRows(200, kDim, 11);
  for (size_t i = 0; i < 200; ++i) {
    if (!Upsert(loaded, added.data() + i * kDim, -1, &idx, "add")) break;
    shadow.insert(shadow.end(), added.begin() + i * kDim,
                  added.begin() + (i + 1) * kDim);
  }
  const Vec updates = RandomRows(50, kDim, 12);
  for (size_t i = 0; i < 50; ++i) {
    const DatapointIndex target = rng() % (shadow.size() / kDim);
    if (!Upsert(loaded, updates.data() + i * kDim, target, &idx, "update")) break;
    std::copy(updates.begin() + i * kDim, updates.begin() + (i + 1) * kDim,
              shadow.begin() + target * kDim);
  }
  for (int i = 0; i < 60; ++i) {
    const size_t n = shadow.size() / kDim;
    const DatapointIndex target = rng() % n;
    auto mutator = loaded.GetMutator();
    if (!Ok(mutator.status(), "GetMutator") ||
        !Ok((*mutator)->RemoveDatapoint(target), "delete"))
      break;
    std::copy(shadow.end() - kDim, shadow.end(), shadow.begin() + target * kDim);
    shadow.resize(shadow.size() - kDim);
  }
  CheckAgainstShadow(loaded, shadow, "after mutations");

  // Retraining keeps the reduction.
  auto retrained = loaded.RetrainAndReindex("");
  if (Ok(retrained.status(), "RetrainAndReindex")) {
    if (retrained->l2_as_dot_product().scale() != kScale ||
        retrained->l2_as_dot_product().center() != center ||
        retrained->distance_measure().distance_measure() != "SquaredL2Distance")
      Fail(absl::StrCat("config after retraining:\n", retrained->DebugString()));
    CheckAgainstShadow(loaded, shadow, "after retraining");
  }
  // With a builder config (scale and center unset: the index's are used).
  {
    ConfigBuilder b(kK, DistanceMeasure::kSquaredL2, kDim);
    auto t = Tree();
    t.num_leaves = 10;
    b.Tree(t).ScoreAh(Ah()).Reorder({200}).L2AsDotProduct();
    Ok(loaded.RetrainAndReindex(MustBuild(b, "retrain config")).status(),
       "RetrainAndReindex with a builder config");
    CheckAgainstShadow(loaded, shadow, "after retraining into 10 leaves");
    ConfigBuilder other(kK, DistanceMeasure::kSquaredL2, kDim);
    other.Tree(t).ScoreAh(Ah()).Reorder({200}).L2AsDotProduct({kScale * 2, {}});
    ExpectError(loaded.RetrainAndReindex(MustBuild(other, "c")).status(),
                "can't change", "retraining with another scale");
    ConfigBuilder plain(kK, DistanceMeasure::kSquaredL2, kDim);
    auto plain_tree = t;
    plain_tree.avq.reset();
    plain.Tree(plain_tree).ScoreAh(Ah()).Reorder({200});
    ExpectError(loaded.RetrainAndReindex(MustBuild(plain, "c")).status(),
                "must have l2_as_dot_product",
                "retraining without l2_as_dot_product");
    ExpectError(man.RetrainAndReindex(opt_config).status(), "can't be added",
                "adding l2_as_dot_product by retraining");
  }

  // Serialize the mutated, retrained index and load it again.
  const fs::path dir2 = TestDir("persist2");
  if (Ok(loaded.SerializeToDirectory(dir2.string(), false), "serialize 2")) {
    auto a2 = ScannInterface::LoadArtifacts(dir2.string());
    ScannInterface again;
    if (Ok(a2.status(), "LoadArtifacts 2") &&
        Ok(again.Initialize(*a2), "Initialize 2"))
      CheckAgainstShadow(again, shadow, "reloaded after mutations");
  }
  fs::remove_all(dir);
  fs::remove_all(dir2);
}

// Brute force through the reduction is exact: recall 1.
void TestBruteForce() {
  const Vec data = RandomRows(kN, kDim, 21);
  const Vec queries = RandomRows(kQueries, kDim, 22);
  ConfigBuilder b(kK, DistanceMeasure::kSquaredL2, kDim);
  b.ScoreBruteForce().L2AsDotProduct();
  ScannInterface s;
  if (!Ok(s.Initialize(data, kN, MustBuild(b, "bf config"), 0), "bf build"))
    return;
  // The defaults: center = mean |x|^2, scale = 0.4 sqrt(center).
  double center = 0;
  for (size_t i = 0; i < kN; ++i) center += SqNorm(data.data() + i * kDim, kDim);
  center /= kN;
  if (std::abs(s.l2_as_dot_product()->center - center) > 1e-9 * center ||
      std::abs(s.l2_as_dot_product()->scale - 0.4 * std::sqrt(center)) >
          1e-9 * center)
    Fail(absl::StrCat("default parameters: ", s.l2_as_dot_product()->scale,
                      " ", s.l2_as_dot_product()->center));
  const Searches r = SearchAll(s, queries, kDim, -1, -1, "bf");
  for (size_t i = 0; i < kQueries; ++i) {
    const float* q = queries.data() + i * kDim;
    for (const auto* res : {&r.single[i], &r.batched[i], &r.parallel[i]}) {
      if (Recall(data, kDim, q, *res, kK) != 1.0)
        Fail(absl::StrCat("brute force recall < 1, query ", i));
      CheckDistances(data, q, *res, s.l2_as_dot_product()->center,
                     absl::StrCat("brute force distances, query ", i));
    }
  }
}

// An empty index (explicit scale, dimensionality from the config), grown by
// upserts.
void TestEmptyIndex() {
  const std::string config = absl::StrCat(
      "num_neighbors: 10 distance_measure { distance_measure: "
      "\"SquaredL2Distance\" } brute_force { fixed_point { enabled: false } } "
      "input_output { pure_dynamic_config { vector_type: DENSE "
      "dimensionality: ", kDim, " } } l2_as_dot_product { scale: 50 }");
  ScannInterface s;
  if (!Ok(s.Initialize(ConstSpan<float>(), 0, config, 1), "empty build"))
    return;
  if (s.dimensionality() != kDim || s.l2_as_dot_product()->center != 0 ||
      s.config()->input_output().pure_dynamic_config().dimensionality() != kDim)
    Fail("empty index dimensionality / center / config");
  const Vec data = RandomRows(300, kDim, 31);
  Vec shadow;
  for (size_t i = 0; i < 300; ++i) {
    DatapointIndex idx;
    if (!Upsert(s, data.data() + i * kDim, -1, &idx, "grow")) return;
    shadow.insert(shadow.end(), data.begin() + i * kDim,
                  data.begin() + (i + 1) * kDim);
  }
  CheckAgainstShadow(s, shadow, "grown from empty");

  ExpectError(ScannInterface().Initialize(
                  ConstSpan<float>(), 0,
                  absl::StrCat("num_neighbors: 10 distance_measure { "
                               "distance_measure: \"SquaredL2Distance\" } "
                               "brute_force { } input_output { "
                               "pure_dynamic_config { dimensionality: ",
                               kDim, " } } l2_as_dot_product { }"),
                  1),
              "needs a scale", "empty dataset without a scale");
}

void TestConfigErrors() {
  const Vec data = RandomRows(500, kDim, 41);
  auto build = [&](const std::string& config) {
    ScannInterface s;
    return s.Initialize(data, 500, config, 1);
  };
  const std::string bf = " brute_force { } ";
  ExpectError(build("num_neighbors: 10 distance_measure { distance_measure: "
                    "\"DotProductDistance\" } l2_as_dot_product { }" + bf),
              "must be SquaredL2Distance", "dot product + l2_as_dot_product");
  ExpectError(build("num_neighbors: 10 distance_measure { distance_measure: "
                    "\"SquaredL2Distance\" } l2_as_dot_product { scale: -1 }" +
                    bf),
              "scale must be positive", "negative scale");
  ExpectError(
      build("num_neighbors: 10 distance_measure { distance_measure: "
            "\"SquaredL2Distance\" } l2_as_dot_product { } partitioning { "
            "num_children: 4 partitioning_type: SPHERICAL query_spilling { "
            "spilling_type: FIXED_NUMBER_OF_CENTERS max_spill_centers: 1 } }" +
            bf),
      "spherical", "spherical partitioning");
  ExpectError(build("num_neighbors: 10 epsilon_distance: 5 distance_measure { "
                    "distance_measure: \"SquaredL2Distance\" } "
                    "l2_as_dot_product { }" + bf),
              "epsilon_distance", "epsilon_distance");
  // A config with the reduction but no parameters reaching the artifacts
  // path (e.g. a hand-edited saved config).
  ScannConfig config;
  google::protobuf::TextFormat::ParseFromString(
      "num_neighbors: 10 distance_measure { distance_measure: "
      "\"SquaredL2Distance\" } l2_as_dot_product { } brute_force { }",
      &config);
  auto ds = std::make_shared<research_scann::DenseDataset<float>>(
      Augment(data, kDim, 0, 1), 500);
  ScannInterface s;
  ExpectError(s.Initialize(std::make_tuple(
                  config, ds, research_scann::SingleMachineFactoryOptions())),
              "needs its scale and center", "artifacts without parameters");
}

}  // namespace

int main() {
  TestAgainstManualAndPersistence();
  TestBruteForce();
  TestEmptyIndex();
  TestConfigErrors();
  std::printf("l2_as_dot_product: %s (%d failure(s))\n",
              g_failures ? "FAILED" : "PASSED", g_failures);
  return g_failures ? 1 : 0;
}
