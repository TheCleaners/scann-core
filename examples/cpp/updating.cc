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

// Keeping a scann-core index up to date from C++: add, update and remove
// points through the mutator, retrain, and save / reload.
//
// ScannInterface identifies points by row index, as the Rust bindings do
// (the Python wrapper keeps a docid <-> index map on top). Removing a point
// moves the last row into its slot, so keep your own id mapping in step.
//
// Build (from the scann-core build tree):
//   cmake --build <build> --target scann_core_example_updating
//   ./<build>/examples/scann_core_example_updating

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann_core/config_builder.h"

using research_scann::DatapointIndex;
using research_scann::DatapointPtr;
using research_scann::DenseDataset;
using research_scann::NNResultsVector;
using research_scann::ScannInterface;

namespace {

constexpr size_t kN = 3000, kDim = 32, kLeaves = 60;

// Unit-norm points around 40 fixed random centers, row-major n x kDim.
std::vector<float> MakeData(size_t n, uint32_t seed) {
  std::mt19937 center_rng(42), rng(seed);
  std::normal_distribution<float> gauss;
  std::vector<float> centers(40 * kDim);
  for (float& c : centers) c = gauss(center_rng);
  std::vector<float> data(n * kDim);
  for (size_t i = 0; i < n; ++i) {
    float* row = &data[i * kDim];
    const size_t c = rng() % 40;
    double norm = 0;
    for (size_t d = 0; d < kDim; ++d) {
      row[d] = centers[c * kDim + d] + 0.3f * gauss(rng);
      norm += row[d] * row[d];
    }
    for (size_t d = 0; d < kDim; ++d) row[d] /= std::sqrt(norm);
  }
  return data;
}

DatapointPtr<float> Row(const std::vector<float>& data, size_t i) {
  return DatapointPtr<float>(nullptr, data.data() + i * kDim, kDim, kDim);
}

// Stops the example with a message on an error, or on a failed check.
void Check(const absl::Status& s, const char* what) {
  if (s.ok()) return;
  std::fprintf(stderr, "%s: %s\n", what, s.ToString().c_str());
  std::exit(1);
}
void Check(bool ok, const char* what) {
  if (!ok) Check(absl::InternalError("check failed"), what);
}

// The index of the nearest stored point, searching every leaf and
// reordering 1000 candidates: effectively exact, so a stored vector always
// finds itself.
DatapointIndex Nearest(const ScannInterface& index, DatapointPtr<float> q) {
  NNResultsVector res;
  Check(index.Search(q, &res, /*final_nn=*/1, /*pre_reorder_nn=*/1000,
                     /*leaves=*/kLeaves),
        "search");
  Check(!res.empty(), "search returned nothing");
  return res[0].first;
}

void PrintHealth(const char* when, const ScannInterface& index) {
  auto stats = index.GetHealthStats();
  Check(stats.status(), "health stats");
  std::printf("%-28s %5zu points, imbalance %.3f, quantization error %.4f\n", when,
              index.n_points(), stats->partition_avg_relative_positive_imbalance,
              stats->avg_quantization_error);
}

using Mutator = research_scann::SingleMachineSearcherBase<float>::Mutator;

// After a batch of changes, give the index a chance to maintain itself.
// Incremental maintenance can decide that the tree has drifted enough to
// need retraining; that is on the caller, as ScaNN's Python and Rust
// bindings do it. Retraining replaces the searcher, and with it the
// mutator, so this returns the one to continue with.
Mutator* MaintainOrRetrain(ScannInterface& index, Mutator* mutator) {
  auto maintenance = mutator->IncrementalMaintenance();
  Check(maintenance.status(), "incremental maintenance");
  if (!maintenance->has_value()) return mutator;
  Check(index.RetrainAndReindex("").status(), "retrain");
  auto fresh = index.GetMutator();
  Check(fresh.status(), "mutator");
  return *fresh;
}

}  // namespace

int main() {
  const std::vector<float> initial = MakeData(kN, 1);

  // Python: builder(db, 10, "dot_product")
  //           .tree(num_leaves=60, num_leaves_to_search=10, random_init=False)
  //           .score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100)
  scann_core::TreeOptions tree;
  tree.num_leaves = kLeaves;
  tree.num_leaves_to_search = 10;
  tree.random_init = false;
  scann_core::AhOptions ah;
  ah.anisotropic_quantization_threshold = 0.2;
  scann_core::ReorderOptions reorder;
  reorder.reordering_num_neighbors = 100;
  auto config = scann_core::ConfigBuilder(10, scann_core::DistanceMeasure::kDotProduct, kDim)
                    .Tree(tree)
                    .ScoreAh(ah)
                    .Reorder(reorder)
                    .BuildText(kN);
  Check(config.status(), "config");

  ScannInterface index;
  Check(index.Initialize(initial, kN, *config, /*training_threads=*/0), "build");
  PrintHealth("built:", index);

  // The mutator changes the index in place. It isn't safe to use while
  // other threads search or mutate the same index: serialize access
  // yourself, e.g. with a reader/writer lock around searches and changes
  // (the Python module does that; in Rust, &mut self does).
  auto got = index.GetMutator();
  Check(got.status(), "mutator");
  Mutator* mutator = *got;

  // --- Add -----------------------------------------------------------------
  // AddDatapoint returns the new point's index: n_points() before the add.
  // (For a tree with spherical=true, normalize the vector first with
  // index.NormalizeDatapoints(), as the bindings do.) The docid argument is
  // for ScaNN's own docid tracking, which ScannInterface doesn't use: pass "".
  const std::vector<float> added = MakeData(1000, 2);
  for (size_t i = 0; i < 1000; ++i) {
    auto idx = mutator->AddDatapoint(Row(added, i), "");
    Check(idx.status(), "add");
    Check(*idx == kN + i, "added points are appended");
  }
  mutator = MaintainOrRetrain(index, mutator);
  Check(index.n_points() == kN + 1000, "size after adding");
  for (size_t i = 0; i < 1000; ++i)
    Check(Nearest(index, Row(added, i)) == kN + i, "an added point finds itself");
  PrintHealth("after adding 1000:", index);

  // --- Update --------------------------------------------------------------
  // UpdateDatapoint replaces the vector at an index; the index is kept.
  const std::vector<float> moved = MakeData(1, 3);
  auto updated = mutator->UpdateDatapoint(Row(moved, 0), DatapointIndex{0});
  Check(updated.status(), "update");
  mutator = MaintainOrRetrain(index, mutator);
  Check(Nearest(index, Row(moved, 0)) == 0, "point 0 is found at its new vector");
  Check(Nearest(index, Row(initial, 0)) != 0, "and not at its old one");
  std::printf("updated point 0\n");

  // --- Remove --------------------------------------------------------------
  // RemoveDatapoint(i) moves the last point into slot i. Removing from the
  // highest index down keeps every index you still hold valid.
  const DatapointIndex last = index.n_points() - 1;  // added[999]
  Check(mutator->RemoveDatapoint(DatapointIndex{5}), "remove");
  mutator = MaintainOrRetrain(index, mutator);
  Check(index.n_points() == kN + 999, "size after removing");
  Check(Nearest(index, Row(added, 999)) == 5, "the last point moved into slot 5");
  std::printf("removed point 5; point %u moved into its slot\n", last);

  // --- Retrain -------------------------------------------------------------
  // RetrainAndReindex("") retrains the tree and AH codebooks on the current
  // data (a new config text would switch configs). It replaces the searcher:
  // fetch a new mutator afterwards. Health stats are recomputed.
  Check(index.RetrainAndReindex("").status(), "retrain");
  got = index.GetMutator();
  Check(got.status(), "mutator");
  mutator = *got;
  PrintHealth("after RetrainAndReindex:", index);

  // --- Save, change, save again, reload ------------------------------------
  // SerializeToDirectory writes into an existing directory and replaces an
  // index already there: the files are staged and fsynced first, and the
  // manifest (scann_assets.pbtxt) is renamed in last, so an interrupted save
  // leaves the old index, the new one, or a directory that fails to load;
  // never a mix. relative_path=true makes the directory movable.
  const auto dir = std::filesystem::temp_directory_path() / "scann-core-example-updating";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  Check(index.SerializeToDirectory(dir.string(), /*relative_path=*/true), "serialize");
  auto extra = mutator->AddDatapoint(Row(moved, 0), "");
  Check(extra.status(), "add");
  mutator = MaintainOrRetrain(index, mutator);
  Check(index.SerializeToDirectory(dir.string(), /*relative_path=*/true), "re-serialize");

  auto artifacts = ScannInterface::LoadArtifacts(dir.string());
  Check(artifacts.status(), "load");
  ScannInterface reloaded;
  Check(reloaded.Initialize(*std::move(artifacts)), "reload");
  Check(reloaded.n_points() == index.n_points(), "reloaded size");

  // The reloaded index gives the same answers.
  const std::vector<float> queries = MakeData(50, 4);
  DenseDataset<float> batch(std::vector<float>(queries), 50);
  std::vector<NNResultsVector> a(50), b(50);
  Check(index.SearchBatched(batch, research_scann::MakeMutableSpan(a), -1, -1, -1), "search");
  Check(reloaded.SearchBatched(batch, research_scann::MakeMutableSpan(b), -1, -1, -1),
        "search reloaded");
  Check(a == b, "reloaded index returns identical results");
  std::printf("re-saved and reloaded %zu points from %s: identical results\n",
              reloaded.n_points(), dir.c_str());
  std::filesystem::remove_all(dir);
  return 0;
}
