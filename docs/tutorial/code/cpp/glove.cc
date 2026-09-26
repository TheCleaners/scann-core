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

// Part 7: the tutorial's GloVe pipeline in C++.
//
//   glove <data dir>
//
// <data dir> is what part7_export.py wrote: glove_{train,test,neighbors}.npy
// and glove-index/ (an index saved from Python). Builds the part 3 index
// with scann_core::ConfigBuilder, evaluates it, then loads and evaluates the
// Python-built index.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "cnpy/cnpy.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann_core/config_builder.h"

using research_scann::DenseDataset;
using research_scann::NNResultsVector;
using research_scann::ScannInterface;

namespace {

double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

// recall@k of `found` against the first k of each row of `truth`.
double Recall(const std::vector<NNResultsVector>& found,
              const cnpy::NpyArray& truth, size_t k) {
  const size_t truth_cols = truth.shape[1];
  size_t hits = 0;
  for (size_t q = 0; q < found.size(); ++q) {
    const uint32_t* t = truth.data<uint32_t>() + q * truth_cols;
    for (const auto& [index, distance] : found[q])
      hits += std::count(t, t + k, index);
  }
  return static_cast<double>(hits) / (found.size() * k);
}

void Evaluate(const char* name, ScannInterface& index,
              const DenseDataset<float>& queries, const cnpy::NpyArray& truth) {
  index.SetNumThreads(std::thread::hardware_concurrency());
  std::vector<NNResultsVector> found(queries.size());
  // Best of 3, like tutorial_data.evaluate().
  double best = 1e9;
  for (int pass = 0; pass < 3; ++pass) {
    auto start = std::chrono::steady_clock::now();
    auto status = index.SearchBatchedParallel(
        queries, research_scann::MakeMutableSpan(found), /*final_nn=*/-1,
        /*pre_reorder_nn=*/-1, /*leaves=*/-1);
    best = std::min(best, Seconds(start));
    if (!status.ok()) {
      std::fprintf(stderr, "search failed: %s\n", status.ToString().c_str());
      std::exit(1);
    }
  }
  std::printf("%-26s recall@10 %.4f  %8.0f QPS\n", name,
              Recall(found, truth, 10), queries.size() / best);
}

void Check(const absl::Status& status, const char* what) {
  if (!status.ok()) {
    std::fprintf(stderr, "%s: %s\n", what, status.ToString().c_str());
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <data dir written by part7_export.py>\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  cnpy::NpyArray train = cnpy::npy_load(dir + "/glove_train.npy");
  cnpy::NpyArray test = cnpy::npy_load(dir + "/glove_test.npy");
  cnpy::NpyArray truth = cnpy::npy_load(dir + "/glove_neighbors.npy");
  const size_t n = train.shape[0], dim = train.shape[1];
  std::printf("dataset %zu x %zu, %zu queries\n", n, dim, test.shape[0]);

  // The part 3 configuration.
  scann_core::TreeOptions tree;
  tree.num_leaves = 2000;
  tree.num_leaves_to_search = 100;
  tree.training_sample_size = 250000;
  scann_core::AhOptions ah;
  ah.dimensions_per_block = 2;
  ah.anisotropic_quantization_threshold = 0.2;
  scann_core::ReorderOptions reorder;
  reorder.reordering_num_neighbors = 100;
  auto config = scann_core::ConfigBuilder(10, scann_core::DistanceMeasure::kDotProduct, dim)
                    .Tree(tree)
                    .ScoreAh(ah)
                    .Reorder(reorder)
                    .BuildText(n);
  Check(config.status(), "config");

  DenseDataset<float> queries(
      std::vector<float>(test.data<float>(), test.data<float>() + test.num_vals),
      test.shape[0]);

  ScannInterface built;
  auto start = std::chrono::steady_clock::now();
  Check(built.Initialize({train.data<float>(), train.num_vals}, n, *config,
                         /*training_threads=*/0),
        "build");
  std::printf("built in %.1f s\n", Seconds(start));
  Evaluate("built in C++", built, queries, truth);

  // The index part7_export.py saved from Python.
  start = std::chrono::steady_clock::now();
  auto artifacts = ScannInterface::LoadArtifacts(dir + "/glove-index");
  Check(artifacts.status(), "load");
  ScannInterface loaded;
  Check(loaded.Initialize(*std::move(artifacts)), "load");
  std::printf("loaded in %.1f s\n", Seconds(start));
  Evaluate("built in Python, loaded", loaded, queries, truth);
  return 0;
}
