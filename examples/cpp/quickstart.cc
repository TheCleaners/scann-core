// scann-core from C++: build a tree + asymmetric-hashing index with the
// config builder, search it (one query, a batch, a parallel batch), add a
// point, and save / reload it.
//
// Build (from the scann-core build tree):
//   cmake --build <build> --target scann_core_example_quickstart
//   ./<build>/examples/scann_core_example_quickstart
//
// In your own CMake project, add_subdirectory(scann-core) and link
// scann::core (static, whole-archive) or scann::core_shared.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "google/protobuf/text_format.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann_core/config_builder.h"

using research_scann::DatapointPtr;
using research_scann::DenseDataset;
using research_scann::NNResultsVector;
using research_scann::ScannInterface;

namespace {

constexpr size_t kN = 20000, kDim = 64, kQueries = 5, kK = 5;

// Unit-norm points around 50 random centers, row-major kN x kDim.
std::vector<float> MakeDataset(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> gauss;
  std::vector<float> centers(50 * kDim);
  for (float& c : centers) c = gauss(rng);
  std::vector<float> data(n * kDim);
  for (size_t i = 0; i < n; ++i) {
    float* row = &data[i * kDim];
    double norm = 0;
    for (size_t d = 0; d < kDim; ++d) {
      row[d] = centers[(i % 50) * kDim + d] + 0.3f * gauss(rng);
      norm += row[d] * row[d];
    }
    for (size_t d = 0; d < kDim; ++d) row[d] /= std::sqrt(norm);
  }
  return data;
}

DatapointPtr<float> Row(const std::vector<float>& data, size_t i) {
  return DatapointPtr<float>(nullptr, data.data() + i * kDim, kDim, kDim);
}

// ScannInterface returns internal scores (dot products are negated so that
// smaller is better); ReshapeNNResult converts back, as the Python API does.
void Print(const ScannInterface& index, const NNResultsVector& res) {
  std::vector<uint32_t> ids(res.size());
  std::vector<float> dists(res.size());
  index.ReshapeNNResult(res, ids.data(), dists.data());
  for (size_t i = 0; i < res.size(); ++i) std::printf(" %u(%.3f)", ids[i], dists[i]);
  std::printf("\n");
}

int Fail(const char* what, const absl::Status& s) {
  std::fprintf(stderr, "%s: %s\n", what, s.ToString().c_str());
  return 1;
}

}  // namespace

int main() {
  const std::vector<float> dataset = MakeDataset(kN, 1);
  const std::vector<float> queries = MakeDataset(kQueries, 2);

  // The same options as the Python
  //   scann.scann_ops_pybind.builder(db, 5, "dot_product")
  //       .tree(num_leaves=200, num_leaves_to_search=20, random_init=False)
  //       .score_ah(2, anisotropic_quantization_threshold=0.2)
  //       .reorder(100)
  scann_core::TreeOptions tree;
  tree.num_leaves = 200;
  tree.num_leaves_to_search = 20;
  tree.random_init = false;  // k-means++: reproducible training
  scann_core::AhOptions ah;
  ah.dimensions_per_block = 2;
  ah.anisotropic_quantization_threshold = 0.2;
  scann_core::ReorderOptions reorder;
  reorder.reordering_num_neighbors = 100;
  auto config = scann_core::ConfigBuilder(kK, scann_core::DistanceMeasure::kDotProduct, kDim)
                    .Tree(tree)
                    .ScoreAh(ah)
                    .Reorder(reorder)
                    .BuildText(kN);
  if (!config.ok()) return Fail("config", config.status());

  ScannInterface index;
  if (auto s = index.Initialize(dataset, kN, *config, /*training_threads=*/0); !s.ok())
    return Fail("build", s);
  std::printf("built an index of %zu points x %zu dims\n", index.n_points(),
              size_t{index.dimensionality()});

  // One query at a time. -1 = use the value from the config.
  for (size_t q = 0; q < kQueries; ++q) {
    NNResultsVector res;
    if (auto s = index.Search(Row(queries, q), &res, /*final_nn=*/-1,
                              /*pre_reorder_nn=*/-1, /*leaves=*/-1);
        !s.ok())
      return Fail("search", s);
    std::printf("query %zu:", q);
    Print(index, res);
  }

  // A batch, and a batch split across the query thread pool.
  DenseDataset<float> batch(std::vector<float>(queries), kQueries);
  std::vector<NNResultsVector> batched(kQueries), parallel(kQueries);
  if (auto s = index.SearchBatched(batch, research_scann::MakeMutableSpan(batched), -1, -1, -1);
      !s.ok())
    return Fail("batched search", s);
  index.SetNumThreads(4);
  if (auto s = index.SearchBatchedParallel(batch, research_scann::MakeMutableSpan(parallel),
                                           -1, -1, -1, /*batch_size=*/2);
      !s.ok())
    return Fail("parallel search", s);
  std::printf("batched  query 0:");
  Print(index, batched[0]);

  // Add a point; it is immediately searchable.
  auto mutator = index.GetMutator();
  if (!mutator.ok()) return Fail("mutator", mutator.status());
  auto added = (*mutator)->AddDatapoint(Row(queries, 0), "");
  if (!added.ok()) return Fail("add", added.status());
  NNResultsVector res;
  if (auto s = index.Search(Row(queries, 0), &res, 1, -1, -1); !s.ok()) return Fail("search", s);
  std::printf("added point %u; nearest neighbour of query 0 is now:", *added);
  Print(index, res);

  // Save and reload. The directory is also loadable from Python with
  // scann.scann_ops_pybind.load_searcher(dir), and from Rust with
  // ScannIndex::load(dir).
  const auto dir = std::filesystem::temp_directory_path() / "scann-core-quickstart";
  std::filesystem::create_directories(dir);
  auto assets = index.Serialize(dir.string());
  if (!assets.ok()) return Fail("serialize", assets.status());
  std::string assets_text;
  google::protobuf::TextFormat::PrintToString(*assets, &assets_text);
  auto artifacts = ScannInterface::LoadArtifacts(dir.string(), assets_text);
  if (!artifacts.ok()) return Fail("load", artifacts.status());
  ScannInterface reloaded;
  if (auto s = reloaded.Initialize(*std::move(artifacts)); !s.ok()) return Fail("reload", s);
  std::printf("reloaded %zu points from %s\n", reloaded.n_points(), dir.c_str());
  std::filesystem::remove_all(dir);
  return 0;
}
