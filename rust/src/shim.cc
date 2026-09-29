// Copyright 2026 The Google Research Authors.
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
//
// Upsert/delete logic ported from scann/scann_ops/cc/scann_npy.cc.

#include "shim.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "scann-core/rust/src/bridge.rs.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/utils/common.h"
#include "scann/utils/io_oss_wrapper.h"
#include "scann/utils/types.h"

namespace scann_core_ffi {

static_assert(std::is_same_v<research_scann::DatapointIndex, uint32_t>,
              "the Rust bindings assume 32-bit datapoint indices");

namespace {

using research_scann::ConstSpan;
using research_scann::DatapointIndex;
using research_scann::DatapointPtr;
using research_scann::DenseDataset;
using research_scann::NNResultsVector;

// cxx turns an exception thrown from a function declared `-> Result<T>` in
// bridge.rs into `Err(cxx::Exception)` on the Rust side.
void ThrowIfNotOk(const absl::Status& status, absl::string_view context) {
  if (!status.ok())
    throw std::runtime_error(absl::StrCat(context, ": ", status.ToString()));
}

template <typename T>
T ValueOrThrow(absl::StatusOr<T> status_or, absl::string_view context) {
  ThrowIfNotOk(status_or.status(), context);
  return *std::move(status_or);
}

absl::string_view View(rust::Str s) { return {s.data(), s.size()}; }

Neighbors ToNeighbors(const ScannIndex& idx, const NNResultsVector& res) {
  std::vector<DatapointIndex> indices(res.size());
  std::vector<float> distances(res.size());
  // Applies the result multiplier (dot product distances are negated
  // internally), exactly like the Python API.
  idx.ReshapeNNResult(res, indices.data(), distances.data());
  Neighbors out;
  out.indices.reserve(res.size());
  out.distances.reserve(res.size());
  for (size_t i = 0; i < res.size(); ++i) {
    out.indices.push_back(indices[i]);
    out.distances.push_back(distances[i]);
  }
  return out;
}

research_scann::SingleMachineSearcherBase<float>::Mutator* GetMutator(
    ScannIndex& idx) {
  return ValueOrThrow(idx.GetMutator(), "Failed to fetch mutator");
}

// Retrains when the mutator reports that incremental maintenance isn't
// enough; mirrors ScannNumpy::Upsert/Delete (scann_npy.cc). Returns the
// mutator to continue with (retraining replaces the searcher).
research_scann::SingleMachineSearcherBase<float>::Mutator* MaintainOrRetrain(
    ScannIndex& idx,
    research_scann::SingleMachineSearcherBase<float>::Mutator* mutator,
    bool attach_pool) {
  auto maintenance = ValueOrThrow(mutator->IncrementalMaintenance(),
                                  "Error performing incremental maintenance");
  if (!maintenance.has_value()) return mutator;
  ThrowIfNotOk(idx.RetrainAndReindex("").status(), "Failed to retrain searcher");
  mutator = GetMutator(idx);
  if (attach_pool) mutator->set_mutation_threadpool(idx.parallel_query_pool());
  return mutator;
}

scann_core::Quantization ToCore(Quantization q) {
  switch (q) {
    case Quantization::Int8: return scann_core::Quantization::kInt8;
    case Quantization::Bfloat16: return scann_core::Quantization::kBfloat16;
    default: return scann_core::Quantization::kFloat32;
  }
}

std::optional<double> OptF(double v) {
  return std::isnan(v) ? std::nullopt : std::optional<double>(v);
}

}  // namespace

// --- construction -----------------------------------------------------------

std::unique_ptr<ScannIndex> scann_new(rust::Slice<const float> dataset,
                                      uint64_t n_points, rust::Str config,
                                      int32_t training_threads) {
  // Datapoint indices are 32-bit; don't let the cast below truncate.
  // (ScannIndex::with_training_threads already checks this.)
  if (n_points > research_scann::kInvalidDatapointIndex)
    throw std::invalid_argument(absl::StrCat(
        n_points, " datapoints exceed the 32-bit index space"));
  auto idx = std::make_unique<ScannIndex>();
  ThrowIfNotOk(idx->Initialize(ConstSpan<float>(dataset.data(), dataset.size()),
                               static_cast<DatapointIndex>(n_points),
                               View(config), training_threads),
               "Error initializing searcher");
  return idx;
}

std::unique_ptr<ScannIndex> scann_load(rust::Str dir) {
  auto artifacts = ValueOrThrow(
      ScannIndex::LoadArtifacts(std::string(View(dir))), "Error loading artifacts");
  auto idx = std::make_unique<ScannIndex>();
  ThrowIfNotOk(idx->Initialize(std::move(artifacts)), "Error initializing searcher");
  return idx;
}

// --- search -----------------------------------------------------------------

Neighbors scann_search(const ScannIndex& idx, rust::Slice<const float> query,
                       int32_t final_nn, int32_t pre_reorder_nn,
                       int32_t leaves) {
  DatapointPtr<float> ptr(nullptr, query.data(), query.size(), query.size());
  NNResultsVector res;
  ThrowIfNotOk(idx.Search(ptr, &res, final_nn, pre_reorder_nn, leaves),
               "Error during search");
  return ToNeighbors(idx, res);
}

rust::Vec<Neighbors> scann_search_batched(const ScannIndex& idx,
                                          rust::Slice<const float> queries,
                                          uint64_t n_queries, int32_t final_nn,
                                          int32_t pre_reorder_nn,
                                          int32_t leaves, bool parallel,
                                          int32_t batch_size) {
  rust::Vec<Neighbors> out;
  if (n_queries == 0) return out;
  // The queries are read in place (no copy).
  std::vector<NNResultsVector> res(n_queries);
  ThrowIfNotOk(
      idx.SearchBatchedRows(
          research_scann::ConstSpan<float>(queries.data(), queries.size()),
          queries.size() / n_queries, research_scann::MakeMutableSpan(res),
          final_nn, pre_reorder_nn, leaves, parallel, parallel ? batch_size : 256),
      "Error during search");
  out.reserve(res.size());
  for (const auto& r : res) out.push_back(ToNeighbors(idx, r));
  return out;
}

// --- accessors --------------------------------------------------------------

uint64_t scann_size(const ScannIndex& idx) { return idx.n_points(); }

uint64_t scann_dimensionality(const ScannIndex& idx) {
  return idx.dimensionality();
}

rust::String scann_config(ScannIndex& idx) {
  std::string text;
  google::protobuf::TextFormat::PrintToString(*idx.config(), &text);
  return rust::String(text);
}

HealthStats scann_health_stats(const ScannIndex& idx) {
  auto s = ValueOrThrow(idx.GetHealthStats(), "Failed to get health stats");
  return HealthStats{s.partition_weighted_avg_relative_imbalance,
                     s.partition_avg_relative_positive_imbalance,
                     s.avg_quantization_error, s.sum_partition_sizes};
}

// --- persistence and mutation -------------------------------------------------

void scann_serialize(ScannIndex& idx, rust::Str dir, bool relative_path) {
  // Staged and committed so an interrupted serialize can't leave a directory
  // that loads a mix of two indexes (see SerializeToDirectory).
  ThrowIfNotOk(idx.SerializeToDirectory(std::string(View(dir)), relative_path),
               "Failed to serialize searcher");
}

void scann_set_num_threads(ScannIndex& idx, int32_t num_threads) {
  idx.SetNumThreads(num_threads);
}

void scann_reserve(ScannIndex& idx, uint64_t n_points) {
  GetMutator(idx)->Reserve(n_points);
}

rust::Vec<uint32_t> scann_upsert(ScannIndex& idx, rust::Slice<const int64_t> ids,
                                 rust::Slice<const float> vectors,
                                 int32_t batch_size) {
  const size_t n = ids.size();
  const size_t dim = idx.dimensionality();
  if (batch_size < 1) throw std::invalid_argument("batch_size must be >= 1");
  if (vectors.size() != n * dim)
    throw std::invalid_argument(absl::StrCat(
        "upsert: ", vectors.size(), " floats for ", n, " rows of dimensionality ", dim));
  const DatapointIndex size = idx.n_points();
  uint64_t n_adds = 0;
  for (int64_t id : ids) {
    if (id >= static_cast<int64_t>(size))
      throw std::invalid_argument(
          absl::StrCat("upsert: index ", id, " out of range (", size, " points)"));
    if (id < 0) ++n_adds;
  }
  if (size + n_adds > research_scann::kInvalidDatapointIndex)
    throw std::invalid_argument(absl::StrCat(
        "upsert: ", size + n_adds, " datapoints exceed the 32-bit index space"));
  // Validate every row before mutating anything: a NaN/infinity vector gets
  // partition token -1 in tree indexes (upstream's mutator then indexed
  // leaf_mutators_[-1]), and a row failing partway through a batch would
  // leave the earlier rows applied.
  for (size_t i = 0; i < vectors.size(); ++i)
    if (!std::isfinite(vectors[i]))
      throw std::invalid_argument(absl::StrCat(
          "upsert: vector at row ", i / dim,
          " contains NaN or infinity; ScaNN only supports finite values"));

  // The vectors as the index stores them: unit vectors with spherical
  // partitioning, with l2_as_dot_product's extra coordinate; see
  // ScannInterface::ToStoredDatapoints.
  std::vector<float> stored;
  const float* rows = vectors.data();
  if (idx.TransformsDatapoints()) {
    idx.ToStoredDatapoints(ConstSpan<float>(vectors.data(), vectors.size()),
                           &stored);
    rows = stored.data();
  }
  const size_t stored_dim = idx.stored_dimensionality();

  const bool attach_pool = batch_size > 1;
  auto* mutator = GetMutator(idx);
  if (attach_pool) mutator->set_mutation_threadpool(idx.parallel_query_pool());
  rust::Vec<uint32_t> result;
  result.reserve(n);
  for (size_t begin = 0; begin < n; begin += batch_size) {
    const size_t bs = std::min<size_t>(n - begin, batch_size);
    DenseDataset<float> ds(
        std::vector<float>(rows + begin * stored_dim,
                           rows + (begin + bs) * stored_dim),
        bs);
    auto precomputed =
        mutator->ComputePrecomputedMutationArtifacts(ds, idx.parallel_query_pool());
    for (size_t i = 0; i < bs; ++i) {
      DatapointPtr<float> dptr(nullptr, rows + (begin + i) * stored_dim,
                               stored_dim, stored_dim);
      research_scann::UntypedSingleMachineSearcherBase::MutationOptions mo{.precomputed_mutation_artifacts =
                                             precomputed[i].get()};
      const int64_t id = ids[begin + i];
      result.push_back(
          id < 0 ? ValueOrThrow(mutator->AddDatapoint(dptr, "", mo),
                                "Failed to add datapoint")
                 : ValueOrThrow(mutator->UpdateDatapoint(
                                    dptr, static_cast<DatapointIndex>(id), mo),
                                "Failed to update datapoint"));
    }
  }
  // Incremental maintenance once per call, after every row is in (as
  // ScannNumpy::Upsert; it ran after every batch before 0.2.1).
  MaintainOrRetrain(idx, mutator, attach_pool);
  return result;
}

void scann_delete(ScannIndex& idx, rust::Slice<const uint32_t> ids) {
  auto* mutator = GetMutator(idx);
  mutator->set_mutation_threadpool(idx.parallel_query_pool());
  for (uint32_t id : ids) {
    ThrowIfNotOk(mutator->RemoveDatapoint(id), "Failed to delete datapoint");
    mutator = MaintainOrRetrain(idx, mutator, /*attach_pool=*/true);
  }
}

void scann_rebalance(ScannIndex& idx, rust::Str config) {
  ThrowIfNotOk(idx.RetrainAndReindex(std::string(View(config))).status(),
               "Failed to retrain searcher");
}

void scann_initialize_health_stats(ScannIndex& idx) {
  ThrowIfNotOk(idx.InitializeHealthStats(), "Failed to initialize health stats");
}

// --- ConfigBuilder ------------------------------------------------------------

std::unique_ptr<ConfigBuilder> config_builder_new(int32_t num_neighbors,
                                                  DistanceMeasure distance,
                                                  uint32_t dimensionality) {
  return std::make_unique<ConfigBuilder>(
      num_neighbors,
      distance == DistanceMeasure::SquaredL2 ? scann_core::DistanceMeasure::kSquaredL2
                                             : scann_core::DistanceMeasure::kDotProduct,
      dimensionality);
}

void config_builder_tree(ConfigBuilder& b, const FfiTreeOptions& o) {
  scann_core::TreeOptions t;
  t.num_leaves = o.num_leaves;
  t.num_leaves_to_search = o.num_leaves_to_search;
  t.training_sample_size = o.training_sample_size;
  t.min_partition_size = o.min_partition_size;
  t.training_iterations = o.training_iterations;
  t.spherical = o.spherical;
  t.quantize_centroids = o.quantize_centroids;
  t.random_init = o.random_init;
  if (o.incremental_threshold_points >= 0)
    t.incremental_threshold_points = o.incremental_threshold_points;
  t.incremental_threshold_fraction = OptF(o.incremental_threshold_fraction);
  t.avq = OptF(o.avq);
  t.soar_lambda = OptF(o.soar_lambda);
  t.overretrieve_factor = OptF(o.overretrieve_factor);
  b.Tree(t);
}

void config_builder_upper_tree(ConfigBuilder& b, const FfiUpperTreeOptions& o) {
  scann_core::UpperTreeOptions u;
  u.num_leaves = o.num_leaves;
  u.num_leaves_to_search = o.num_leaves_to_search;
  u.avq = o.avq;  // NaN = unset, as in the C++ struct
  u.soar_lambda = OptF(o.soar_lambda);
  u.overretrieve_factor = OptF(o.overretrieve_factor);
  u.scoring_mode = ToCore(o.scoring_mode);
  u.anisotropic_quantization_threshold = o.anisotropic_quantization_threshold;
  b.UpperTree(u);
}

void config_builder_score_ah(ConfigBuilder& b, const FfiAhOptions& o) {
  scann_core::AhOptions a;
  a.dimensions_per_block = o.dimensions_per_block;
  a.anisotropic_quantization_threshold = o.anisotropic_quantization_threshold;
  a.training_sample_size = o.training_sample_size;
  a.hash_type = o.hash_type == HashType::Lut256 ? scann_core::HashType::kLut256
                                                : scann_core::HashType::kLut16;
  a.training_iterations = o.training_iterations;
  if (o.has_residual_quantization) a.residual_quantization = o.residual_quantization;
  b.ScoreAh(a);
}

void config_builder_score_brute_force(ConfigBuilder& b, Quantization quantize) {
  b.ScoreBruteForce(ToCore(quantize));
}

void config_builder_reorder(ConfigBuilder& b, const FfiReorderOptions& o) {
  scann_core::ReorderOptions r;
  r.reordering_num_neighbors = o.reordering_num_neighbors;
  r.quantize = ToCore(o.quantize);
  r.anisotropic_quantization_threshold = o.anisotropic_quantization_threshold;
  b.Reorder(r);
}

void config_builder_pca(ConfigBuilder& b, const FfiPcaOptions& o) {
  scann_core::PcaOptions p;
  if (o.reduction_dim >= 0) p.reduction_dim = o.reduction_dim;
  p.pca_significance_threshold = OptF(o.pca_significance_threshold);
  p.pca_truncation_threshold = o.pca_truncation_threshold;
  b.Pca(p);
}

void config_builder_truncate(ConfigBuilder& b, int32_t reduction_dim) {
  b.Truncate(reduction_dim);
}

void config_builder_autopilot(ConfigBuilder& b, IncrementalMode mode,
                              Quantization quantize) {
  b.Autopilot(mode == IncrementalMode::Online ? scann_core::IncrementalMode::kOnline
              : mode == IncrementalMode::OnlineIncremental
                  ? scann_core::IncrementalMode::kOnlineIncremental
                  : scann_core::IncrementalMode::kNone,
              ToCore(quantize));
}

void config_builder_l2_as_dot_product(ConfigBuilder& b, double scale,
                                      double center) {
  scann_core::L2AsDotProductOptions o;
  o.scale = OptF(scale);
  o.center = OptF(center);
  b.L2AsDotProduct(o);
}

rust::String config_builder_build(const ConfigBuilder& b, uint64_t num_points) {
  return rust::String(ValueOrThrow(b.BuildText(num_points), "Invalid ScaNN config"));
}

}  // namespace scann_core_ffi
