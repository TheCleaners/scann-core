// Copyright 2026 The Google Research Authors.
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
// Modified in 2026 by Elias Benali (@ebenali) and TheCleaners for
// scann-core (a derived work of ScaNN, not an official Google product);
// see NOTICE.

#include "scann/scann_ops/cc/scann_npy.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "pybind11/gil.h"
#include "pybind11/pytypes.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/common.h"
#include "scann/utils/io_oss_wrapper.h"
#include "scann/utils/single_machine_autopilot.h"
#include "scann/utils/types.h"

namespace research_scann {
using MutationOptions = UntypedSingleMachineSearcherBase::MutationOptions;
using PrecomputedMutationArtifacts =
    UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts;

void RuntimeErrorIfNotOk(const char* prefix, const Status& status) {
  if (!status.ok()) {
    std::string msg = prefix + std::string(status.message());
    throw std::runtime_error(msg);
  }
}

template <typename T>
T ValueOrRuntimeError(StatusOr<T> status_or, const char* prefix) {
  RuntimeErrorIfNotOk(prefix, status_or.status());
  return status_or.value();
}

// scann-core: the most datapoints an index can hold. Indices are 32-bit and
// kInvalidDatapointIndex (the largest) is reserved, so the last valid index
// is kInvalidDatapointIndex - 1.
constexpr uint64_t kMaxDatapoints = kInvalidDatapointIndex;

ScannNumpy::ScannNumpy(const std::string& artifacts_dir,
                       const std::string& scann_assets_pbtxt) {
  auto status_or =
      ScannInterface::LoadArtifacts(artifacts_dir, scann_assets_pbtxt);
  RuntimeErrorIfNotOk("Error loading artifacts: ", status_or.status());
  RuntimeErrorIfNotOk("Error initializing searcher: ",
                      scann_.Initialize(status_or.value()));
}

ScannNumpy::ScannNumpy(const np_row_major_arr<float>& np_dataset,
                       absl::string_view config, int training_threads) {
  if (np_dataset.ndim() != 2)
    throw std::invalid_argument("Dataset input must be two-dimensional");
  // scann-core: datapoint indices are 32-bit. Upstream passed the row count
  // on truncated to 32 bits, building an index over the wrong number of
  // rows (or failing with an unrelated shape error).
  if (static_cast<uint64_t>(np_dataset.shape()[0]) > kMaxDatapoints)
    throw std::invalid_argument(absl::StrCat(
        "Dataset has ", np_dataset.shape()[0], " rows; ScaNN supports at most ",
        kMaxDatapoints, " (datapoint indices are 32-bit)"));
  ConstSpan<float> dataset(np_dataset.data(), np_dataset.size());
  pybind11::gil_scoped_release gil_release;
  RuntimeErrorIfNotOk("Error initializing searcher: ",
                      scann_.Initialize(dataset, np_dataset.shape()[0], config,
                                        training_threads));
}

vector<DatapointIndex> ScannNumpy::Upsert(
    vector<std::optional<DatapointIndex>> indices,
    vector<np_row_major_arr<float>>& vecs, int batch_size) {
  if (batch_size < 1)
    throw std::invalid_argument("Upsert batch_size must be >= 1.");
  pybind11::gil_scoped_release gil_release;
  absl::MutexLock lock(&mu_);
  if (indices.size() != vecs.size())
    throw std::runtime_error("Upsert input size must match.");
  // scann-core: validate every row before mutating anything. Upstream
  // checked nothing here: a wrong-sized vector failed deep in the mutator
  // with an uninformative "SCANN_RET_CHECK failure"; a NaN/infinity vector
  // got partition token -1 in tree indexes and the mutator then indexed
  // leaf_mutators_[-1] (segfault); and any row failing partway through a
  // batch left the earlier rows applied, so the index and the caller's docid
  // bookkeeping diverged.
  const DatapointIndex n_points = scann_.n_points();
  const uint64_t n_adds =
      std::count_if(indices.begin(), indices.end(),
                    [](const auto& index) { return !index.has_value(); });
  if (n_points + n_adds > kMaxDatapoints)
    throw std::invalid_argument(absl::StrCat(
        "Upsert would grow the index to ", n_points + n_adds,
        " datapoints; ScaNN supports at most ", kMaxDatapoints,
        " (datapoint indices are 32-bit)"));
  for (size_t row : Seq(vecs.size())) {
    const auto& vec = vecs[row];
    if (vec.size() != scann_.dimensionality())
      throw std::invalid_argument(absl::StrCat(
          "Upsert vector has dimensionality ", vec.size(),
          ", but the dataset has ", scann_.dimensionality(), " (row ", row,
          ")"));
    const float* data = vec.data();
    for (size_t d : Seq(vec.size()))
      if (!std::isfinite(data[d]))
        throw std::invalid_argument(absl::StrCat(
            "Upsert vector at row ", row,
            " contains NaN or infinity; ScaNN only supports finite values."));
    if (indices[row].has_value() && indices[row].value() >= n_points)
      throw std::invalid_argument(absl::StrCat(
          "Upsert index ", indices[row].value(), " at row ", row,
          " is out of range for an index with ", n_points, " points"));
  }
  auto mutator =
      ValueOrRuntimeError(scann_.GetMutator(), "Failed to fetch mutator: ");
  if (batch_size > 1)
    mutator->set_mutation_threadpool(scann_.parallel_query_pool());

  DatapointIndex n = vecs.size();
  vector<DatapointIndex> result;

  // scann-core: an index with spherical partitioning stores unit vectors;
  // see ScannInterface::NormalizeDatapoints.
  const size_t dim = scann_.dimensionality();
  vector<float> normalized;
  if (scann_.NormalizesDatapoints()) {
    normalized.resize(vecs.size() * dim);
    for (size_t row : Seq(vecs.size()))
      std::copy(vecs[row].data(), vecs[row].data() + dim,
                normalized.begin() + row * dim);
    scann_.NormalizeDatapoints(MakeMutableSpan(normalized));
  }
  auto row_ptr = [&](size_t row) {
    return MakeDatapointPtr(
        normalized.empty() ? vecs[row].data() : normalized.data() + row * dim,
        dim);
  };

  for (size_t b : Seq(DivRoundUp(n, batch_size))) {
    size_t begin = batch_size * b;
    size_t bs = std::min<DatapointIndex>(n - begin, batch_size);
    DenseDataset<float> ds;
    for (size_t i : Seq(bs))
      RuntimeErrorIfNotOk("Error appending datapoint.",
                          ds.Append(row_ptr(begin + i)));
    auto precomputed = mutator->ComputePrecomputedMutationArtifacts(
        ds, scann_.parallel_query_pool());

    for (size_t i : Seq(bs)) {
      auto& index = indices[begin + i];
      auto mo = MutationOptions{.precomputed_mutation_artifacts =
                                    precomputed[i].get()};
      if (!index.has_value()) {
        result.push_back(ValueOrRuntimeError(
            mutator->AddDatapoint(row_ptr(begin + i), "", mo),
            "Failed to add datapoint: "));
      } else {
        result.push_back(ValueOrRuntimeError(
            mutator->UpdateDatapoint(row_ptr(begin + i), index.value(), mo),
            "Failed to update datapoint: "));
      }
    }
    auto statusor = mutator->IncrementalMaintenance();
    RuntimeErrorIfNotOk("Error performing incremental maintenance ",
                        statusor.status());
    if (statusor.value().has_value()) {
      RebalanceLocked("");
      mutator =
          ValueOrRuntimeError(scann_.GetMutator(), "Failed to fetch mutator: ");
      if (batch_size > 1)
        mutator->set_mutation_threadpool(scann_.parallel_query_pool());
    }
  }
  return result;
}

vector<DatapointIndex> ScannNumpy::Delete(vector<DatapointIndex> indices) {
  pybind11::gil_scoped_release gil_release;
  absl::MutexLock lock(&mu_);
  auto mutator =
      ValueOrRuntimeError(scann_.GetMutator(), "Failed to fetch mutator: ");
  mutator->set_mutation_threadpool(scann_.parallel_query_pool());
  vector<DatapointIndex> result;
  for (const auto& index : indices) {
    RuntimeErrorIfNotOk("Failed to delete datapoint: ",
                        mutator->RemoveDatapoint(index));
    auto statusor = mutator->IncrementalMaintenance();
    RuntimeErrorIfNotOk("Error performing incremental maintenance ",
                        statusor.status());
    if (statusor.value().has_value()) {
      RebalanceLocked("");
      mutator =
          ValueOrRuntimeError(scann_.GetMutator(), "Failed to fetch mutator: ");
      mutator->set_mutation_threadpool(scann_.parallel_query_pool());
    }
    result.push_back(scann_.n_points());
  }
  return result;
}

int ScannNumpy::Rebalance(const string& config) {
  pybind11::gil_scoped_release gil_release;
  absl::MutexLock lock(&mu_);
  return RebalanceLocked(config);
}

int ScannNumpy::RebalanceLocked(const string& config) {
  auto statusor = scann_.RetrainAndReindex(config);
  if (!statusor.ok()) {
    RuntimeErrorIfNotOk("Failed to retrain searcher: ", statusor.status());
    return -1;
  }

  return scann_.n_points();
}

size_t ScannNumpy::Size() const {
  pybind11::gil_scoped_release gil_release;
  absl::ReaderMutexLock lock(&mu_);
  return scann_.n_points();
}

void ScannNumpy::Reserve(size_t num_datapoints) {
  pybind11::gil_scoped_release gil_release;
  absl::MutexLock lock(&mu_);
  auto mutator =
      ValueOrRuntimeError(scann_.GetMutator(), "Failed to fetch mutator: ");
  mutator->Reserve(num_datapoints);
}

void ScannNumpy::SetNumThreads(int num_threads) {
  pybind11::gil_scoped_release gil_release;
  absl::MutexLock lock(&mu_);
  scann_.SetNumThreads(num_threads);
}

string ScannNumpy::SuggestAutopilot(absl::string_view config_str,
                                    DatapointIndex n, DimensionIndex dim) {
  ScannConfig config;
  RuntimeErrorIfNotOk("Failed to parse config: ",
                      ParseTextProto(&config, config_str));
  auto status_or = Autopilot(config, nullptr, n, dim);
  RuntimeErrorIfNotOk("Failed to suggest autopilot config: ",
                      status_or.status());
  std::string result;
  google::protobuf::TextFormat::PrintToString(status_or.value(), &result);
  return result;
}

string ScannNumpy::Config() {
  pybind11::gil_scoped_release gil_release;
  // Exclusive: ScannInterface::config() refreshes a cached copy.
  absl::MutexLock lock(&mu_);
  std::string config_str;
  google::protobuf::TextFormat::PrintToString(*scann_.config(), &config_str);
  return config_str;
}

std::pair<pybind11::array_t<DatapointIndex>, pybind11::array_t<float>>
ScannNumpy::Search(const np_row_major_arr<float>& query, int final_nn,
                   int pre_reorder_nn, int leaves) {
  if (query.ndim() != 1)
    throw std::invalid_argument("Query must be one-dimensional");

  DatapointPtr<float> ptr(nullptr, query.data(), query.size(), query.size());
  vector<DatapointIndex> idx;
  vector<float> dis;
  {
    pybind11::gil_scoped_release gil_release;
    absl::ReaderMutexLock lock(&mu_);
    NNResultsVector res;
    auto status = scann_.Search(ptr, &res, final_nn, pre_reorder_nn, leaves);
    RuntimeErrorIfNotOk("Error during search: ", status);
    idx.resize(res.size());
    dis.resize(res.size());
    scann_.ReshapeNNResult(res, idx.data(), dis.data());
  }
  return {pybind11::array_t<DatapointIndex>(idx.size(), idx.data()),
          pybind11::array_t<float>(dis.size(), dis.data())};
}

std::pair<pybind11::array_t<DatapointIndex>, pybind11::array_t<float>>
ScannNumpy::SearchBatched(const np_row_major_arr<float>& queries, int final_nn,
                          int pre_reorder_nn, int leaves, bool parallel,
                          int batch_size) {
  if (queries.ndim() != 2)
    throw std::invalid_argument("Queries must be in two-dimensional array");

  // scann-core: upstream turned zero queries into a dataset of
  // dimensionality 0 and failed with a misleading dimensionality error (or a
  // bare RET_CHECK failure from the parallel path). Return empty results
  // shaped like a normal call's, (0, k), after the same argument checks.
  if (queries.shape()[0] == 0) {
    if (final_nn == 0)
      throw std::invalid_argument("final_num_neighbors must be > 0");
    int k;
    {
      pybind11::gil_scoped_release gil_release;
      absl::ReaderMutexLock lock(&mu_);
      if (static_cast<size_t>(queries.shape()[1]) != scann_.dimensionality())
        throw std::invalid_argument(absl::StrCat(
            "Queries have dimensionality ", queries.shape()[1],
            ", but the dataset has ", scann_.dimensionality()));
      k = final_nn > 0 ? final_nn : scann_.default_num_neighbors();
    }
    std::vector<long> shape = {0, static_cast<long>(k)};
    return {pybind11::array_t<DatapointIndex>(shape),
            pybind11::array_t<float>(shape)};
  }

  vector<float> queries_vec(queries.data(), queries.data() + queries.size());
  auto query_dataset =
      DenseDataset<float>(std::move(queries_vec), queries.shape()[0]);

  std::vector<NNResultsVector> res(query_dataset.size());
  vector<DatapointIndex> idx;
  vector<float> dis;
  {
    pybind11::gil_scoped_release gil_release;
    absl::ReaderMutexLock lock(&mu_);
    Status status;
    if (parallel)
      status = scann_.SearchBatchedParallel(query_dataset, MakeMutableSpan(res),
                                            final_nn, pre_reorder_nn, leaves,
                                            batch_size);
    else
      status = scann_.SearchBatched(query_dataset, MakeMutableSpan(res),
                                    final_nn, pre_reorder_nn, leaves);
    RuntimeErrorIfNotOk("Error during search: ", status);

    for (const auto& nn_res : res)
      final_nn = std::max<int>(final_nn, nn_res.size());
    idx.resize(query_dataset.size() * std::max(final_nn, 0));
    dis.resize(idx.size());
    scann_.ReshapeBatchedNNResult(MakeConstSpan(res), idx.data(), dis.data(),
                                  final_nn);
  }
  std::vector<long> shape = {static_cast<long>(query_dataset.size()),
                             static_cast<long>(final_nn)};
  return {pybind11::array_t<DatapointIndex>(shape, idx.data()),
          pybind11::array_t<float>(shape, dis.data())};
}

void ScannNumpy::Serialize(std::string path, bool relative_path) {
  pybind11::gil_scoped_release gil_release;
  absl::MutexLock lock(&mu_);
  StatusOr<ScannAssets> assets_or = scann_.Serialize(path, relative_path);
  RuntimeErrorIfNotOk("Failed to extract SingleMachineFactoryOptions: ",
                      assets_or.status());
  std::string assets_or_text;
  google::protobuf::TextFormat::PrintToString(*assets_or, &assets_or_text);
  RuntimeErrorIfNotOk("Failed to write ScannAssets proto: ",
                      OpenSourceableFileWriter(path + "/scann_assets.pbtxt")
                          .Write(assets_or_text));
}

pybind11::dict ScannNumpy::GetHealthStats() const {
  StatusOr<ScannInterface::ScannHealthStats> r;
  {
    pybind11::gil_scoped_release gil_release;
    absl::MutexLock lock(&mu_);
    r = scann_.GetHealthStats();
  }
  RuntimeErrorIfNotOk("Error getting health stats: ", r.status());

  using namespace pybind11::literals;

  return pybind11::dict("avg_quantization_error"_a = r->avg_quantization_error,
                        "partition_weighted_avg_relative_imbalance"_a =
                            r->partition_weighted_avg_relative_imbalance,
                        "partition_avg_relative_positive_imbalance"_a =
                            r->partition_avg_relative_positive_imbalance,
                        "sum_partition_sizes"_a = r->sum_partition_sizes);
}

void ScannNumpy::InitializeHealthStats() {
  pybind11::gil_scoped_release gil_release;
  absl::MutexLock lock(&mu_);
  Status status = scann_.InitializeHealthStats();
  RuntimeErrorIfNotOk("Error initializing health stats: ", status);
}

}  // namespace research_scann
