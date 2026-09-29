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

#ifndef SCANN_SCANN_OPS_CC_SCANN_H_
#define SCANN_SCANN_OPS_CC_SCANN_H_

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "google/protobuf/io/tokenizer.h"
#include "google/protobuf/text_format.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/base/single_machine_factory_scann.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/scann_assets.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/threads.h"

namespace research_scann {

class ScannInterface {
 public:
  using ScannArtifacts =
      std::tuple<ScannConfig, shared_ptr<DenseDataset<float>>,
                 SingleMachineFactoryOptions>;

  static StatusOr<ScannArtifacts> LoadArtifacts(const ScannConfig& config,
                                                const ScannAssets& orig_assets);
  static StatusOr<ScannArtifacts> LoadArtifacts(
      const std::string& artifacts_dir,
      const std::string& scann_assets_pbtxt = "");
  // scann-core: LoadArtifacts(artifacts_dir) from the directory's files in
  // memory. `files` maps each file name, as SerializeToDirectory writes it
  // (scann_config.pb, scann_assets.pbtxt and the assets it lists), to its
  // contents; other entries are ignored. An asset path from the manifest is
  // looked up as is, then by its last component, so both relative_path
  // serializations and absolute ones load. The contents are only read
  // during the call. The TensorFlow op (tf_op/) loads indexes this way.
  static StatusOr<ScannArtifacts> LoadArtifactsFromMemory(
      const absl::flat_hash_map<std::string, absl::string_view>& files);

  static StatusOr<std::unique_ptr<SingleMachineSearcherBase<float>>>
  CreateSearcher(ScannArtifacts artifacts);

  Status Initialize(absl::string_view config_pbtxt,
                    absl::string_view scann_assets_pbtxt);
  Status Initialize(ScannConfig config, SingleMachineFactoryOptions opts,
                    ConstSpan<float> dataset,
                    ConstSpan<int32_t> datapoint_to_token,
                    ConstSpan<uint8_t> hashed_dataset,
                    ConstSpan<int8_t> int8_dataset,
                    ConstSpan<float> int8_multipliers,
                    ConstSpan<float> dp_norms, DatapointIndex n_points);
  Status Initialize(ConstSpan<float> dataset, DatapointIndex n_points,
                    absl::string_view config, int training_threads);
  Status Initialize(ScannArtifacts artifacts);

  StatusOr<typename SingleMachineSearcherBase<float>::Mutator*> GetMutator()
      const {
    return scann_->GetMutator();
  }

  StatusOr<ScannConfig> RetrainAndReindex(const string& config);

  // scann-core: an index with spherical partitioning stores unit vectors
  // (Initialize normalizes the dataset). These normalize the vectors of an
  // upsert the same way; ScannNumpy::Upsert and the Rust bindings use them,
  // and C++ callers that mutate through GetMutator() should too.
  bool NormalizesDatapoints() const;
  void NormalizeDatapoints(MutableSpan<float> rows) const;

  Status Search(const DatapointPtr<float> query, NNResultsVector* res,
                int final_nn, int pre_reorder_nn, int leaves) const;
  Status SearchBatched(const DenseDataset<float>& queries,
                       MutableSpan<NNResultsVector> res, int final_nn,
                       int pre_reorder_nn, int leaves) const;
  Status SearchBatchedParallel(const DenseDataset<float>& queries,
                               MutableSpan<NNResultsVector> res, int final_nn,
                               int pre_reorder_nn, int leaves,
                               int batch_size = 256) const;
  // scann-core: SearchBatched (parallel = false) or SearchBatchedParallel
  // over `queries`, row-major with `query_dim` (= dimensionality()) values
  // per row, which are read in place (not copied). When `on_chunk` is set, it is called
  // with (first query, that chunk's results) as soon as each chunk is done,
  // on the thread that searched it (concurrently from several threads when
  // `parallel`), e.g. to write the results out; not after a failed chunk.
  using ChunkCallback =
      std::function<void(size_t, ConstSpan<NNResultsVector>)>;
  Status SearchBatchedRows(ConstSpan<float> queries, size_t query_dim,
                           MutableSpan<NNResultsVector> res, int final_nn,
                           int pre_reorder_nn, int leaves, bool parallel,
                           int batch_size = 256,
                           const ChunkCallback& on_chunk = nullptr) const;
  StatusOr<ScannAssets> Serialize(std::string path, bool relative_path = false);
  // scann-core: writes the whole index into the existing directory `dir`,
  // including scann_assets.pbtxt, so that an interrupted write never leaves
  // a directory that loads a mix of two indexes. Serialize() writes the files
  // in place, one by one, and leaves the manifest to the caller.
  //
  // Everything is written and fsynced in a staging directory inside `dir`
  // first. Then scann_assets.pbtxt is atomically replaced by a marker that
  // makes loading fail with FailedPreconditionError, the files are renamed
  // into place, and the new scann_assets.pbtxt is renamed over the marker
  // last. Each rename is atomic, the sequence isn't: a crash midway leaves
  // a directory that fails to load, cleanly, until the next successful
  // serialize (and possibly a leftover ".scann_staging_*" directory).
  // Asset files and a scann_docids.pkl of a previous index that this one
  // doesn't have are removed.
  //
  // `extra_files` (name, contents) are committed along with the index; the
  // Python wrapper passes its scann_docids.pkl this way. Loading while
  // another thread or process serializes into the same directory, or two
  // concurrent serializes into it, aren't supported.
  Status SerializeToDirectory(
      const std::string& dir, bool relative_path = false,
      const std::vector<std::pair<std::string, std::string>>& extra_files =
          {});
  StatusOr<SingleMachineFactoryOptions> ExtractOptions();

  template <typename T_idx>
  void ReshapeNNResult(const NNResultsVector& res, T_idx* indices,
                       float* distances) const;
  template <typename T_idx>
  void ReshapeBatchedNNResult(ConstSpan<NNResultsVector> res, T_idx* indices,
                              float* distances, int neighbors_per_query) const;

  StatusOr<shared_ptr<const DenseDataset<float>>> Float32DatasetIfNeeded() {
    return scann_->SharedFloatDatasetIfNeeded();
  }

  size_t n_points() const { return scann_->DatasetSize().value(); }
  DimensionIndex dimensionality() const { return dimensionality_; }
  // scann-core: how many neighbors a search with final_nn = -1 returns per
  // query (fewer if the index has fewer points).
  int default_num_neighbors() const {
    return config_.has_exact_reordering()
               ? scann_->default_post_reordering_num_neighbors()
               : scann_->default_pre_reordering_num_neighbors();
  }
  const ScannConfig* config() {
    if (scann_->config().has_value()) config_ = *scann_->config();
    return &config_;
  }

  // scann-core: the query pool (search_batched_parallel, mutations), created
  // on first use rather than by Initialize: a process holding several
  // indexes, or one that never searches in parallel, doesn't keep idle
  // threads. Null when the pool has fewer than 2 workers (everything then
  // runs on the calling thread). Thread-safe.
  std::shared_ptr<ThreadPool> parallel_query_pool() const;

  // scann-core: the number of workers (threads, including the calling one)
  // that parallel searches and mutations use: `num_threads` - 1 pool threads
  // plus the caller; 0 or 1 means no pool. Upstream started `num_threads`
  // pool threads (so num_threads + 1 workers), while its default was
  // NumCPUs() - 1 pool threads (NumCPUs() workers); the default is now
  // scann_core::AvailableCPUs() workers (see available_cpus.h), evaluated
  // by each Initialize.
  void SetNumThreads(int num_threads);
  int NumThreads() const { return num_threads_; }

  using ScannHealthStats = SingleMachineSearcherBase<float>::HealthStats;
  StatusOr<ScannHealthStats> GetHealthStats() const;
  Status InitializeHealthStats();

 private:
  SearchParameters GetSearchParameters(int final_nn, int pre_reorder_nn,
                                       int leaves) const;
  vector<SearchParameters> GetSearchParametersBatched(
      int batch_size, int final_nn, int pre_reorder_nn, int leaves,
      bool set_unspecified) const;
  // scann-core: SearchBatched on a view.
  Status SearchBatchedView(const DefaultDenseDatasetView<float>& queries,
                           MutableSpan<NNResultsVector> res, int final_nn,
                           int pre_reorder_nn, int leaves) const;
  DimensionIndex dimensionality_;
  std::unique_ptr<SingleMachineSearcherBase<float>> scann_;
  ScannConfig config_;

  float result_multiplier_;

  size_t min_batch_size_;

  // scann-core: see parallel_query_pool() and SetNumThreads().
  int num_threads_ = 1;
  // The training_threads the index was built with (0: the default), which
  // RetrainAndReindex also uses; the query pool's size when 0.
  int training_threads_ = 0;
  mutable absl::Mutex pool_mu_;
  mutable std::shared_ptr<ThreadPool> parallel_query_pool_
      ABSL_GUARDED_BY(pool_mu_);
  mutable bool pool_started_ ABSL_GUARDED_BY(pool_mu_) = false;
};

template <typename T_idx>
void ScannInterface::ReshapeNNResult(const NNResultsVector& res, T_idx* indices,
                                     float* distances) const {
  for (const auto& p : res) {
    *(indices++) = static_cast<T_idx>(p.first);
    *(distances++) = result_multiplier_ * p.second;
  }
}

template <typename T_idx>
void ScannInterface::ReshapeBatchedNNResult(ConstSpan<NNResultsVector> res,
                                            T_idx* indices, float* distances,
                                            int neighbors_per_query) const {
  for (const auto& result_vec : res) {
    DCHECK_LE(result_vec.size(), neighbors_per_query);
    for (const auto& pair : result_vec) {
      *(indices++) = static_cast<T_idx>(pair.first);
      *(distances++) = result_multiplier_ * pair.second;
    }

    for (int i = result_vec.size(); i < neighbors_per_query; i++) {
      *(indices++) = 0;
      *(distances++) = std::numeric_limits<float>::quiet_NaN();
    }
  }
}

// scann-core: upstream discarded ParseFromString's result, so a malformed or
// misspelled config (or scann_assets.pbtxt) was reported as OK and ScaNN ran
// with whatever part of it had parsed before the error.
class TextProtoErrorCollector : public ::google::protobuf::io::ErrorCollector {
 public:
  void RecordError(int line, ::google::protobuf::io::ColumnNumber column,
                   absl::string_view message) override {
    if (!errors_.empty()) errors_ += "; ";
    absl::StrAppend(&errors_, line + 1, ":", column + 1, ": ", message);
  }
  const std::string& errors() const { return errors_; }

 private:
  std::string errors_;
};

template <typename T>
Status ParseTextProto(T* proto, absl::string_view proto_str) {
  TextProtoErrorCollector errors;
  ::google::protobuf::TextFormat::Parser parser;
  parser.RecordErrorsTo(&errors);
  if (!parser.ParseFromString(proto_str, proto)) {
    return InvalidArgumentError(absl::StrCat("Failed to parse ",
                                             proto->GetTypeName(),
                                             " text proto: ", errors.errors()));
  }
  return OkStatus();
}

}  // namespace research_scann

#endif
