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

#include "scann/scann_ops/cc/scann.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <system_error>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "google/protobuf/message.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/partitioning/partitioner.pb.h"
#include "scann/proto/brute_force.pb.h"
#include "scann/proto/centers.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/scann_ops/scann_assets.pb.h"
#include "scann/trees/kmeans_tree/kmeans_tree.pb.h"
#include "scann/tree_x_hybrid/tree_x_params.h"
#include "scann/utils/common.h"
#include "scann/utils/io_npy.h"
#include "scann/utils/io_oss_wrapper.h"
#include "scann/utils/scann_config_utils.h"
#include "scann/utils/single_machine_retraining.h"
#include "scann/utils/threads.h"
#include "scann/utils/types.h"
#include "scann_core/available_cpus.h"

namespace research_scann {
namespace {

constexpr const int32_t kSoarEmptyToken = -1;

bool HasSoar(const ScannConfig& config) {
  return config.partitioning().database_spilling().spilling_type() ==
         DatabaseSpillingConfig::TWO_CENTER_ORTHOGONALITY_AMPLIFIED;
}

// scann-core: upstream used absl::base_internal::NumCPUs(), the online CPU
// count cached for the whole process, which ignores the CPU affinity (taskset,
// cpusets) and container CPU quotas: a process pinned to 4 of 64 CPUs started
// 63 query threads per index and trained with 64. See available_cpus.h.
int GetNumCPUs() { return scann_core::AvailableCPUs(); }

unique_ptr<DenseDataset<float>> InitDataset(
    ConstSpan<float> dataset, DatapointIndex n_points,
    DimensionIndex n_dim = kInvalidDimension) {
  if (dataset.empty() && n_dim == kInvalidDimension) return nullptr;

  vector<float> dataset_vec(dataset.data(), dataset.data() + dataset.size());
  auto ds =
      std::make_unique<DenseDataset<float>>(std::move(dataset_vec), n_points);
  if (n_dim != kInvalidDimension) {
    ds->set_dimensionality(n_dim);
  }
  return ds;
}

// scann-core: DenseDataset infers dimensionality as size / n_points. With
// n_points == 0 that divides by zero, and a size that isn't a multiple of
// n_points silently yields the wrong dimensionality (misaligning every row)
// -- only a DCHECK catches it, in debug builds, by aborting. Both are
// reachable through the C++ and Rust APIs and via truncated artifact files.
template <typename T>
Status CheckDenseShape(ConstSpan<T> data, DatapointIndex n_points,
                       absl::string_view what) {
  if (data.empty()) return OkStatus();
  if (n_points == 0)
    return InvalidArgumentError(absl::StrCat(
        what, " has ", data.size(), " values but n_points is 0"));
  if (data.size() % n_points != 0)
    return InvalidArgumentError(absl::StrCat(
        what, " has ", data.size(), " values, which is not a multiple of ",
        "n_points (", n_points, ")"));
  return OkStatus();
}

// scann-core: returns InvalidArgumentError naming the first row of the
// row-major `data` (rows of `dim` values) that holds a NaN or infinity.
Status CheckAllFinite(ConstSpan<float> data, DimensionIndex dim,
                      absl::string_view what) {
  for (size_t i = 0; i < data.size(); ++i) {
    if (!std::isfinite(data[i]))
      return InvalidArgumentError(absl::StrCat(
          what, " row ", dim == 0 ? 0 : i / dim,
          " contains NaN or infinity; ScaNN only supports finite values."));
  }
  return OkStatus();
}

Status AddTokenizationToOptions(SingleMachineFactoryOptions& opts,
                                ConstSpan<int32_t> tokenization,
                                const int spilling_mult = 1) {
  if (tokenization.empty()) return OkStatus();
  if (opts.serialized_partitioner == nullptr)
    return FailedPreconditionError(
        "Non-empty tokenization but no serialized partitioner is present.");
  // scann-core: upstream bounds-checked tokens with vector::at(), which
  // throws std::out_of_range from this Status-returning path (std::terminate
  // for C++ callers), and let a non-SOAR token of -1 silently drop a point.
  const int32_t n_tokens = opts.serialized_partitioner->n_tokens();
  if (n_tokens <= 0)
    return InvalidArgumentError(absl::StrCat(
        "The serialized partitioner has ", n_tokens, " partitions."));
  if (tokenization.size() % spilling_mult != 0)
    return InvalidArgumentError(absl::StrCat(
        "The tokenization has ", tokenization.size(),
        " entries; a SOAR tokenization has two per datapoint."));
  for (auto [i, token] : Enumerate(tokenization)) {
    // With SOAR, the second entry of a datapoint may be empty.
    const bool may_be_empty = spilling_mult > 1 && i % spilling_mult != 0;
    if ((token < 0 || token >= n_tokens) &&
        !(may_be_empty && token == kSoarEmptyToken))
      return InvalidArgumentError(absl::StrCat(
          "Datapoint ", i / spilling_mult, " has partition token ", token,
          ", outside [0, ", n_tokens, ")."));
    if (may_be_empty && token == tokenization[i - 1])
      return InvalidArgumentError(absl::StrCat(
          "Datapoint ", i / spilling_mult, " is assigned to partition ", token,
          " twice."));
  }
  opts.datapoints_by_token =
      std::make_shared<vector<std::vector<DatapointIndex>>>(n_tokens);
  for (auto [dp_idx, token] : Enumerate(tokenization)) {
    if (token != kSoarEmptyToken) {
      (*opts.datapoints_by_token)[token].push_back(dp_idx / spilling_mult);
    }
  }
  return OkStatus();
}

}  // namespace

namespace {

// scann-core: validation for LoadArtifacts. Upstream trusted every file of
// an artifacts directory. Damaged or inconsistent directories -- typically
// an interrupted re-serialize, which leaves a new scann_config.pb next to
// the previous index's assets -- crashed the process (segfault, SIGFPE,
// LOG(FATAL), heap overflow, uncaught exceptions) or loaded silently wrong
// data. Each asset is now checked on its own (see NumpyToVectorAndShape)
// and against the others and the config.

// SerializeToDirectory() puts a scann_assets.pbtxt containing this field
// (unknown to the ScannAssets proto, so it never parses) in place while it
// replaces the other files.
constexpr absl::string_view kIncompleteSerializationMarker =
    "scann_core_incomplete_serialization";

Status CheckSerializationComplete(absl::string_view assets_pbtxt) {
  if (absl::StrContains(assets_pbtxt, kIncompleteSerializationMarker))
    return FailedPreconditionError(
        "This index directory is incomplete: serialize() was interrupted "
        "while writing it, or is still writing it. Serialize the index "
        "again.");
  return OkStatus();
}

// scann-core: where LoadArtifacts reads the asset files from: the file
// system, or a caller's in-memory copies of the files
// (LoadArtifactsFromMemory).
class AssetReader {
 public:
  using Files = absl::flat_hash_map<std::string, absl::string_view>;

  AssetReader() = default;
  explicit AssetReader(const Files* files) : files_(files) {}

  template <typename T>
  StatusOr<pair<vector<T>, vector<size_t>>> Npy(absl::string_view path) const {
    if (files_ == nullptr) return NumpyToVectorAndShape<T>(path);
    SCANN_ASSIGN_OR_RETURN(absl::string_view bytes, Get(path));
    return NumpyBytesToVectorAndShape<T>(path, bytes);
  }

  Status Proto(absl::string_view path,
               google::protobuf::Message* message) const {
    if (files_ == nullptr) return ReadProtobufFromFile(path, message);
    SCANN_ASSIGN_OR_RETURN(absl::string_view bytes, Get(path));
    if (!message->ParseFromString(bytes))
      return InvalidArgumentError(
          absl::StrCat("Failed to parse ", message->GetTypeName(),
                       " proto from ", path));
    return OkStatus();
  }

  // The file `path`, or the file named like its last component: the
  // manifest of an index serialized with relative_path=False lists absolute
  // paths.
  StatusOr<absl::string_view> Get(absl::string_view path) const {
    auto it = files_->find(path);
    if (it == files_->end())
      it = files_->find(
          std::filesystem::path(std::string(path)).filename().string());
    if (it == files_->end())
      return NotFoundError(
          absl::StrCat("The index files have no ", path, "."));
    return it->second;
  }

 private:
  const Files* files_ = nullptr;
};

template <typename T>
StatusOr<pair<vector<T>, vector<size_t>>> LoadNpyOfRank(
    const AssetReader& reader, absl::string_view path, size_t rank) {
  SCANN_ASSIGN_OR_RETURN(auto v, reader.Npy<T>(path));
  if (v.second.size() != rank)
    return InvalidArgumentError(absl::StrCat(path, " has ", v.second.size(),
                                             " dimensions; expected ", rank,
                                             "."));
  return v;
}

// A row-major (n, d) .npy file as a dataset. Keeps d when n == 0. (An empty
// index can have an empty dataset of unknown dimensionality, (0, 0).)
template <typename T>
StatusOr<shared_ptr<DenseDataset<T>>> LoadDenseNpy(const AssetReader& reader,
                                                   absl::string_view path) {
  SCANN_ASSIGN_OR_RETURN(auto v, LoadNpyOfRank<T>(reader, path, 2));
  const size_t n = v.second[0], d = v.second[1];
  if (d == 0 && n > 0)
    return InvalidArgumentError(absl::StrCat(path, " has 0 columns."));
  if (n >= kInvalidDatapointIndex)
    return InvalidArgumentError(
        absl::StrCat(path, " has ", n, " rows, more than ScaNN supports."));
  auto ds = make_shared<DenseDataset<T>>(std::move(v.first), n);
  if (n == 0) ds->set_dimensionality(d);
  return ds;
}

// Checks that every quantity added has the same value (optionally ignoring
// zeros: an empty dataset can have an unknown dimensionality).
class ConsistentCount {
 public:
  ConsistentCount(absl::string_view quantity, bool ignore_zero)
      : quantity_(quantity), ignore_zero_(ignore_zero) {}
  Status Add(absl::string_view source, size_t value) {
    if (ignore_zero_ && value == 0) return OkStatus();
    if (source_.empty()) {
      source_ = std::string(source);
      value_ = value;
      return OkStatus();
    }
    if (value == value_) return OkStatus();
    return InvalidArgumentError(absl::StrCat(
        "Inconsistent index files: ", source, " has ", value, " ", quantity_,
        " but ", source_, " has ", value_,
        ". The directory may mix files of different indexes (e.g. from an "
        "interrupted serialize()); serialize the index again."));
  }
  bool has_value() const { return !source_.empty(); }
  size_t value() const { return value_; }

 private:
  std::string quantity_, source_;
  bool ignore_zero_;
  size_t value_ = 0;
};

Status ValidatePartitioner(const SerializedPartitioner& partitioner,
                           const ScannConfig& config, DimensionIndex dim) {
  if (partitioner.n_tokens() <= 0)
    return InvalidArgumentError(absl::StrCat(
        "The serialized partitioner has ", partitioner.n_tokens(),
        " partitions."));
  if (!partitioner.has_kmeans() || !partitioner.kmeans().has_kmeans_tree() ||
      partitioner.kmeans().has_next_bottom_up_level()) {
    // Not a plain k-means tree: at least don't let a garbage n_tokens
    // allocate unbounded memory for the partitions.
    if (partitioner.n_tokens() > (1 << 26))
      return InvalidArgumentError(absl::StrCat(
          "The serialized partitioner has ", partitioner.n_tokens(),
          " partitions."));
    return OkStatus();
  }
  const SerializedKMeansTree::Node& root =
      partitioner.kmeans().kmeans_tree().root();
  size_t n_leaves = 0;
  vector<const SerializedKMeansTree::Node*> stack = {&root};
  while (!stack.empty()) {
    const SerializedKMeansTree::Node* node = stack.back();
    stack.pop_back();
    if (node->children_size() == 0) ++n_leaves;
    for (const auto& child : node->children()) stack.push_back(&child);
  }
  if (n_leaves != static_cast<size_t>(partitioner.n_tokens()))
    return InvalidArgumentError(absl::StrCat(
        "The serialized partitioner says it has ", partitioner.n_tokens(),
        " partitions, but its tree has ", n_leaves, " leaves."));
  // K-means trains at most num_children leaves (fewer for tiny datasets).
  if (config.partitioning().has_num_children() &&
      partitioner.n_tokens() > config.partitioning().num_children())
    return InvalidArgumentError(absl::StrCat(
        "The serialized partitioner has ", partitioner.n_tokens(),
        " partitions, but the config asks for at most ",
        config.partitioning().num_children(),
        ". The directory may mix files of different indexes."));
  if (dim != kInvalidDimension && !partitioner.uses_projection() &&
      !config.partitioning().has_projection()) {
    for (const auto& center : root.centers()) {
      const size_t center_dim = std::max<size_t>(
          center.dimension_size(), center.float_dimension_size());
      if (center_dim != dim)
        return InvalidArgumentError(absl::StrCat(
            "The serialized partitioner's centers have dimensionality ",
            center_dim, ", but the dataset has ", dim,
            ". The directory may mix files of different indexes."));
    }
  }
  return OkStatus();
}

Status ValidateHashedDataset(const CentersForAllSubspaces& codebook,
                             const DenseDataset<uint8_t>& hashed,
                             absl::string_view what) {
  const size_t n_blocks = codebook.subspace_centers_size();
  if (hashed.empty()) return OkStatus();
  if (hashed.dimensionality() != n_blocks)
    return InvalidArgumentError(absl::StrCat(
        "The ", what, " has ", hashed.dimensionality(),
        " codes per datapoint, but the AH codebook has ", n_blocks,
        " blocks. The directory may mix files of different indexes."));
  ConstSpan<uint8_t> codes = hashed.data();
  for (size_t i = 0; i < codes.size(); ++i) {
    const size_t block = i % n_blocks;
    if (codes[i] >= codebook.subspace_centers(block).center_size())
      return InvalidArgumentError(absl::StrCat(
          "The ", what, " has code ", codes[i], " for block ", block,
          " of datapoint ", i / n_blocks, ", but the AH codebook has only ",
          codebook.subspace_centers(block).center_size(),
          " centers there."));
  }
  return OkStatus();
}

}  // namespace

namespace {

StatusOr<ScannInterface::ScannArtifacts> LoadArtifactsWith(
    const AssetReader& reader, const ScannConfig& config,
    const ScannAssets& assets) {
  SingleMachineFactoryOptions opts;
  const bool soar = HasSoar(config);

  absl::flat_hash_map<int, std::string> paths;
  vector<int32_t> tokenization;
  pair<vector<uint8_t>, vector<size_t>> soar_hashed;
  shared_ptr<DenseDataset<float>> dataset;
  auto fp = make_shared<PreQuantizedFixedPoint>();
  for (const ScannAsset& asset : assets.assets()) {
    const string_view asset_path = asset.asset_path();
    const ScannAsset::AssetType type = asset.asset_type();
    switch (type) {
      case ScannAsset::AH_CENTERS:
      case ScannAsset::PARTITIONER:
      case ScannAsset::TOKENIZATION_NPY:
      case ScannAsset::AH_DATASET_NPY:
      case ScannAsset::AH_DATASET_SOAR_NPY:
      case ScannAsset::DATASET_NPY:
      case ScannAsset::INT8_DATASET_NPY:
      case ScannAsset::INT8_MULTIPLIERS_NPY:
      case ScannAsset::INT8_NORMS_NPY:
      case ScannAsset::BF16_DATASET_NPY:
        if (!paths.emplace(type, asset_path).second)
          return InvalidArgumentError(absl::StrCat(
              "The assets list more than one ",
              ScannAsset::AssetType_Name(type), " file."));
        break;
      default:
        break;
    }
    switch (type) {
      case ScannAsset::AH_CENTERS:
        opts.ah_codebook = std::make_shared<CentersForAllSubspaces>();
        SCANN_RETURN_IF_ERROR(
            reader.Proto(asset_path, opts.ah_codebook.get()));
        break;
      case ScannAsset::PARTITIONER:
        opts.serialized_partitioner = std::make_shared<SerializedPartitioner>();
        SCANN_RETURN_IF_ERROR(
            reader.Proto(asset_path, opts.serialized_partitioner.get()));
        break;
      case ScannAsset::TOKENIZATION_NPY: {
        SCANN_ASSIGN_OR_RETURN(auto v,
                               LoadNpyOfRank<int32_t>(reader, asset_path, 1));
        tokenization = std::move(v.first);
        break;
      }
      case ScannAsset::AH_DATASET_NPY: {
        SCANN_ASSIGN_OR_RETURN(opts.hashed_dataset,
                               LoadDenseNpy<uint8_t>(reader, asset_path));
        break;
      }
      case ScannAsset::AH_DATASET_SOAR_NPY: {
        if (!soar)
          return InvalidArgumentError(
              "The assets include a SOAR hashed dataset, but the config has "
              "no SOAR spilling. The directory may mix files of different "
              "indexes.");
        SCANN_ASSIGN_OR_RETURN(soar_hashed,
                               LoadNpyOfRank<uint8_t>(reader, asset_path, 2));
        break;
      }
      case ScannAsset::DATASET_NPY: {
        SCANN_ASSIGN_OR_RETURN(dataset,
                               LoadDenseNpy<float>(reader, asset_path));
        break;
      }
      case ScannAsset::INT8_DATASET_NPY: {
        SCANN_ASSIGN_OR_RETURN(fp->fixed_point_dataset,
                               LoadDenseNpy<int8_t>(reader, asset_path));
        break;
      }
      case ScannAsset::INT8_MULTIPLIERS_NPY: {
        SCANN_ASSIGN_OR_RETURN(auto v,
                               LoadNpyOfRank<float>(reader, asset_path, 1));
        fp->multiplier_by_dimension =
            make_shared<vector<float>>(std::move(v.first));
        break;
      }
      case ScannAsset::INT8_NORMS_NPY: {
        SCANN_ASSIGN_OR_RETURN(auto v,
                               LoadNpyOfRank<float>(reader, asset_path, 1));
        fp->squared_l2_norm_by_datapoint =
            make_shared<vector<float>>(std::move(v.first));
        break;
      }
      case ScannAsset::BF16_DATASET_NPY: {
        SCANN_ASSIGN_OR_RETURN(opts.bfloat16_dataset,
                               LoadDenseNpy<int16_t>(reader, asset_path));
        break;
      }
      default:
        break;
    }
  }
  const auto has = [&paths](ScannAsset::AssetType type) {
    return paths.contains(type);
  };
  const auto name = [&paths](ScannAsset::AssetType type) {
    return absl::StrCat(ScannAsset::AssetType_Name(type), " (", paths[type],
                        ")");
  };

  // Assets the config can't use come from another index.
  const auto mixed = [](absl::string_view what, absl::string_view section) {
    return InvalidArgumentError(absl::StrCat(
        "The assets include ", what, ", but the config has no ", section,
        ". The directory may mix files of different indexes (e.g. from an "
        "interrupted serialize()); serialize the index again."));
  };
  if ((has(ScannAsset::AH_CENTERS) || has(ScannAsset::AH_DATASET_NPY)) &&
      !config.has_hash())
    return mixed("asymmetric-hashing data", "hash section");
  if ((has(ScannAsset::PARTITIONER) || has(ScannAsset::TOKENIZATION_NPY)) &&
      !config.has_partitioning())
    return mixed("a partitioner or tokenization", "partitioning section");
  if (has(ScannAsset::INT8_DATASET_NPY) &&
      !config.brute_force().fixed_point().enabled() &&
      !config.brute_force().scalar_quantized() &&
      !config.exact_reordering().fixed_point().enabled() &&
      !config.exact_reordering().use_fixed_point_if_possible())
    return mixed("an int8 dataset", "int8 (fixed-point) scoring");
  if (has(ScannAsset::BF16_DATASET_NPY) &&
      !config.brute_force().bfloat16().enabled() &&
      !config.exact_reordering().bfloat16().enabled())
    return mixed("a bfloat16 dataset", "bfloat16 scoring");

  // Every per-datapoint asset must have the same number of rows, and every
  // per-dimension one the same dimensionality.
  ConsistentCount rows("datapoints", /*ignore_zero=*/false);
  ConsistentCount dims("dimensions", /*ignore_zero=*/true);
  if (dataset) {
    SCANN_RETURN_IF_ERROR(rows.Add(name(ScannAsset::DATASET_NPY),
                                   dataset->size()));
    SCANN_RETURN_IF_ERROR(dims.Add(name(ScannAsset::DATASET_NPY),
                                   dataset->dimensionality()));
  }
  if (opts.hashed_dataset)
    SCANN_RETURN_IF_ERROR(rows.Add(name(ScannAsset::AH_DATASET_NPY),
                                   opts.hashed_dataset->size()));
  if (fp->fixed_point_dataset) {
    SCANN_RETURN_IF_ERROR(rows.Add(name(ScannAsset::INT8_DATASET_NPY),
                                   fp->fixed_point_dataset->size()));
    SCANN_RETURN_IF_ERROR(dims.Add(name(ScannAsset::INT8_DATASET_NPY),
                                   fp->fixed_point_dataset->dimensionality()));
    if (fp->multiplier_by_dimension == nullptr ||
        (fp->fixed_point_dataset->dimensionality() > 0 &&
         fp->multiplier_by_dimension->size() !=
             fp->fixed_point_dataset->dimensionality()))
      return InvalidArgumentError(absl::StrCat(
          "The int8 dataset has ", fp->fixed_point_dataset->dimensionality(),
          " dimensions, but the assets have ",
          fp->multiplier_by_dimension ? fp->multiplier_by_dimension->size() : 0,
          " int8 multipliers (INT8_MULTIPLIERS_NPY)."));
  }
  if (fp->multiplier_by_dimension)
    SCANN_RETURN_IF_ERROR(dims.Add(name(ScannAsset::INT8_MULTIPLIERS_NPY),
                                   fp->multiplier_by_dimension->size()));
  if (fp->squared_l2_norm_by_datapoint &&
      !fp->squared_l2_norm_by_datapoint->empty())
    SCANN_RETURN_IF_ERROR(rows.Add(name(ScannAsset::INT8_NORMS_NPY),
                                   fp->squared_l2_norm_by_datapoint->size()));
  if (opts.bfloat16_dataset) {
    SCANN_RETURN_IF_ERROR(rows.Add(name(ScannAsset::BF16_DATASET_NPY),
                                   opts.bfloat16_dataset->size()));
    SCANN_RETURN_IF_ERROR(dims.Add(name(ScannAsset::BF16_DATASET_NPY),
                                   opts.bfloat16_dataset->dimensionality()));
  }
  if (has(ScannAsset::AH_DATASET_SOAR_NPY))
    SCANN_RETURN_IF_ERROR(
        rows.Add(name(ScannAsset::AH_DATASET_SOAR_NPY), soar_hashed.second[0]));
  const int spilling_mult = soar ? 2 : 1;
  if (has(ScannAsset::TOKENIZATION_NPY)) {
    if (tokenization.size() % spilling_mult != 0)
      return InvalidArgumentError(absl::StrCat(
          "The SOAR tokenization ", paths[ScannAsset::TOKENIZATION_NPY],
          " has ", tokenization.size(),
          " entries; it needs two per datapoint."));
    SCANN_RETURN_IF_ERROR(rows.Add(
        soar ? absl::StrCat("the SOAR tokenization ",
                            paths[ScannAsset::TOKENIZATION_NPY])
             : name(ScannAsset::TOKENIZATION_NPY),
        tokenization.size() / spilling_mult));
  }
  if (config.input_output().pure_dynamic_config().has_dimensionality())
    SCANN_RETURN_IF_ERROR(
        dims.Add("the config (pure_dynamic_config.dimensionality)",
                 config.input_output().pure_dynamic_config().dimensionality()));

  if (opts.serialized_partitioner != nullptr)
    SCANN_RETURN_IF_ERROR(ValidatePartitioner(
        *opts.serialized_partitioner, config,
        dims.has_value() ? dims.value() : kInvalidDimension));
  SCANN_RETURN_IF_ERROR(
      AddTokenizationToOptions(opts, tokenization, spilling_mult));
  // An index with every datapoint deleted: keep its (empty) partitions
  // rather than re-tokenizing, which needs a float dataset.
  if (has(ScannAsset::TOKENIZATION_NPY) && tokenization.empty() &&
      opts.serialized_partitioner != nullptr)
    opts.datapoints_by_token =
        std::make_shared<vector<std::vector<DatapointIndex>>>(
            opts.serialized_partitioner->n_tokens());

  if (opts.ah_codebook != nullptr && opts.hashed_dataset != nullptr)
    SCANN_RETURN_IF_ERROR(ValidateHashedDataset(
        *opts.ah_codebook, *opts.hashed_dataset, "hashed dataset"));
  if (has(ScannAsset::AH_DATASET_SOAR_NPY)) {
    if (!has(ScannAsset::TOKENIZATION_NPY) || !opts.hashed_dataset ||
        !opts.ah_codebook)
      return InvalidArgumentError(
          "A SOAR hashed dataset needs the tokenization, the hashed dataset "
          "and the AH codebook, which the assets lack.");
    // The SOAR hashed dataset's docids hold each datapoint's second token.
    auto docids = std::make_unique<FixedLengthDocidCollection>(4);
    docids->Reserve(tokenization.size() / 2);
    for (size_t i = 1; i < tokenization.size(); i += 2)
      SCANN_RETURN_IF_ERROR(
          docids->Append(strings::Int32ToKey(tokenization[i])));
    const size_t n = soar_hashed.second[0];
    if (n > 0 && soar_hashed.second[1] != opts.hashed_dataset->dimensionality())
      return InvalidArgumentError(absl::StrCat(
          "The SOAR hashed dataset has ", soar_hashed.second[1],
          " codes per datapoint, the hashed dataset ",
          opts.hashed_dataset->dimensionality(), "."));
    opts.soar_hashed_dataset = make_shared<DenseDataset<uint8_t>>(
        std::move(soar_hashed.first), std::move(docids));
    if (n == 0)
      opts.soar_hashed_dataset->set_dimensionality(soar_hashed.second[1]);
    SCANN_RETURN_IF_ERROR(ValidateHashedDataset(*opts.ah_codebook,
                                                *opts.soar_hashed_dataset,
                                                "SOAR hashed dataset"));
  }

  if (fp->fixed_point_dataset != nullptr) {
    if (fp->squared_l2_norm_by_datapoint == nullptr)
      fp->squared_l2_norm_by_datapoint = make_shared<vector<float>>();
    opts.pre_quantized_fixed_point = fp;
  }
  return std::make_tuple(config, std::move(dataset), std::move(opts));
}

}  // namespace

StatusOr<ScannInterface::ScannArtifacts> ScannInterface::LoadArtifacts(
    const ScannConfig& config, const ScannAssets& assets) {
  return LoadArtifactsWith(AssetReader(), config, assets);
}

StatusOr<ScannInterface::ScannArtifacts>
ScannInterface::LoadArtifactsFromMemory(
    const absl::flat_hash_map<std::string, absl::string_view>& files) {
  const AssetReader reader(&files);
  ScannAssets assets;
  SCANN_ASSIGN_OR_RETURN(absl::string_view assets_pbtxt,
                         reader.Get("scann_assets.pbtxt"));
  SCANN_RETURN_IF_ERROR(CheckSerializationComplete(assets_pbtxt));
  SCANN_RETURN_IF_ERROR(ParseTextProto(&assets, assets_pbtxt));
  ScannConfig config;
  SCANN_RETURN_IF_ERROR(reader.Proto("scann_config.pb", &config));
  return LoadArtifactsWith(reader, config, assets);
}

std::string RewriteAssetFilenameIfRelative(const string& artifacts_dir,
                                           const string& asset_path) {
  std::filesystem::path path(asset_path);
  if (path.is_relative()) {
    return (artifacts_dir / path).string();
  } else {
    return asset_path;
  }
}

StatusOr<ScannInterface::ScannArtifacts> ScannInterface::LoadArtifacts(
    const std::string& artifacts_dir, const std::string& scann_assets_pbtxt) {
  ScannAssets assets;
  if (scann_assets_pbtxt.empty()) {
    SCANN_ASSIGN_OR_RETURN(auto assets_pbtxt,
                           GetContents(artifacts_dir + "/scann_assets.pbtxt"));
    SCANN_RETURN_IF_ERROR(CheckSerializationComplete(assets_pbtxt));
    SCANN_RETURN_IF_ERROR(ParseTextProto(&assets, assets_pbtxt));
  } else {
    SCANN_RETURN_IF_ERROR(CheckSerializationComplete(scann_assets_pbtxt));
    SCANN_RETURN_IF_ERROR(ParseTextProto(&assets, scann_assets_pbtxt));
  }
  ScannConfig config;
  SCANN_RETURN_IF_ERROR(
      ReadProtobufFromFile(artifacts_dir + "/scann_config.pb", &config));
  for (auto i : Seq(assets.assets_size())) {
    auto new_path = RewriteAssetFilenameIfRelative(
        artifacts_dir, assets.assets(i).asset_path());
    assets.mutable_assets(i)->set_asset_path(new_path);
  }
  return LoadArtifacts(config, assets);
}

StatusOr<std::unique_ptr<SingleMachineSearcherBase<float>>>
ScannInterface::CreateSearcher(ScannArtifacts artifacts) {
  auto [config, dataset, opts] = std::move(artifacts);

  if (dataset && config.has_partitioning() &&
      config.partitioning().partitioning_type() ==
          PartitioningConfig::SPHERICAL)
    dataset->set_normalization_tag(research_scann::UNITL2NORM);

  SCANN_ASSIGN_OR_RETURN(auto searcher, SingleMachineFactoryScann<float>(
                                            config, dataset, std::move(opts)));
  SCANN_RETURN_IF_ERROR(searcher->InitializeHealthStats());
  searcher->MaybeReleaseDataset();
  return searcher;
}

Status ScannInterface::Initialize(absl::string_view config_pbtxt,
                                  absl::string_view scann_assets_pbtxt) {
  SCANN_RETURN_IF_ERROR(ParseTextProto(&config_, config_pbtxt));
  ScannAssets assets;
  SCANN_RETURN_IF_ERROR(CheckSerializationComplete(scann_assets_pbtxt));
  SCANN_RETURN_IF_ERROR(ParseTextProto(&assets, scann_assets_pbtxt));
  SCANN_ASSIGN_OR_RETURN(auto dataset_and_opts, LoadArtifacts(config_, assets));
  auto [_, dataset, opts] = std::move(dataset_and_opts);
  return Initialize(std::tie(config_, dataset, opts));
}

Status ScannInterface::Initialize(
    ScannConfig config, SingleMachineFactoryOptions opts,
    ConstSpan<float> dataset, ConstSpan<int32_t> datapoint_to_token,
    ConstSpan<uint8_t> hashed_dataset, ConstSpan<int8_t> int8_dataset,
    ConstSpan<float> int8_multipliers, ConstSpan<float> dp_norms,
    DatapointIndex n_points) {
  config_ = config;
  SCANN_RETURN_IF_ERROR(CheckDenseShape(dataset, n_points, "dataset"));
  SCANN_RETURN_IF_ERROR(CheckDenseShape(int8_dataset, n_points, "int8 dataset"));
  if (opts.ah_codebook != nullptr) {
    if (hashed_dataset.empty())
      return InvalidArgumentError("AH codebook present but hashed dataset is empty");
    SCANN_RETURN_IF_ERROR(CheckDenseShape(hashed_dataset, n_points, "hashed dataset"));
    vector<uint8_t> hashed_db(hashed_dataset.data(),
                              hashed_dataset.data() + hashed_dataset.size());
    opts.hashed_dataset =
        std::make_shared<DenseDataset<uint8_t>>(std::move(hashed_db), n_points);
  }
  const int spilling_mult = HasSoar(config_) ? 2 : 1;
  SCANN_RETURN_IF_ERROR(
      AddTokenizationToOptions(opts, datapoint_to_token, spilling_mult));
  if (!int8_dataset.empty()) {
    auto int8_data = std::make_shared<PreQuantizedFixedPoint>();
    vector<int8_t> int8_vec(int8_dataset.data(),
                            int8_dataset.data() + int8_dataset.size());
    int8_data->fixed_point_dataset =
        std::make_shared<DenseDataset<int8_t>>(std::move(int8_vec), n_points);

    int8_data->multiplier_by_dimension = make_shared<vector<float>>(
        int8_multipliers.begin(), int8_multipliers.end());

    int8_data->squared_l2_norm_by_datapoint =
        make_shared<vector<float>>(dp_norms.begin(), dp_norms.end());
    opts.pre_quantized_fixed_point = int8_data;
  }

  DimensionIndex n_dim = kInvalidDimension;
  if (config.input_output().pure_dynamic_config().has_dimensionality())
    n_dim = config.input_output().pure_dynamic_config().dimensionality();
  return Initialize(std::make_tuple(
      config_, InitDataset(dataset, n_points, n_dim), std::move(opts)));
}

Status ScannInterface::Initialize(ConstSpan<float> dataset,
                                  DatapointIndex n_points,
                                  absl::string_view config,
                                  int training_threads) {
  SCANN_RETURN_IF_ERROR(ParseTextProto(&config_, config));
  if (training_threads < 0)
    return InvalidArgumentError("training_threads must be non-negative");
  SCANN_RETURN_IF_ERROR(CheckDenseShape(dataset, n_points, "dataset"));
  training_threads_ = training_threads;
  if (training_threads == 0) training_threads = GetNumCPUs();
  SingleMachineFactoryOptions opts;

  opts.parallelization_pool =
      StartThreadPool("scann_threadpool", training_threads - 1);

  DimensionIndex n_dim = kInvalidDimension;
  if (config_.input_output().pure_dynamic_config().has_dimensionality())
    n_dim = config_.input_output().pure_dynamic_config().dimensionality();
  shared_ptr<DenseDataset<float>> ds = InitDataset(dataset, n_points, n_dim);
  // scann-core: spherical partitioning needs a dataset tagged unit-L2-norm
  // (CreateSearcher sets the tag), but upstream never normalized it. Points
  // upserted later were then normalized in some configurations (where the
  // tag reached the stored float data) and stored as given in others, while
  // the original points were stored as given: the same vector scored
  // differently depending on when it was added. Store unit vectors
  // throughout: normalize the dataset here and upserts in
  // NormalizeIfSpherical.
  if (ds && IsSphericalPartitioning(config_))
    for (size_t i = 0; i < ds->size(); ++i)
      NormalizeForSphericalPartitioning(ds->mutable_data(i));
  return Initialize(std::make_tuple(config_, std::move(ds), std::move(opts)));
}

Status ScannInterface::Initialize(ScannInterface::ScannArtifacts artifacts) {
  auto [config, dataset, opts] = std::move(artifacts);
  config_ = config;
  // scann-core: with max_spill_centers = 0 (e.g. the Python builder's
  // tree(num_leaves_to_search=0)), the searcher built fine and then every
  // search failed with a bare "SCANN_RET_CHECK failure".
  if (config_.has_partitioning() &&
      config_.partitioning().query_spilling().spilling_type() ==
          QuerySpillingConfig::FIXED_NUMBER_OF_CENTERS &&
      config_.partitioning().query_spilling().max_spill_centers() < 1)
    return InvalidArgumentError(
        "partitioning.query_spilling.max_spill_centers (the number of leaves "
        "to search) must be > 0.");
  // scann-core: upstream accepted NaN/infinity in the dataset. Training a
  // partitioner on it then died on a QCHECK in gmm_utils.cc (process abort),
  // and non-finite points break every distance computed against them.
  if (dataset != nullptr)
    SCANN_RETURN_IF_ERROR(
        CheckAllFinite(dataset->data(), dataset->dimensionality(), "dataset"));
  SCANN_ASSIGN_OR_RETURN(dimensionality_, opts.ComputeConsistentDimensionality(
                                              config_, dataset.get()));
  SCANN_ASSIGN_OR_RETURN(scann_,
                         CreateSearcher(std::tie(config_, dataset, opts)));
  if (scann_->config().has_value()) config_ = scann_->config().value();

  absl::string_view distance = config_.distance_measure().distance_measure();
  const absl::flat_hash_set<std::string> negated_distances{
      "DotProductDistance", "BinaryDotProductDistance", "AbsDotProductDistance",
      "LimitedInnerProductDistance"};
  result_multiplier_ =
      negated_distances.find(distance) == negated_distances.end() ? 1 : -1;

  if (config_.has_partitioning()) {
    min_batch_size_ = 1;
  } else {
    if (config_.has_hash())
      min_batch_size_ = 16;
    else
      min_batch_size_ = 256;
  }
  // scann-core: the query pool is started on first use (see
  // parallel_query_pool()), sized from the CPUs available now.
  SetNumThreads(GetNumCPUs());
  return OkStatus();
}

void ScannInterface::SetNumThreads(int num_threads) {
  absl::MutexLock lock(&pool_mu_);
  num_threads_ = std::max(num_threads, 1);
  parallel_query_pool_.reset();
  pool_started_ = false;
}

std::shared_ptr<ThreadPool> ScannInterface::parallel_query_pool() const {
  absl::MutexLock lock(&pool_mu_);
  if (!pool_started_) {
    parallel_query_pool_ =
        StartThreadPool("ScannQueryingPool", num_threads_ - 1);
    pool_started_ = true;
  }
  return parallel_query_pool_;
}

SearchParameters ScannInterface::GetSearchParameters(int final_nn,
                                                     int pre_reorder_nn,
                                                     int leaves) const {
  SearchParameters params;
  bool has_reordering = config_.has_exact_reordering();
  int post_reorder_nn = -1;
  if (has_reordering) {
    post_reorder_nn = final_nn;
  } else {
    pre_reorder_nn = final_nn;
  }
  params.set_pre_reordering_num_neighbors(pre_reorder_nn);
  params.set_post_reordering_num_neighbors(post_reorder_nn);
  // scann-core: upstream attached TreeXOptionalParameters whenever leaves > 0,
  // even for non-tree searchers; the int8 brute-force searcher then
  // down_cast them to its own parameter type and segfaulted. leaves_to_search
  // only means something for a partitioned (tree) index; ignore it otherwise.
  if (leaves > 0 && config_.has_partitioning()) {
    auto tree_params = std::make_shared<TreeXOptionalParameters>();
    tree_params->set_num_partitions_to_search_override(leaves);
    params.set_searcher_specific_optional_parameters(tree_params);
  }
  return params;
}

vector<SearchParameters> ScannInterface::GetSearchParametersBatched(
    int batch_size, int final_nn, int pre_reorder_nn, int leaves,
    bool set_unspecified) const {
  vector<SearchParameters> params(batch_size);
  bool has_reordering = config_.has_exact_reordering();
  int post_reorder_nn = -1;
  if (has_reordering) {
    post_reorder_nn = final_nn;
  } else {
    pre_reorder_nn = final_nn;
  }
  std::shared_ptr<research_scann::TreeXOptionalParameters> tree_params;
  // scann-core: only for tree indexes; see GetSearchParameters.
  if (leaves > 0 && config_.has_partitioning()) {
    tree_params = std::make_shared<TreeXOptionalParameters>();
    tree_params->set_num_partitions_to_search_override(leaves);
  }

  for (auto& p : params) {
    p.set_pre_reordering_num_neighbors(pre_reorder_nn);
    p.set_post_reordering_num_neighbors(post_reorder_nn);
    if (tree_params) p.set_searcher_specific_optional_parameters(tree_params);
    if (set_unspecified) scann_->SetUnspecifiedParametersToDefaults(&p);
  }
  return params;
}

StatusOr<ScannConfig> ScannInterface::RetrainAndReindex(const string& config) {
  absl::Mutex mu;
  ScannConfig new_config = config_;
  if (!config.empty())
    SCANN_RETURN_IF_ERROR(ParseTextProto(&new_config, config));

  // scann-core: retrain with the index's training_threads (upstream used the
  // query pool whatever training_threads was); for an index built with the
  // default (0) or loaded, with the query pool's workers, so
  // SetNumThreads(1) also keeps a rebalance single-threaded.
  std::shared_ptr<ThreadPool> pool =
      training_threads_ > 0
          ? std::shared_ptr<ThreadPool>(
                StartThreadPool("scann_threadpool", training_threads_ - 1))
          : parallel_query_pool();
  auto status_or =
      RetrainAndReindexSearcher(scann_.get(), &mu, new_config, pool);
  if (!status_or.ok()) return status_or.status();
  // scann-core: on success RetrainAndReindexSearcher returns with `mu`
  // write-locked so the caller can swap the searcher pointer under it.
  // Upstream never unlocked, destroying a held absl::Mutex (reported by
  // TSan as "destroy of a locked mutex").
  scann_.reset(static_cast<SingleMachineSearcherBase<float>*>(
      std::move(status_or.value().release())));
  mu.WriterUnlock();
  if (scann_->config().has_value()) config_ = scann_->config().value();
  // scann-core: health stats first, while the searcher still has its float
  // dataset, as CreateSearcher() does. Upstream released the dataset first,
  // so for trees that don't keep it (no float reordering) the quantization
  // error after a rebalance was 0.
  SCANN_RETURN_IF_ERROR(scann_->InitializeHealthStats());
  scann_->MaybeReleaseDataset();
  return config_;
}

bool ScannInterface::NormalizesDatapoints() const {
  return IsSphericalPartitioning(config_);
}

void ScannInterface::NormalizeDatapoints(MutableSpan<float> rows) const {
  if (!NormalizesDatapoints() || dimensionality_ == 0) return;
  for (size_t begin = 0; begin + dimensionality_ <= rows.size();
       begin += dimensionality_)
    NormalizeForSphericalPartitioning(rows.subspan(begin, dimensionality_));
}

Status ScannInterface::Search(const DatapointPtr<float> query,
                              NNResultsVector* res, int final_nn,
                              int pre_reorder_nn, int leaves) const {
  if (query.dimensionality() != dimensionality_)
    return InvalidArgumentError(
        absl::StrCat("Query has dimensionality ", query.dimensionality(),
                     ", but the dataset has ", dimensionality_));
  SearchParameters params =
      GetSearchParameters(final_nn, pre_reorder_nn, leaves);
  scann_->SetUnspecifiedParametersToDefaults(&params);
  return scann_->FindNeighbors(query, params, res);
}

Status ScannInterface::SearchBatched(const DenseDataset<float>& queries,
                                     MutableSpan<NNResultsVector> res,
                                     int final_nn, int pre_reorder_nn,
                                     int leaves) const {
  if (queries.dimensionality() != dimensionality_)
    return InvalidArgumentError(
        absl::StrCat("Queries have dimensionality ", queries.dimensionality(),
                     ", but the dataset has ", dimensionality_));
  return SearchBatchedView(DefaultDenseDatasetView<float>(queries), res,
                           final_nn, pre_reorder_nn, leaves);
}

Status ScannInterface::SearchBatchedView(
    const DefaultDenseDatasetView<float>& queries,
    MutableSpan<NNResultsVector> res, int final_nn, int pre_reorder_nn,
    int leaves) const {
  // scann-core: single-query FindNeighbors rejects NaN/infinity queries, but
  // upstream's batched path never checked, returning garbage neighbors.
  SCANN_RETURN_IF_ERROR(
      CheckAllFinite(queries.data(), dimensionality_, "query"));
  if (!std::isinf(scann_->default_pre_reordering_epsilon()) ||
      !std::isinf(scann_->default_post_reordering_epsilon()))
    return InvalidArgumentError("Batch querying isn't supported with epsilon");
  auto params = GetSearchParametersBatched(queries.size(), final_nn,
                                           pre_reorder_nn, leaves, true);
  return scann_->FindNeighborsBatched(queries, params, MakeMutableSpan(res));
}

Status ScannInterface::SearchBatchedParallel(const DenseDataset<float>& queries,
                                             MutableSpan<NNResultsVector> res,
                                             int final_nn, int pre_reorder_nn,
                                             int leaves, int batch_size) const {
  // scann-core: the same error as SearchBatched; upstream failed with a bare
  // "SCANN_RET_CHECK_EQ failure".
  if (queries.dimensionality() != dimensionality_)
    return InvalidArgumentError(
        absl::StrCat("Queries have dimensionality ", queries.dimensionality(),
                     ", but the dataset has ", dimensionality_));
  return SearchBatchedRows(queries.data(), queries.dimensionality(), res,
                           final_nn, pre_reorder_nn, leaves, /*parallel=*/true,
                           batch_size);
}

Status ScannInterface::SearchBatchedRows(ConstSpan<float> queries,
                                         size_t query_dim,
                                         MutableSpan<NNResultsVector> res,
                                         int final_nn, int pre_reorder_nn,
                                         int leaves, bool parallel,
                                         int batch_size,
                                         const ChunkCallback& on_chunk) const {
  if (query_dim != dimensionality_)
    return InvalidArgumentError(
        absl::StrCat("Queries have dimensionality ", query_dim,
                     ", but the dataset has ", dimensionality_));
  if (query_dim == 0 || queries.size() % query_dim != 0)
    return InvalidArgumentError(absl::StrCat(
        "Queries have ", queries.size(), " values, not a multiple of ",
        query_dim));
  const size_t num_queries = queries.size() / query_dim;
  if (res.size() != num_queries)
    return InvalidArgumentError(absl::StrCat(
        "Result span has ", res.size(), " entries for ", num_queries,
        " queries"));
  if (batch_size < 1)
    return InvalidArgumentError(
        absl::StrCat("batch_size must be >= 1, got ", batch_size));
  if (num_queries == 0) return OkStatus();

  // Chunks [begin, begin + size) are views of `queries`: no copy.
  auto search_chunk = [&](size_t begin, size_t size) -> Status {
    Status status;
    if (size == 1) {
      // scann-core: a single query is searched with Search():
      // FindNeighborsBatched's fixed cost (batched tokenization, lookup
      // table and top-N setup) made a batch of one about 13 % slower on
      // GloVe-100. (Its distances can differ from Search()'s in the last
      // bits; a one-query chunk now returns exactly what search() does.)
      DatapointPtr<float> q(nullptr, queries.data() + begin * dimensionality_,
                            dimensionality_, dimensionality_);
      status = Search(q, &res[begin], final_nn, pre_reorder_nn, leaves);
    } else {
      status = SearchBatchedView(
          DefaultDenseDatasetView<float>(
              queries.subspan(begin * dimensionality_, size * dimensionality_),
              dimensionality_),
          res.subspan(begin, size), final_nn, pre_reorder_nn, leaves);
    }
    if (status.ok() && on_chunk) on_chunk(begin, res.subspan(begin, size));
    return status;
  };

  // On failure, a NaN/infinity query is reported first, by its row in
  // `queries` (the first such row, as a serial check would).
  auto report = [&](Status status) {
    if (status.ok()) return status;
    Status finite = CheckAllFinite(queries, dimensionality_, "query");
    return finite.ok() ? status : finite;
  };
  if (!parallel) return report(search_chunk(0, num_queries));

  std::shared_ptr<ThreadPool> pool = parallel_query_pool();
  // scann-core: the calling thread works too (ParallelFor runs chunks on
  // it), so there are NumThreads() + 1 workers. Upstream made NumThreads()
  // chunks, leaving one worker idle.
  const size_t workers = pool ? pool->NumThreads() + 1 : 1;
  // Chunks as upstream: one per worker (at most batch_size queries), or
  // min_batch_size_ for indexes without a tree. scann-core: except that
  // tree indexes search chunks of up to kMaxSingleQueryChunk queries as
  // single queries (chunks of one), which spreads small batches over more
  // workers for about the same work (a batch of 8 costs 0.96x the time of 8
  // Search() calls on GloVe-100). Batches of 16-128 queries: 1.1-1.35x the
  // throughput with 8 and 16 workers; larger ones are unchanged. (More
  // chunks per worker, tried too, cost 2-12 % for 128-4096 queries on one
  // CCD: smaller chunks make the batched kernels less efficient.)
  constexpr size_t kMaxSingleQueryChunk = 8;
  const size_t cap = static_cast<size_t>(batch_size);
  size_t chunk = std::min(
      std::max(min_batch_size_, DivRoundUp(num_queries, workers)), cap);
  if (min_batch_size_ == 1 && chunk <= kMaxSingleQueryChunk) chunk = 1;

  // scann-core: one chunk per worker fetch (ParallelFor<1>). Upstream's
  // ParallelForWithStatus<1> drops its template argument and batches
  // dynamically, handing each worker chunks / 4 / pool threads chunks at a
  // time: harmless with a pool of NumCPUs() - 1 threads, but with 4 workers
  // and 40 chunks one worker was left with 3 chunks while the others idled.
  // Finiteness is checked per chunk, inside the parallel region (upstream
  // checked every query serially first, then again per chunk).
  Status status = OkStatus();
  size_t failed_chunk = std::numeric_limits<size_t>::max();
  std::atomic<bool> failed{false};
  absl::Mutex status_mu;
  ParallelFor<1>(Seq(DivRoundUp(num_queries, chunk)), pool.get(),
                 [&](size_t i) {
                   if (failed.load(std::memory_order_relaxed)) return;
                   const size_t begin = chunk * i;
                   Status chunk_status = search_chunk(
                       begin, std::min(num_queries - begin, chunk));
                   if (!chunk_status.ok()) {
                     absl::MutexLock lock(&status_mu);
                     // The first failing chunk's error, as a serial search
                     // would report (among those that ran).
                     if (i < failed_chunk) {
                       failed_chunk = i;
                       status = chunk_status;
                     }
                     failed.store(true, std::memory_order_relaxed);
                   }
                 });
  return report(status);
}

StatusOr<ScannAssets> ScannInterface::Serialize(std::string path,
                                                bool relative_path) {
  SCANN_ASSIGN_OR_RETURN(auto opts,
                         scann_->ExtractSingleMachineFactoryOptions());
  ScannAssets assets;
  const auto add_asset = [&assets](absl::string_view fpath,
                                   ScannAsset::AssetType type) {
    ScannAsset* asset = assets.add_assets();
    asset->set_asset_type(type);
    asset->set_asset_path(fpath);
  };

  const auto convert_path = [&path, &relative_path](const std::string& fpath) {
    std::string absolute_path = path + "/" + fpath;
    return std::pair(relative_path ? fpath : absolute_path, absolute_path);
  };

  SCANN_RETURN_IF_ERROR(
      WriteProtobufToFile(path + "/scann_config.pb", config_));
  if (opts.ah_codebook != nullptr) {
    auto [rpath, fpath] = convert_path("ah_codebook.pb");
    add_asset(rpath, ScannAsset::AH_CENTERS);
    SCANN_RETURN_IF_ERROR(WriteProtobufToFile(fpath, *opts.ah_codebook));
  }
  if (opts.serialized_partitioner != nullptr) {
    auto [rpath, fpath] = convert_path("serialized_partitioner.pb");
    add_asset(rpath, ScannAsset::PARTITIONER);
    SCANN_RETURN_IF_ERROR(
        WriteProtobufToFile(fpath, *opts.serialized_partitioner));
  }
  if (opts.datapoints_by_token != nullptr) {
    vector<int32_t> datapoint_to_token;
    if (HasSoar(config_)) {
      datapoint_to_token = vector<int32_t>(2 * n_points(), kSoarEmptyToken);
      for (const auto& [token_idx, dps] :
           Enumerate(*opts.datapoints_by_token)) {
        for (auto dp_idx : dps) {
          dp_idx *= 2;
          if (datapoint_to_token[dp_idx] != -1) dp_idx++;
          DCHECK_EQ(datapoint_to_token[dp_idx], -1);
          datapoint_to_token[dp_idx] = token_idx;
        }
      }
    } else {
      datapoint_to_token = vector<int32_t>(n_points());
      for (const auto& [token_idx, dps] : Enumerate(*opts.datapoints_by_token))
        for (auto dp_idx : dps) datapoint_to_token[dp_idx] = token_idx;
    }
    auto [rpath, fpath] = convert_path("datapoint_to_token.npy");
    add_asset(rpath, ScannAsset::TOKENIZATION_NPY);
    SCANN_RETURN_IF_ERROR(VectorToNumpy(fpath, datapoint_to_token));
  }
  if (opts.hashed_dataset != nullptr) {
    auto [rpath, fpath] = convert_path("hashed_dataset.npy");
    add_asset(rpath, ScannAsset::AH_DATASET_NPY);
    SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *(opts.hashed_dataset)));

    if (opts.soar_hashed_dataset != nullptr) {
      DCHECK(HasSoar(config_));
      auto [rpath, fpath] = convert_path("hashed_dataset_soar.npy");
      add_asset(rpath, ScannAsset::AH_DATASET_SOAR_NPY);
      SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *(opts.soar_hashed_dataset)));
    }
  }
  if (opts.bfloat16_dataset != nullptr) {
    auto [rpath, fpath] = convert_path("bfloat16_dataset.npy");
    add_asset(rpath, ScannAsset::BF16_DATASET_NPY);
    SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *(opts.bfloat16_dataset)));
  }
  if (opts.pre_quantized_fixed_point != nullptr) {
    auto fixed_point = opts.pre_quantized_fixed_point;
    auto dataset = fixed_point->fixed_point_dataset;
    if (dataset != nullptr) {
      auto [rpath, fpath] = convert_path("int8_dataset.npy");
      add_asset(rpath, ScannAsset::INT8_DATASET_NPY);
      SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *dataset));
    }
    auto multipliers = fixed_point->multiplier_by_dimension;
    if (multipliers != nullptr) {
      auto [rpath, fpath] = convert_path("int8_multipliers.npy");
      add_asset(rpath, ScannAsset::INT8_MULTIPLIERS_NPY);
      SCANN_RETURN_IF_ERROR(VectorToNumpy(fpath, *multipliers));
    }
    auto norms = fixed_point->squared_l2_norm_by_datapoint;
    if (norms != nullptr) {
      auto [rpath, fpath] = convert_path("dp_norms.npy");
      add_asset(rpath, ScannAsset::INT8_NORMS_NPY);
      SCANN_RETURN_IF_ERROR(VectorToNumpy(fpath, *norms));
    }
  }
  SCANN_ASSIGN_OR_RETURN(auto dataset, Float32DatasetIfNeeded());
  if (dataset != nullptr) {
    auto [rpath, fpath] = convert_path("dataset.npy");
    add_asset(rpath, ScannAsset::DATASET_NPY);
    SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *dataset));
  }
  return assets;
}

namespace {

// scann-core: helpers for SerializeToDirectory.

// Every asset file name Serialize() can write, and the Python wrapper's
// docids.
constexpr absl::string_view kIndexFileNames[] = {
    "ah_codebook.pb",          "serialized_partitioner.pb",
    "datapoint_to_token.npy",  "hashed_dataset.npy",
    "hashed_dataset_soar.npy", "bfloat16_dataset.npy",
    "int8_dataset.npy",        "int8_multipliers.npy",
    "dp_norms.npy",            "dataset.npy",
    "scann_docids.pkl"};

Status WriteWholeFile(const std::string& path, absl::string_view contents) {
  std::ofstream out(path, std::ofstream::binary | std::ofstream::trunc);
  if (!out.write(contents.data(), contents.size()) || !out.flush())
    return InternalError(absl::StrCat("Failed to write ", path));
  return OkStatus();
}

Status FsyncPath(const std::string& path, bool directory) {
  const int fd =
      ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (directory ? O_DIRECTORY : 0));
  if (fd < 0)
    return InternalError(absl::StrCat("Failed to open ", path,
                                      " for fsync: ", std::strerror(errno)));
  const int rc = ::fsync(fd);
  const int err = errno;
  ::close(fd);
  if (rc != 0)
    return InternalError(
        absl::StrCat("fsync of ", path, " failed: ", std::strerror(err)));
  return OkStatus();
}

Status RenamePath(const std::string& from, const std::string& to) {
  std::error_code ec;
  std::filesystem::rename(from, to, ec);
  if (ec)
    return InternalError(absl::StrCat("Failed to rename ", from, " to ", to,
                                      ": ", ec.message()));
  return OkStatus();
}

}  // namespace

Status ScannInterface::SerializeToDirectory(
    const std::string& dir, bool relative_path,
    const std::vector<std::pair<std::string, std::string>>& extra_files) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (!fs::is_directory(dir, ec))
    return NotFoundError(
        absl::StrCat(dir, " isn't an existing directory; create it first."));
  for (const auto& [name, contents] : extra_files) {
    if (name.empty() || name[0] == '.' || name.find('/') != std::string::npos ||
        name == "scann_config.pb" || name == "scann_assets.pbtxt" ||
        (name != "scann_docids.pkl" &&
         absl::c_linear_search(kIndexFileNames, name)))
      return InvalidArgumentError(
          absl::StrCat("Invalid extra file name \"", name, "\"."));
  }

  // Staging directories left by an interrupted serialize. (Non-throwing
  // iteration: a range-for would use the throwing operator++.)
  vector<fs::path> leftovers;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
       it.increment(ec)) {
    if (absl::StartsWith(it->path().filename().string(), ".scann_staging_"))
      leftovers.push_back(it->path());
  }
  for (const fs::path& leftover : leftovers) fs::remove_all(leftover, ec);

  std::string staging;
  std::random_device rd;
  for (int attempt = 0;; ++attempt) {
    staging = absl::StrCat(dir, "/.scann_staging_",
                           absl::Hex(rd(), absl::kZeroPad8),
                           absl::Hex(rd(), absl::kZeroPad8));
    if (fs::create_directory(staging, ec)) break;
    if (ec || attempt == 10)
      return InternalError(absl::StrCat("Failed to create a staging directory ",
                                        "in ", dir, ": ", ec.message()));
  }
  struct RemoveOnExit {
    std::string path;
    ~RemoveOnExit() {
      std::error_code ec;
      fs::remove_all(path, ec);
    }
  } remove_staging{staging};

  // Stage everything. Serialize() names each asset relative to `staging`;
  // the manifest records the final paths the caller asked for.
  SCANN_ASSIGN_OR_RETURN(ScannAssets assets,
                         Serialize(staging, /*relative_path=*/true));
  // Absolute paths are absolute even when `dir` is relative: the loader
  // resolves a relative asset path against the index directory, so upstream's
  // `dir + "/" + name` made it look for dir/dir/name.
  fs::path absolute_dir;
  if (!relative_path) {
    absolute_dir = fs::absolute(dir, ec);
    if (ec)
      return InternalError(absl::StrCat("Failed to make ", dir,
                                        " an absolute path: ", ec.message()));
  }
  vector<std::string> files = {"scann_config.pb"};
  for (ScannAsset& asset : *assets.mutable_assets()) {
    files.push_back(asset.asset_path());
    if (!relative_path)
      asset.set_asset_path(
          (absolute_dir / asset.asset_path()).lexically_normal().string());
  }
  for (const auto& [name, contents] : extra_files) {
    SCANN_RETURN_IF_ERROR(WriteWholeFile(staging + "/" + name, contents));
    files.push_back(name);
  }
  std::string manifest;
  if (!google::protobuf::TextFormat::PrintToString(assets, &manifest))
    return InternalError("Failed to print the ScannAssets proto.");
  SCANN_RETURN_IF_ERROR(
      WriteWholeFile(staging + "/scann_assets.pbtxt", manifest));
  SCANN_RETURN_IF_ERROR(WriteWholeFile(
      staging + "/incomplete",
      absl::StrCat("# scann-core: serialize() is writing this index "
                   "directory, or was\n# interrupted while writing it. Its "
                   "files may come from different indexes.\n# Serialize the "
                   "index again.\n",
                   kIncompleteSerializationMarker, ": true\n")));
  for (const std::string& f : files)
    SCANN_RETURN_IF_ERROR(FsyncPath(staging + "/" + f, false));
  SCANN_RETURN_IF_ERROR(FsyncPath(staging + "/scann_assets.pbtxt", false));
  SCANN_RETURN_IF_ERROR(FsyncPath(staging + "/incomplete", false));

  // Commit: the marker, then the files (removing a previous index's files
  // that this one lacks, notably a stale scann_docids.pkl), then the
  // manifest.
  const std::string manifest_path = dir + "/scann_assets.pbtxt";
  SCANN_RETURN_IF_ERROR(RenamePath(staging + "/incomplete", manifest_path));
  SCANN_RETURN_IF_ERROR(FsyncPath(dir, true));
  for (absl::string_view name : kIndexFileNames) {
    if (absl::c_linear_search(files, name)) continue;
    const std::string path = absl::StrCat(dir, "/", name);
    fs::remove(path, ec);
    if (ec)
      return InternalError(
          absl::StrCat("Failed to remove ", path, ": ", ec.message()));
  }
  for (const std::string& f : files)
    SCANN_RETURN_IF_ERROR(RenamePath(staging + "/" + f, dir + "/" + f));
  SCANN_RETURN_IF_ERROR(FsyncPath(dir, true));
  SCANN_RETURN_IF_ERROR(
      RenamePath(staging + "/scann_assets.pbtxt", manifest_path));
  return FsyncPath(dir, true);
}

StatusOr<ScannInterface::ScannHealthStats> ScannInterface::GetHealthStats()
    const {
  return scann_->GetHealthStats();
}

Status ScannInterface::InitializeHealthStats() {
  return scann_->InitializeHealthStats();
}

StatusOr<SingleMachineFactoryOptions> ScannInterface::ExtractOptions() {
  return scann_->ExtractSingleMachineFactoryOptions();
}

}  // namespace research_scann
