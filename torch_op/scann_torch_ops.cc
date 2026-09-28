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

// scann-core's native PyTorch ops, the backend of scann.torch when the
// scann-core-torch package is installed (see docs/integrations.md).
//
// Built against LibTorch's stable ABI only (torch/csrc/stable,
// torch/headeronly; TORCH_TARGET_VERSION 2.10): no c10/ATen/torch C++
// symbol is used, so one library runs on every torch >= 2.10 (CPU, CUDA,
// ROCm builds). It imports only the stable C shim functions (aoti_torch_*,
// torch_*) from libtorch_cpu.so, and links scann-core and all of its
// dependencies statically, exporting no symbols (see CMakeLists.txt).
//
// The stable ABI has no custom classes, so, as the TensorFlow op (tf_op/),
// the search ops take the index itself as tensors -- the files
// SerializeToDirectory writes, as
//   index_data     uint8 [total bytes]   the files' contents, concatenated
//   index_offsets  int64 [num_files + 1] file i is data[offsets[i]:offsets[i+1]]
//   index_names    uint8 [...]           the file names, each followed by '\n'
// -- which scann.torch.Searcher keeps as buffers, so state_dict(),
// torch.export and AOTInductor programs carry the index. A kernel builds
// the searcher from them on first use (LoadArtifactsFromMemory) and caches
// it by shared_name (a random id per Searcher, a constant in exported
// graphs) and a fingerprint of the tensors; see ComputeFingerprint for what
// it detects. Later calls with the same name and fingerprint reuse it; a
// different fingerprint under the same name rebuilds (and replaces) it.
//
// Ops (torch.ops.scann.*):
//   search(index_data, index_offsets, index_names, shared_name, query, k,
//          pre_reorder_num_neighbors, leaves_to_search) -> (indices, distances)
//   search_batched(..., queries, k, pre_reorder_num_neighbors,
//                  leaves_to_search, parallel, batch_size) -> (indices, distances)
//   stats(like) -> int64[2]: cached searchers, searchers built so far
//   release(like, shared_name) -> int64[1]: cached searchers dropped (0/1)
// The results are exactly what scann.torch's Python backend returns: int64
// indices and float32 distances, exactly k per query (missing results are
// index -1 and distance NaN), on the queries' device. Any float (or other
// numeric) queries on any device are accepted; the search runs on the CPU.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/types.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

namespace {

using research_scann::ScannInterface;
using torch::headeronly::ScalarType;
using torch::stable::Device;
using torch::stable::DeviceType;
using torch::stable::Tensor;

// Errors become Python RuntimeErrors (the stable ABI turns exceptions from
// a kernel into c10 errors).
[[noreturn]] void Fail(const std::string& msg) {
  throw std::runtime_error(absl::StrCat("scann.torch: ", msg));
}

void Check(const absl::Status& status, absl::string_view prefix = "") {
  if (!status.ok()) Fail(absl::StrCat(prefix, status.message()));
}

// A tensor in host memory, contiguous, of the given dtype (a no-op for one
// that already is).
Tensor Host(const Tensor& t, ScalarType dtype) {
  Tensor host = t;
  if (t.scalar_type() != dtype || !t.is_cpu())
    host = torch::stable::to(t, dtype, std::nullopt, Device(DeviceType::CPU));
  return torch::stable::contiguous(host);
}

int ToInt(int64_t v, const char* name) {
  if (v < std::numeric_limits<int>::min() ||
      v > std::numeric_limits<int>::max())
    Fail(absl::StrCat(name, " is out of range: ", v));
  return static_cast<int>(v);
}

// --- The index tensors and their fingerprint -------------------------------

struct IndexFiles {
  // Host copies (the same tensors when they are already contiguous CPU
  // tensors of the right dtype); the views below point into them.
  Tensor data, offsets, names_blob;
  std::vector<absl::string_view> names, contents;
};

IndexFiles GetIndexFiles(const Tensor& data, const Tensor& offsets,
                         const Tensor& names) {
  if (data.dim() != 1 || data.scalar_type() != ScalarType::Byte ||
      offsets.dim() != 1 || offsets.scalar_type() != ScalarType::Long ||
      names.dim() != 1 || names.scalar_type() != ScalarType::Byte)
    Fail(
        "the index state must be index_data (uint8 [bytes]), index_offsets "
        "(int64 [num_files + 1]) and index_names (uint8 [bytes]) vectors, as "
        "scann.torch.Searcher keeps them.");
  IndexFiles f;
  f.data = Host(data, ScalarType::Byte);
  f.offsets = Host(offsets, ScalarType::Long);
  f.names_blob = Host(names, ScalarType::Byte);
  const int64_t n = f.offsets.numel() - 1;
  const auto* off = static_cast<const int64_t*>(f.offsets.const_data_ptr());
  const auto* bytes = static_cast<const char*>(f.data.const_data_ptr());
  const int64_t size = f.data.numel();
  if (n < 0 || off[0] != 0 || off[n] != size)
    Fail(absl::StrCat("index_offsets must start at 0 and end at the size of "
                      "index_data (",
                      size, " bytes)."));
  for (int64_t i = 0; i < n; ++i) {
    if (off[i + 1] < off[i])
      Fail("index_offsets must be non-decreasing.");
    f.contents.emplace_back(bytes + off[i], off[i + 1] - off[i]);
  }
  absl::string_view all(static_cast<const char*>(f.names_blob.const_data_ptr()),
                        f.names_blob.numel());
  while (!all.empty()) {
    const size_t end = all.find('\n');
    if (end == absl::string_view::npos)
      Fail("index_names must end with a newline.");
    f.names.push_back(all.substr(0, end));
    all.remove_prefix(end + 1);
  }
  if (static_cast<int64_t>(f.names.size()) != n)
    Fail(absl::StrCat("index_names has ", f.names.size(),
                      " names, index_offsets ", n, " files."));
  return f;
}

// Identifies the index tensors' values cheaply enough to compute on every
// search (as the TensorFlow op does), so that a cached searcher is never
// used for other values (after load_state_dict() of another index, or a
// changed buffer). Hashing every byte would cost about as much as reading
// the index; the fingerprint covers
//   - the number of files, each file's name and size (so any change of
//     shape, dtype, number of points or dimensionality),
//   - the full contents of every file up to kFullHashBytes, which includes
//     scann_config.pb and scann_assets.pbtxt,
//   - for larger files, their first and last kEdgeBytes and kSamples
//     windows of kSampleBytes spread evenly over the file.
// Not detected: a change confined to the unsampled bytes of a large file
// that keeps every size, e.g. an index whose data differs from the cached
// one's in a few points only. Two indexes built from different data differ
// nearly everywhere (codes, centers, quantized data), so their samples
// differ. (scann.torch.Searcher never changes its buffers in place; it
// replaces them.)
struct Fingerprint {
  uint64_t hash = 0;
  uint64_t bytes = 0;
  bool operator==(const Fingerprint& o) const {
    return hash == o.hash && bytes == o.bytes;
  }
  bool operator!=(const Fingerprint& o) const { return !(*this == o); }
};

constexpr size_t kFullHashBytes = size_t{64} << 10;
constexpr size_t kEdgeBytes = 4096;
constexpr size_t kSamples = 64;
constexpr size_t kSampleBytes = 64;

Fingerprint ComputeFingerprint(const IndexFiles& index) {
  Fingerprint fp;
  size_t h = absl::HashOf(index.names.size());
  for (size_t i = 0; i < index.names.size(); ++i) {
    const absl::string_view c = index.contents[i];
    fp.bytes += c.size();
    h = absl::HashOf(h, index.names[i], c.size());
    if (c.size() <= kFullHashBytes) {
      h = absl::HashOf(h, c);
      continue;
    }
    h = absl::HashOf(h, c.substr(0, kEdgeBytes),
                     c.substr(c.size() - kEdgeBytes));
    const size_t stride = (c.size() - kSampleBytes) / (kSamples - 1);
    for (size_t s = 0; s < kSamples; ++s)
      h = absl::HashOf(h, c.substr(s * stride, kSampleBytes));
  }
  fp.hash = h;
  return fp;
}

// --- The searcher cache -----------------------------------------------------

// One searcher per shared_name: the one built from the values last seen
// under that name. Searches hold a shared_ptr to it, so replacing or
// releasing it never pulls it from under a running search.
struct Entry {
  explicit Entry(const Fingerprint& f) : fp(f) {}
  const Fingerprint fp;
  std::mutex build_mu;
  // Set once, under build_mu; read-only afterwards (searches are const).
  std::shared_ptr<const ScannInterface> scann;
};

std::mutex g_cache_mu;
std::unordered_map<std::string, std::shared_ptr<Entry>> g_cache;
std::atomic<int64_t> g_builds{0};

std::unique_ptr<ScannInterface> BuildSearcher(const IndexFiles& index) {
  absl::flat_hash_map<std::string, absl::string_view> files;
  for (size_t i = 0; i < index.names.size(); ++i)
    files[std::string(index.names[i])] = index.contents[i];
  auto artifacts = ScannInterface::LoadArtifactsFromMemory(files);
  Check(artifacts.status(),
        "can't build the ScaNN searcher from the index tensors: ");
  auto scann = std::make_unique<ScannInterface>();
  Check(scann->Initialize(*std::move(artifacts)),
        "can't build the ScaNN searcher from the index tensors: ");
  return scann;
}

std::shared_ptr<const ScannInterface> GetSearcher(
    const std::string& shared_name, const Tensor& data, const Tensor& offsets,
    const Tensor& names) {
  const IndexFiles index = GetIndexFiles(data, offsets, names);
  const Fingerprint fp = ComputeFingerprint(index);
  std::shared_ptr<Entry> entry, previous;
  {
    std::lock_guard<std::mutex> lock(g_cache_mu);
    std::shared_ptr<Entry>& slot = g_cache[shared_name];
    if (!slot || slot->fp != fp) {
      previous = std::move(slot);  // Freed outside the lock.
      slot = std::make_shared<Entry>(fp);
    }
    entry = slot;
  }
  previous.reset();
  // Concurrent first searches build once; other indexes aren't held up.
  std::lock_guard<std::mutex> lock(entry->build_mu);
  if (!entry->scann) {
    entry->scann = BuildSearcher(index);
    ++g_builds;
  }
  return entry->scann;
}

// --- Results ------------------------------------------------------------------

Tensor EmptyCpu(std::vector<int64_t> shape, ScalarType dtype) {
  return torch::stable::empty(shape, dtype, std::nullopt,
                              Device(DeviceType::CPU));
}

// Writes one query's results into k slots: the neighbors (at most k), then
// index -1 / distance NaN. As scann.torch's Python backend (which pads the
// pybind searcher's results and maps every NaN distance to index -1).
void WriteRow(const ScannInterface& scann,
              const research_scann::NNResultsVector& res, int64_t k,
              int64_t* indices, float* distances) {
  const int64_t n = std::min<int64_t>(res.size(), k);
  if (static_cast<int64_t>(res.size()) <= k) {
    scann.ReshapeNNResult(res, indices, distances);
  } else {
    std::vector<int64_t> i(res.size());
    std::vector<float> d(res.size());
    scann.ReshapeNNResult(res, i.data(), d.data());
    std::copy_n(i.begin(), n, indices);
    std::copy_n(d.begin(), n, distances);
  }
  for (int64_t j = n; j < k; ++j) {
    indices[j] = -1;
    distances[j] = std::numeric_limits<float>::quiet_NaN();
  }
  for (int64_t j = 0; j < n; ++j)
    if (std::isnan(distances[j])) indices[j] = -1;
}

std::tuple<Tensor, Tensor> ToDevice(Tensor indices, Tensor distances,
                                    const Tensor& queries) {
  if (!queries.is_cpu()) {
    indices = torch::stable::to(indices, queries.device());
    distances = torch::stable::to(distances, queries.device());
  }
  return {indices, distances};
}

// --- Ops --------------------------------------------------------------------

std::tuple<Tensor, Tensor> Search(Tensor index_data, Tensor index_offsets,
                                  Tensor index_names, std::string shared_name,
                                  Tensor query, int64_t k,
                                  int64_t pre_reorder_num_neighbors,
                                  int64_t leaves_to_search) {
  if (query.dim() != 1)
    Fail(absl::StrCat("search() expects a 1-dimensional query, got ",
                      query.dim(), " dimensions."));
  if (k < 1) Fail(absl::StrCat("k must be > 0, got ", k, "."));
  const int final_nn = ToInt(k, "k");
  const int pre = ToInt(pre_reorder_num_neighbors, "pre_reorder_num_neighbors");
  const int leaves = ToInt(leaves_to_search, "leaves_to_search");
  std::shared_ptr<const ScannInterface> scann =
      GetSearcher(shared_name, index_data, index_offsets, index_names);
  Tensor q = Host(query, ScalarType::Float);
  const auto dim = static_cast<size_t>(q.numel());
  research_scann::DatapointPtr<float> ptr(
      nullptr, static_cast<const float*>(q.const_data_ptr()), dim, dim);
  research_scann::NNResultsVector res;
  Check(scann->Search(ptr, &res, final_nn, pre, leaves), "search failed: ");
  Tensor indices = EmptyCpu({k}, ScalarType::Long);
  Tensor distances = EmptyCpu({k}, ScalarType::Float);
  WriteRow(*scann, res, k, static_cast<int64_t*>(indices.mutable_data_ptr()),
           static_cast<float*>(distances.mutable_data_ptr()));
  return ToDevice(indices, distances, query);
}

std::tuple<Tensor, Tensor> SearchBatched(
    Tensor index_data, Tensor index_offsets, Tensor index_names,
    std::string shared_name, Tensor queries, int64_t k,
    int64_t pre_reorder_num_neighbors, int64_t leaves_to_search,
    bool parallel, int64_t batch_size) {
  if (queries.dim() != 2)
    Fail(absl::StrCat("search_batched() expects 2-dimensional queries, got ",
                      queries.dim(), " dimensions."));
  if (k < 1) Fail(absl::StrCat("k must be > 0, got ", k, "."));
  const int final_nn = ToInt(k, "k");
  const int pre = ToInt(pre_reorder_num_neighbors, "pre_reorder_num_neighbors");
  const int leaves = ToInt(leaves_to_search, "leaves_to_search");
  const int bs = ToInt(batch_size, "batch_size");
  std::shared_ptr<const ScannInterface> scann =
      GetSearcher(shared_name, index_data, index_offsets, index_names);
  const int64_t num_queries = queries.size(0), dim = queries.size(1);
  Tensor indices = EmptyCpu({num_queries, k}, ScalarType::Long);
  Tensor distances = EmptyCpu({num_queries, k}, ScalarType::Float);
  if (num_queries == 0) {
    // Empty results after the same check as a search (as the pybind
    // searcher).
    if (static_cast<size_t>(dim) != scann->dimensionality())
      Fail(absl::StrCat("Queries have dimensionality ", dim,
                        ", but the dataset has ", scann->dimensionality()));
    return ToDevice(indices, distances, queries);
  }
  Tensor q = Host(queries, ScalarType::Float);
  const auto* data = static_cast<const float*>(q.const_data_ptr());
  research_scann::DenseDataset<float> dataset(
      std::vector<float>(data, data + num_queries * dim), num_queries);
  std::vector<research_scann::NNResultsVector> res(num_queries);
  auto span = research_scann::MakeMutableSpan(res);
  Check(parallel ? scann->SearchBatchedParallel(dataset, span, final_nn, pre,
                                                leaves, bs)
                 : scann->SearchBatched(dataset, span, final_nn, pre, leaves),
        "search failed: ");
  auto* ip = static_cast<int64_t*>(indices.mutable_data_ptr());
  auto* dp = static_cast<float*>(distances.mutable_data_ptr());
  for (int64_t r = 0; r < num_queries; ++r)
    WriteRow(*scann, res[r], k, ip + r * k, dp + r * k);
  return ToDevice(indices, distances, queries);
}

Tensor Stats(Tensor like) {
  (void)like;
  int64_t live = 0;
  {
    std::lock_guard<std::mutex> lock(g_cache_mu);
    for (const auto& [name, entry] : g_cache) {
      std::lock_guard<std::mutex> build_lock(entry->build_mu);
      live += entry->scann != nullptr;
    }
  }
  Tensor out = EmptyCpu({2}, ScalarType::Long);
  auto* p = static_cast<int64_t*>(out.mutable_data_ptr());
  p[0] = live;
  p[1] = g_builds.load();
  return out;
}

Tensor Release(Tensor like, std::string shared_name) {
  (void)like;
  std::shared_ptr<Entry> dropped;  // Freed outside the lock.
  {
    std::lock_guard<std::mutex> lock(g_cache_mu);
    auto it = g_cache.find(shared_name);
    if (it != g_cache.end()) {
      dropped = std::move(it->second);
      g_cache.erase(it);
    }
  }
  Tensor out = EmptyCpu({1}, ScalarType::Long);
  *static_cast<int64_t*>(out.mutable_data_ptr()) = dropped != nullptr;
  return out;
}

}  // namespace

STABLE_TORCH_LIBRARY(scann, m) {
  m.def(
      "search(Tensor index_data, Tensor index_offsets, Tensor index_names, "
      "str shared_name, Tensor query, int k, int pre_reorder_num_neighbors, "
      "int leaves_to_search) -> (Tensor, Tensor)");
  m.def(
      "search_batched(Tensor index_data, Tensor index_offsets, "
      "Tensor index_names, str shared_name, Tensor queries, int k, "
      "int pre_reorder_num_neighbors, int leaves_to_search, bool parallel, "
      "int batch_size) -> (Tensor, Tensor)");
  m.def("stats(Tensor like) -> Tensor");
  m.def("release(Tensor like, str shared_name) -> Tensor");
}

// The same kernels for every device: the index state and the queries are
// read in host memory, the results go to the queries' device. ROCm builds
// of PyTorch present their GPUs as CUDA.
STABLE_TORCH_LIBRARY_IMPL(scann, CPU, m) {
  m.impl("search", TORCH_BOX(&Search));
  m.impl("search_batched", TORCH_BOX(&SearchBatched));
  m.impl("stats", TORCH_BOX(&Stats));
  m.impl("release", TORCH_BOX(&Release));
}

STABLE_TORCH_LIBRARY_IMPL(scann, CUDA, m) {
  m.impl("search", TORCH_BOX(&Search));
  m.impl("search_batched", TORCH_BOX(&SearchBatched));
  m.impl("stats", TORCH_BOX(&Stats));
  m.impl("release", TORCH_BOX(&Release));
}
