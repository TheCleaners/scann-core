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

// scann-core's TensorFlow ops: upstream's ScannSearch / ScannSearchBatched
// (scann/scann_ops/cc/kernels/scann_ops.cc and ops/scann_ops.cc), rewritten
// against TensorFlow's C API only (tensorflow/c/kernels.h and ops.h), so
// that no TensorFlow C++ header -- and none of TensorFlow's abseil or
// protobuf -- is involved. This library links its own abseil and protobuf
// and exports no symbols (see CMakeLists.txt).
//
// The C API can't create TensorFlow resources in the pip wheel, so instead
// of a searcher resource the search ops take the index itself: the files
// SerializeToDirectory writes (scann_config.pb, scann_assets.pbtxt, the
// .npy/.pb assets, and optionally scann_docids.pkl, which is ignored), as
// two string tensors of names and contents. The Python side keeps them in
// tf.Variables, so they are saved in SavedModels and checkpoints. A kernel
// builds the searcher from them on first use (LoadArtifactsFromMemory) and
// caches it in a registry keyed by the op's index_id attr (a uuid per
// index) and a fingerprint of the tensors; see Fingerprint below for what
// the fingerprint does and doesn't detect. (The attr plays the role of a
// resource's shared_name, but isn't called that: TensorFlow's Python
// SavedModel loader appends "_load_<n>" to every shared_name attr, so a
// loaded model's functions would no longer share the searcher with a
// searcher_from_module() of the same model.)
//
// Ops (Python names via tf.load_op_library):
//   ScannCoreSearch        scann_core_search
//   ScannCoreSearchBatched scann_core_search_batched
//   ScannCoreStats         scann_core_stats

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
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
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
// Header-only; TF_StringGetDataPointer and friends are only exported from
// libtensorflow_cc, which a custom op library shouldn't depend on.
#include "tsl/platform/ctstring.h"

namespace {

using research_scann::ScannInterface;

// --- TensorFlow C API helpers ---------------------------------------------

struct Status {
  TF_Status* s = TF_NewStatus();
  Status() = default;
  Status(const Status&) = delete;
  Status& operator=(const Status&) = delete;
  ~Status() { TF_DeleteStatus(s); }
  bool ok() const { return TF_GetCode(s) == TF_OK; }
};

// Owns a TF_Tensor from TF_GetInput / TF_AllocateOutput.
struct Tensor {
  TF_Tensor* t = nullptr;
  Tensor() = default;
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  ~Tensor() {
    if (t) TF_DeleteTensor(t);
  }
  int dims() const { return TF_NumDims(t); }
  int64_t dim(int i) const { return TF_Dim(t, i); }
  TF_DataType type() const { return TF_TensorType(t); }
  const void* data() const { return TF_TensorData(t); }
  void* mutable_data() { return TF_TensorData(t); }
};

void Fail(TF_OpKernelContext* ctx, TF_Code code, const std::string& msg) {
  Status st;
  TF_SetStatus(st.s, code, msg.c_str());
  TF_OpKernelContext_Failure(ctx, st.s);
}

void Fail(TF_OpKernelContext* ctx, const absl::Status& status,
          absl::string_view prefix = "") {
  // absl::StatusCode and TF_Code share the canonical code numbers.
  Fail(ctx, static_cast<TF_Code>(static_cast<int>(status.code())),
       absl::StrCat(prefix, status.message()));
}

bool GetInput(TF_OpKernelContext* ctx, int i, Tensor* out) {
  Status st;
  TF_GetInput(ctx, i, &out->t, st.s);
  if (!st.ok()) TF_OpKernelContext_Failure(ctx, st.s);
  return st.ok();
}

bool Allocate(TF_OpKernelContext* ctx, int i, TF_DataType type,
              const std::vector<int64_t>& dims, size_t element_size,
              Tensor* out) {
  size_t n = 1;
  for (int64_t d : dims) n *= static_cast<size_t>(d);
  Status st;
  out->t = TF_AllocateOutput(ctx, i, type, dims.data(),
                             static_cast<int>(dims.size()), n * element_size,
                             st.s);
  if (!st.ok()) TF_OpKernelContext_Failure(ctx, st.s);
  return st.ok();
}

absl::string_view StringAt(const Tensor& t, int64_t i) {
  const auto* s = static_cast<const TF_TString*>(t.data()) + i;
  return absl::string_view(TF_TString_GetDataPointer(s),
                           TF_TString_GetSize(s));
}

// A scalar input of the given type, or an InvalidArgument failure.
template <typename T>
bool GetScalar(TF_OpKernelContext* ctx, int i, TF_DataType type,
               const char* name, T* out) {
  Tensor t;
  if (!GetInput(ctx, i, &t)) return false;
  if (t.type() != type || t.dims() != 0) {
    Fail(ctx, TF_INVALID_ARGUMENT, absl::StrCat(name, " must be a scalar."));
    return false;
  }
  *out = *static_cast<const T*>(t.data());
  return true;
}

// --- The index tensors and their fingerprint -------------------------------

constexpr int kAssetNamesInput = 0;
constexpr int kAssetContentsInput = 1;
constexpr int kQueriesInput = 2;

struct IndexTensors {
  Tensor names, contents;
  int64_t size() const { return names.dim(0); }
  absl::string_view name(int64_t i) const { return StringAt(names, i); }
  absl::string_view content(int64_t i) const { return StringAt(contents, i); }
};

bool GetIndexTensors(TF_OpKernelContext* ctx, IndexTensors* index) {
  if (!GetInput(ctx, kAssetNamesInput, &index->names) ||
      !GetInput(ctx, kAssetContentsInput, &index->contents))
    return false;
  if (index->names.type() != TF_STRING || index->names.dims() != 1 ||
      index->contents.type() != TF_STRING || index->contents.dims() != 1 ||
      index->names.dim(0) != index->contents.dim(0)) {
    Fail(ctx, TF_INVALID_ARGUMENT,
         "asset_names and asset_contents must be string vectors of the same "
         "length (the index files' names and contents).");
    return false;
  }
  return true;
}

// Identifies the index tensors' values cheaply enough to compute on every
// search, so that a cached searcher is never used for other values (e.g.
// after assigning the variables, or restoring a checkpoint of another index
// into them). Hashing every byte would cost about as much as reading the
// index; instead the fingerprint covers
//   - the number of files, each file's name and size (so any change of
//     shape, dtype, number of points or dimensionality),
//   - the full contents of every file up to kFullHashBytes, which includes
//     scann_config.pb and scann_assets.pbtxt,
//   - for larger files, their first and last kEdgeBytes and kSamples
//     windows of kSampleBytes spread evenly over the file.
// Not detected: a change confined to the unsampled bytes of a large file
// that keeps every size, e.g. assigning an index whose data differs from
// the cached one's in a few points only. Two indexes built from different
// data differ nearly everywhere (codes, centers, quantized data), so their
// samples differ.
struct Fingerprint {
  uint64_t hash = 0;
  uint64_t bytes = 0;
  bool operator==(const Fingerprint& o) const {
    return hash == o.hash && bytes == o.bytes;
  }
  bool operator<(const Fingerprint& o) const {
    return std::tie(hash, bytes) < std::tie(o.hash, o.bytes);
  }
};

constexpr size_t kFullHashBytes = size_t{64} << 10;
constexpr size_t kEdgeBytes = 4096;
constexpr size_t kSamples = 64;
constexpr size_t kSampleBytes = 64;

Fingerprint ComputeFingerprint(const IndexTensors& index) {
  Fingerprint fp;
  size_t h = absl::HashOf(index.size());
  for (int64_t i = 0; i < index.size(); ++i) {
    const absl::string_view name = index.name(i);
    const absl::string_view c = index.content(i);
    fp.bytes += c.size();
    h = absl::HashOf(h, name, c.size());
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

// --- The searcher registry ----------------------------------------------------

// A searcher built from one set of index tensor values. Kernels hold
// shared_ptrs to the entries they use; the registry only weak ones, so a
// searcher lives as long as some kernel (a graph node, a function, an eager
// op cache entry) that used it, and is shared by all of them.
struct Entry {
  std::mutex build_mu;
  // Set once, under build_mu; read-only afterwards (searches are const).
  std::unique_ptr<ScannInterface> scann;
};

std::mutex g_registry_mu;
std::map<std::tuple<std::string, Fingerprint>, std::weak_ptr<Entry>>
    g_registry;
std::atomic<int64_t> g_builds{0};

std::shared_ptr<Entry> FindOrCreateEntry(const std::string& index_id,
                                         const Fingerprint& fp) {
  std::lock_guard<std::mutex> lock(g_registry_mu);
  for (auto it = g_registry.begin(); it != g_registry.end();) {
    if (it->second.expired())
      it = g_registry.erase(it);
    else
      ++it;
  }
  std::weak_ptr<Entry>& slot = g_registry[{index_id, fp}];
  std::shared_ptr<Entry> entry = slot.lock();
  if (!entry) {
    entry = std::make_shared<Entry>();
    slot = entry;
  }
  return entry;
}

absl::StatusOr<std::unique_ptr<ScannInterface>> BuildSearcher(
    const IndexTensors& index) {
  absl::flat_hash_map<std::string, absl::string_view> files;
  for (int64_t i = 0; i < index.size(); ++i)
    files[std::string(index.name(i))] = index.content(i);
  auto artifacts = ScannInterface::LoadArtifactsFromMemory(files);
  if (!artifacts.ok()) return artifacts.status();
  auto scann = std::make_unique<ScannInterface>();
  absl::Status st = scann->Initialize(*std::move(artifacts));
  if (!st.ok()) return st;
  // The outputs are int32, as upstream's.
  if (scann->n_points() >
      static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    return absl::InvalidArgumentError(absl::StrCat(
        "The index has ", scann->n_points(),
        " points; the TensorFlow op's int32 indices support up to 2^31 - 1."));
  return scann;
}

// --- Search kernels ---------------------------------------------------------

struct SearchKernel {
  std::string index_id;
  std::mutex mu;
  Fingerprint fp;                // guarded by mu
  std::shared_ptr<Entry> entry;  // guarded by mu
};

void* CreateSearchKernel(TF_OpKernelConstruction* ctx) {
  auto* k = new SearchKernel;
  Status st;
  int32_t list_size = 0, total_size = 0;
  TF_OpKernelConstruction_GetAttrSize(ctx, "index_id", &list_size,
                                      &total_size, st.s);
  if (st.ok() && total_size > 0) {
    k->index_id.resize(total_size);
    TF_OpKernelConstruction_GetAttrString(ctx, "index_id",
                                          k->index_id.data(), total_size,
                                          st.s);
  }
  if (!st.ok()) TF_OpKernelConstruction_Failure(ctx, st.s);
  return k;
}

void DeleteSearchKernel(void* kernel) {
  delete static_cast<SearchKernel*>(kernel);
}

// The searcher for this call's index tensors: the kernel's own if the
// fingerprint matches, else a shared one from the registry, built on first
// use. Returns null after reporting a failure.
std::shared_ptr<Entry> GetSearcher(SearchKernel* k, TF_OpKernelContext* ctx) {
  IndexTensors index;
  if (!GetIndexTensors(ctx, &index)) return nullptr;
  const Fingerprint fp = ComputeFingerprint(index);
  {
    std::lock_guard<std::mutex> lock(k->mu);
    if (k->entry && k->fp == fp) return k->entry;
  }
  std::shared_ptr<Entry> entry = FindOrCreateEntry(k->index_id, fp);
  {
    std::lock_guard<std::mutex> lock(entry->build_mu);
    if (!entry->scann) {
      auto scann = BuildSearcher(index);
      if (!scann.ok()) {
        Fail(ctx, scann.status(),
             "Can't build the ScaNN searcher from the index tensors: ");
        return nullptr;
      }
      entry->scann = *std::move(scann);
      ++g_builds;
    }
  }
  std::shared_ptr<Entry> previous;
  {
    std::lock_guard<std::mutex> lock(k->mu);
    previous = std::move(k->entry);  // Freed outside the lock.
    k->entry = entry;
    k->fp = fp;
  }
  return entry;
}

struct SearchParams {
  int32_t final_nn, pre_reorder_nn, leaves;
};

bool GetSearchParams(TF_OpKernelContext* ctx, SearchParams* p) {
  return GetScalar(ctx, 3, TF_INT32, "final_num_neighbors", &p->final_nn) &&
         GetScalar(ctx, 4, TF_INT32, "pre_reordering_num_neighbors",
                   &p->pre_reorder_nn) &&
         GetScalar(ctx, 5, TF_INT32, "leaves_to_search", &p->leaves);
}

// ScannCoreSearch: one query, [dim] -> index: int32[n], distance: float[n],
// n <= k (upstream's ScannSearch).
void ComputeSearch(void* kernel, TF_OpKernelContext* ctx) {
  auto* k = static_cast<SearchKernel*>(kernel);
  Tensor q;
  SearchParams p;
  if (!GetInput(ctx, kQueriesInput, &q) || !GetSearchParams(ctx, &p)) return;
  if (q.type() != TF_FLOAT || q.dims() != 1)
    return Fail(ctx, TF_INVALID_ARGUMENT,
                "The query must be a one-dimensional float32 tensor; use "
                "search_batched for batches.");
  std::shared_ptr<Entry> e = GetSearcher(k, ctx);
  if (!e) return;
  const ScannInterface& scann = *e->scann;

  const auto* query = static_cast<const float*>(q.data());
  const size_t dim = static_cast<size_t>(q.dim(0));
  research_scann::DatapointPtr<float> ptr(nullptr, query, dim, dim);
  research_scann::NNResultsVector res;
  absl::Status st =
      scann.Search(ptr, &res, p.final_nn, p.pre_reorder_nn, p.leaves);
  if (!st.ok()) return Fail(ctx, st);

  const int64_t n = static_cast<int64_t>(res.size());
  Tensor index, distance;
  if (!Allocate(ctx, 0, TF_INT32, {n}, sizeof(int32_t), &index) ||
      !Allocate(ctx, 1, TF_FLOAT, {n}, sizeof(float), &distance))
    return;
  scann.ReshapeNNResult(res, static_cast<int32_t*>(index.mutable_data()),
                        static_cast<float*>(distance.mutable_data()));
}

// ScannCoreSearchBatched: [num_queries, dim] -> indices: int32[num_queries,
// w], distances: float[num_queries, w]. Unlike upstream's ScannSearchBatched
// (always as wide as the longest row), w is final_num_neighbors when it's
// given (> 0), so the width is known in advance; rows with fewer results
// are padded with index 0 and distance NaN. With the default k (-1), w is
// the longest row's length. This is what the pybind searcher returns.
void ComputeSearchBatched(void* kernel, TF_OpKernelContext* ctx) {
  auto* k = static_cast<SearchKernel*>(kernel);
  Tensor q;
  SearchParams p;
  bool parallel = false;
  int32_t batch_size = 0;
  if (!GetInput(ctx, kQueriesInput, &q) || !GetSearchParams(ctx, &p) ||
      !GetScalar(ctx, 6, TF_BOOL, "parallel", &parallel) ||
      !GetScalar(ctx, 7, TF_INT32, "batch_size", &batch_size))
    return;
  if (q.type() != TF_FLOAT || q.dims() != 2)
    return Fail(ctx, TF_INVALID_ARGUMENT,
                "The queries must be a two-dimensional float32 tensor.");
  std::shared_ptr<Entry> e = GetSearcher(k, ctx);
  if (!e) return;
  const ScannInterface& scann = *e->scann;

  const int64_t num_queries = q.dim(0), dim = q.dim(1);
  Tensor indices, distances;
  if (num_queries == 0) {
    // Empty (0, k) results after the same checks as a search, as the
    // pybind searcher returns.
    if (p.final_nn == 0)
      return Fail(ctx, TF_INVALID_ARGUMENT, "final_num_neighbors must be > 0");
    if (static_cast<size_t>(dim) != scann.dimensionality())
      return Fail(ctx, TF_INVALID_ARGUMENT,
                  absl::StrCat("Queries have dimensionality ", dim,
                               ", but the dataset has ",
                               scann.dimensionality()));
    const int64_t w =
        p.final_nn > 0 ? p.final_nn : scann.default_num_neighbors();
    Allocate(ctx, 0, TF_INT32, {0, w}, sizeof(int32_t), &indices) &&
        Allocate(ctx, 1, TF_FLOAT, {0, w}, sizeof(float), &distances);
    return;
  }
  const auto* data = static_cast<const float*>(q.data());
  research_scann::DenseDataset<float> queries(
      std::vector<float>(data, data + num_queries * dim), num_queries);
  std::vector<research_scann::NNResultsVector> res(num_queries);
  auto res_span = research_scann::MakeMutableSpan(res);
  absl::Status st =
      parallel ? scann.SearchBatchedParallel(queries, res_span, p.final_nn,
                                             p.pre_reorder_nn, p.leaves,
                                             batch_size)
               : scann.SearchBatched(queries, res_span, p.final_nn,
                                     p.pre_reorder_nn, p.leaves);
  if (!st.ok()) return Fail(ctx, st);

  int64_t w = std::max<int64_t>(p.final_nn, 0);
  for (const auto& r : res) w = std::max<int64_t>(w, r.size());
  if (!Allocate(ctx, 0, TF_INT32, {num_queries, w}, sizeof(int32_t),
                &indices) ||
      !Allocate(ctx, 1, TF_FLOAT, {num_queries, w}, sizeof(float), &distances))
    return;
  scann.ReshapeBatchedNNResult(research_scann::MakeConstSpan(res),
                               static_cast<int32_t*>(indices.mutable_data()),
                               static_cast<float*>(distances.mutable_data()),
                               static_cast<int>(w));
}

// ScannCoreStats: () -> live_searchers: int64, builds: int64. The number of
// searchers currently cached and the number built since the library was
// loaded; for tests and for checking that a model builds its index once.
void ComputeStats(void*, TF_OpKernelContext* ctx) {
  int64_t live = 0;
  {
    std::lock_guard<std::mutex> lock(g_registry_mu);
    for (const auto& [key, entry] : g_registry) {
      auto e = entry.lock();
      live += e && e->scann != nullptr;
    }
  }
  const int64_t values[2] = {live, g_builds.load()};
  for (int i = 0; i < 2; ++i) {
    Tensor out;
    if (!Allocate(ctx, i, TF_INT64, {}, sizeof(int64_t), &out)) return;
    *static_cast<int64_t*>(out.mutable_data()) = values[i];
  }
}

// --- Shape functions ------------------------------------------------------------

struct ShapeHandles {
  std::vector<TF_ShapeHandle*> handles;
  TF_ShapeHandle* New() {
    handles.push_back(TF_NewShapeHandle());
    return handles.back();
  }
  ~ShapeHandles() {
    for (TF_ShapeHandle* h : handles) TF_DeleteShapeHandle(h);
  }
};

bool Ok(TF_Status* s) { return TF_GetCode(s) == TF_OK; }

// Checks the index inputs are vectors.
void CheckIndexShapes(TF_ShapeInferenceContext* ctx, ShapeHandles& h,
                      TF_Status* status) {
  for (int i : {kAssetNamesInput, kAssetContentsInput}) {
    TF_ShapeHandle* in = h.New();
    TF_ShapeHandle* v = h.New();
    if (Ok(status)) TF_ShapeInferenceContextGetInput(ctx, i, in, status);
    if (Ok(status)) TF_ShapeInferenceContextWithRank(ctx, in, 1, v, status);
  }
}

// [?]: the number of results isn't known before the search.
void SearchShape(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  ShapeHandles h;
  CheckIndexShapes(ctx, h, status);
  TF_ShapeHandle* q = h.New();
  TF_ShapeHandle* q1 = h.New();
  TF_ShapeHandle* out = h.New();
  if (Ok(status))
    TF_ShapeInferenceContextGetInput(ctx, kQueriesInput, q, status);
  if (Ok(status)) TF_ShapeInferenceContextWithRank(ctx, q, 1, q1, status);
  if (Ok(status)) TF_ShapeInferenceContextWithRank(ctx, h.New(), 1, out, status);
  for (int i = 0; i < 2 && Ok(status); ++i)
    TF_ShapeInferenceContextSetOutput(ctx, i, out, status);
}

// [num_queries, ?]: the width depends on final_num_neighbors, an input the
// C API's shape inference can't read; the Python wrapper sets it when it's
// a Python int.
void SearchBatchedShape(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  ShapeHandles h;
  CheckIndexShapes(ctx, h, status);
  TF_ShapeHandle* q = h.New();
  TF_ShapeHandle* q2 = h.New();
  TF_ShapeHandle* nq = h.New();
  TF_ShapeHandle* unknown1 = h.New();
  TF_ShapeHandle* out = h.New();
  if (Ok(status))
    TF_ShapeInferenceContextGetInput(ctx, kQueriesInput, q, status);
  if (Ok(status)) TF_ShapeInferenceContextWithRank(ctx, q, 2, q2, status);
  if (Ok(status)) TF_ShapeInferenceContextSubshape(ctx, q2, 0, 1, nq, status);
  if (Ok(status))
    TF_ShapeInferenceContextWithRank(ctx, h.New(), 1, unknown1, status);
  if (Ok(status))
    TF_ShapeInferenceContextConcatenateShapes(ctx, nq, unknown1, out, status);
  for (int i = 0; i < 2 && Ok(status); ++i)
    TF_ShapeInferenceContextSetOutput(ctx, i, out, status);
}

void ScalarsShape(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  TF_ShapeHandle* s = TF_ShapeInferenceContextScalar(ctx);
  for (int i = 0; i < 2 && Ok(status); ++i)
    TF_ShapeInferenceContextSetOutput(ctx, i, s, status);
  TF_DeleteShapeHandle(s);
}

// --- Registration ------------------------------------------------------------------

void CheckRegistered(TF_Status* s, const char* what) {
  if (TF_GetCode(s) != TF_OK) {
    std::fprintf(stderr, "scann-core TensorFlow ops: registering %s failed: %s\n",
                 what, TF_Message(s));
    std::abort();
  }
}

void RegisterOp(const char* name, const std::vector<const char*>& inputs,
                const std::vector<const char*>& outputs,
                const std::vector<const char*>& attrs, bool stateful,
                void (*shape_fn)(TF_ShapeInferenceContext*, TF_Status*)) {
  TF_OpDefinitionBuilder* b = TF_NewOpDefinitionBuilder(name);
  for (const char* i : inputs) TF_OpDefinitionBuilderAddInput(b, i);
  for (const char* o : outputs) TF_OpDefinitionBuilderAddOutput(b, o);
  for (const char* a : attrs) TF_OpDefinitionBuilderAddAttr(b, a);
  TF_OpDefinitionBuilderSetIsStateful(b, stateful);
  TF_OpDefinitionBuilderSetShapeInferenceFunction(b, shape_fn);
  Status st;
  TF_RegisterOpDefinition(b, st.s);  // Takes ownership of b.
  CheckRegistered(st.s, name);
}

void RegisterKernel(const char* name,
                    void* (*create)(TF_OpKernelConstruction*),
                    void (*compute)(void*, TF_OpKernelContext*),
                    void (*del)(void*)) {
  TF_KernelBuilder* kb = TF_NewKernelBuilder(name, "CPU", create, compute, del);
  Status st;
  TF_RegisterKernelBuilder(name, kb, st.s);  // Takes ownership of kb.
  CheckRegistered(st.s, name);
}

void* NoKernelState(TF_OpKernelConstruction*) { return nullptr; }
void NoDelete(void*) {}

bool Register() {
  // The searches aren't stateful (upstream's weren't either): the results
  // depend only on the inputs; the cache is an implementation detail.
  RegisterOp("ScannCoreSearch",
             {"asset_names: string", "asset_contents: string",
              "queries: float32", "final_num_neighbors: int32",
              "pre_reordering_num_neighbors: int32",
              "leaves_to_search: int32"},
             {"index: int32", "distance: float32"}, {"index_id: string"},
             /*stateful=*/false, SearchShape);
  RegisterOp("ScannCoreSearchBatched",
             {"asset_names: string", "asset_contents: string",
              "queries: float32", "final_num_neighbors: int32",
              "pre_reordering_num_neighbors: int32",
              "leaves_to_search: int32", "parallel: bool",
              "batch_size: int32"},
             {"indices: int32", "distances: float32"}, {"index_id: string"},
             /*stateful=*/false, SearchBatchedShape);
  RegisterOp("ScannCoreStats", {}, {"live_searchers: int64", "builds: int64"},
             {}, /*stateful=*/true, ScalarsShape);
  RegisterKernel("ScannCoreSearch", CreateSearchKernel, ComputeSearch,
                 DeleteSearchKernel);
  RegisterKernel("ScannCoreSearchBatched", CreateSearchKernel,
                 ComputeSearchBatched, DeleteSearchKernel);
  RegisterKernel("ScannCoreStats", NoKernelState, ComputeStats, NoDelete);
  return true;
}

[[maybe_unused]] const bool registered = Register();

}  // namespace
