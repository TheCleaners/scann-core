#include "shim.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "scann-core/src/bridge.rs.h"
#include "scann/data_format/datapoint.h"
#include "scann/utils/types.h"

namespace scann_core_ffi {

namespace {

// cxx turns a thrown exception into `Result::Err` on the Rust side for any
// bridge function declared with a `Result<T>` return type -- this is the
// StatusOr/Status -> Result adapter mentioned in bridge.rs's module doc.
void ThrowIfNotOk(const absl::Status& status, const char* context) {
  if (!status.ok()) {
    throw std::runtime_error(std::string(context) + ": " +
                              std::string(status.message()));
  }
}

}  // namespace

std::unique_ptr<ScannIndex> scann_new(rust::Slice<const float> dataset,
                                       uint64_t n_points, rust::Str config,
                                       int32_t training_threads) {
  auto idx = std::make_unique<ScannIndex>();
  research_scann::ConstSpan<float> dataset_span(dataset.data(),
                                                  dataset.size());
  absl::string_view config_view(config.data(), config.size());
  auto status =
      idx->Initialize(dataset_span, static_cast<research_scann::DatapointIndex>(n_points),
                       config_view, training_threads);
  ThrowIfNotOk(status, "Error initializing scann-core index");
  return idx;
}

SearchResult scann_search(const ScannIndex& idx, rust::Slice<const float> query,
                           int32_t final_nn, int32_t pre_reorder_nn,
                           int32_t leaves) {
  research_scann::DatapointPtr<float> ptr(nullptr, query.data(), query.size(),
                                           query.size());
  research_scann::NNResultsVector res;
  // Search() is const; ReshapeNNResult() below is not (it's a template
  // method, not marked const, even though it only reads `res`) -- hence
  // the const_cast only on that call.
  auto status = idx.Search(ptr, &res, final_nn, pre_reorder_nn, leaves);
  ThrowIfNotOk(status, "Error during search");

  SearchResult out;
  out.indices.reserve(res.size());
  out.distances.reserve(res.size());
  std::vector<research_scann::DatapointIndex> idx_buf(res.size());
  std::vector<float> dist_buf(res.size());
  const_cast<ScannIndex&>(idx).ReshapeNNResult(res, idx_buf.data(), dist_buf.data());
  for (size_t i = 0; i < res.size(); ++i) {
    out.indices.push_back(static_cast<uint32_t>(idx_buf[i]));
    out.distances.push_back(dist_buf[i]);
  }
  return out;
}

uint64_t scann_size(const ScannIndex& idx) {
  return static_cast<uint64_t>(idx.n_points());
}

}  // namespace scann_core_ffi
