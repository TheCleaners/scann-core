#pragma once

#include <cstdint>
#include <memory>

#include "rust/cxx.h"
#include "scann/scann_ops/cc/scann.h"

namespace scann_core_ffi {

struct SearchResult;

// Opaque to Rust: cxx only ever holds this behind a UniquePtr<ScannIndex>.
// Aliasing the real facade class directly (rather than wrapping it in
// another struct) keeps this shim as thin as possible -- ScannInterface
// already has no pybind11/Python dependency (see scann/scann_ops/cc/scann.h),
// so there's nothing left to adapt structurally, only the types crossing
// the FFI boundary itself.
using ScannIndex = ::research_scann::ScannInterface;

std::unique_ptr<ScannIndex> scann_new(rust::Slice<const float> dataset,
                                       uint64_t n_points, rust::Str config,
                                       int32_t training_threads);

SearchResult scann_search(const ScannIndex& idx, rust::Slice<const float> query,
                           int32_t final_nn, int32_t pre_reorder_nn,
                           int32_t leaves);

uint64_t scann_size(const ScannIndex& idx);

}  // namespace scann_core_ffi
