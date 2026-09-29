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

#pragma once

#include <cstdint>
#include <memory>

#include "rust/cxx.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann_core/config_builder.h"

namespace scann_core_ffi {

struct Neighbors;
struct HealthStats;
struct FfiTreeOptions;
struct FfiUpperTreeOptions;
struct FfiAhOptions;
struct FfiReorderOptions;
struct FfiPcaOptions;
struct FfiAutopilotOptions;
enum class DistanceMeasure : uint8_t;
enum class Quantization : uint8_t;
enum class IncrementalMode : uint8_t;

// Opaque to Rust, which only holds them behind a UniquePtr. ScannInterface
// has no pybind11/Python dependency, so it is bridged as is.
using ScannIndex = ::research_scann::ScannInterface;
using ConfigBuilder = ::scann_core::ConfigBuilder;

std::unique_ptr<ScannIndex> scann_new(rust::Slice<const float> dataset,
                                      uint64_t n_points, rust::Str config,
                                      int32_t training_threads);
std::unique_ptr<ScannIndex> scann_load(rust::Str dir);

Neighbors scann_search(const ScannIndex& idx, rust::Slice<const float> query,
                       int32_t final_nn, int32_t pre_reorder_nn,
                       int32_t leaves);
rust::Vec<Neighbors> scann_search_batched(const ScannIndex& idx,
                                          rust::Slice<const float> queries,
                                          uint64_t n_queries, int32_t final_nn,
                                          int32_t pre_reorder_nn,
                                          int32_t leaves, bool parallel,
                                          int32_t batch_size);

uint64_t scann_size(const ScannIndex& idx);
uint64_t scann_dimensionality(const ScannIndex& idx);
rust::String scann_config(ScannIndex& idx);
HealthStats scann_health_stats(const ScannIndex& idx);

void scann_serialize(ScannIndex& idx, rust::Str dir, bool relative_path);
void scann_set_num_threads(ScannIndex& idx, int32_t num_threads);
void scann_reserve(ScannIndex& idx, uint64_t n_points);
rust::Vec<uint32_t> scann_upsert(ScannIndex& idx, rust::Slice<const int64_t> ids,
                                 rust::Slice<const float> vectors,
                                 int32_t batch_size);
void scann_delete(ScannIndex& idx, rust::Slice<const uint32_t> ids);
void scann_rebalance(ScannIndex& idx, rust::Str config);
void scann_initialize_health_stats(ScannIndex& idx);

std::unique_ptr<ConfigBuilder> config_builder_new(int32_t num_neighbors,
                                                  DistanceMeasure distance,
                                                  uint32_t dimensionality);
void config_builder_tree(ConfigBuilder& b, const FfiTreeOptions& o);
void config_builder_upper_tree(ConfigBuilder& b, const FfiUpperTreeOptions& o);
void config_builder_score_ah(ConfigBuilder& b, const FfiAhOptions& o);
void config_builder_score_brute_force(ConfigBuilder& b, Quantization quantize);
void config_builder_reorder(ConfigBuilder& b, const FfiReorderOptions& o);
void config_builder_pca(ConfigBuilder& b, const FfiPcaOptions& o);
void config_builder_truncate(ConfigBuilder& b, int32_t reduction_dim);
void config_builder_autopilot(ConfigBuilder& b, const FfiAutopilotOptions& o);
void config_builder_l2_as_dot_product(ConfigBuilder& b, double scale,
                                      double center);
rust::String config_builder_build(const ConfigBuilder& b, uint64_t num_points);

}  // namespace scann_core_ffi
