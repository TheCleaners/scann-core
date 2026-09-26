// Copyright 2026 ebenali and TheCleaners.
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

//! cxx bridge declarations. The C++ side (`shim.h`/`shim.cc`) adapts
//! `research_scann::ScannInterface` (the pybind-free facade in
//! `src/scann/scann_ops/cc/scann.h`) and `scann_core::ConfigBuilder`
//! (`core/scann_core/config_builder.h`) to cxx-bridgeable types: slices
//! instead of `absl::Span`, `Result<T>` (a thrown `std::runtime_error` for a
//! non-OK `absl::Status`) instead of `absl::StatusOr`, and plain structs
//! instead of `NNResultsVector` / option structs with `std::optional`.
//!
//! Everything here is crate-private; `lib.rs` is the public, safe API.

#[cxx::bridge(namespace = "scann_core_ffi")]
pub(crate) mod ffi {
    /// Nearest neighbours of one query, best first.
    #[derive(Debug, Clone, Default, PartialEq)]
    struct Neighbors {
        indices: Vec<u32>,
        distances: Vec<f32>,
    }

    /// `SingleMachineSearcherBase::HealthStats`.
    #[derive(Debug, Clone, Copy, Default, PartialEq)]
    struct HealthStats {
        partition_weighted_avg_relative_imbalance: f64,
        partition_avg_relative_positive_imbalance: f64,
        avg_quantization_error: f64,
        sum_partition_sizes: u64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum DistanceMeasure {
        DotProduct,
        SquaredL2,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Quantization {
        Float32,
        Int8,
        Bfloat16,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum HashType {
        Lut16,
        Lut256,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum IncrementalMode {
        None,
        Online,
        OnlineIncremental,
    }

    // FFI mirrors of the ConfigBuilder option structs. cxx has no Option,
    // so "unset" is NaN for floating-point fields, a negative value for
    // integer fields, and a separate flag for booleans. lib.rs converts
    // from the public Option-based structs.
    struct FfiTreeOptions {
        num_leaves: i32,
        num_leaves_to_search: i32,
        training_sample_size: i64,
        min_partition_size: i32,
        training_iterations: i32,
        spherical: bool,
        quantize_centroids: bool,
        random_init: bool,
        incremental_threshold_points: i64,
        incremental_threshold_fraction: f64,
        avq: f64,
        soar_lambda: f64,
        overretrieve_factor: f64,
    }

    struct FfiUpperTreeOptions {
        num_leaves: i32,
        num_leaves_to_search: i32,
        avq: f64,
        soar_lambda: f64,
        overretrieve_factor: f64,
        scoring_mode: Quantization,
        anisotropic_quantization_threshold: f64,
    }

    struct FfiAhOptions {
        dimensions_per_block: i32,
        anisotropic_quantization_threshold: f64,
        training_sample_size: i64,
        hash_type: HashType,
        training_iterations: i32,
        has_residual_quantization: bool,
        residual_quantization: bool,
    }

    struct FfiReorderOptions {
        reordering_num_neighbors: i32,
        quantize: Quantization,
        anisotropic_quantization_threshold: f64,
    }

    struct FfiPcaOptions {
        reduction_dim: i32,
        pca_significance_threshold: f64,
        pca_truncation_threshold: f64,
    }

    unsafe extern "C++" {
        include!("shim.h");

        /// `research_scann::ScannInterface`.
        type ScannIndex;
        /// `scann_core::ConfigBuilder`.
        type ConfigBuilder;

        // --- construction -------------------------------------------------
        fn scann_new(
            dataset: &[f32],
            n_points: u64,
            config: &str,
            training_threads: i32,
        ) -> Result<UniquePtr<ScannIndex>>;
        fn scann_load(dir: &str) -> Result<UniquePtr<ScannIndex>>;

        // --- search (const; safe to call concurrently) ---------------------
        fn scann_search(
            idx: &ScannIndex,
            query: &[f32],
            final_nn: i32,
            pre_reorder_nn: i32,
            leaves: i32,
        ) -> Result<Neighbors>;
        #[allow(clippy::too_many_arguments)]
        fn scann_search_batched(
            idx: &ScannIndex,
            queries: &[f32],
            n_queries: u64,
            final_nn: i32,
            pre_reorder_nn: i32,
            leaves: i32,
            parallel: bool,
            batch_size: i32,
        ) -> Result<Vec<Neighbors>>;

        // --- accessors -----------------------------------------------------
        fn scann_size(idx: &ScannIndex) -> u64;
        fn scann_dimensionality(idx: &ScannIndex) -> u64;
        fn scann_config(idx: Pin<&mut ScannIndex>) -> String;
        fn scann_health_stats(idx: &ScannIndex) -> Result<HealthStats>;

        // --- persistence and mutation --------------------------------------
        fn scann_serialize(idx: Pin<&mut ScannIndex>, dir: &str, relative_path: bool)
            -> Result<()>;
        fn scann_set_num_threads(idx: Pin<&mut ScannIndex>, num_threads: i32);
        fn scann_reserve(idx: Pin<&mut ScannIndex>, n_points: u64) -> Result<()>;
        /// `ids[i] < 0` adds `vectors` row i as a new point; otherwise the
        /// row replaces point `ids[i]`. Returns the index of every row.
        fn scann_upsert(
            idx: Pin<&mut ScannIndex>,
            ids: &[i64],
            vectors: &[f32],
            batch_size: i32,
        ) -> Result<Vec<u32>>;
        fn scann_delete(idx: Pin<&mut ScannIndex>, ids: &[u32]) -> Result<()>;
        /// `config` empty: retrain with the current config.
        fn scann_rebalance(idx: Pin<&mut ScannIndex>, config: &str) -> Result<()>;
        fn scann_initialize_health_stats(idx: Pin<&mut ScannIndex>) -> Result<()>;

        // --- ConfigBuilder --------------------------------------------------
        fn config_builder_new(
            num_neighbors: i32,
            distance: DistanceMeasure,
            dimensionality: u32,
        ) -> UniquePtr<ConfigBuilder>;
        fn config_builder_tree(b: Pin<&mut ConfigBuilder>, o: &FfiTreeOptions);
        fn config_builder_upper_tree(b: Pin<&mut ConfigBuilder>, o: &FfiUpperTreeOptions);
        fn config_builder_score_ah(b: Pin<&mut ConfigBuilder>, o: &FfiAhOptions);
        fn config_builder_score_brute_force(b: Pin<&mut ConfigBuilder>, quantize: Quantization);
        fn config_builder_reorder(b: Pin<&mut ConfigBuilder>, o: &FfiReorderOptions);
        fn config_builder_pca(b: Pin<&mut ConfigBuilder>, o: &FfiPcaOptions);
        fn config_builder_truncate(b: Pin<&mut ConfigBuilder>, reduction_dim: i32);
        fn config_builder_autopilot(
            b: Pin<&mut ConfigBuilder>,
            mode: IncrementalMode,
            quantize: Quantization,
        );
        fn config_builder_build(b: &ConfigBuilder, num_points: u64) -> Result<String>;
    }
}
