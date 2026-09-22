//! cxx bridge declarations. The C++ side (`shim.h`/`shim.cc`) adapts
//! `research_scann::ScannInterface` (scann-core's pybind-free facade class,
//! see `src/scann/scann_ops/cc/scann.h`) to cxx-bridgeable types: `Span`s
//! instead of `absl::Span`, `Result<T>` (thrown `std::runtime_error` on a
//! non-OK `absl::Status`) instead of `absl::StatusOr`, plain structs instead
//! of `NNResultsVector`/`DatapointPtr`.
//!
//! This is a first cut covering construction and single-query search --
//! enough to prove the pipeline end-to-end. Batched search, serialize/
//! load, and upsert/delete follow the same pattern and are natural
//! follow-ups, not fundamental blockers.

#[cxx::bridge(namespace = "scann_core_ffi")]
mod ffi {
    /// Mirrors `ScannInterface::Search`'s output after
    /// `ReshapeNNResult` flattens it into parallel arrays.
    struct SearchResult {
        indices: Vec<u32>,
        distances: Vec<f32>,
    }

    unsafe extern "C++" {
        include!("shim.h");

        /// Opaque handle around `research_scann::ScannInterface`.
        type ScannIndex;

        /// Builds an index from a flat, row-major `n_points * dim` dataset
        /// and a text-proto ScaNN config (the same config string produced
        /// by `ScannBuilder.create_config()` on the Python side --
        /// see docs/api_reference.md).
        fn scann_new(
            dataset: &[f32],
            n_points: u64,
            config: &str,
            training_threads: i32,
        ) -> Result<UniquePtr<ScannIndex>>;

        /// Searches a single query vector.  `-1` for `pre_reorder_nn` /
        /// `leaves` means "use the value baked in at build time" (same
        /// sentinel convention as the Python `search()` method).
        fn scann_search(
            idx: &ScannIndex,
            query: &[f32],
            final_nn: i32,
            pre_reorder_nn: i32,
            leaves: i32,
        ) -> Result<SearchResult>;

        /// Number of indexed datapoints.
        fn scann_size(idx: &ScannIndex) -> u64;
    }
}

pub use ffi::{scann_new, scann_search, scann_size, ScannIndex, SearchResult};
