//! Safe Rust wrapper around scann-core's C++ facade (`ScannInterface`),
//! via the cxx bridge in `bridge.rs`. See docs/api_reference.md in the
//! parent scann/ tree for what these parameters mean; this wraps the same
//! underlying config/search semantics, just without the Python layer.

mod bridge;

use std::fmt;

pub use bridge::SearchResult;

/// An error returned by scann-core (a non-OK `absl::Status`, surfaced as a
/// thrown C++ exception across the cxx boundary and converted here).
#[derive(Debug)]
pub struct ScannError(String);

impl fmt::Display for ScannError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ScannError {}

impl From<cxx::Exception> for ScannError {
    fn from(e: cxx::Exception) -> Self {
        ScannError(e.what().to_string())
    }
}

/// A built ScaNN index. Owns the underlying C++ `ScannInterface`.
pub struct ScannIndex(cxx::UniquePtr<bridge::ScannIndex>);

impl ScannIndex {
    /// Builds an index from a flat, row-major `n_points * dim` dataset and
    /// a text-proto ScaNN config -- the same config string
    /// `ScannBuilder.create_config()` produces on the Python side (see
    /// docs/api_reference.md). `training_threads` mirrors
    /// `.set_n_training_threads()`; 0 means "implementation default".
    pub fn new(
        dataset: &[f32],
        n_points: u64,
        config: &str,
        training_threads: i32,
    ) -> Result<Self, ScannError> {
        let idx = bridge::scann_new(dataset, n_points, config, training_threads)?;
        Ok(ScannIndex(idx))
    }

    /// Searches a single query vector. Pass `-1` for `pre_reorder_nn` /
    /// `leaves` to use the value baked in at build time (matching the
    /// Python `search()` sentinel convention).
    pub fn search(
        &self,
        query: &[f32],
        final_nn: i32,
        pre_reorder_nn: i32,
        leaves: i32,
    ) -> Result<SearchResult, ScannError> {
        Ok(bridge::scann_search(
            &self.0,
            query,
            final_nn,
            pre_reorder_nn,
            leaves,
        )?)
    }

    /// Number of indexed datapoints.
    pub fn len(&self) -> u64 {
        bridge::scann_size(&self.0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
