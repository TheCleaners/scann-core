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

//! Rust bindings for scann-core, the TensorFlow- and Python-independent core
//! of [ScaNN](https://github.com/google-research/google-research/tree/master/scann).
//!
//! Two types do the work:
//!
//! * [`ConfigBuilder`] turns high-level options into a ScaNN config, like
//!   `scann.scann_ops_pybind.builder(...)` in Python (it is the same builder
//!   ported to C++; see `core/scann_core/config_builder.h`).
//! * [`ScannIndex`] is a built, searchable, mutable index.
//!
//! ```no_run
//! use scann_core::{AhOptions, ConfigBuilder, DistanceMeasure, ReorderOptions,
//!                  SearchOptions, TreeOptions};
//!
//! # fn main() -> Result<(), scann_core::ScannError> {
//! let dim = 128;
//! let dataset: Vec<f32> = vec![0.0; 10_000 * dim]; // row-major, n x dim
//! let index = ConfigBuilder::new(10, DistanceMeasure::DotProduct, dim)
//!     .tree(TreeOptions::new(100, 10))
//!     .score_ah(AhOptions::new(2).anisotropic_quantization_threshold(0.2))
//!     .reorder(ReorderOptions::new(100))
//!     .build_index(&dataset)?;
//! let neighbors = index.search(&dataset[..dim], SearchOptions::default())?;
//! println!("{:?}", neighbors.indices);
//! # Ok(())
//! # }
//! ```
//!
//! Distances are reported the way the Python API reports them: dot product
//! as the (positive) inner product, squared L2 as the squared distance.

mod bridge;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;
use std::sync::Mutex;

use bridge::ffi;

pub use ffi::{HealthStats, Neighbors};

/// Error from scann-core, or from argument validation in these bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScannError {
    /// Rejected by the bindings before reaching C++ (shape mismatch, value
    /// out of range, ...).
    InvalidArgument(String),
    /// A non-OK `absl::Status` (or C++ exception) from scann-core.
    Scann(String),
}

impl fmt::Display for ScannError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScannError::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            ScannError::Scann(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ScannError {}

impl From<cxx::Exception> for ScannError {
    fn from(e: cxx::Exception) -> Self {
        ScannError::Scann(e.what().to_string())
    }
}

pub type Result<T, E = ScannError> = std::result::Result<T, E>;

fn invalid<T>(msg: impl Into<String>) -> Result<T> {
    Err(ScannError::InvalidArgument(msg.into()))
}

/// `Some(n)` -> n as i32; `None` -> -1 (scann-core's "use the default").
fn opt_i32(v: Option<usize>, what: &str) -> Result<i32> {
    match v {
        None => Ok(-1),
        Some(n) => i32::try_from(n).or_else(|_| invalid(format!("{what} = {n} does not fit in i32"))),
    }
}

fn to_i32(n: usize, what: &str) -> Result<i32> {
    i32::try_from(n).or_else(|_| invalid(format!("{what} = {n} does not fit in i32")))
}

/// Number of rows in a row-major `rows x dim` buffer.
fn rows(data: &[f32], dim: usize, what: &str) -> Result<usize> {
    if dim == 0 {
        return invalid("dimensionality must be > 0");
    }
    if !data.len().is_multiple_of(dim) {
        return invalid(format!(
            "{what}: {} floats is not a multiple of the dimensionality {dim}",
            data.len()
        ));
    }
    Ok(data.len() / dim)
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// Per-query search overrides. `None` uses the value from the config the
/// index was built with (Python: passing `None`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[must_use = "search options do nothing until passed to a search"]
pub struct SearchOptions {
    /// Neighbours to return.
    pub final_num_neighbors: Option<usize>,
    /// Candidates kept for reordering (Python: `pre_reorder_num_neighbors`).
    pub pre_reorder_num_neighbors: Option<usize>,
    /// Partitions to search (Python: `leaves_to_search`).
    pub leaves_to_search: Option<usize>,
}

impl SearchOptions {
    /// Return `k` neighbours, everything else from the config.
    pub fn k(k: usize) -> Self {
        SearchOptions { final_num_neighbors: Some(k), ..Default::default() }
    }
    pub fn pre_reorder_num_neighbors(mut self, n: usize) -> Self {
        self.pre_reorder_num_neighbors = Some(n);
        self
    }
    pub fn leaves_to_search(mut self, n: usize) -> Self {
        self.leaves_to_search = Some(n);
        self
    }

    fn ffi(&self) -> Result<(i32, i32, i32)> {
        Ok((
            opt_i32(self.final_num_neighbors, "final_num_neighbors")?,
            opt_i32(self.pre_reorder_num_neighbors, "pre_reorder_num_neighbors")?,
            opt_i32(self.leaves_to_search, "leaves_to_search")?,
        ))
    }
}

// ---------------------------------------------------------------------------
// ScannIndex
// ---------------------------------------------------------------------------

/// A built ScaNN index (`research_scann::ScannInterface`).
///
/// Searching takes `&self` and may run concurrently from several threads;
/// everything that changes the index takes `&mut self`.
pub struct ScannIndex {
    inner: cxx::UniquePtr<ffi::ScannIndex>,
    /// Serializes `health_stats`: see the SAFETY note below.
    health_stats_lock: Mutex<()>,
}

// SAFETY: ScannInterface has no thread affinity. What &self reaches:
// search (ScaNN serves concurrent queries on one searcher by design, and
// the Python binding releases the GIL around them), size and
// dimensionality (plain reads), and health_stats. The last is a const
// member function in C++, but it recomputes cached imbalance figures in a
// `mutable` HealthStatsCollector, so two concurrent calls would race;
// health_stats_lock serializes them (searches don't touch the collector).
// All mutation goes through &mut self, so Rust's aliasing rules keep it
// exclusive.
unsafe impl Send for ScannIndex {}
unsafe impl Sync for ScannIndex {}

impl fmt::Debug for ScannIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScannIndex")
            .field("len", &self.len())
            .field("dimensionality", &self.dimensionality())
            .finish()
    }
}

impl ScannIndex {
    fn wrap(inner: cxx::UniquePtr<ffi::ScannIndex>) -> Self {
        ScannIndex { inner, health_stats_lock: Mutex::new(()) }
    }

    /// Builds an index over `dataset` (row-major, `n x dimensionality`)
    /// with a text-format ScaNN config, e.g. from [`ConfigBuilder::build`].
    pub fn new(dataset: &[f32], dimensionality: usize, config: &str) -> Result<Self> {
        Self::with_training_threads(dataset, dimensionality, config, 0)
    }

    /// Like [`new`](Self::new), training with `training_threads` threads
    /// (0: scann-core's default).
    pub fn with_training_threads(
        dataset: &[f32],
        dimensionality: usize,
        config: &str,
        training_threads: usize,
    ) -> Result<Self> {
        let n = rows(dataset, dimensionality, "dataset")?;
        if n == 0 {
            return invalid("dataset is empty");
        }
        if u32::try_from(n).is_err() {
            return invalid(format!("{n} datapoints exceed the 32-bit index space"));
        }
        let threads = to_i32(training_threads, "training_threads")?;
        Ok(ScannIndex::wrap(ffi::scann_new(dataset, n as u64, config, threads)?))
    }

    /// Loads an index written by [`serialize`](Self::serialize) -- or by
    /// Python's `searcher.serialize(dir)`; the on-disk format is the same.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = path_str(dir.as_ref())?;
        Ok(ScannIndex::wrap(ffi::scann_load(dir)?))
    }

    /// Writes the index to `dir` (which must exist), readable by
    /// [`load`](Self::load) and by Python's `scann_ops_pybind.load_searcher`.
    /// `relative_path` records asset paths relative to `dir`, so the
    /// directory can be moved.
    ///
    /// An index already in `dir` is replaced: the files are staged, then
    /// renamed into place with the manifest last, so an interrupted
    /// `serialize` leaves the old index, the new one, or a directory that
    /// [`load`](Self::load) rejects, never a mix of both. Index files the
    /// new index doesn't have (including a Python `scann_docids.pkl`) are
    /// removed.
    pub fn serialize(&mut self, dir: impl AsRef<Path>, relative_path: bool) -> Result<()> {
        let dir = path_str(dir.as_ref())?;
        Ok(ffi::scann_serialize(self.inner.pin_mut(), dir, relative_path)?)
    }

    /// Nearest neighbours of one query.
    pub fn search(&self, query: &[f32], options: SearchOptions) -> Result<Neighbors> {
        self.check_query(query.len())?;
        let (k, pre, leaves) = options.ffi()?;
        Ok(ffi::scann_search(&self.inner, query, k, pre, leaves)?)
    }

    /// Nearest neighbours of each row of `queries` (row-major,
    /// `n x dimensionality`), searched as one batch.
    pub fn search_batched(&self, queries: &[f32], options: SearchOptions) -> Result<Vec<Neighbors>> {
        self.batched(queries, options, false, 256)
    }

    /// Like [`search_batched`](Self::search_batched), splitting the batch
    /// into chunks of at most `batch_size` queries searched on the index's
    /// thread pool (see [`set_num_threads`](Self::set_num_threads)).
    pub fn search_batched_parallel(
        &self,
        queries: &[f32],
        options: SearchOptions,
        batch_size: usize,
    ) -> Result<Vec<Neighbors>> {
        if batch_size == 0 {
            return invalid("batch_size must be > 0");
        }
        self.batched(queries, options, true, batch_size)
    }

    fn batched(
        &self,
        queries: &[f32],
        options: SearchOptions,
        parallel: bool,
        batch_size: usize,
    ) -> Result<Vec<Neighbors>> {
        let n = rows(queries, self.dimensionality(), "queries")?;
        let (k, pre, leaves) = options.ffi()?;
        let bs = to_i32(batch_size, "batch_size")?;
        Ok(ffi::scann_search_batched(&self.inner, queries, n as u64, k, pre, leaves, parallel, bs)?)
    }

    fn check_query(&self, len: usize) -> Result<()> {
        let dim = self.dimensionality();
        if len != dim {
            return invalid(format!("query has {len} dimensions, the index has {dim}"));
        }
        Ok(())
    }

    /// Number of datapoints in the index.
    pub fn len(&self) -> usize {
        ffi::scann_size(&self.inner) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dimensionality(&self) -> usize {
        ffi::scann_dimensionality(&self.inner) as usize
    }

    /// The config the index currently runs with, as ScaNN text format.
    pub fn config(&mut self) -> String {
        ffi::scann_config(self.inner.pin_mut())
    }

    /// Resizes the query thread pool used by
    /// [`search_batched_parallel`](Self::search_batched_parallel) and by
    /// mutation. 0 disables the pool (everything runs on the calling thread).
    pub fn set_num_threads(&mut self, num_threads: usize) -> Result<()> {
        let n = to_i32(num_threads, "num_threads")?;
        ffi::scann_set_num_threads(self.inner.pin_mut(), n);
        Ok(())
    }

    /// Reserves space for `n_points` datapoints in total.
    pub fn reserve(&mut self, n_points: usize) -> Result<()> {
        Ok(ffi::scann_reserve(self.inner.pin_mut(), n_points as u64)?)
    }

    /// Adds the rows of `vectors`; returns their indices.
    pub fn add(&mut self, vectors: &[f32]) -> Result<Vec<u32>> {
        let n = rows(vectors, self.dimensionality(), "vectors")?;
        self.upsert(&vec![None; n], vectors, 1)
    }

    /// For each row of `vectors`: replaces datapoint `ids[i]`, or adds a new
    /// one if `ids[i]` is `None`. Rows are processed `batch_size` at a time
    /// (> 1 parallelizes on the thread pool). Returns each row's index.
    ///
    /// Python's `searcher.upsert` maps docids to these indices on top.
    ///
    /// An index listed more than once is rejected before anything changes,
    /// as in [`delete`](Self::delete) (and Python's `upsert` with a repeated
    /// docid).
    pub fn upsert(&mut self, ids: &[Option<u32>], vectors: &[f32], batch_size: usize) -> Result<Vec<u32>> {
        let n = rows(vectors, self.dimensionality(), "vectors")?;
        if n != ids.len() {
            return invalid(format!("{} ids for {n} vectors", ids.len()));
        }
        if batch_size == 0 {
            return invalid("batch_size must be > 0");
        }
        let mut seen = HashSet::new();
        if let Some(dup) = ids.iter().flatten().find(|&&id| !seen.insert(id)) {
            return invalid(format!("upsert: index {dup} listed more than once"));
        }
        let ids: Vec<i64> = ids.iter().map(|id| id.map_or(-1, i64::from)).collect();
        let bs = to_i32(batch_size, "batch_size")?;
        Ok(ffi::scann_upsert(self.inner.pin_mut(), &ids, vectors, bs)?)
    }

    /// Deletes the datapoints at `ids`, each an index as it is before the
    /// call. As in ScaNN, the last datapoint moves into each freed slot, so
    /// some remaining datapoints change index: the result lists each of
    /// them as `(old index, new index)`, sorted by old index.
    ///
    /// Duplicate or out-of-range ids are rejected before anything changes.
    pub fn delete(&mut self, ids: &[u32]) -> Result<Vec<(u32, u32)>> {
        let len = self.len();
        if let Some(&bad) = ids.iter().find(|&&id| id as usize >= len) {
            return invalid(format!("delete: index {bad} out of range ({len} points)"));
        }
        // Highest first: the datapoint that fills a freed slot then always
        // comes from above every remaining id, so each id still refers to
        // the datapoint it named before the call.
        let mut order = ids.to_vec();
        order.sort_unstable_by(|a, b| b.cmp(a));
        if let Some(w) = order.windows(2).find(|w| w[0] == w[1]) {
            return invalid(format!("delete: index {} listed more than once", w[0]));
        }
        ffi::scann_delete(self.inner.pin_mut(), &order)?;

        // Replay the moves: current position -> original index, for the
        // datapoints that moved.
        let mut origin: HashMap<u32, u32> = HashMap::new();
        let mut remaining = len as u32;
        for &id in &order {
            let last = remaining - 1;
            let moved = origin.remove(&last).unwrap_or(last);
            origin.remove(&id);
            if id != last {
                origin.insert(id, moved);
            }
            remaining -= 1;
        }
        let mut moves: Vec<(u32, u32)> = origin.into_iter().map(|(now, was)| (was, now)).collect();
        moves.sort_unstable();
        Ok(moves)
    }

    /// Retrains the partitioning and quantization on the current data, with
    /// `config` (text format) or, if `None`, the current config.
    pub fn rebalance(&mut self, config: Option<&str>) -> Result<()> {
        Ok(ffi::scann_rebalance(self.inner.pin_mut(), config.unwrap_or(""))?)
    }

    /// Partition balance and quantization error, as tracked by scann-core.
    ///
    /// `avg_quantization_error` is NaN when it can't be known: a tree without
    /// float reordering drops its float data once the index is created (built,
    /// loaded, rebalanced), so after an upsert or delete (or
    /// [`initialize_health_stats`](Self::initialize_health_stats)) it is NaN
    /// until the next [`rebalance`](Self::rebalance), and AH without
    /// reordering saves no float data (NaN after loading). NaN != NaN, so
    /// compare two `HealthStats` field by field. For a tree with a
    /// PCA/TRUNCATE projection it is measured in the projected space.
    pub fn health_stats(&self) -> Result<HealthStats> {
        let _guard = self.health_stats_lock.lock().unwrap_or_else(|e| e.into_inner());
        Ok(ffi::scann_health_stats(&self.inner)?)
    }

    /// Recomputes the statistics behind [`health_stats`](Self::health_stats)
    /// from scratch.
    pub fn initialize_health_stats(&mut self) -> Result<()> {
        Ok(ffi::scann_initialize_health_stats(self.inner.pin_mut())?)
    }
}

fn path_str(p: &Path) -> Result<&str> {
    p.to_str()
        .ok_or_else(|| ScannError::InvalidArgument(format!("path {p:?} is not valid UTF-8")))
}

// ---------------------------------------------------------------------------
// ConfigBuilder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMeasure {
    DotProduct,
    SquaredL2,
}

/// How vectors are stored for scoring (Python: `scann.ReorderType`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Quantization {
    #[default]
    Float32,
    Int8,
    Bfloat16,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HashType {
    #[default]
    Lut16,
    Lut256,
}

/// Python: `scann_builder.IncrementalMode`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum IncrementalMode {
    #[default]
    None,
    Online,
    OnlineIncremental,
}

/// When a partition has grown enough to be retrained in place (Python's
/// `incremental_threshold`: an int is a point count, a float a fraction).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IncrementalThreshold {
    Points(u64),
    Fraction(f64),
}

fn q(q: Quantization) -> ffi::Quantization {
    match q {
        Quantization::Float32 => ffi::Quantization::Float32,
        Quantization::Int8 => ffi::Quantization::Int8,
        Quantization::Bfloat16 => ffi::Quantization::Bfloat16,
    }
}

fn nan_if_none(v: Option<f64>) -> f64 {
    v.unwrap_or(f64::NAN)
}

/// Python: `ScannBuilder.tree()`. Defaults match Python's.
#[derive(Debug, Clone, PartialEq)]
#[must_use = "options do nothing until passed to ConfigBuilder"]
pub struct TreeOptions {
    pub num_leaves: u32,
    pub num_leaves_to_search: u32,
    pub training_sample_size: u64,
    pub min_partition_size: u32,
    pub training_iterations: u32,
    pub spherical: bool,
    pub quantize_centroids: bool,
    /// Random center initialization (Python's default). Upstream's random
    /// initialization depends on hash-table iteration order, so training
    /// isn't reproducible run to run; `false` (k-means++) is.
    pub random_init: bool,
    pub incremental_threshold: Option<IncrementalThreshold>,
    /// Anisotropic vector quantization of the centers (dot product only).
    pub avq: Option<f64>,
    /// SOAR spilling (dot product only).
    pub soar_lambda: Option<f64>,
    pub overretrieve_factor: Option<f64>,
}

impl TreeOptions {
    pub fn new(num_leaves: u32, num_leaves_to_search: u32) -> Self {
        TreeOptions {
            num_leaves,
            num_leaves_to_search,
            training_sample_size: 100_000,
            min_partition_size: 50,
            training_iterations: 12,
            spherical: false,
            quantize_centroids: false,
            random_init: true,
            incremental_threshold: None,
            avq: None,
            soar_lambda: None,
            overretrieve_factor: None,
        }
    }
    pub fn training_sample_size(mut self, n: u64) -> Self {
        self.training_sample_size = n;
        self
    }
    pub fn min_partition_size(mut self, n: u32) -> Self {
        self.min_partition_size = n;
        self
    }
    pub fn training_iterations(mut self, n: u32) -> Self {
        self.training_iterations = n;
        self
    }
    /// Spherical k-means. The index then stores unit vectors: rows are
    /// L2-normalized at build time and on upsert.
    pub fn spherical(mut self, v: bool) -> Self {
        self.spherical = v;
        self
    }
    pub fn quantize_centroids(mut self, v: bool) -> Self {
        self.quantize_centroids = v;
        self
    }
    pub fn random_init(mut self, v: bool) -> Self {
        self.random_init = v;
        self
    }
    pub fn incremental_threshold(mut self, t: IncrementalThreshold) -> Self {
        self.incremental_threshold = Some(t);
        self
    }
    pub fn avq(mut self, eta: f64) -> Self {
        self.avq = Some(eta);
        self
    }
    pub fn soar_lambda(mut self, lambda: f64) -> Self {
        self.soar_lambda = Some(lambda);
        self
    }
    pub fn overretrieve_factor(mut self, f: f64) -> Self {
        self.overretrieve_factor = Some(f);
        self
    }

    fn ffi(&self) -> Result<ffi::FfiTreeOptions> {
        let (points, fraction) = match self.incremental_threshold {
            None => (-1, f64::NAN),
            Some(IncrementalThreshold::Points(p)) => {
                (i64::try_from(p).or_else(|_| invalid("incremental_threshold too large"))?, f64::NAN)
            }
            Some(IncrementalThreshold::Fraction(f)) => (-1, f),
        };
        Ok(ffi::FfiTreeOptions {
            num_leaves: to_i32(self.num_leaves as usize, "num_leaves")?,
            num_leaves_to_search: to_i32(self.num_leaves_to_search as usize, "num_leaves_to_search")?,
            training_sample_size: i64::try_from(self.training_sample_size)
                .or_else(|_| invalid("training_sample_size too large"))?,
            min_partition_size: to_i32(self.min_partition_size as usize, "min_partition_size")?,
            training_iterations: to_i32(self.training_iterations as usize, "training_iterations")?,
            spherical: self.spherical,
            quantize_centroids: self.quantize_centroids,
            random_init: self.random_init,
            incremental_threshold_points: points,
            incremental_threshold_fraction: fraction,
            avq: nan_if_none(self.avq),
            soar_lambda: nan_if_none(self.soar_lambda),
            overretrieve_factor: nan_if_none(self.overretrieve_factor),
        })
    }
}

/// Python: `ScannBuilder.upper_tree()`: a second partitioning level above
/// the tree. Requires [`ConfigBuilder::tree`].
#[derive(Debug, Clone, PartialEq)]
#[must_use = "options do nothing until passed to ConfigBuilder"]
pub struct UpperTreeOptions {
    pub num_leaves: u32,
    pub num_leaves_to_search: u32,
    pub avq: Option<f64>,
    pub soar_lambda: Option<f64>,
    pub overretrieve_factor: Option<f64>,
    pub scoring_mode: Quantization,
    pub anisotropic_quantization_threshold: Option<f64>,
}

impl UpperTreeOptions {
    pub fn new(num_leaves: u32, num_leaves_to_search: u32) -> Self {
        UpperTreeOptions {
            num_leaves,
            num_leaves_to_search,
            avq: None,
            soar_lambda: None,
            overretrieve_factor: None,
            scoring_mode: Quantization::Int8,
            anisotropic_quantization_threshold: None,
        }
    }
    pub fn avq(mut self, eta: f64) -> Self {
        self.avq = Some(eta);
        self
    }
    pub fn soar_lambda(mut self, lambda: f64) -> Self {
        self.soar_lambda = Some(lambda);
        self
    }
    pub fn overretrieve_factor(mut self, f: f64) -> Self {
        self.overretrieve_factor = Some(f);
        self
    }
    pub fn scoring_mode(mut self, m: Quantization) -> Self {
        self.scoring_mode = m;
        self
    }
    pub fn anisotropic_quantization_threshold(mut self, t: f64) -> Self {
        self.anisotropic_quantization_threshold = Some(t);
        self
    }

    fn ffi(&self) -> Result<ffi::FfiUpperTreeOptions> {
        Ok(ffi::FfiUpperTreeOptions {
            num_leaves: to_i32(self.num_leaves as usize, "num_leaves")?,
            num_leaves_to_search: to_i32(self.num_leaves_to_search as usize, "num_leaves_to_search")?,
            avq: nan_if_none(self.avq),
            soar_lambda: nan_if_none(self.soar_lambda),
            overretrieve_factor: nan_if_none(self.overretrieve_factor),
            scoring_mode: q(self.scoring_mode),
            anisotropic_quantization_threshold: nan_if_none(self.anisotropic_quantization_threshold),
        })
    }
}

/// Python: `ScannBuilder.score_ah()`: asymmetric-hashing (product
/// quantization) scoring.
#[derive(Debug, Clone, PartialEq)]
#[must_use = "options do nothing until passed to ConfigBuilder"]
pub struct AhOptions {
    pub dimensions_per_block: u32,
    /// Anisotropic quantization (dot product); `None` = plain k-means PQ.
    pub anisotropic_quantization_threshold: Option<f64>,
    pub training_sample_size: u64,
    pub hash_type: HashType,
    pub training_iterations: u32,
    /// `None`: on iff a tree is configured and the distance is dot product.
    pub residual_quantization: Option<bool>,
}

impl AhOptions {
    pub fn new(dimensions_per_block: u32) -> Self {
        AhOptions {
            dimensions_per_block,
            anisotropic_quantization_threshold: None,
            training_sample_size: 100_000,
            hash_type: HashType::Lut16,
            training_iterations: 10,
            residual_quantization: None,
        }
    }
    pub fn anisotropic_quantization_threshold(mut self, t: f64) -> Self {
        self.anisotropic_quantization_threshold = Some(t);
        self
    }
    pub fn training_sample_size(mut self, n: u64) -> Self {
        self.training_sample_size = n;
        self
    }
    pub fn hash_type(mut self, h: HashType) -> Self {
        self.hash_type = h;
        self
    }
    pub fn training_iterations(mut self, n: u32) -> Self {
        self.training_iterations = n;
        self
    }
    pub fn residual_quantization(mut self, v: bool) -> Self {
        self.residual_quantization = Some(v);
        self
    }

    fn ffi(&self) -> Result<ffi::FfiAhOptions> {
        Ok(ffi::FfiAhOptions {
            dimensions_per_block: to_i32(self.dimensions_per_block as usize, "dimensions_per_block")?,
            anisotropic_quantization_threshold: nan_if_none(self.anisotropic_quantization_threshold),
            training_sample_size: i64::try_from(self.training_sample_size)
                .or_else(|_| invalid("training_sample_size too large"))?,
            hash_type: match self.hash_type {
                HashType::Lut16 => ffi::HashType::Lut16,
                HashType::Lut256 => ffi::HashType::Lut256,
            },
            training_iterations: to_i32(self.training_iterations as usize, "training_iterations")?,
            has_residual_quantization: self.residual_quantization.is_some(),
            residual_quantization: self.residual_quantization.unwrap_or(false),
        })
    }
}

/// Python: `ScannBuilder.reorder()`: rescore the top candidates exactly (or
/// with a finer quantization).
#[derive(Debug, Clone, PartialEq)]
#[must_use = "options do nothing until passed to ConfigBuilder"]
pub struct ReorderOptions {
    pub reordering_num_neighbors: u32,
    pub quantize: Quantization,
    pub anisotropic_quantization_threshold: Option<f64>,
}

impl ReorderOptions {
    pub fn new(reordering_num_neighbors: u32) -> Self {
        ReorderOptions {
            reordering_num_neighbors,
            quantize: Quantization::Float32,
            anisotropic_quantization_threshold: None,
        }
    }
    pub fn quantize(mut self, q: Quantization) -> Self {
        self.quantize = q;
        self
    }
    pub fn anisotropic_quantization_threshold(mut self, t: f64) -> Self {
        self.anisotropic_quantization_threshold = Some(t);
        self
    }
}

/// Python: `ScannBuilder.pca()`. Set at most one of `reduction_dim` and
/// `pca_significance_threshold`; with neither, the threshold is 0.8.
#[derive(Debug, Clone, PartialEq)]
#[must_use = "options do nothing until passed to ConfigBuilder"]
pub struct PcaOptions {
    pub reduction_dim: Option<u32>,
    pub pca_significance_threshold: Option<f64>,
    pub pca_truncation_threshold: f64,
}

impl Default for PcaOptions {
    fn default() -> Self {
        PcaOptions { reduction_dim: None, pca_significance_threshold: None, pca_truncation_threshold: 0.6 }
    }
}

impl PcaOptions {
    pub fn reduction_dim(dim: u32) -> Self {
        PcaOptions { reduction_dim: Some(dim), ..Default::default() }
    }
    pub fn significance_threshold(t: f64) -> Self {
        PcaOptions { pca_significance_threshold: Some(t), ..Default::default() }
    }
}

/// Builds a ScaNN config from high-level options, like Python's
/// `scann_ops_pybind.builder(db, num_neighbors, distance_measure)`.
///
/// Where Python silently ignores an option (an upper tree or PCA without a
/// tree, autopilot combined with manual options, an option set twice, ...)
/// [`build`](Self::build) returns an error instead.
#[must_use = "a ConfigBuilder does nothing until build() or build_index() is called"]
pub struct ConfigBuilder {
    inner: cxx::UniquePtr<ffi::ConfigBuilder>,
    dimensionality: usize,
    training_threads: usize,
    error: Option<ScannError>,
}

impl fmt::Debug for ConfigBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConfigBuilder").field("dimensionality", &self.dimensionality).finish()
    }
}

impl ConfigBuilder {
    pub fn new(num_neighbors: usize, distance: DistanceMeasure, dimensionality: usize) -> Self {
        let mut error = None;
        let k = to_i32(num_neighbors, "num_neighbors").unwrap_or_else(|e| {
            error = Some(e);
            0
        });
        let dim = u32::try_from(dimensionality).unwrap_or_else(|_| {
            error = Some(ScannError::InvalidArgument("dimensionality too large".into()));
            0
        });
        let d = match distance {
            DistanceMeasure::DotProduct => ffi::DistanceMeasure::DotProduct,
            DistanceMeasure::SquaredL2 => ffi::DistanceMeasure::SquaredL2,
        };
        ConfigBuilder {
            inner: ffi::config_builder_new(k, d, dim),
            dimensionality,
            training_threads: 0,
            error,
        }
    }

    // Conversion errors are deferred to build(), like the C++ builder's own
    // validation, so the chain stays infallible.
    fn apply<T>(mut self, v: Result<T>, f: impl FnOnce(std::pin::Pin<&mut ffi::ConfigBuilder>, &T)) -> Self {
        match v {
            Ok(v) => f(self.inner.pin_mut(), &v),
            Err(e) => {
                self.error.get_or_insert(e);
            }
        }
        self
    }

    /// Partition the dataset with a k-means tree.
    pub fn tree(self, o: TreeOptions) -> Self {
        let v = o.ffi();
        self.apply(v, ffi::config_builder_tree)
    }

    /// Add a second partitioning level above the tree.
    pub fn upper_tree(self, o: UpperTreeOptions) -> Self {
        let v = o.ffi();
        self.apply(v, ffi::config_builder_upper_tree)
    }

    /// Score with asymmetric hashing.
    pub fn score_ah(self, o: AhOptions) -> Self {
        let v = o.ffi();
        self.apply(v, ffi::config_builder_score_ah)
    }

    /// Score exactly (optionally on quantized vectors).
    pub fn score_brute_force(self, quantize: Quantization) -> Self {
        self.apply(Ok(q(quantize)), |b, &qq| ffi::config_builder_score_brute_force(b, qq))
    }

    pub fn reorder(self, o: ReorderOptions) -> Self {
        let v = to_i32(o.reordering_num_neighbors as usize, "reordering_num_neighbors").map(|k| {
            ffi::FfiReorderOptions {
                reordering_num_neighbors: k,
                quantize: q(o.quantize),
                anisotropic_quantization_threshold: nan_if_none(o.anisotropic_quantization_threshold),
            }
        });
        self.apply(v, ffi::config_builder_reorder)
    }

    /// Project with PCA before partitioning (requires a tree).
    pub fn pca(self, o: PcaOptions) -> Self {
        let v = match o.reduction_dim {
            None => Ok(-1),
            Some(d) => to_i32(d as usize, "reduction_dim"),
        }
        .map(|d| ffi::FfiPcaOptions {
            reduction_dim: d,
            pca_significance_threshold: nan_if_none(o.pca_significance_threshold),
            pca_truncation_threshold: o.pca_truncation_threshold,
        });
        self.apply(v, ffi::config_builder_pca)
    }

    /// Keep only the first `reduction_dim` dimensions (requires a tree).
    pub fn truncate(self, reduction_dim: usize) -> Self {
        let v = to_i32(reduction_dim, "reduction_dim");
        self.apply(v, |b, &d| ffi::config_builder_truncate(b, d))
    }

    /// Let scann-core pick the configuration for the dataset size. Can't be
    /// combined with manual options.
    pub fn autopilot(self, mode: IncrementalMode, quantize: Quantization) -> Self {
        let m = match mode {
            IncrementalMode::None => ffi::IncrementalMode::None,
            IncrementalMode::Online => ffi::IncrementalMode::Online,
            IncrementalMode::OnlineIncremental => ffi::IncrementalMode::OnlineIncremental,
        };
        self.apply(Ok((m, q(quantize))), |b, &(m, qq)| ffi::config_builder_autopilot(b, m, qq))
    }

    /// Training threads for [`build_index`](Self::build_index) (0: default).
    pub fn training_threads(mut self, n: usize) -> Self {
        self.training_threads = n;
        self
    }

    /// The config as ScaNN text format. `num_points` only matters with
    /// [`autopilot`](Self::autopilot), which sizes the config to the data.
    pub fn build(&self, num_points: u64) -> Result<String> {
        if let Some(e) = &self.error {
            return Err(e.clone());
        }
        Ok(ffi::config_builder_build(&self.inner, num_points)?)
    }

    /// Builds the config for `dataset` and an index over it.
    pub fn build_index(&self, dataset: &[f32]) -> Result<ScannIndex> {
        let n = rows(dataset, self.dimensionality, "dataset")?;
        let config = self.build(n as u64)?;
        ScannIndex::with_training_threads(dataset, self.dimensionality, &config, self.training_threads)
    }
}
