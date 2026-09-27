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

//! Keeping a scann-core index up to date from Rust: upsert (insert and
//! update), delete, rebalance, and save over an existing index.
//!
//! `ScannIndex` identifies points by row index. Deleting moves the last
//! row into each freed slot, and `delete` returns those moves, so an
//! application-side id table (here, `docids`) stays in step.
//!
//! Run through CMake (`cmake --build <build> --target scann_core_rust`
//! builds it), or directly:
//!   SCANN_CORE_BUILD_ENV=<build>/rust/scann_core_rust_build.env \
//!     cargo run --release --example updating

use scann_core::{
    AhOptions, ConfigBuilder, DistanceMeasure, ReorderOptions, ScannError, ScannIndex,
    SearchOptions, TreeOptions,
};

const DIM: usize = 32;
const LEAVES: u32 = 60;

/// A small deterministic generator (xorshift), uniform in [-1, 1).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
    }
}

/// `n` unit-norm points around 40 fixed centers, row-major n x DIM.
fn make_data(n: usize, seed: u64) -> Vec<f32> {
    let mut centers_rng = Rng::new(42);
    let centers: Vec<f32> = (0..40 * DIM).map(|_| centers_rng.uniform()).collect();
    let mut rng = Rng::new(seed);
    let mut data = Vec::with_capacity(n * DIM);
    for _ in 0..n {
        let c = (rng.0 % 40) as usize;
        let row: Vec<f32> =
            centers[c * DIM..][..DIM].iter().map(|x| x + 0.2 * rng.uniform()).collect();
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        data.extend(row.iter().map(|x| x / norm));
    }
    data
}

/// The docid of each query's nearest stored point, searching every leaf and
/// reordering 1000 candidates: effectively exact, so a stored vector always
/// finds itself.
fn nearest(index: &ScannIndex, docids: &[String], queries: &[f32]) -> Vec<String> {
    let options =
        SearchOptions::k(1).leaves_to_search(LEAVES as usize).pre_reorder_num_neighbors(1000);
    let found = index.search_batched(queries, options).expect("search");
    found.iter().map(|nn| docids[nn.indices[0] as usize].clone()).collect()
}

fn print_health(when: &str, index: &ScannIndex) -> Result<(), ScannError> {
    let s = index.health_stats()?;
    println!(
        "{when:24} {:5} points, imbalance {:.3}, quantization error {:.4}",
        index.len(),
        s.partition_avg_relative_positive_imbalance,
        s.avg_quantization_error
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let initial = make_data(3000, 1);
    // The application's ids, one per row: row i of the index is docids[i].
    let mut docids: Vec<String> = (0..3000).map(|i| format!("doc-{i}")).collect();

    let builder = ConfigBuilder::new(10, DistanceMeasure::DotProduct, DIM)
        .tree(TreeOptions::new(LEAVES, 10).random_init(false))
        .score_ah(AhOptions::new(2).anisotropic_quantization_threshold(0.2))
        .reorder(ReorderOptions::new(100));
    let mut index = builder.build_index(&initial)?;
    print_health("built:", &index)?;

    // --- Insert ----------------------------------------------------------
    // upsert: `None` adds a row, `Some(i)` replaces row i. The result is
    // each row's index; new rows are appended. batch_size > 1 assigns the
    // batch to leaves on the index's thread pool. An upsert (or delete) can
    // trigger incremental maintenance and, when that asks for it, a full
    // retrain, as in Python.
    let added = make_data(1000, 2);
    let added_docids: Vec<String> = (0..1000).map(|i| format!("new-{i}")).collect();
    let new_rows = index.upsert(&[None; 1000], &added, 256)?;
    assert_eq!(new_rows, (3000..4000).collect::<Vec<u32>>());
    docids.extend(added_docids.iter().cloned());
    assert_eq!(nearest(&index, &docids, &added), added_docids);
    print_health("after inserting 1000:", &index)?;

    // --- Update ----------------------------------------------------------
    // Replace row 0's vector; it keeps its index (and so its docid).
    let moved = make_data(1, 3);
    index.upsert(&[Some(0)], &moved, 1)?;
    assert_eq!(nearest(&index, &docids, &moved), ["doc-0"]);
    assert_ne!(nearest(&index, &docids, &initial[..DIM]), ["doc-0"]);
    println!("updated doc-0: found at its new vector, not at its old one");

    // An index listed twice in one upsert is rejected before anything
    // changes, like a repeated docid in Python.
    let err = index.upsert(&[Some(1), Some(1)], &make_data(2, 4), 1).unwrap_err();
    assert!(matches!(err, ScannError::InvalidArgument(_)));
    println!("repeated index rejected: {err}");
    assert_eq!(index.len(), 4000);

    // --- Delete ----------------------------------------------------------
    // delete takes indices as they are before the call and returns the
    // (old index, new index) of every row that moved into a freed slot.
    // Apply the moves to the id table, then drop its tail.
    let doomed: Vec<u32> = (1..=500).collect();
    let moves = index.delete(&doomed)?;
    for &(from, to) in &moves {
        docids[to as usize] = docids[from as usize].clone();
    }
    docids.truncate(index.len());
    assert_eq!(index.len(), 3500);
    println!("deleted {} points; {} others moved to new indices", doomed.len(), moves.len());
    // The added points still find themselves, under their own docids.
    assert_eq!(nearest(&index, &docids, &added), added_docids);
    print_health("after deleting 500:", &index)?;

    // --- Rebalance -------------------------------------------------------
    // Retrains the tree and AH codebooks on the current data (a full
    // retrain, as slow as a build) and recomputes the health stats. Indices
    // don't change.
    index.rebalance(None)?;
    print_health("after rebalance:", &index)?;
    assert_eq!(nearest(&index, &docids, &moved), ["doc-0"]);

    // --- Save over an existing index, reload -----------------------------
    // serialize replaces an index already in the directory: files are
    // staged, then renamed into place with the manifest last, so an
    // interrupted save leaves the old index, the new one, or a directory
    // that `load` rejects; never a mix. `true`: relative paths, movable.
    let dir = std::env::temp_dir().join(format!("scann-core-rust-updating-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    index.serialize(&dir, true)?;
    index.add(&make_data(1, 5))?;
    docids.push("late-arrival".to_string());
    index.serialize(&dir, true)?;

    let reloaded = ScannIndex::load(&dir)?;
    assert_eq!(reloaded.len(), docids.len());
    let queries = make_data(50, 6);
    let a = index.search_batched(&queries, SearchOptions::default())?;
    let b = reloaded.search_batched(&queries, SearchOptions::default())?;
    assert_eq!(a, b);
    println!("re-saved and reloaded {} points: identical results", reloaded.len());
    // The ids are the application's to store (Python's scann_docids.pkl is
    // written only by the Python wrapper).
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
