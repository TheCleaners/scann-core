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

//! Part 7: the tutorial's GloVe pipeline in Rust.
//!
//!   cargo run --release --example tutorial_glove -- <data dir>
//!
//! <data dir> is what part7_export.py wrote: glove_{train,test,neighbors}.npy
//! and glove-index/ (an index saved from Python). Builds the part 3 index,
//! evaluates it, loads and evaluates the Python-built index, then measures
//! concurrent search() calls from many threads.

use scann_core::{
    AhOptions, ConfigBuilder, DistanceMeasure, Neighbors, ReorderOptions, ScannIndex,
    SearchOptions, TreeOptions,
};
use std::path::Path;
use std::time::Instant;

/// A 2-D little-endian .npy array of 4-byte elements (f32 or u32), as
/// numpy.save writes them: magic, version, header dict, raw data.
struct Npy {
    rows: usize,
    cols: usize,
    bytes: Vec<u8>,
}

impl Npy {
    fn load(path: &Path) -> Npy {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(&bytes[..6], b"\x93NUMPY", "{}: not a .npy file", path.display());
        let (header_len, start) = match bytes[6] {
            1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
            _ => (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12),
        };
        let header = std::str::from_utf8(&bytes[start..start + header_len]).unwrap();
        assert!(header.contains("'<f4'") || header.contains("'<u4'"), "{header}");
        assert!(header.contains("'fortran_order': False"), "{header}");
        let shape = &header[header.find("'shape': (").unwrap() + 10..];
        let dims: Vec<usize> = shape[..shape.find(')').unwrap()]
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        Npy { rows: dims[0], cols: dims[1], bytes: bytes[start + header_len..].to_vec() }
    }

    fn f32s(&self) -> Vec<f32> {
        self.bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    }

    fn u32s(&self) -> Vec<u32> {
        self.bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect()
    }
}

/// recall@10 of `found` against the first 10 of each row of the ground truth.
fn recall(found: &[Neighbors], truth: &[u32], truth_cols: usize) -> f64 {
    let hits: usize = found
        .iter()
        .enumerate()
        .map(|(q, nn)| {
            let t = &truth[q * truth_cols..][..10];
            nn.indices.iter().filter(|i| t.contains(i)).count()
        })
        .sum();
    hits as f64 / (found.len() * 10) as f64
}

fn evaluate(name: &str, index: &mut ScannIndex, queries: &[f32], truth: &[u32], cols: usize) {
    index.set_num_threads(std::thread::available_parallelism().map_or(1, |n| n.get())).unwrap();
    let n_queries = queries.len() / index.dimensionality();
    let mut best = f64::INFINITY;
    let mut found = Vec::new();
    for _ in 0..3 {
        // Best of 3, like tutorial_data.evaluate().
        let start = Instant::now();
        found = index.search_batched_parallel(queries, SearchOptions::default(), 256).unwrap();
        best = best.min(start.elapsed().as_secs_f64());
    }
    println!(
        "{name:26} recall@10 {:.4}  {:8.0} QPS",
        recall(&found, truth, cols),
        n_queries as f64 / best
    );
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: tutorial_glove <data dir>");
    let dir = Path::new(&dir);
    let train = Npy::load(&dir.join("glove_train.npy"));
    let test = Npy::load(&dir.join("glove_test.npy"));
    let truth = Npy::load(&dir.join("glove_neighbors.npy"));
    println!("dataset {} x {}, {} queries", train.rows, train.cols, test.rows);
    let (dataset, queries, truth_ids) = (train.f32s(), test.f32s(), truth.u32s());

    // The part 3 configuration.
    let start = Instant::now();
    let mut built = ConfigBuilder::new(10, DistanceMeasure::DotProduct, train.cols)
        .tree(TreeOptions::new(2000, 100).training_sample_size(250_000))
        .score_ah(AhOptions::new(2).anisotropic_quantization_threshold(0.2))
        .reorder(ReorderOptions::new(100))
        .build_index(&dataset)
        .unwrap();
    println!("built in {:.1} s", start.elapsed().as_secs_f64());
    evaluate("built in Rust", &mut built, &queries, &truth_ids, truth.cols);

    // The index part7_export.py saved from Python.
    let start = Instant::now();
    let mut loaded = ScannIndex::load(dir.join("glove-index")).unwrap();
    println!("loaded in {:.1} s", start.elapsed().as_secs_f64());
    evaluate("built in Python, loaded", &mut loaded, &queries, &truth_ids, truth.cols);

    // Concurrent single-query search() from many threads, sharing one index:
    // ScannIndex is Send + Sync, and there is no interpreter lock.
    let dim = loaded.dimensionality();
    let index = &loaded;
    for threads in [1, 8, 32, 64] {
        let start = Instant::now();
        std::thread::scope(|s| {
            for t in 0..threads {
                let queries = &queries;
                s.spawn(move || {
                    // Thread t takes every threads-th query.
                    for q in queries.chunks(dim).skip(t).step_by(threads) {
                        index.search(q, SearchOptions::default()).unwrap();
                    }
                });
            }
        });
        let qps = (queries.len() / dim) as f64 / start.elapsed().as_secs_f64();
        println!("concurrent search() from {threads:2} threads: {qps:8.0} QPS");
    }
}
