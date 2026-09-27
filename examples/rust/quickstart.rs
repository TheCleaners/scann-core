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

//! scann-core from Rust: build a tree + asymmetric-hashing index with the
//! config builder, search it (one query, a batch, from several threads),
//! add and delete points, and save / reload it.
//!
//! Run through CMake (`cmake --build <build> --target scann_core_rust`
//! builds it), or directly:
//!   SCANN_CORE_BUILD_ENV=<build>/rust/scann_core_rust_build.env \
//!     cargo run --release --example quickstart

use scann_core::{
    AhOptions, ConfigBuilder, DistanceMeasure, ReorderOptions, ScannIndex, SearchOptions,
    TreeOptions,
};

const N: usize = 20_000;
const DIM: usize = 64;

/// Unit-norm points around 50 pseudo-random centers, row-major N x DIM.
fn make_dataset(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut uniform = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
    };
    let centers: Vec<f32> = (0..50 * DIM).map(|_| uniform()).collect();
    let mut data = Vec::with_capacity(n * DIM);
    for i in 0..n {
        let row: Vec<f32> =
            centers[(i % 50) * DIM..][..DIM].iter().map(|c| c + 0.3 * uniform()).collect();
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        data.extend(row.iter().map(|x| x / norm));
    }
    data
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = make_dataset(N, 1);
    let queries = make_dataset(5, 2);

    // The same options as the Python
    //   scann.scann_ops_pybind.builder(db, 5, "dot_product")
    //       .tree(num_leaves=200, num_leaves_to_search=20, random_init=False)
    //       .score_ah(2, anisotropic_quantization_threshold=0.2)
    //       .reorder(100).build()
    let builder = ConfigBuilder::new(5, DistanceMeasure::DotProduct, DIM)
        .tree(TreeOptions::new(200, 20).random_init(false)) // k-means++: reproducible
        .score_ah(AhOptions::new(2).anisotropic_quantization_threshold(0.2))
        .reorder(ReorderOptions::new(100));
    println!("config:\n{}", builder.build(N as u64)?);
    let mut index = builder.build_index(&dataset)?;
    println!("built {index:?}");

    // One query at a time; SearchOptions overrides the config per query.
    for (q, query) in queries.chunks(DIM).enumerate() {
        let nn = index.search(query, SearchOptions::default())?;
        println!("query {q}: {:?} {:?}", nn.indices, nn.distances);
    }
    let wide = index.search(&queries[..DIM], SearchOptions::k(10).leaves_to_search(50))?;
    println!("query 0, k=10 over 50 leaves: {:?}", wide.indices);

    // A batch, and the same batch split across the index's thread pool.
    let batched = index.search_batched(&queries, SearchOptions::default())?;
    index.set_num_threads(4)?;
    let parallel = index.search_batched_parallel(&queries, SearchOptions::default(), 2)?;
    assert_eq!(batched[0].indices, parallel[0].indices);

    // ScannIndex is Send + Sync: search it from several threads at once.
    let index_ref = &index;
    let from_threads: Vec<Vec<u32>> = std::thread::scope(|s| {
        let handles: Vec<_> = queries
            .chunks(DIM)
            .map(|q| s.spawn(move || index_ref.search(q, SearchOptions::k(1)).unwrap().indices))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    println!("top-1 per query, searched concurrently: {from_threads:?}");

    // Mutation: add a point (immediately searchable), then delete it.
    let added = index.add(&queries[..DIM])?;
    let nn = index.search(&queries[..DIM], SearchOptions::k(1))?;
    println!("added point {:?}; query 0's nearest neighbour is now {:?}", added, nn.indices);
    index.delete(&added)?;
    println!("deleted it again: {} points", index.len());

    // Save and reload. The directory is also loadable from Python with
    // scann.scann_ops_pybind.load_searcher(dir).
    let dir = std::env::temp_dir().join("scann-core-rust-quickstart");
    std::fs::create_dir_all(&dir)?;
    index.serialize(&dir, true)?;
    let reloaded = ScannIndex::load(&dir)?;
    println!("reloaded {reloaded:?} from {}", dir.display());
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
