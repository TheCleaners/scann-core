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

//! Self-contained tests of the public API on synthetic data: correctness
//! against a naive exact search, agreement between search modes,
//! serialize/load, mutation, concurrency and error handling.

use scann_core::{
    AhOptions, ConfigBuilder, DistanceMeasure, IncrementalMode, Neighbors, Quantization,
    ReorderOptions, ScannError, ScannIndex, SearchOptions, TreeOptions, UpperTreeOptions,
};
use std::path::PathBuf;

const DIM: usize = 32;
const N: usize = 3000;
const K: usize = 10;

/// Deterministic xorshift-based data: `clusters` Gaussian-ish blobs,
/// unit-normalized rows.
fn dataset(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let clusters = 20;
    let centers: Vec<f32> = (0..clusters * dim).map(|_| (next() * 2.0 - 1.0) as f32).collect();
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = &centers[(i % clusters) * dim..][..dim];
        let row: Vec<f32> = c.iter().map(|&x| x + 0.3 * (next() * 2.0 - 1.0) as f32).collect();
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        out.extend(row.iter().map(|x| x / norm));
    }
    out
}

fn row(data: &[f32], i: usize) -> &[f32] {
    &data[i * DIM..(i + 1) * DIM]
}

fn naive(data: &[f32], q: &[f32], k: usize, distance: DistanceMeasure) -> Vec<(u32, f32)> {
    let mut all: Vec<(u32, f32)> = (0..data.len() / DIM)
        .map(|i| {
            let r = row(data, i);
            let d = match distance {
                DistanceMeasure::DotProduct => r.iter().zip(q).map(|(a, b)| a * b).sum(),
                DistanceMeasure::SquaredL2 => r.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum(),
            };
            (i as u32, d)
        })
        .collect();
    match distance {
        DistanceMeasure::DotProduct => all.sort_by(|a, b| b.1.total_cmp(&a.1)),
        DistanceMeasure::SquaredL2 => all.sort_by(|a, b| a.1.total_cmp(&b.1)),
    }
    all.truncate(k);
    all
}

fn recall(got: &[Neighbors], want: &[Vec<(u32, f32)>]) -> f64 {
    let hits: usize = got
        .iter()
        .zip(want)
        .map(|(g, w)| w.iter().filter(|(i, _)| g.indices.contains(i)).count())
        .sum();
    hits as f64 / want.iter().map(Vec::len).sum::<usize>() as f64
}

fn assert_close(a: &[Neighbors], b: &[Neighbors], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: result count");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.indices.len(), y.indices.len(), "{what}: query {i} length");
        for (dx, dy) in x.distances.iter().zip(&y.distances) {
            assert!((dx - dy).abs() <= 1e-5 * dx.abs().max(1.0), "{what}: query {i}: {dx} vs {dy}");
        }
    }
}

fn tree_ah_index(data: &[f32]) -> ScannIndex {
    ConfigBuilder::new(K, DistanceMeasure::DotProduct, DIM)
        .tree(TreeOptions::new(30, 8).random_init(false))
        .score_ah(AhOptions::new(2).anisotropic_quantization_threshold(0.2))
        .reorder(ReorderOptions::new(100))
        .training_threads(1)
        .build_index(data)
        .expect("build")
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("scann-core-rust-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn brute_force_is_exact() {
    let data = dataset(N, DIM, 1);
    let queries = dataset(50, DIM, 2);
    for distance in [DistanceMeasure::DotProduct, DistanceMeasure::SquaredL2] {
        let index = ConfigBuilder::new(K, distance, DIM)
            .score_brute_force(Quantization::Float32)
            .build_index(&data)
            .unwrap();
        assert_eq!(index.len(), N);
        assert_eq!(index.dimensionality(), DIM);
        for qi in 0..50 {
            let q = row(&queries, qi);
            let got = index.search(q, SearchOptions::default()).unwrap();
            let want = naive(&data, q, K, distance);
            assert_eq!(got.indices, want.iter().map(|p| p.0).collect::<Vec<_>>(), "{distance:?} q{qi}");
            // Reported like the Python API: the dot product itself / the
            // squared distance, not scann's internal negated score.
            for (d, (_, w)) in got.distances.iter().zip(&want) {
                assert!((d - w).abs() < 1e-4, "{distance:?} q{qi}: {d} vs {w}");
            }
        }
    }
}

#[test]
fn tree_ah_has_good_recall_and_modes_agree() {
    let data = dataset(N, DIM, 3);
    let queries = dataset(100, DIM, 4);
    let mut index = tree_ah_index(&data);

    let single: Vec<Neighbors> =
        (0..100).map(|i| index.search(row(&queries, i), SearchOptions::default()).unwrap()).collect();
    let batched = index.search_batched(&queries, SearchOptions::default()).unwrap();
    index.set_num_threads(4).unwrap();
    let parallel = index.search_batched_parallel(&queries, SearchOptions::default(), 16).unwrap();
    assert_close(&single, &batched, "single vs batched");
    assert_close(&batched, &parallel, "batched vs parallel");

    let want: Vec<_> = (0..100).map(|i| naive(&data, row(&queries, i), K, DistanceMeasure::DotProduct)).collect();
    let r = recall(&batched, &want);
    assert!(r > 0.9, "recall@{K} = {r}");

    // Overrides: fewer neighbours, more leaves.
    let few = index.search(row(&queries, 0), SearchOptions::k(3).leaves_to_search(30)).unwrap();
    assert_eq!(few.indices.len(), 3);
    assert_eq!(few.indices, naive(&data, row(&queries, 0), 3, DistanceMeasure::DotProduct)
        .iter().map(|p| p.0).collect::<Vec<_>>());

    // No thread pool: parallel search runs inline (used to crash upstream).
    index.set_num_threads(0).unwrap();
    let inline = index.search_batched_parallel(&queries, SearchOptions::default(), 16).unwrap();
    assert_close(&batched, &inline, "batched vs parallel without a pool");

    assert!(index.config().contains("num_neighbors: 10"));
}

#[test]
fn serialize_and_load_round_trip() {
    let data = dataset(N, DIM, 5);
    let queries = dataset(40, DIM, 6);
    let mut index = tree_ah_index(&data);
    let before = index.search_batched(&queries, SearchOptions::default()).unwrap();
    for relative in [false, true] {
        let dir = temp_dir(&format!("serialize-{relative}"));
        index.serialize(&dir, relative).unwrap();
        assert!(dir.join("scann_assets.pbtxt").exists());
        let loaded = ScannIndex::load(&dir).unwrap();
        assert_eq!(loaded.len(), N);
        assert_eq!(loaded.search_batched(&queries, SearchOptions::default()).unwrap(), before);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn mutation() {
    let data = dataset(N, DIM, 7);
    let extra = dataset(20, DIM, 8);
    let mut index = tree_ah_index(&data);

    let added = index.add(&extra[..10 * DIM]).unwrap();
    assert_eq!(added, (N as u32..N as u32 + 10).collect::<Vec<_>>());
    assert_eq!(index.len(), N + 10);
    for (i, &id) in added.iter().enumerate() {
        let got = index.search(row(&extra, i), SearchOptions::k(1)).unwrap();
        assert_eq!(got.indices, vec![id], "added point {i} finds itself");
    }

    // Replace point 0 with extra[10], add one more, batched.
    let ids = index.upsert(&[Some(0), None], &extra[10 * DIM..12 * DIM], 2).unwrap();
    assert_eq!(ids, vec![0, N as u32 + 10]);
    assert_eq!(index.search(row(&extra, 10), SearchOptions::k(1)).unwrap().indices, vec![0]);
    assert_eq!(index.len(), N + 11);

    index.delete(&[5, 6, 7]).unwrap();
    assert_eq!(index.len(), N + 8);

    index.reserve(N + 100).unwrap();
    index.rebalance(None).unwrap();
    assert_eq!(index.len(), N + 8);
    index.initialize_health_stats().unwrap();
    let stats = index.health_stats().unwrap();
    assert_eq!(stats.sum_partition_sizes as usize, N + 8);

    let out = index.search_batched(&extra, SearchOptions::default()).unwrap();
    assert!(out.iter().flat_map(|n| &n.indices).all(|&i| (i as usize) < index.len()));
}

/// Exact index where every stored vector finds itself at distance 0.
fn exact_l2_index(data: &[f32]) -> ScannIndex {
    ConfigBuilder::new(1, DistanceMeasure::SquaredL2, DIM)
        .score_brute_force(Quantization::Float32)
        .build_index(data)
        .unwrap()
}

#[test]
fn delete_ids_refer_to_the_index_before_the_call() {
    let data = dataset(200, DIM, 13);
    let mut index = exact_l2_index(&data);
    // A middle point, the last point, the one just below it, and the first:
    // deleted one by one in this order, the second id would already be out
    // of range.
    let gone = [5u32, 199, 198, 0];
    let moves = index.delete(&gone).unwrap();
    assert_eq!(index.len(), 196);
    assert_eq!(moves, vec![(196, 0), (197, 5)]);
    let moved: std::collections::HashMap<u32, u32> = moves.iter().copied().collect();
    for i in 0..200u32 {
        let got = index.search(row(&data, i as usize), SearchOptions::default()).unwrap();
        if gone.contains(&i) {
            assert!(got.distances[0] > 1e-6, "deleted point {i} still found");
        } else {
            assert_eq!(got.indices, vec![moved.get(&i).copied().unwrap_or(i)], "point {i}");
            assert!(got.distances[0] < 1e-6, "point {i}");
        }
    }
    // Rejected before anything changes.
    assert!(is_invalid_argument(index.delete(&[3, 3])));
    assert!(is_invalid_argument(index.delete(&[1, 196])));
    assert_eq!(index.len(), 196);

    // A point that moves twice is reported once, from where it started.
    let mut small = exact_l2_index(&data[..10 * DIM]);
    assert_eq!(small.delete(&[7, 8]).unwrap(), vec![(9, 7)]);
    assert_eq!(small.search(row(&data, 9), SearchOptions::default()).unwrap().indices, vec![7]);
}

#[test]
fn health_stats_from_many_threads() {
    let data = dataset(N, DIM, 14);
    let mut index = tree_ah_index(&data);
    index.initialize_health_stats().unwrap();
    let want = index.health_stats().unwrap();
    let (index, data) = (&index, &data);
    std::thread::scope(|s| {
        for t in 0..8 {
            s.spawn(move || {
                for i in 0..50 {
                    assert_eq!(index.health_stats().unwrap(), want);
                    index.search(row(data, (t * 50 + i) % N), SearchOptions::default()).unwrap();
                }
            });
        }
    });
}

#[test]
fn concurrent_search_matches_serial() {
    let data = dataset(N, DIM, 9);
    let queries = dataset(64, DIM, 10);
    let index = tree_ah_index(&data);
    let serial: Vec<Neighbors> =
        (0..64).map(|i| index.search(row(&queries, i), SearchOptions::default()).unwrap()).collect();
    let index = &index;
    let per_thread: Vec<Vec<Neighbors>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                s.spawn(|| {
                    (0..64).map(|i| index.search(row(&queries, i), SearchOptions::default()).unwrap()).collect()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for r in per_thread {
        assert_eq!(r, serial);
    }
}

#[test]
fn autopilot_builds() {
    let data = dataset(N, DIM, 11);
    let index = ConfigBuilder::new(K, DistanceMeasure::SquaredL2, DIM)
        .autopilot(IncrementalMode::None, Quantization::Float32)
        .build_index(&data)
        .unwrap();
    assert_eq!(index.search(row(&data, 3), SearchOptions::k(1)).unwrap().indices, vec![3]);
}

fn is_invalid_argument<T: std::fmt::Debug>(r: Result<T, ScannError>) -> bool {
    matches!(r, Err(ScannError::InvalidArgument(_)))
}

#[test]
fn errors_not_crashes() {
    let data = dataset(500, DIM, 12);
    let bf = ConfigBuilder::new(K, DistanceMeasure::DotProduct, DIM)
        .score_brute_force(Quantization::Float32)
        .build(500)
        .unwrap();

    // Config problems are reported by scann-core.
    assert!(matches!(ScannIndex::new(&data, DIM, "this is { not a config"), Err(ScannError::Scann(_))));
    assert!(matches!(ScannIndex::new(&data, DIM, &format!("{bf}\nnum_neighbours: 3\n")), Err(ScannError::Scann(_))));
    // Shape problems are caught before reaching C++.
    assert!(is_invalid_argument(ScannIndex::new(&data[..data.len() - 1], DIM, &bf)));
    assert!(is_invalid_argument(ScannIndex::new(&[], DIM, &bf)));
    assert!(is_invalid_argument(ScannIndex::new(&data, 0, &bf)));

    let mut index = ScannIndex::new(&data, DIM, &bf).unwrap();
    assert!(is_invalid_argument(index.search(&data[..DIM - 1], SearchOptions::default())));
    assert!(is_invalid_argument(index.search_batched(&data[..DIM + 1], SearchOptions::default())));
    assert!(is_invalid_argument(index.search_batched_parallel(&data[..DIM], SearchOptions::default(), 0)));
    assert!(is_invalid_argument(index.delete(&[500])));
    assert!(is_invalid_argument(index.upsert(&[Some(0)], &data[..2 * DIM], 1)));
    assert!(index.upsert(&[Some(10_000)], &data[..DIM], 1).is_err());
    assert!(index.search_batched(&[], SearchOptions::default()).unwrap().is_empty());
    // Asking for more neighbours than points returns what there is.
    let all = index.search(&data[..DIM], SearchOptions::k(1000)).unwrap();
    assert!(all.indices.len() <= 500 && all.indices.iter().all(|&i| i < 500));
}

fn is_non_finite_error<T: std::fmt::Debug>(r: Result<T, ScannError>) -> bool {
    matches!(&r, Err(ScannError::Scann(m)) if m.contains("NaN or infinity"))
}

/// NaN/infinity inputs and tree-only options on other index types used to
/// crash the process (segfault or QCHECK abort); they must be errors.
#[test]
fn non_finite_inputs_and_tree_options_are_errors() {
    const N_SMALL: usize = 600;
    let data = dataset(N_SMALL, DIM, 21);
    let tree_bf = || {
        ConfigBuilder::new(K, DistanceMeasure::DotProduct, DIM)
            .tree(TreeOptions::new(12, 4).training_sample_size(N_SMALL as u64))
            .score_brute_force(Quantization::Float32)
            .training_threads(1)
    };

    // Building a tree on NaN data aborted in k-means training.
    let mut bad_data = data.clone();
    bad_data[7 * DIM + 2] = f32::NAN;
    assert!(is_non_finite_error(tree_bf().build_index(&bad_data)));
    bad_data[7 * DIM + 2] = f32::INFINITY;
    assert!(is_non_finite_error(tree_bf().build_index(&bad_data)));

    // Upserting NaN/inf into a tree read leaf_mutators_[-1]. A batch that
    // fails on a later row must not apply the earlier ones.
    let mut index = tree_bf().build_index(&data).unwrap();
    let mut two = row(&data, 3).to_vec();
    two.extend([f32::NAN; DIM]);
    for (ids, vectors) in [
        (vec![None], &two[DIM..]),
        (vec![Some(5)], &two[DIM..]),
        (vec![None, None], &two[..]),
        (vec![Some(1), None], &two[..]),
    ] {
        for batch_size in [1, 2] {
            assert!(is_non_finite_error(index.upsert(&ids, vectors, batch_size)));
            assert_eq!(index.len(), N_SMALL);
        }
    }
    let inf = vec![f32::NEG_INFINITY; DIM];
    assert!(is_non_finite_error(index.upsert(&[None], &inf, 1)));
    assert_eq!(index.len(), N_SMALL);
    assert_eq!(index.search(row(&data, 3), SearchOptions::k(1)).unwrap().indices, vec![3]);

    // Batched search never checked queries (single search does).
    let mut queries = data[..4 * DIM].to_vec();
    queries[DIM + 1] = f32::NAN;
    assert!(is_non_finite_error(index.search_batched(&queries, SearchOptions::default())));
    assert!(is_non_finite_error(index.search_batched_parallel(&queries, SearchOptions::default(), 1)));
    assert!(index.search(&queries[DIM..2 * DIM], SearchOptions::default()).is_err());

    // leaves_to_search on a non-tree index: the int8 brute-force searcher
    // misread the tree parameters (segfault). It is ignored now.
    let int8 = ConfigBuilder::new(K, DistanceMeasure::DotProduct, DIM)
        .score_brute_force(Quantization::Int8)
        .build_index(&data)
        .unwrap();
    let opts = SearchOptions::k(K).leaves_to_search(5);
    let want = int8.search(row(&data, 0), SearchOptions::k(K)).unwrap();
    assert_eq!(int8.search(row(&data, 0), opts).unwrap().indices, want.indices);
    let batched = int8.search_batched(&data[..4 * DIM], opts).unwrap();
    assert_eq!(batched[0].indices, want.indices);
    let parallel = int8.search_batched_parallel(&data[..4 * DIM], opts, 2).unwrap();
    assert_eq!(parallel[0].indices, want.indices);
}

#[test]
fn builder_rejects_what_python_silently_ignores() {
    let b = || ConfigBuilder::new(K, DistanceMeasure::DotProduct, DIM);
    let ah = || AhOptions::new(2);
    let tree = || TreeOptions::new(30, 5);
    assert!(b().upper_tree(UpperTreeOptions::new(10, 2)).score_ah(ah()).build(1000).is_err());
    assert!(b().autopilot(IncrementalMode::None, Quantization::Float32).tree(tree()).build(1000).is_err());
    assert!(b().tree(tree()).build(1000).is_err(), "no scoring method");
    assert!(b().score_ah(ah()).score_brute_force(Quantization::Float32).build(1000).is_err());
    assert!(ConfigBuilder::new(K, DistanceMeasure::SquaredL2, DIM)
        .tree(tree().avq(1.0))
        .score_ah(ah())
        .build(1000)
        .is_err());
    assert!(b().truncate(DIM).tree(tree()).score_ah(ah()).build(1000).is_err());
    // Values Python passes through and ScaNN then mishandles.
    assert!(b().tree(TreeOptions::new(30, 0)).score_ah(ah()).build(1000).is_err());
    assert!(b().score_ah(ah()).reorder(ReorderOptions::new(K as u32 - 1)).build(1000).is_err());
    assert!(b().score_ah(ah().residual_quantization(true)).build(1000).is_err());
    assert!(b().tree(tree()).score_ah(ah()).build(1000).is_ok());
}
