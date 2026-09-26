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

//! Rust-side equivalence check: builds each fixture's index through the
//! cxx bridge and compares single-query search results against those the
//! *reference wheel* (the Bazel-built scann 1.4.2, `scann.scann_ops_pybind`)
//! produced for the same dataset, config and queries.
//!
//! Fixtures are written by tests/equivalence/run.py; run via
//! `cmake --build <build> --target scann_core_rust_test`.

use scann_core::{ScannIndex, SearchOptions};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn read_f32(path: &Path) -> Vec<f32> {
    fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

struct Meta {
    n: usize,
    dim: usize,
    nq: usize,
    k: usize,
}

fn read_meta(path: &Path) -> Meta {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let v: Vec<usize> = text.split_whitespace().map(|s| s.parse().unwrap()).collect();
    Meta { n: v[0], dim: v[1], nq: v[2], k: v[3] }
}

#[test]
fn matches_reference_wheel() {
    let Some(root) = env::var_os("SCANN_CORE_EQUIV_FIXTURES").map(PathBuf::from) else {
        eprintln!("skipped: SCANN_CORE_EQUIV_FIXTURES not set (fixtures come from tests/equivalence/run.py)");
        return;
    };
    let mut cases: Vec<PathBuf> = fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("{}: {e}", root.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("meta.txt").exists())
        .collect();
    cases.sort();
    assert!(!cases.is_empty(), "no fixtures under {}", root.display());

    let mut failures = Vec::new();
    for case in &cases {
        let name = case.file_name().unwrap().to_string_lossy().to_string();
        let m = read_meta(&case.join("meta.txt"));
        let dataset = read_f32(&case.join("dataset.f32"));
        let queries = read_f32(&case.join("queries.f32"));
        let config = fs::read_to_string(case.join("config.pbtxt")).unwrap();
        let ref_idx = read_u32(&case.join("ref_indices.u32"));
        let ref_dist = read_f32(&case.join("ref_distances.f32"));
        assert_eq!(dataset.len(), m.n * m.dim);
        assert_eq!(queries.len(), m.nq * m.dim);

        let index = ScannIndex::new(&dataset, m.dim, &config)
            .unwrap_or_else(|e| panic!("{name}: build failed: {e}"));
        assert_eq!(index.len(), m.n);

        let (mut identical, mut max_delta) = (0usize, 0f32);
        for q in 0..m.nq {
            let query = &queries[q * m.dim..(q + 1) * m.dim];
            let res = index
                .search(query, SearchOptions::k(m.k))
                .unwrap_or_else(|e| panic!("{name}: search failed: {e}"));
            let want_idx = &ref_idx[q * m.k..(q + 1) * m.k];
            let want_dist = &ref_dist[q * m.k..(q + 1) * m.k];
            if res.indices.as_slice() == want_idx {
                identical += 1;
                for (a, b) in res.distances.iter().zip(want_dist) {
                    max_delta = max_delta.max((a - b).abs());
                }
            }
        }
        println!(
            "rust {name:22} identical neighbour lists {identical}/{} max |dist delta| {max_delta:.3e}",
            m.nq
        );
        if identical != m.nq || max_delta > 1e-5 {
            failures.push(name);
        }
    }
    assert!(failures.is_empty(), "mismatch vs reference wheel in: {failures:?}");
}
