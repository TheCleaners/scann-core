# Part 7: C++ and Rust

Programs: [`code/cpp/glove.cc`](code/cpp/glove.cc),
[`code/rust/glove.rs`](code/rust/glove.rs); setup:
[`code/part7_export.py`](code/part7_export.py)

Python is convenient for experimenting. A service often wants the index
inside a C++ or Rust program instead, with no interpreter and no GIL. Both
use the same library, the same configurations and the same saved indexes.

This part runs the part 3 pipeline in both languages. Each program does
two things:

1. builds its own index from the same data and evaluates it;
2. loads the index that Python built and saved, and evaluates that.

## Setup

C++ and Rust don't read HDF5, so `part7_export.py` writes the data as `.npy`
files. It also builds the part 3 index in Python and saves it, with
`relative_path=True`:

```sh
python part7_export.py
```

```
wrote .npy files and ~/.cache/scann-core-tutorial/glove-index; recall@10 from Python: 0.9007
```

Both programs are built by scann-core's CMake build:

```sh
cmake --build build --target scann_core_tutorial_glove   # C++
cmake --build build --target scann_core_rust             # Rust (crate + examples)

build/examples/scann_core_tutorial_glove ~/.cache/scann-core-tutorial
build/rust/target/release/examples/tutorial_glove ~/.cache/scann-core-tutorial
```

For the Rust program you can also use plain cargo from the repository root:
`cargo run --release --example tutorial_glove -- <data dir>`. Without
`SCANN_CORE_BUILD_ENV` the crate builds the C++ library itself; with
`SCANN_CORE_BUILD_ENV=$PWD/build/rust/scann_core_rust_build.env` it reuses
the CMake build.

## C++

The C++ API has two parts. `scann_core::ConfigBuilder` is the builder
from Python. `research_scann::ScannInterface` is the index.

```cpp
#include "scann/scann_ops/cc/scann.h"
#include "scann_core/config_builder.h"

scann_core::TreeOptions tree;
tree.num_leaves = 2000;
tree.num_leaves_to_search = 100;
tree.training_sample_size = 250000;
scann_core::AhOptions ah;
ah.dimensions_per_block = 2;
ah.anisotropic_quantization_threshold = 0.2;
scann_core::ReorderOptions reorder;
reorder.reordering_num_neighbors = 100;
auto config = scann_core::ConfigBuilder(10, scann_core::DistanceMeasure::kDotProduct, dim)
                  .Tree(tree).ScoreAh(ah).Reorder(reorder)
                  .BuildText(n);          // absl::StatusOr<std::string>

research_scann::ScannInterface index;
absl::Status status = index.Initialize({data, n * dim}, n, *config, /*training_threads=*/0);
```

`BuildText` returns the same text configuration Python's `create_config()`
does. The tests check that for 75 option combinations. Where Python silently
ignores an option, for example `upper_tree` without `tree` or a builder
method called twice, `ConfigBuilder` returns an error instead.

Searching:

```cpp
index.SetNumThreads(64);
std::vector<research_scann::NNResultsVector> found(num_queries);
status = index.SearchBatchedParallel(queries, research_scann::MakeMutableSpan(found),
                                     /*final_nn=*/-1, /*pre_reorder_nn=*/-1, /*leaves=*/-1);
```

`-1` means "use the value from the configuration", like passing `None` in
Python. Every call returns an `absl::Status`, and bad input (wrong
dimensions, a malformed config, `batch_size = 0`) is an error status, not a
crash.

**One difference from Python:** `NNResultsVector` holds ScaNN's *internal*
scores, which for dot product are negated so that smaller is always better.
`index.ReshapeNNResult(result, indices, distances)` converts to Python's
convention; [`examples/cpp/quickstart.cc`](../../examples/cpp/quickstart.cc)
shows it.

Loading the Python-built index:

```cpp
auto artifacts = research_scann::ScannInterface::LoadArtifacts(dir + "/glove-index");
research_scann::ScannInterface loaded;
status = loaded.Initialize(*std::move(artifacts));
```

Output:

```
dataset 1183514 x 100, 10000 queries
built in 3.2 s
built in C++               recall@10 0.8988    470672 QPS
loaded in 0.5 s
built in Python, loaded    recall@10 0.9007    415904 QPS
```

The loaded index gives **exactly** Python's recall (0.9007): it is the same
index. The index built in C++ differs slightly (0.8988) because training
uses random initialization, and so does every rebuild in Python.

To link against scann-core in your own CMake project, `add_subdirectory` it
and link `scann::core`. The [README](../../README.md#building) explains the
whole-archive requirement if you link the static library by hand.

## Rust

The `scann-core` crate wraps the same library in a safe API:

```rust
use scann_core::{AhOptions, ConfigBuilder, DistanceMeasure, Neighbors, ReorderOptions,
                 ScannIndex, SearchOptions, TreeOptions};

let mut index = ConfigBuilder::new(10, DistanceMeasure::DotProduct, dim)
    .tree(TreeOptions::new(2000, 100).training_sample_size(250_000))
    .score_ah(AhOptions::new(2).anisotropic_quantization_threshold(0.2))
    .reorder(ReorderOptions::new(100))
    .build_index(&dataset)?;              // dataset: &[f32], row-major

index.set_num_threads(64)?;
let found: Vec<Neighbors> =
    index.search_batched_parallel(&queries, SearchOptions::default(), 256)?;
// found[i].indices, found[i].distances

let loaded = ScannIndex::load(dir.join("glove-index"))?;
```

* Options are builder structs with Python's defaults. `SearchOptions` holds
  the per-query overrides: `SearchOptions::k(20).leaves_to_search(200)`.
* Distances follow Python's convention; there's no negation to undo.
* Errors, whether a shape mismatch caught in Rust or a failed status from
  C++, are `ScannError` values.
* `ScannIndex` is `Send + Sync`. Searching takes `&self`, and mutation
  (`upsert`, `delete`, `rebalance`, ...) takes `&mut self`, so the compiler
  rules out searching while the index is being modified.

Output:

```
dataset 1183514 x 100, 10000 queries
built in 3.3 s
built in Rust              recall@10 0.8999    380945 QPS
loaded in 0.5 s
built in Python, loaded    recall@10 0.9007    379369 QPS
concurrent search() from  1 threads:    14489 QPS
concurrent search() from  8 threads:   105798 QPS
concurrent search() from 32 threads:   281285 QPS
concurrent search() from 64 threads:   343017 QPS
```

The last four lines are the experiment Python couldn't do well in
[part 5](05-saving-and-serving.md#serving-batch-size-latency-and-throughput).
Many threads call `search()` on one shared index, one query at a time:

```rust
std::thread::scope(|s| {
    for t in 0..threads {
        s.spawn(move || {
            for q in queries.chunks(dim).skip(t).step_by(threads) {
                index.search(q, SearchOptions::default()).unwrap();
            }
        });
    }
});
```

Python's threads flattened out at about 100k QPS with the GIL, and reached
327k on free-threaded Python. Rust scales to 343k QPS on 64 threads, 24× one
thread and close to batched throughput (379–381k), with no batching at all. So a Rust service can simply run one search per request,
on whatever thread handles it. Each query takes about 0.07 ms on an idle
machine, and about 0.19 ms (64 threads ÷ 343,017 QPS) with all 64 hardware
threads busy, since two hyperthreads share each core.

## Python, C++ and Rust side by side

| | Python | C++ | Rust |
|---|---|---|---|
| configure | `scann_ops_pybind.builder(db, k, "dot_product").tree(...)...` | `scann_core::ConfigBuilder(k, kDotProduct, dim).Tree(...)...BuildText(n)` | `ConfigBuilder::new(k, DotProduct, dim).tree(...)...` |
| build | `.build()` | `ScannInterface::Initialize(data, n, config, threads)` | `.build_index(&data)` or `ScannIndex::new(&data, dim, &config)` |
| search one | `search(q, final_num_neighbors=...)` | `Search(q, &res, k, pre, leaves)` | `search(q, SearchOptions::k(..))` |
| search many | `search_batched(qs)` / `search_batched_parallel(qs)` | `SearchBatched` / `SearchBatchedParallel` | `search_batched` / `search_batched_parallel` |
| save / load | `serialize(dir)` / `load_searcher(dir)` | `SerializeToDirectory(dir)`\* / `LoadArtifacts(dir)` + `Initialize` | `serialize(dir, relative)` / `ScannIndex::load(dir)` |
| update | `upsert(docids, vecs)`, `delete(docids)` | `GetMutator()` → `AddDatapoint` / `UpdateDatapoint` / `RemoveDatapoint` | `add`, `upsert(ids, vecs, batch)`, `delete(ids)` |
| retrain | `rebalance()` | `RetrainAndReindex("")` | `rebalance(None)` |
| distances | Python convention | internal (use `ReshapeNNResult`) | Python convention |

\* `SerializeToDirectory` (scann-core) writes the index files and the
`scann_assets.pbtxt` manifest that `LoadArtifacts(dir)` reads, staged and
committed so that an interrupted save never leaves a directory that loads a
mix of two indexes (the Python and Rust wrappers use it; see
[`serialize`](../api_reference.md#persistence-serialize--load_searcher)).
Upstream's `ScannInterface::Serialize` is still there: it writes the index
files in place and *returns* the manifest for you to write.
[`examples/cpp/quickstart.cc`](../../examples/cpp/quickstart.cc) shows the
C++ side.

Docids are a Python-side feature, stored in `scann_docids.pkl`. C++ and Rust
identify points by index. If you delete, remember that the last point moves
into the freed slot, so keep your own id mapping. In Rust, `delete` returns
each point that moved as `(old index, new index)`; in C++, Python's
`upsert`/`delete` in `scann_ops_pybind.py` are a template for one.

That's the tutorial. For every option in detail, see
[api_reference.md](../api_reference.md). For more on the ideas behind them,
see [algorithms.md](../algorithms.md) and the
[anisotropic quantization explainer](../anisotropic_quantization_explained.md).
