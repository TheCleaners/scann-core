# scann-core (Rust)

Rust bindings for [scann-core](https://github.com/TheCleaners/scann-core):
the nearest-neighbour search core of Google's
[ScaNN](https://github.com/google-research/google-research/tree/master/scann),
without TensorFlow.

> scann-core is a derived work of ScaNN. It is not an official Google product
> and is not affiliated with or endorsed by Google.

```rust
use scann_core::{AhOptions, ConfigBuilder, DistanceMeasure, ReorderOptions,
                 SearchOptions, TreeOptions};

let index = ConfigBuilder::new(10, DistanceMeasure::DotProduct, dim)
    .tree(TreeOptions::new(2000, 100))
    .score_ah(AhOptions::new(2).anisotropic_quantization_threshold(0.2))
    .reorder(ReorderOptions::new(100))
    .build_index(&dataset)?;            // &[f32], row-major n x dim
let nn = index.search(query, SearchOptions::default())?;
println!("{:?} {:?}", nn.indices, nn.distances);
```

`ScannIndex` supports single, batched and parallel batched search, adding,
updating and deleting points, retraining, and saving and loading. Indexes are
interchangeable with scann-core's C++ and Python APIs (and upstream ScaNN's
Python package). It is `Send + Sync`: search from as many threads as you like.

## Building

The crate builds the C++ library with CMake the first time it compiles, so
you need:

* **CMake ≥ 3.27** and **clang ≥ 19** or **GCC ≥ 13** (clang is preferred
  when `CXX` isn't set);
* **Linux x86-64** (the only platform tested so far);
* **network access** to download the C++ dependencies (abseil, protobuf,
  highway, Eigen, zlib). To build offline, point CMake at local copies:
  `SCANN_CORE_CMAKE_ARGS="-DFETCHCONTENT_SOURCE_DIR_ABSL=/src/abseil-cpp ..."`
  (see the repository README).

The first build takes a few minutes; later builds reuse it. The default
instruction set is AVX + FMA; pass
`SCANN_CORE_CMAKE_ARGS="-DSCANN_ARCH_FLAGS=-march=native"` to use everything
your CPU has.

## Documentation

* API docs: `cargo doc --open`, or [docs.rs/scann-core](https://docs.rs/scann-core).
* A seven-part tutorial with Python, C++ and Rust:
  [docs/tutorial](https://github.com/TheCleaners/scann-core/tree/main/docs/tutorial).

## License

Apache 2.0. ScaNN is Copyright The Google Research Authors; scann-core's
additions and modifications are Copyright 2026 Elias Benali
([@ebenali](https://github.com/ebenali)) and TheCleaners. See
[NOTICE](https://github.com/TheCleaners/scann-core/blob/main/NOTICE).
