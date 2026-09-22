# scann-core tutorial

A hands-on walk through ScaNN on a real dataset: from exact search to a tuned
index serving a quarter of a million queries per second, then saving it,
updating it, and running it from C++ and Rust.

| Part | You'll learn |
|---|---|
| [1. A first index](01-first-index.md) | Building and querying a brute-force index; what the results mean |
| [2. Measuring](02-measuring.md) | Recall, throughput and latency, measured properly; quantized brute force |
| [3. The pipeline](03-the-pipeline.md) | Partitioning, asymmetric hashing and reordering, one stage at a time |
| [4. Tuning](04-tuning.md) | Trading speed for recall, at query time and at build time |
| [5. Saving and serving](05-saving-and-serving.md) | Serialization, index size, batching and threads |
| [6. Updating](06-updating.md) | Inserting, updating and deleting points in a live index |
| [7. C++ and Rust](07-cpp-and-rust.md) | The same pipeline from C++ and Rust, sharing indexes with Python |

Every number in this tutorial comes from running the scripts in
[`code/`](code/). There is one script per part, and each part shows its
script's actual output. They were run on an AMD Threadripper PRO 7975WX
(32 cores, 64 threads). Your absolute speeds will differ, but the
comparisons between configurations should hold.

Each output is from a single run, with file paths shortened. Rerunning a
script gives throughput within a few percent. Recall is usually identical,
but it can move by a few thousandths when an index is rebuilt: tree
training starts from randomly chosen centres, the Python default.

For background on *why* the techniques work, the tutorial links to
[algorithms.md](../algorithms.md) and the
[anisotropic quantization explainer](../anisotropic_quantization_explained.md);
for every option's exact meaning, [api_reference.md](../api_reference.md).

## Setup

Build scann-core with the Python package (see the
[README](../../README.md#building)):

```sh
cmake -S . -B build -G Ninja -DSCANN_ARCH_FLAGS="-march=native"
cmake --build build
```

`-march=native` lets the compiler use every instruction set your CPU has.
The portable default (`-mavx -mfma`) works too, just more slowly.

Then make a Python environment with numpy, protobuf and h5py, and point it at
the package:

```sh
python -m venv .venv
.venv/bin/pip install numpy "protobuf>=7.36.2" h5py
export PYTHONPATH=$PWD/build/python
cd docs/tutorial/code
../../../.venv/bin/python part1_first_index.py
```

### The dataset

The tutorial uses **GloVe-100** from
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks):

* 1,183,514 word vectors of dimension 100;
* 10,000 held-out query vectors;
* for every query, its true 100 nearest neighbours by cosine similarity.

The ScaNN paper benchmarks on this dataset. The first script to run
downloads it (485 MB) into `~/.cache/scann-core-tutorial`, or into
`$SCANN_TUTORIAL_DATA` if set. [`code/tutorial_data.py`](code/tutorial_data.py)
loads it and provides two helpers every part uses: `recall()` and
`evaluate()`.

The vectors are L2-normalized on load, so that cosine similarity, which
the ground truth uses, is the same as the dot product, which ScaNN searches
with. [Part 1](01-first-index.md) explains why that matters.
