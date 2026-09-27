# Using scann-core with TensorFlow

scann-core has no TensorFlow op (see [why](#why-there-is-no-tensorflow-op)).
There are two ways to use it with TensorFlow models:

* **`scann.tf`**, a small Python wrapper that makes the searcher usable
  from TensorFlow code: eager mode, `tf.function` and `tf.data`. Use it
  for training-time evaluation, batch retrieval and notebooks. It can't
  be saved in a SavedModel.
* **For serving**, export only the query tower as a SavedModel, and run
  scann-core (Python, Rust or C++) next to it.

## Install

```sh
pip install 'scann-core[tf]'    # scann-core + tensorflow>=2.21
```

or install `tensorflow-cpu` (or `tensorflow`) yourself next to
`scann-core`. `import scann` never imports TensorFlow; only `import
scann.tf` does, and without TensorFlow it raises an `ImportError` that says
so.

The constraint is protobuf. scann-core's generated `_pb2` modules need
protobuf ≥ 7.36.2. TensorFlow 2.21 requires protobuf ≥ 6.31.1, < 8, so the
two install together; that's the tested combination. TensorFlow 2.19 and
older require protobuf < 6 and can't be installed with scann-core. 2.20
declares protobuf ≥ 5.28.0 without an upper bound, but hasn't been tested
with scann-core. A future TensorFlow that caps protobuf below 7.36 would
conflict the same way.

Don't install upstream's `scann` wheel in the same environment: both
provide the `scann` package.

## `scann.tf`

The API is upstream's `scann_ops` module (the TensorFlow op wrapper):
`builder()`, `create_searcher()`, and a `ScannSearcher` with `search`,
`search_batched` and `search_batched_parallel`. Porting is a change of
import:

| upstream (`pip install scann[tf]`) | scann-core |
|---|---|
| `from scann.scann_ops.py import scann_ops` | `from scann import tf as scann_ops` |
| `scann.scann_ops.builder(...)` | `scann.tf.builder(...)` |

```python
import numpy as np
import tensorflow as tf
from scann import tf as scann_tf

db = np.random.rand(10000, 64).astype(np.float32)   # or an eager tf.Tensor
searcher = (scann_tf.builder(db, 10, "dot_product")
            .tree(num_leaves=100, num_leaves_to_search=10)
            .score_ah(2)
            .reorder(100)
            .build())

queries = tf.random.uniform([32, 64])
indices, distances = searcher.search_batched(queries)   # int32, float32; [32, 10]

@tf.function(input_signature=[tf.TensorSpec([None, 64], tf.float32)])
def retrieve(q):
  return searcher.search_batched_parallel(q, final_num_neighbors=5)
```

Runnable version, with docids: [`examples/python/tensorflow_wrapper.py`](../examples/python/tensorflow_wrapper.py).

What the search methods return:

* A namedtuple of an int32 and a float32 tensor, named like upstream's op
  outputs: `(index, distance)` from `search`, `(indices, distances)` from
  the batched methods.
* Exactly the pybind searcher's results, converted: `search` gives `[n]`
  (n ≤ k), the batched methods `[num_queries, k]`. Rows with fewer than k
  results are padded with index 0 and distance NaN. With an explicit
  `final_num_neighbors` the width is exactly that k; with the default
  (`None`) it is the longest row. Upstream's op made it the longest row in
  both cases.
* Static shapes where they are known: the batch dimension from the query
  tensor, and k when `final_num_neighbors` is a Python int. Search
  parameters may also be int32 tensors.

Other behaviour:

* The searcher is built eagerly, from a numpy array or an eager tensor.
  Building inside `tf.function` or a v1 graph (possible with upstream's op)
  raises `ValueError`.
* `searcher.searcher` is the underlying `scann_ops_pybind.ScannSearcher`.
  Use it for everything else (`upsert`, `delete`, `rebalance`,
  `set_num_threads`, `config`); later searches through the wrapper see the
  changes. `scann_tf.ScannSearcher(pybind_searcher)` wraps an existing one,
  and `scann_tf.load_searcher(dir)` loads a saved index. `serialize(dir)`
  saves it; the directory loads in Python, Rust and C++.
* The builder is `scann_ops_pybind`'s, so `autopilot()`, SOAR
  (`tree(soar_lambda=...)`, which upstream's TF builder rejected) and
  `build(docids=...)` work.
* Results are indices into the index, never docids, as with upstream's op.
  If the searcher has docids, map indices with `searcher.searcher.docids`,
  in the graph for example with
  `tf.gather(tf.constant(searcher.searcher.docids), indices)`. That table is
  a snapshot: rebuild it after `upsert` or `delete`, which move points to
  other indices. Padding maps to `docids[0]`; check for NaN distances.
* Errors: eagerly, the pybind searcher's exception (`ValueError`,
  `RuntimeError`). Inside `tf.function`, a `tf.errors.OpError` with the same
  message. A query tensor of the wrong rank is a `ValueError` when the
  function is traced.
* `serialize_to_module()` and `searcher_from_module()` raise
  `NotImplementedError`.

### Limits

* **No SavedModel, no TensorFlow Serving.** The search is a
  [`tf.numpy_function`](https://www.tensorflow.org/api_docs/python/tf/numpy_function):
  the graph holds a reference to a Python function, which isn't saved.
  `tf.saved_model.save` of a function that searches succeeds, but calling it
  after `tf.saved_model.load` fails with `Could not find callback with
  key=pyfunc_0 in the registry`. The same applies to anything else that
  serializes the graph for another process.
* **Python in the graph.** Each search call runs a Python function. It
  holds the GIL only around the call; the search itself releases it, so
  concurrent calls (several threads calling a `tf.function`, or a
  `tf.data` map with `num_parallel_calls`) search in parallel: 8 at a time
  ran 5.6 times as fast as 1 (batches of 8 queries, brute force over 200k
  points). `tests/python/test_tf.py` checks that results stay correct.
* **Program order.** The search is marked stateful (the index can change
  between calls), so TensorFlow doesn't constant-fold or deduplicate it,
  and two searches in one `tf.function` run one after the other. Put the
  queries in one batch and use `search_batched_parallel` instead.
* **CPU only, no XLA.** The search runs on the host; queries on a GPU are
  copied to host memory. A function that searches can't be compiled with
  `jit_compile=True` (XLA has no kernel for the Python callback).

`tf.py_function` would work too, but it converts every input and output
through eager tensors and supports gradients, neither of which a search
needs; `tf.numpy_function` passes numpy arrays straight through.

### Replacing TensorFlow Recommenders' ScaNN layer

TensorFlow Recommenders' `tfrs.layers.factorized_top_k.ScaNN` layer is
built on upstream's op. Its `index()` calls `scann_ops.builder(...)` and
`serialize_to_module()`, so with scann-core installed it fails (checked
against tensorflow-recommenders 0.7.7 source, which does
`from scann import scann_ops`; in scann-core that is an empty package, so
the import succeeds and `index()` raises `AttributeError`). TFRS's
`BruteForce` layer doesn't use ScaNN and is unaffected.

The layer builds a tree + AH searcher over the candidate embeddings and
returns `(scores, identifiers)`. The same with `scann.tf`, with the layer's
defaults (`num_leaves=100`, `num_leaves_to_search=10`,
`training_iterations=12`, `dimensions_per_block=2`, parallel batched
search):

```python
# Instead of:
#   index = tfrs.layers.factorized_top_k.ScaNN(query_model, k=10)
#   index.index_from_dataset(candidates)  # batches of (identifier, embedding)
#   scores, titles = index(features)
from scann import tf as scann_tf

identifiers = tf.concat([ids for ids, _ in candidates], axis=0)
embeddings = tf.concat([emb for _, emb in candidates], axis=0)
searcher = (scann_tf.builder(embeddings, 10, "dot_product")
            .tree(num_leaves=100, num_leaves_to_search=10,
                  training_iterations=12)
            .score_ah(2)
            .build())

@tf.function
def retrieve(features, k=10):
  result = searcher.search_batched_parallel(query_model(features),
                                            final_num_neighbors=k)
  return result.distances, tf.gather(identifiers, result.indices)

scores, titles = retrieve(features)
```

The layer's `num_reordering_candidates` is `.reorder(n)`. Unlike the
layer, this can't be saved with the model; for serving, see below.

## Serving: query tower + scann-core

A two-tower retrieval model splits cleanly at serving time. The query
tower is a TensorFlow model; the candidate tower is only needed to embed
the items once, offline. So export the query tower as a SavedModel, index
the item embeddings with scann-core, and search next to the model:

* The query tower runs in TensorFlow (in-process, or in TensorFlow Serving).
* The index runs in scann-core: Python, or Rust or C++ in a separate
  service. All three load the same index directory.
* Each can be updated on its own schedule, as long as the index is rebuilt
  whenever the embedding space changes (a retrained model).

Export (a tiny untrained Keras query tower, for brevity):

```python
import os

import numpy as np
import keras
import scann

rng = np.random.default_rng(0)
num_items, dim = 10_000, 32

# The query tower: user features -> embedding.
query_model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(dim),
    keras.layers.UnitNormalization(),
])

# Item embeddings: the candidate tower applied to every item.
items = rng.standard_normal((num_items, dim)).astype(np.float32)
items /= np.linalg.norm(items, axis=1, keepdims=True)
docids = [f"item-{i}" for i in range(num_items)]

query_model.export("export/query_model")  # a SavedModel, endpoint "serve"
index = (scann.scann_ops_pybind.builder(items, 10, "dot_product")
         .tree(num_leaves=100, num_leaves_to_search=10)
         .score_ah(2, anisotropic_quantization_threshold=0.2)
         .reorder(100)
         .build(docids=docids))
os.makedirs("export/index", exist_ok=True)
index.serialize("export/index", relative_path=True)  # movable directory
```

Serve:

```python
import numpy as np
import tensorflow as tf
import scann

query_model = tf.saved_model.load("export/query_model")
index = scann.scann_ops_pybind.load_searcher("export/index")

def recommend(features, k=10):
  embeddings = query_model.serve(features).numpy()
  return index.search_batched(embeddings, final_num_neighbors=k)

docids, scores = recommend(np.random.rand(2, 16).astype(np.float32))
# docids[0][:3] == ['item-947', 'item-7436', 'item-7641']
```

Both steps in one runnable script: [`examples/python/tensorflow_serving.py`](../examples/python/tensorflow_serving.py).

Notes:

* `relative_path=True` makes the index directory movable, so it can be
  shipped to the serving hosts with the model.
* Docids are stored in `scann_docids.pkl`, which only Python reads (and
  which, being a pickle, must come from a trusted source). A Rust or C++
  service gets indices; ship the id list in a format of your choice. See
  [tutorial part 7](tutorial/07-cpp-and-rust.md).
* With TensorFlow Serving hosting the query tower, the retrieval service
  calls its predict API and then searches; the index never needs
  TensorFlow.
* Batch size, threads and throughput:
  [tutorial part 5](tutorial/05-saving-and-serving.md#serving-batch-size-latency-and-throughput).

## Why there is no TensorFlow op

A TensorFlow custom op is a shared library loaded into TensorFlow's
process. It must be built against the exact TensorFlow version it runs
with (headers and ABI), and it shares TensorFlow's copies of abseil and
protobuf, so it has to be built with those versions too. scann-core
upgrades both (see [Dependencies](../README.md#dependencies)), builds with
CMake instead of TensorFlow's toolchain, and would have to ship one build
per TensorFlow release. The op is the one part of upstream it leaves out
([Intentional differences](../README.md#intentional-differences-from-upstream)).

If you need the searcher inside a SavedModel or TensorFlow Serving,
upstream's wheel with its op is still an option: `pip install scann[tf]`
(it pins `tensorflow~=2.20.0`), and upstream's
[`tf_serving`](https://github.com/google-research/google-research/tree/master/scann/tf_serving)
directory builds a TensorFlow Serving image with the op. It doesn't have
scann-core's fixes, and can't be installed next to scann-core.
