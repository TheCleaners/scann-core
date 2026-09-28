# Using scann-core with TensorFlow

`scann.tf` is upstream's TensorFlow API (`scann_ops`: `builder()`,
`create_searcher()`, `ScannSearcher.search*()`, `serialize_to_module()`,
`searcher_from_module()`) on scann-core. It has two backends, with the
same API and the same results:

* **The Python backend**, in the wheel: the search runs the pybind
  searcher through `tf.numpy_function`. It works eagerly, in `tf.function`
  and in `tf.data`: training-time evaluation, batch retrieval, notebooks.
  It can't be saved in a SavedModel.
* **The op backend**: scann-core's TensorFlow custom op (the
  `scann_tf_ops` package), which you build from source against your
  TensorFlow. Searches are graph ops, and the index is TensorFlow state,
  so a model that searches saves as a SavedModel, index included. Not in
  the wheel, unsupported, and not loadable by TensorFlow Serving; see
  [The op backend](#the-op-backend-scann_tf_ops-build-from-source).

`import scann.tf` uses the op backend when `scann_tf_ops` is importable,
and the Python backend otherwise; [Backends](#backends) has the details.
For serving with TensorFlow Serving, export only the query tower as a
SavedModel and run scann-core next to it
([Serving](#serving-query-tower--scann-core)).

## Install

```sh
pip install 'scann-core[tf]'    # scann-core + tensorflow>=2.21
```

or install `tensorflow-cpu` (or `tensorflow`) yourself next to
`scann-core`. That gives the Python backend; the op backend is
[built from source](#building-it). `import scann` never imports
TensorFlow; only `import scann.tf` does, and without TensorFlow it raises
an `ImportError` that says so.

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

The API is upstream's `scann_ops` module (the TensorFlow op wrapper), so
porting is a change of import:

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

scann_tf.backend()   # "op" or "python"
```

Runnable version, with docids and an update, on either backend:
[`examples/python/tensorflow_wrapper.py`](../examples/python/tensorflow_wrapper.py).
Saving a model with the op backend:
[`examples/python/tensorflow_op.py`](../examples/python/tensorflow_op.py).

The API, the same with both backends:

| | |
|---|---|
| `builder(db, k, distance).….build()` | a `ScannSearcher`; the builder is `scann_ops_pybind`'s (autopilot, SOAR, `build(docids=...)`) |
| `create_searcher(db, config, training_threads=0, container="", shared_name=None, docids=None)` | as upstream; `container` is ignored, `shared_name` is the op backend's cache key |
| `load_searcher(dir, assets_backcompat_shim=True, shared_name=None)` | an index `serialize()` wrote (any binding) |
| `from_pybind(searcher)`, `ScannSearcher(searcher)` | from a `scann_ops_pybind.ScannSearcher` |
| `searcher.search(q, final_num_neighbors, pre_reorder_num_neighbors, leaves_to_search)` | `(index, distance)`, int32/float32, `[n]` with n ≤ k |
| `searcher.search_batched(q, ...)`, `search_batched_parallel(q, ..., batch_size=256)` | `(indices, distances)`, `[num_queries, k]` |
| `searcher.to_pybind()`, `searcher.serialize(dir, relative_path=False)` | the index as a `scann_ops_pybind` searcher (with its docids), or saved to a directory that loads in Python, Rust and C++ |
| `searcher.serialize_to_module()`, `searcher_from_module(module)` | the searcher's state as a `tf.Module`, and the searcher back from it, also after `tf.saved_model.load`. Op backend only; the Python backend raises `NotImplementedError` |
| `backend()`, `get_backend(name)`, `available_backends()` | the backend in use, either backend's module, the backends that import here |

What the search methods return, with either backend:

* A namedtuple of an int32 and a float32 tensor, named like upstream's op
  outputs: `(index, distance)` from `search`, `(indices, distances)` from
  the batched methods.
* Exactly the pybind searcher's results, converted: `search` gives `[n]`
  (n ≤ k), the batched methods `[num_queries, k]`. Rows with fewer than k
  results are padded with index 0 and distance NaN. With an explicit
  `final_num_neighbors` the width is exactly that k; with the default
  (`None`) it is the longest row. Upstream's op made it the longest row in
  both cases. The tests compare both backends with the pybind searcher,
  and with each other, bit for bit.
* Static shapes where they are known: the batch dimension from the query
  tensor, and k when `final_num_neighbors` is a Python int (with a
  tensor, the width stays unknown). Search parameters may be Python ints
  or int32 tensors; `None` means the searcher's default.

Other behaviour:

* The searcher is built eagerly, from a numpy array or an eager tensor.
  Building inside `tf.function` or a v1 graph (possible with upstream's op)
  raises `ValueError`.
* The builder is `scann_ops_pybind`'s, so `autopilot()`, SOAR
  (`tree(soar_lambda=...)`, which upstream's TF builder rejected) and
  `build(docids=...)` work. Every index scann-core builds is supported,
  including int8 and bfloat16 brute force and reordering, and trees with
  every point deleted.
* Results are indices into the index, never docids, as with upstream's op.
  If the searcher has docids, map indices with `searcher.to_pybind().docids`
  (or the list you built it with), in the graph for example with
  `tf.gather(tf.constant(docids), indices)`. That table is a snapshot:
  rebuild it after `upsert` or `delete`, which move points to other
  indices. Padding maps to `docids[0]`; check for NaN distances.
* Updating the index: mutate the pybind searcher and make a searcher from
  it again, which works with both backends:
  `p = searcher.to_pybind(); p.upsert(...); searcher = scann_tf.from_pybind(p)`.
  Functions traced with the old searcher keep searching the old index with
  the op backend, so make them again too (the example does).
* Errors: a query tensor of the wrong rank is a `ValueError` (when the
  function is traced). Search errors (wrong dimensionality, NaN queries,
  `final_num_neighbors=0`, `batch_size=0`) are
  `tf.errors.InvalidArgumentError`, eagerly and in graphs, with ScaNN's
  message.
* The indices are int32, as upstream's, so an index with more than
  2^31 - 1 points doesn't fit: the op backend fails on its first search,
  the Python backend on a search that returns an index beyond 2^31 - 1,
  both with `InvalidArgumentError`. Use `scann_ops_pybind` for such an
  index.

## Backends

### Choosing one

`import scann.tf` picks a backend once, when it is first imported:

* `"op"` if the `scann_tf_ops` package is importable (it is built next to
  the `scann` package in `build/python/` with `-DSCANN_BUILD_TF_OP=ON`),
* `"python"` otherwise. If `scann_tf_ops` is installed but fails to import
  (built for another TensorFlow, or copied from another scann-core
  version), with a `RuntimeWarning` giving the error.

`scann.tf.backend()` says which. The `SCANN_TF_BACKEND` environment
variable overrides the choice: `op` (then a failing op import is an
`ImportError`), `python`, or `auto` (the default). Either backend is
also available explicitly, whatever `scann.tf` picked:

```python
import scann.tf

scann.tf.backend()                  # "op" or "python"
scann.tf.available_backends()       # ("op", "python") or ("python",)
py = scann.tf.get_backend("python") # a module with the scann.tf API
searcher = py.builder(db, 10, "dot_product").score_brute_force().build()
op = scann.tf.get_backend("op")     # ImportError without scann_tf_ops
```

Code that imported `scann_tf_ops` directly keeps working: it is the op
backend's module (`scann.tf.get_backend("op") is scann_tf_ops`), and its
objects are the ones `scann.tf` returns with the op backend.
`scann_tf_ops.stats()` reports the op's searcher cache.

### Differences between the backends

The results are the same; these differences are intentional:

| | Python backend | op backend |
|---|---|---|
| where | in the wheel | built from source against one TensorFlow |
| SavedModel | no: `serialize_to_module()` and `searcher_from_module()` raise `NotImplementedError` | yes, index included |
| the searcher | wraps a `scann_ops_pybind` searcher, `searcher.searcher` | a `tf.Module`; the index files in `tf.Variable`s (`asset_names`, `asset_contents`, `index_id`) |
| `to_pybind()`, `from_pybind()` | the wrapped searcher itself; wrap without copying | copies of the index |
| changes to the pybind searcher | seen by later searches, also through functions traced before | not seen: make a new searcher (`from_pybind()`) |
| in a graph | a Python callback (`tf.numpy_function`, stateful: searches run in program order) | a graph op (no Python, no GIL) |
| memory | the pybind searcher | the index files in the variables, plus the searcher built from them on the first search (about twice as much) |
| `shared_name` | ignored | the op's cache key |

### Limits of the Python backend

* **No SavedModel, no TensorFlow Serving.** The search is a
  [`tf.numpy_function`](https://www.tensorflow.org/api_docs/python/tf/numpy_function):
  the graph holds a reference to a Python function, which isn't saved.
  `tf.saved_model.save` of a function that searches succeeds, but calling it
  after `tf.saved_model.load` fails with `Could not find callback with
  key=pyfunc_0 in the registry`. The same applies to anything else that
  serializes the graph for another process. Build the
  [op backend](#the-op-backend-scann_tf_ops-build-from-source) for that.
* **Python in the graph.** Each search call in a graph runs a Python
  function. It holds the GIL only around the call; the search itself
  releases it, so concurrent calls (several threads calling a
  `tf.function`, or a `tf.data` map with `num_parallel_calls`) search in
  parallel: 8 at a time ran 5.6 times as fast as 1 (batches of 8 queries,
  brute force over 200k points). `tests/python/test_tf.py` checks that
  results stay correct. (Eagerly, searches call the pybind searcher
  directly.)
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

## The op backend (`scann_tf_ops`, build from source)

scann-core's TensorFlow custom op is in [`tf_op/`](../tf_op/). It is
**source-only and unsupported**: not part of the wheel, off by default,
Linux only, and built against the one TensorFlow version installed where
you build it. What it adds over the Python backend:

* **SavedModel.** The searcher is TensorFlow state (`tf.Variable`s holding
  the whole index), and searches are graph ops, so a model that searches
  saves as one SavedModel, index included, and loads and searches in
  another process.
* **No Python in the graph.** No `tf.numpy_function`, no GIL, no Python
  callback registry: the graph runs the same in a loaded SavedModel as in
  the process that built it.

Verified with TensorFlow 2.21.0 only (Python 3.12, x86-64 Linux; built
with clang and with GCC): built against the `tensorflow-cpu` 2.21.0 wheel,
the same library also loads and passes its tests in the CUDA-built
`tensorflow` 2.21.0 wheel, also with a GPU in use (`tensorflow[and-cuda]`,
one NVIDIA GPU). Other TensorFlow versions are untested; build the op
against the version you run.

### Building it

In a virtual environment with TensorFlow (the op is built against the
TensorFlow in `Python_EXECUTABLE`):

```sh
python3 -m venv tfenv
tfenv/bin/pip install tensorflow-cpu==2.21.0 numpy 'protobuf>=7.36.2'
cmake -S . -B build -G Ninja -DSCANN_BUILD_TF_OP=ON \
      -DPython_EXECUTABLE="$PWD/tfenv/bin/python"
cmake --build build
ctest --test-dir build -R 'tf|tensorflow' --output-on-failure
```

That builds `build/python/scann_tf_ops/` (`__init__.py` and
`_scann_tf_ops.so`) next to the `scann` package in `build/python/`. Use
them together (`PYTHONPATH=build/python`; then `scann.tf.backend()` is
`"op"`), or copy `scann_tf_ops/` next to an installed scann-core of the
same version: it uses the `scann` package for building indexes, and
`scann.tf` finds it there. `-DSCANN_BUILD_TF_OP=ON` needs the Python
package and the static library (`SCANN_BUILD_PYTHON`,
`SCANN_BUILD_STATIC`, both on by default).

The library is built only against TensorFlow's C API
(`tensorflow/c/kernels.h`, `ops.h`), never its C++ headers, so TensorFlow's
own abseil and protobuf don't matter: scann-core's are linked in
statically and kept private. The library exports no symbols, imports only
TensorFlow's `TF_*` C functions (from `libtensorflow_framework.so.2`, which
TensorFlow has loaded already; no rpath), and the `tf_op_symbols` test
checks that. It does need the build machine's glibc and libstdc++, or
newer.

### Saving a model that searches

```python
import numpy as np
import tensorflow as tf
import scann.tf   # upstream: from scann.scann_ops.py import scann_ops

assert scann.tf.backend() == "op"
db = np.random.rand(20000, 32).astype(np.float32)
searcher = (scann.tf.builder(db, 10, "dot_product")
            .tree(num_leaves=150, num_leaves_to_search=15)
            .score_ah(2)
            .reorder(100)
            .build())

class Retrieval(tf.Module):
  def __init__(self, searcher):
    super().__init__()
    self.index = searcher.serialize_to_module()   # a tf.Module

  @tf.function(input_signature=[tf.TensorSpec([None, 32], tf.float32)])
  def retrieve(self, queries):
    searcher = scann.tf.searcher_from_module(self.index)
    return searcher.search_batched_parallel(queries, final_num_neighbors=10)

model = Retrieval(searcher)
tf.saved_model.save(model, "export/retrieval",
                    signatures={"serving_default": model.retrieve})

# Another process: import scann.tf first (with the op backend, it
# registers the op).
import scann.tf
loaded = tf.saved_model.load("export/retrieval")
indices, distances = loaded.retrieve(tf.random.uniform([4, 32]))
same = scann.tf.searcher_from_module(loaded.index)   # the same searcher
```

Runnable, with the reload in a fresh process:
[`examples/python/tensorflow_op.py`](../examples/python/tensorflow_op.py).

`serialize_to_module()` returns the searcher itself (its variables hold
the index); `searcher_from_module()` accepts it or the object
`tf.saved_model.load` restored for it, and shares its variables. Besides
the common API, the op backend's searcher has `files()` (the index files,
`{name: bytes}`) and `shared_name`, and `scann_tf_ops.stats()` reports
`{"live_searchers", "builds"}` of the op's cache in this process.

Results are what the pybind searcher returns: the tests compare them bit
for bit for brute force (float, int8, bfloat16), AH, and trees with each.
For SOAR with AH and reordering they agree to the last bit of a distance,
which varies between identical searches with the pybind searcher itself.

### How it works

TensorFlow's C API can't create TensorFlow resources from a pip-installed
TensorFlow, so there is no searcher resource. Instead the search ops take
the index itself: the files `serialize()` writes (`scann_config.pb`,
`scann_assets.pbtxt`, the `.npy` and `.pb` assets, and
`scann_docids.pkl` if there are docids) as two string tensors, file names
and contents. The `ScannSearcher` keeps them in two `tf.Variable`s, plus a
third holding a random index id. On its first search, the op builds the
ScaNN searcher from the tensors in memory (no temporary files) and caches
it, keyed by the index id and a fingerprint of the tensors. Every graph,
function and eager call that searches the same index with the same
variable values shares that one searcher; a loaded SavedModel builds it on
its first search, once. It is freed when the last function using it is
(eager searches go through functions owned by the `ScannSearcher`, so it
is freed with the searcher).

### Limits of the op

* **No mutation through the op** (upstream's op had none either). Mutate
  with the pybind searcher, then make a new searcher and re-export:
  `p = searcher.to_pybind(); p.upsert(...); searcher = scann.tf.from_pybind(p)`.
* **Static k only from Python.** The op's shape function can't read
  `final_num_neighbors` (an input, and the C API's shape inference
  can't read attributes either), so the op's outputs are
  `[num_queries, ?]`. The Python methods set the static shape to
  `[num_queries, k]` when `final_num_neighbors` is a Python int, as the
  Python backend does; with a tensor, the width stays unknown.
* **The index is in memory twice** once searched: in the variables (the
  serialized files) and in the searcher built from them. For a 400,000 ×
  128 tree + AH + reorder index (223 MB of files), `load_searcher()` added
  238 MB of RSS, and the first search 254 MB more, 492 MB in all; the
  pybind searcher alone takes 226 MB. The first search takes as long as
  `scann_ops_pybind.load_searcher()` (about 220 ms here). Making a searcher
  from a pybind one (`builder().build()`, `from_pybind()`) peaks higher:
  the pybind searcher, `serialize()`'s temporary copies (about 560 MB for
  this index; freed afterwards but kept by the allocator), and the
  variables. For large indexes, build and `serialize()` with
  `scann_ops_pybind`, then `load_searcher(dir)` where the model is made.
* **Stale variables.** The cached searcher is used only while the tensors'
  fingerprint matches, so assigning other values to the variables (another
  index, or restoring a checkpoint of another index) rebuilds the searcher
  on the next search. The fingerprint covers the number of files, each
  file's name and size, the whole of every file up to 64 KiB (including
  the config and the manifest), and for larger files their first and last
  4 KiB and 64 evenly spaced 64-byte samples. It does not detect a change
  confined to the unsampled bytes of a large file that keeps every size:
  editing the variables in place (a few points' data) isn't seen. Indexes
  built from different data differ nearly everywhere and are detected (the
  tests use two brute-force indexes whose files all have the same sizes).
  Hashing everything on every search would be exact but cost 7.3 ms per
  search for the 223 MB index above; the fingerprint takes 2.6 µs.
* **Per-call overhead.** For the index above, one query took 130 µs in a
  `tf.function` against 44 µs from the pybind searcher (eagerly: 258 µs);
  100 queries 4.0 ms against 3.5 ms (`search_batched`) and 0.57 ms against
  0.40 ms (`search_batched_parallel`). The Python backend in a
  `tf.function` measured 161 µs and 4.1 ms.
* **Loading needs the op.** `import scann.tf` (with the op backend) or
  `import scann_tf_ops` before `tf.saved_model.load`, which otherwise
  fails with `Op type not registered 'ScannCoreSearchBatched'`. The
  SavedModel only loads where the op library is built for that
  TensorFlow.
* **CPU only, no XLA.** The op has a CPU kernel only; in a GPU TensorFlow
  it is placed on the host and queries computed on the GPU are copied
  there (checked with a GPU: the query tower's ops ran on the GPU, the
  search on the CPU, results unchanged). It can't be compiled with
  `jit_compile=True`.
* **TensorFlow Serving can't load it.** `tensorflow_model_server` has no
  option for loading op libraries, and is one static binary (checked with
  the `tensorflow/serving:latest` image, TensorFlow Serving 2.20.0):
  there is no `libtensorflow_framework.so.2` for the op library to link
  against, so `LD_PRELOAD` fails to load it. A copy without that
  dependency finds every `TF_*` function it needs in the server binary,
  but crashes it at startup: its registrations run before the server's
  own static initializers. Requests to a model using the op fail with
  `Op type not registered`. TensorFlow Serving's route for custom ops is
  building the model server from source with the op linked in; that is
  untested here. With TensorFlow Serving, keep the index next to the model
  ([below](#serving-query-tower--scann-core)).

## Replacing TensorFlow Recommenders' ScaNN layer

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

The layer's `num_reordering_candidates` is `.reorder(n)`. Like the layer,
this can be saved with the model when `scann.tf` uses the
[op backend](#the-op-backend-scann_tf_ops-build-from-source) (keep
`searcher.serialize_to_module()` on the model and search through
`scann_tf.searcher_from_module()`, as above); with the Python backend it
can't. For serving, see below.

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

## Why the wheel has no TensorFlow op

A TensorFlow custom op built against TensorFlow's C++ API, as upstream's
is, must be built against the exact TensorFlow version it runs with
(headers and ABI), and shares TensorFlow's copies of abseil and protobuf,
so it has to be built with those versions too. scann-core upgrades both
(see [Dependencies](../README.md#dependencies)) and builds with CMake
instead of TensorFlow's toolchain. The source-built `scann_tf_ops` avoids
the abseil/protobuf conflict by using only TensorFlow's C API, but it is
still built against one TensorFlow at a time and runs only where that
TensorFlow is, so shipping it would mean one build per TensorFlow release
(and none of them would load in TensorFlow Serving). The wheel leaves it
out ([Intentional differences](../README.md#intentional-differences-from-upstream)),
so `scann.tf` from the wheel uses the Python backend.

If you need the searcher inside a SavedModel, build `scann_tf_ops`. For
TensorFlow Serving, upstream's wheel with its op is still an option:
`pip install scann[tf]` (it pins `tensorflow~=2.20.0`), and upstream's
[`tf_serving`](https://github.com/google-research/google-research/tree/master/scann/tf_serving)
directory builds a TensorFlow Serving image with the op. It doesn't have
scann-core's fixes, and can't be installed next to scann-core.
