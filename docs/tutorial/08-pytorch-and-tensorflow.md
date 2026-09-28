# Part 8: PyTorch and TensorFlow

Scripts: [`code/part8_torch.py`](code/part8_torch.py),
[`code/part8_tensorflow.py`](code/part8_tensorflow.py)

So far the queries were rows of a numpy array. In an application they
usually come out of a model: a query encoder (a "query tower") turns text,
a user or an image into an embedding, and the index finds its neighbours.
There are three ways to put the two together:

1. **Plain Python**: run the model, convert its output to numpy, and call
   the `scann_ops_pybind` searcher, as in parts 1 to 6.
2. **`scann.torch`**: the searcher as a `torch.nn.Module`. Tensors in,
   tensors out, on the queries' device; the search is a PyTorch op, so a
   model that encodes and searches compiles with `torch.compile`, and with
   the native backend it exports with `torch.export`, index included.
3. **`scann.tf`**: upstream ScaNN's TensorFlow API. Searches work eagerly
   and in `tf.function`; with the op backend, a model that searches saves
   as one SavedModel, index included.

This part runs all three on the part 3 index (tree + AH + reorder, recall
0.90) and a stand-in query encoder, checks that they return the same
neighbours, and measures what each costs per call.

## Setup

```sh
pip install 'scann-core[torch]'     # scann-core + PyTorch
pip install scann-core-torch        # optional: scann.torch's native backend
pip install 'scann-core[tf]'        # scann-core + TensorFlow
```

From a CMake build, as in the [setup](README.md#setup):
`-DSCANN_BUILD_TORCH_OP=ON` builds the native PyTorch backend (what the
`scann-core-torch` wheel holds), and `-DSCANN_BUILD_TF_OP=ON` builds
`scann.tf`'s op backend, which is source-only (see
[tensorflow.md](../tensorflow.md#building-it)). Each is built against the
framework installed in the build's Python, so this part used two builds
and two environments: PyTorch 2.14.0+cu132 on Python 3.14, and
TensorFlow 2.21.0 (`tensorflow-cpu`) on Python 3.12, both with numpy,
protobuf and h5py. The GPU is an NVIDIA RTX 3060 Ti.

**About the timings in this part.** They were measured while other jobs
(compilations and a fuzzer) kept the machine busy, so absolute times are
higher and noisier than in the other parts. Each table was measured
round-robin, all columns interleaved, so the columns of one row saw the
same load and can be compared with each other; compare the differences,
not the absolute values, and don't compare rows between the two scripts.
Recall and the "identical" checks don't depend on load.

## PyTorch

### The index as a module

```python
import torch
import scann.torch as scann_torch

searcher = (scann_torch.builder(torch.from_numpy(dataset), 10, "dot_product")
            .tree(num_leaves=2000, num_leaves_to_search=100,
                  training_sample_size=250000)
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(100)
            .build())

indices, distances = searcher.search_batched_parallel(torch.from_numpy(queries))
```

```
torch 2.14.0+cu132; scann.torch backend: native
built in 6.3 s: Searcher(size=1183514, default_num_neighbors=10, backend=native, index_bytes=538967556)
search_batched_parallel: torch.int64 (10000, 10), torch.float32; recall@10 0.9007
identical to the pybind searcher: True
Python backend identical: True
```

The builder is the same as `scann_ops_pybind`'s; `build()` returns a
`scann.torch.Searcher`. A float32 CPU tensor is read without a copy, and
the results are int64 indices and float32 distances, on the queries'
device. Every row has exactly k entries; if the index had fewer than k
points, the missing ones would be index -1 and distance NaN.

`searcher.searcher` is the `scann_ops_pybind` searcher underneath, for
everything that isn't a search: `upsert`, `delete`, `docids`,
`serialize`. The results are bit for bit those of that searcher, and of
the Python backend (`Searcher(pybind, backend="python")`).

**The two backends.** `scann.torch` has two implementations of the
search op, with the same API and the same results:

* **native**, when `scann-core-torch` is installed (or the op is built):
  C++ ops, and the index in the module's buffers, 514 MiB here
  (`index_bytes`). That is what makes `state_dict()`, `torch.save` and
  `torch.export` carry the index (below). The build took longer than
  part 3's 3.1 s partly because the index is also serialized into the
  buffers (and partly the busy machine).
* **python**, otherwise: the op is a Python function calling the pybind
  searcher. It compiles, but can't be exported or pickled.

### A model that encodes, then searches

A real query encoder is trained. This one is a stand-in built so that we
know the right answers: the model's input is a GloVe query hidden behind
a random rotation, and the encoder, one linear layer plus normalization,
undoes it. The model's output is therefore the GloVe query again (up to
rounding), and the ground truth still applies.

```python
rotation, _ = torch.linalg.qr(torch.randn(100, 100, generator=...))
features = torch.from_numpy(queries) @ rotation      # what the model sees


class QueryEncoder(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.proj = torch.nn.Linear(100, 100, bias=False)

  def forward(self, x):
    return torch.nn.functional.normalize(self.proj(x), dim=-1)


class Retriever(torch.nn.Module):
  def __init__(self, encoder, index):
    super().__init__()
    self.encoder = encoder
    self.index = index            # a Searcher is a submodule like any other

  def forward(self, features):
    return self.index.search_batched_parallel(self.encoder(features), 10)


encoder = QueryEncoder().eval()
with torch.no_grad():
  encoder.proj.weight.copy_(rotation)   # x @ rotation.T undoes the rotation
retriever = Retriever(encoder, searcher).eval()
compiled = torch.compile(retriever, fullgraph=True, dynamic=True)
```

```
compiled Retriever, 3 batch sizes, first calls: 8.1 s (mostly compiling)
  recall@10 0.9007; 12 of 100,000 results differ from searching the GloVe queries directly
```

`fullgraph=True` means no graph breaks: the search is part of the
compiled graph, as a custom op. With `dynamic=True` one compilation
serves every batch size (the script uses 7, 993 and 9,000). The 8.1 s is
Inductor compiling on first use; a second run of the script, with
Inductor's cache warm, took 1.3 s.

Recall is unchanged. 12 of the 100,000 neighbours differ from searching
the original queries, because the encoder's output differs from them in
the last bits, and a few near-ties flip.

### Queries on a GPU

Move the encoder to the GPU and give it GPU features: the encoder runs
there, the search runs on the CPU, and the results come back on the GPU.

```python
gpu_retriever = Retriever(QueryEncoder().cuda(), searcher)   # same weights
indices, distances = torch.compile(gpu_retriever, fullgraph=True,
                                   dynamic=True)(features.cuda())
```

```
GPU (NVIDIA GeForce RTX 3060 Ti): results on cuda:0, recall@10 0.9007; 3 of 100,000 results differ
  index buffers stay on: cpu
```

ScaNN has no GPU kernels. The queries are copied to host memory (which
waits for the GPU to finish computing them), and the results are copied
back. `model.to("cuda")` leaves the `Searcher`'s buffers in host memory,
where the search reads them.

### What a call costs

Median time per call of the whole `Retriever` (encoder, then
`search_batched_parallel`), against the plain-Python way: run the same
encoder, convert its output to numpy, and call the pybind searcher. The
"comp." columns are the `torch.compile`d model.

```
µs per call, median (Retriever = encoder + search_batched_parallel)
                                  pybind    native    python comp. nat  comp. py
  1 query, CPU                     139.0     151.0     166.5     198.2     203.3
  1,000 queries, CPU                7197      7587      7413      7631      7641
  1 query, GPU queries             205.6     250.6     265.7     296.9     307.0
  1,000 queries, GPU queries        4927      5534      5494      5613      5675
```

* **For one query**, `scann.torch` adds about 12 µs on the native backend
  and 28 µs on the Python backend. The compiled model costs another 40
  to 50 µs: a compiled function has fixed costs of its own (guards, the
  compiled wrapper), and this "model" is a single matrix product, so
  there is nothing for the compiler to win back. In a real model, that
  cost is shared with the rest of the forward pass.
* **For 1,000 queries** the search dominates, and on the CPU the columns
  are within 6% of each other, about this machine's noise today.
* **With GPU queries**, `scann.torch` also copies the results back to the
  GPU, and waits for it: 45 µs more for one query, and about 12% for
  1,000. The pybind column leaves its results in host memory.

These are relative costs, measured under load; rerun the script on an
idle machine for absolute numbers. (For a smaller index on an idle
machine, [integrations.md](../integrations.md#overhead) measured about
10 µs of dispatch per eager call, and 8 to 20 µs less on the native
backend than on the Python one.)

### Saving the model with its index

The native backend's `Searcher` keeps the index in three buffers, so a
model that searches carries the index everywhere a model's state goes:

```python
state = retriever.state_dict()   # index.index_data, index.index_offsets, ...

program = torch.export.export(
    retriever, (features[:8],),
    dynamic_shapes={"features": {0: torch.export.Dim("batch")}})
torch.export.save(program, "retriever.pt2")

# Another process: import scann.torch (it registers the ops), load, run.
import torch, scann.torch
model = torch.export.load("retriever.pt2").module()
indices, distances = model(features)
```

```
state_dict: ['encoder.proj.weight', 'index.index_data', 'index.index_offsets', 'index.index_names']; index_data 514.0 MiB
Python backend, torch.export: RuntimeError: scann.torch searches on the Python backend can't be exported: the export...
exported and saved in 0.7 s: retriever.pt2, 517.9 MiB
fresh process: loaded and searched 10,000 queries in 1.9 s; recall@10 0.9007; identical to eager: True
```

The exported program is 518 MiB: the index is almost all of it, and
`dataset.npy`, the float32 vectors for reordering, is 451 MiB of that.
Reordering in bfloat16 ([part 5](05-saving-and-serving.md#making-the-index-smaller))
would halve it. The fresh process needed nothing but the `.pt2` file
and `import scann.torch`. Its 1.9 s include reading the 518 MiB file
and building the searcher from the buffers on the first search.

On the Python backend, exporting raises: the op there finds its searcher
in the current process, so an exported program would hold a handle to
nothing. `searcher.index_in_state_dict = False` keeps a large index out
of frequent training checkpoints. AOTInductor works the same way as
`torch.export`; see
[integrations.md](../integrations.md#the-native-backend-scann-core-torch)
for it, for pickling, and for changing the index of a `Searcher` that
compiled or exported code uses (call `sync()`).

## TensorFlow

### Two backends, one API

`scann.tf` is upstream ScaNN's `scann_ops` API. From the wheel, it runs
the pybind searcher through `tf.numpy_function` (the **Python backend**);
with scann-core's TensorFlow op built, it uses the op (the **op
backend**), whose searchers are TensorFlow state and save in SavedModels.
`scann.tf.get_backend(name)` returns either one's module, so the script
can run both on the same index:

```python
import scann.tf

pybind = scann.scann_ops_pybind.builder(dataset, 10, "dot_product")...build()
for name in scann.tf.available_backends():       # ("op", "python")
  s = scann.tf.get_backend(name).from_pybind(pybind)

  @tf.function(input_signature=[tf.TensorSpec([None, 100], tf.float32)])
  def retrieve(q):
    return s.search_batched_parallel(q, final_num_neighbors=10)
```

```
TensorFlow 2.21.0; scann.tf backend: op; available: ('op', 'python')
built in 4.3 s
pybind searcher: recall@10 0.9007
op     backend: from_pybind 1.1 s; tf.function gives int32 (10000, 10), float32; identical to pybind: True
python backend: from_pybind 0.0 s; tf.function gives int32 (10000, 10), float32; identical to pybind: True
```

The results are int32 indices, as in upstream's op, and the static shape
`[None, 10]` is known when the function is traced. The Python backend
wraps the pybind searcher (no copy, hence 0.0 s); the op backend copies
the index into `tf.Variable`s, which takes a second for 514 MiB.

### A model that encodes, searches, and saves

The same stand-in encoder as a Keras model (a `Dense` layer with the
inverse rotation as its kernel, and `UnitNormalization`), and a
`tf.Module` whose `retrieve` function encodes and searches:

```python
class Retrieval(tf.Module):
  def __init__(self, encoder, searcher):
    super().__init__()
    self.encoder = encoder
    self.index = searcher.serialize_to_module()    # op backend only

  @tf.function(input_signature=[tf.TensorSpec([None, 100], tf.float32)])
  def retrieve(self, features):
    searcher = scann.tf.searcher_from_module(self.index)
    return searcher.search_batched_parallel(self.encoder(features),
                                            final_num_neighbors=10)


model = Retrieval(encoder, searchers["op"])
tf.saved_model.save(model, export_dir,
                    signatures={"serving_default": model.retrieve})

# Another process: import scann.tf (it registers the op), load, run.
loaded = tf.saved_model.load(export_dir)
out = loaded.signatures["serving_default"](features=...)
```

```
Python backend, serialize_to_module(): NotImplementedError: serialize_to_module() is not supported by scann.tf's Python ...
Retrieval.retrieve: recall@10 0.9007; 0 of 100,000 results differ from searching the GloVe queries directly

SavedModel written in 0.5 s: 514.1 MiB
fresh process: loaded and searched 10,000 queries in 1.3 s; signature outputs ['distances', 'indices']
  recall@10 0.9007; identical to before saving: True
```

The SavedModel holds the encoder, the graph and the index, and the fresh
process got the same results from it with nothing but the directory and
`import scann.tf`. With the Python backend, the search is a Python
callback that can't be saved, so `serialize_to_module()` raises. The op
can't be loaded by TensorFlow Serving either; for that, export only the
query tower and search next to it
([tensorflow.md](../tensorflow.md#serving-query-tower--scann-core)).

### What a call costs

First the search alone (`search_batched_parallel`, 10 neighbours), from
the pybind searcher on a numpy array, and from `scann.tf` on a tensor,
eagerly and in a `tf.function`, on each backend. Then the encoder plus
the search: the Keras model called eagerly then pybind, the encoder in a
`tf.function` then pybind, and the whole `Retrieval.retrieve` as one
`tf.function` (for the Python backend, a `tf.function` closing over its
searcher).

```
search_batched_parallel, µs per call (median)
                         pybind     op, eager     op, tf.fn python, eager python, tf.fn
  1 query                 117.3         454.3         307.4         257.2         406.1
  1,000 queries            4643          5799          5237          5447          5588

encoder + search, µs per call (median)
                   Keras+pybind  tf.fn+pybind     op, tf.fn python, tf.fn
  1 query                 868.0         307.3         370.7         428.1
  1,000 queries            6548          5653          5813          6174
```

* **TensorFlow's per-call costs are larger than PyTorch's.** One search
  through `scann.tf` cost 140 to 340 µs more than the pybind searcher:
  eager op calls go through a function the searcher owns, and a
  `tf.function` call has fixed costs of its own. For 1,000 queries it is
  13 to 25%.
* **Calling a Keras model eagerly is the slowest step** of all: 868 µs,
  against 307 µs for the same encoder in a `tf.function` followed by the
  pybind searcher. If you search with pybind, still wrap the encoder in a
  `tf.function` (or call a SavedModel's function).
* **For single queries, the encoder in a `tf.function` plus pybind is
  the fastest** (307 µs); the whole model as one `tf.function` with the op
  is 60 µs slower, and with the Python backend 120 µs slower. The op is
  worth it when the model must be one SavedModel; the Python backend
  when the search must happen inside a graph (a `tf.data` map, a training
  step) and saving doesn't matter.

As for PyTorch, these were measured on a busy machine: compare within a
row. [tensorflow.md](../tensorflow.md#limits-of-the-op) has idle-machine
numbers for another index.

## Which to use

| you want | use |
|---|---|
| the lowest latency per query, in a service | the model (compiled, or a `tf.function`), then the pybind searcher on its output; or C++/Rust ([part 7](07-cpp-and-rust.md)) |
| neighbours as tensors inside PyTorch code: a model's `forward`, evaluation in a training loop, a GPU pipeline | `scann.torch` |
| one PyTorch artifact with the index inside (`torch.export`, AOTInductor, `torch.save`, `state_dict`) | `scann.torch` with `scann-core-torch` (native backend) |
| searches inside TensorFlow graphs: `tf.function`, `tf.data` | `scann.tf` (either backend) |
| one SavedModel with the index inside | `scann.tf` with the op backend (built from source) |
| TensorFlow Serving | the query tower in TF Serving, scann-core next to it ([tensorflow.md](../tensorflow.md#serving-query-tower--scann-core)) |
| batch jobs (Ray, Spark, Dask) and web services | the pybind searcher, one per worker process ([frameworks.md](../frameworks.md)) |

Whichever you pick, the index is the same one: every way of searching in
this part found the same neighbours (up to the encoder's rounding), the
directory `serialize()` writes loads in all of them, and the index is
updated through the pybind searcher ([part 6](06-updating.md)). After an
update, call `sync()` before running `scann.torch`'s compiled code
(exported programs keep the index they were exported with), and make a
new searcher (`from_pybind()`) for `scann.tf`'s op backend.

That's the tutorial. For every option in detail, see
[api_reference.md](../api_reference.md). For more on the ideas behind them,
see [algorithms.md](../algorithms.md) and the
[anisotropic quantization explainer](../anisotropic_quantization_explained.md).
For PyTorch and TensorFlow in more depth,
[integrations.md](../integrations.md) and [tensorflow.md](../tensorflow.md);
for data and serving frameworks, [frameworks.md](../frameworks.md).
