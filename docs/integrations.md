# Using scann-core with other libraries

scann-core installs the same `scann` Python module as the upstream
`scann` wheel, with the same `scann_ops_pybind` API. Libraries written
against upstream ScaNN work with it unchanged. This page covers inputs
from other array libraries, searching from PyTorch models
([`scann.torch`](#scanntorch-searching-from-pytorch-models)), the
libraries that have been checked, and what to know when using them.

TensorFlow has its own page: [tensorflow.md](tensorflow.md). For batch
jobs and services (Ray, Spark, Dask, FastAPI, Ray Serve, BentoML,
Triton), inputs from Arrow, pandas and Polars, and hybrid dense + sparse
retrieval, see [frameworks.md](frameworks.md). [Tutorial part
8](tutorial/08-pytorch-and-tensorflow.md) walks through `scann.torch`
and `scann.tf` on a real dataset.

## Installing in place of `scann`

Both packages install a top-level `scann` module, so only one of them can
be installed in an environment. Remove the upstream wheel first:

```sh
pip uninstall scann
pip install scann-core
```

A library that lists `scann` as a requirement will try to install the
upstream wheel again. Install it with `pip install --no-deps`, or install
scann-core after it.

## Arrays from PyTorch and other libraries

Every call that takes vectors (`builder`, `create_searcher`, `search`,
`search_batched`, `search_batched_parallel`, `upsert`) accepts PyTorch
tensors as well as numpy arrays, and returns the same results for the same
values:

```python
import torch
from scann.scann_ops.py import scann_ops_pybind

embeddings = model.encode(corpus)            # a torch.Tensor, CPU or GPU
searcher = scann_ops_pybind.builder(embeddings, 10, "dot_product") \
    .tree(num_leaves=1000, num_leaves_to_search=50) \
    .score_ah(2, anisotropic_quantization_threshold=0.2) \
    .reorder(100).build()
neighbors, distances = searcher.search_batched(model.encode(queries))
```

* **A float32 tensor in CPU memory is read in place**, without a copy.
  Other dtypes (float64, float16, bfloat16) are converted to float32, as
  numpy arrays are.
* **A tensor on a GPU is copied to host memory first.** ScaNN runs on the
  CPU, so the copy is unavoidable; it costs one device-to-host transfer per
  call. Keep queries on the CPU if they are there already.
* **A tensor that requires grad is detached.** Searching doesn't
  participate in autograd; the results are plain numpy arrays (or docids),
  not tensors.

Upstream's Python layer passes vectors to the pybind module as given, so
there CPU float tensors work but bfloat16 tensors, tensors that require
grad and GPU tensors fail with a pybind "incompatible function arguments"
error.

The conversion only uses the tensor's attributes (`requires_grad`,
`device`, `cpu()`, `float()`) and numpy's array protocol, so it doesn't
import PyTorch. Arrays from other libraries that numpy can read work the
same way, for example JAX and TensorFlow CPU arrays; only PyTorch is tested
(`python_torch_input`, with CUDA tensors when a GPU is present). For
Arrow, pandas and Polars columns, and which conversions copy, see
[frameworks.md](frameworks.md#inputs-from-arrow-pandas-and-polars).

For results as tensors, and searching inside `torch.compile`d models, see
[`scann.torch`](#scanntorch-searching-from-pytorch-models) below; for
TensorFlow graphs, [tensorflow.md](tensorflow.md).

## `scann.torch`: searching from PyTorch models

`scann.torch.Searcher` is a `torch.nn.Module` around the pybind searcher.
Its searches take query tensors and return tensors on the queries' device,
and they are PyTorch custom ops, so a model that searches compiles with
`torch.compile(fullgraph=True)`.

```sh
pip install 'scann-core[torch]'    # scann-core + torch>=2.10
pip install scann-core-torch       # optional: the native backend (below)
```

or install `torch` yourself (any build: CPU, CUDA, ROCm). `import scann`
never imports PyTorch; only `import scann.torch` does, and without PyTorch
it raises an `ImportError` that says so.

There are two backends, with the same API and the same results (bit for
bit; `python_torch` runs every check on both):

* **native**, when the `scann-core-torch` package is installed: the
  searches are C++ ops and the `Searcher` keeps the index in its buffers,
  so models that search can be exported (`torch.export`, AOTInductor),
  their `state_dict()` contains the index, and they can be pickled
  (`torch.save(model)`). See
  [The native backend](#the-native-backend-scann-core-torch).
* **python** otherwise (scann-core 0.2.0's): the ops are Python functions
  that call the pybind searcher, found through a handle into a registry of
  the process; exporting and pickling raise.

`scann.torch.backend()` says which one new `Searcher`s use.
`SCANN_TORCH_BACKEND=native|python|auto` (read at import; `auto`, the
default, is native when it loads) and a `backend="native"|"python"`
argument of `Searcher`, `builder()`, `create_searcher()` and
`load_searcher()` choose; `searcher.backend` says which one a `Searcher`
has.

```python
import torch
import scann.torch as scann_torch

item_embeddings = encoder(items)                    # [n, d], CPU or GPU
searcher = (scann_torch.builder(item_embeddings, 10, "dot_product")
            .tree(num_leaves=1000, num_leaves_to_search=50)
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(100)
            .build())

indices, distances = searcher.search_batched(encoder(queries))  # int64, float32

class Retriever(torch.nn.Module):
  def __init__(self, encoder, searcher):
    super().__init__()
    self.encoder = encoder
    self.index = searcher                           # a submodule

  def forward(self, features):
    return self.index.search_batched_parallel(self.encoder(features), 10)

retriever = torch.compile(Retriever(encoder, searcher), fullgraph=True,
                          dynamic=True)
indices, distances = retriever(features)
```

Runnable version (a toy encoder, recall checks, GPU if present):
[`examples/python/torch_retrieval.py`](../examples/python/torch_retrieval.py).

| | |
|---|---|
| `builder(db, k, distance).….build()` | a `Searcher`; the builder is `scann_ops_pybind`'s (autopilot, SOAR, `build(docids=...)`); `db` a tensor (any float dtype, any device) or array |
| `create_searcher(db, config, training_threads=0, docids=None)` | from a text config, as in `scann_ops_pybind` |
| `load_searcher(dir)`, `Searcher.from_pybind(s)` | from a directory `serialize()` wrote (any binding), or around an existing `scann_ops_pybind` searcher (shared, not copied) |
| `searcher.search(q, final_num_neighbors=None, pre_reorder_num_neighbors=None, leaves_to_search=None)` | one 1-D query: `(indices, distances)` of shape `[k]` |
| `searcher.search_batched(q, ...)`, `searcher(q, ...)` | `[num_queries, dim]` queries: `[num_queries, k]` |
| `searcher.search_batched_parallel(q, ..., batch_size=256)` | the same on the searcher's thread pool |
| `searcher.searcher`, `searcher.to_pybind()` | the `scann_ops_pybind` searcher: `upsert`, `delete`, `docids`, `config`, ...; searches through the module see the changes (native backend: see [changes](#changing-the-index)) |
| `searcher.serialize(dir)`, `searcher.default_num_neighbors` | save the index; k when `final_num_neighbors` is `None` |
| `scann.torch.backend()`, `searcher.backend` | `"native"` or `"python"` |
| `searcher.sync()`, `searcher.index_in_state_dict` | native backend: copy changes made through `searcher.searcher` to the buffers; whether `state_dict()` includes the index (default `True`) |

What the searches return:

* **int64 indices and float32 distances, on the queries' device.** Queries
  can be any float dtype (converted to float32) on any device, or numpy
  arrays. Search parameters are Python ints; `None` (or -1) means the
  index's default, as in `scann_ops_pybind`.
* **Always k per query**, where k is `final_num_neighbors` or, if that is
  `None`, the index's default. When the index has fewer than k points, the
  missing results are **index -1 and distance NaN**. `scann_ops_pybind`
  pads with index 0 instead (and its `search` returns fewer than k, and its
  batched searches with the default k are as wide as the longest row); -1
  can't be mistaken for a real point, and fixed shapes are what
  `torch.compile` needs. Otherwise the results are exactly the pybind
  searcher's: `python_torch` compares them bit for bit for brute force, AH
  and a tree.
* **Indices, never docids.** If the index has docids, map with the pybind
  searcher's list, skipping padding (`docids[-1]` is the last docid, not an
  error):
  `[[docids[i] if i >= 0 else None for i in row] for row in indices.tolist()]`
  with `docids = searcher.searcher.docids`. `upsert` and `delete` move points
  to other indices, so map with the list as it is when you search.
* **No gradients.** Queries that require grad are detached; the results
  don't require grad. The rest of the model trains normally (the encoder in
  the model above gets its gradients from other terms of the loss).

Devices: ScaNN searches on the CPU. Queries on a GPU are copied to host
memory (which waits for the GPU to finish computing them), and the results
are copied back to the GPU. `model.to("cuda")` leaves the `Searcher` as it
is: on the Python backend it has no parameters or buffers, and on the
native backend its buffers (the index) stay in host memory, where the
search reads them. Checked on an NVIDIA GPU (CUDA builds of torch) and an
AMD GPU (ROCm, where the device is also `cuda`).

### `torch.compile`

The searches are custom ops, `scann::search` and `scann::search_batched`
(native backend) or `scann_py::search` and `scann_py::search_batched`
(Python backend, registered with `torch.library.custom_op`), and their
fake implementations give exact output shapes, so:

* `fullgraph=True` works: no graph breaks, in a model that searches or
  around the searcher alone.
* `dynamic=True` (or dynamic shapes after a recompile) keeps the batch size
  symbolic: `python_torch` checks one compilation for batch sizes 2 to 64.
  Batch sizes 0 and 1 are specialized by `torch.compile` itself and compile
  once more each. `final_num_neighbors` and the other parameters are
  compile-time constants.
* With `mode="reduce-overhead"` (CUDA graphs), the ops are tagged as unsafe
  to capture, so Inductor keeps them out of the graphs; results are correct
  (checked on CUDA).
* Inside compiled code the search is still a call into Python on the
  Python backend, and a C++ op call on the native backend. Compiling pays
  off for the rest of the model, not for the search (see the overhead
  below).

Concurrent calls from several threads, eagerly or through a compiled model,
are fine: the search releases the GIL (the native op runs without it), and
`python_torch` checks the results from 8 threads.

### The native backend (scann-core-torch)

```sh
pip install scann-core-torch    # the scann-core of the same version, and torch>=2.10
```

The `scann-core-torch` wheel holds the `scann_torch_ops` package: the ops
`torch.ops.scann.search` and `torch.ops.scann.search_batched` in a C++
library built against LibTorch's **stable ABI** only (targeting 2.10), so
one wheel per architecture (`py3-none-manylinux_2_34_x86_64` and
`_aarch64`) works with every torch >= 2.10, CPU, CUDA or ROCm build, and
every Python >= 3.10. scann-core and all of its dependencies are linked
into it statically and kept private: it exports no symbols, and imports
only the stable C functions of `libtorch_cpu.so` and the C/C++ runtime
(`torch_op_symbols` checks this). With it installed, `scann.torch` uses it
without any change to your code; its version must be scann-core's (pip
installs the matching one; a mismatched or broken install falls back to the
Python backend with a warning).

What it adds:

* **`torch.export` and AOTInductor.** The index is in the `Searcher`'s
  buffers, so an exported program of a model that searches carries it (as
  constants), and runs in another process with only the saved file:

  ```python
  ep = torch.export.export(retriever, (features,),
                           dynamic_shapes={"features": {0: torch.export.Dim("n")}})
  torch.export.save(ep, "retriever.pt2")

  # elsewhere
  import torch, scann.torch     # registers the ops
  retriever = torch.export.load("retriever.pt2").module()
  indices, distances = retriever(features)
  ```

  AOTInductor works the same way
  (`torch._inductor.aoti_compile_and_package(ep)`, then
  `torch._inductor.aoti_load_package(path)` after `import scann.torch`).
  A C++ program running the package without Python loads the op library
  first (`dlopen` of `scann_torch_ops.library_path`, the installed
  `_scann_torch_ops.so`), which needs only libtorch (that path isn't
  tested; the tests load packages from Python). With torch
  2.10, `aoti_load_package()` in a fresh process also needs
  `import torch._inductor.codecache` first (a torch bug, fixed in 2.11).
  For NVIDIA GPUs, Inductor needs a CUDA toolkit's headers; see
  [AOTInductor on NVIDIA GPUs](#aotinductor-on-nvidia-gpus).
* **The index in `state_dict()`**, as three buffers, `index_data` (uint8:
  the files `serialize()` writes, concatenated), `index_offsets` (int64)
  and `index_names` (uint8). `load_state_dict()` replaces the index with
  the state_dict's (another index, or the same one in another process; the
  buffers are replaced, not copied into, so their sizes may differ). A
  state_dict without the index loads with `strict=True` and keeps the
  module's index: one saved with scann-core 0.2.0 or the Python backend,
  or with `searcher.index_in_state_dict = False` (set it to keep large
  indexes out of frequent training checkpoints). A Python-backend
  `Searcher` accepts a native state_dict's index too. Docids in the state
  are unpickled as plain data only (strings, numbers, lists, ...), so a
  state_dict loaded with `torch.load(weights_only=True)` can't run code.
* **Pickling.** `pickle`, `copy.deepcopy` and `torch.save(model)` of a
  model holding a native `Searcher` work (the index goes with it; loading
  with `torch.load` needs `weights_only=False`, as for any pickled model).
  Unpickled where the native backend isn't available (or with
  `SCANN_TORCH_BACKEND=python`), it becomes a Python-backend `Searcher`
  with the same index.

How it works: every search passes the index tensors and a name (a random
`shared_name` per `Searcher`, a constant in exported programs) to the op.
The op builds the ScaNN searcher from the tensors on first use
(`LoadArtifactsFromMemory`), and caches it by that name and a fingerprint
of the tensors: the number, names and sizes of the files, small files
(<= 64 KiB, which includes the config and the manifest) entirely, and of
larger ones the first and last 4 KiB and 64 samples of 64 bytes spread over
the file (as the TensorFlow op). Later searches with
the same name and fingerprint reuse it: a model is built once per process,
however it runs (eagerly, compiled, exported, from several threads at
once). New values under the same name (after `load_state_dict()`, or
`sync()`) are rebuilt. Not detected: a change confined to unsampled bytes
of a large file that keeps every size; `scann.torch` never changes the
buffers in place, it replaces them, so this only matters if you write
into them yourself. Deleting the `Searcher` drops its cached searcher; an
exported program keeps its own for the life of the process
(`scann_torch_ops.stats()` counts cached and built searchers).

#### AOTInductor on NVIDIA GPUs

For a CUDA device, AOTInductor compiles the package's C++ wrapper with
the host compiler against a CUDA toolkit's headers (`cuda_runtime.h` and
what it includes; Triton compiles the kernels, with its own `ptxas`).
Inductor finds the toolkit as `torch.utils.cpp_extension` does:
`$CUDA_HOME` (or `$CUDA_PATH`), else the directory above `nvcc` on `PATH`,
else `/usr/local/cuda`. Without one, `aoti_compile_and_package()` fails
with "CUDA_HOME environment variable is not set" (and `python_torch_native`
skips that case). nvcc itself isn't run, and ROCm builds of torch carry
what they need.

NVIDIA's pip packages are enough; no system install. torch's own
dependencies already install the CUDA runtime's headers, but not the
`crt` headers or CCCL (`nv/target`) they include. Add those, from the
`cuda-toolkit` version your torch pins, to torch's environment:

```sh
# the cuda-toolkit version torch depends on: 13.2.1 for 2.14.0+cu132, 12.8.1 for 2.11.0+cu128
python -c "import importlib.metadata as m; print([r.split(';')[0] for r in m.requires('torch') if r.startswith('cuda-toolkit')])"

# CUDA 13 (torch +cu13x): everything lands in site-packages/nvidia/cu13
pip install "cuda-toolkit[crt,cccl]==13.2.1"
export CUDA_HOME=$(python -c "import nvidia.cu13 as m; print(m.__path__[0])")

# CUDA 12 (torch +cu12x): one directory per package; the crt headers
# come with nvcc's
pip install "cuda-toolkit[nvcc,cccl]==12.8.1"
NV=$(python -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))")
export CUDA_HOME=$NV/cuda_runtime
export CPATH=$NV/cuda_nvcc/include:$NV/cuda_cccl/include
```

A complete toolkit works too, also a newer one than torch's CUDA (13.4
with torch 2.14.0+cu132).

#### Changing the index

Changes through `searcher.searcher` (`upsert`, `delete`, `rebalance`)
reach the buffers at the next eager search, `state_dict()` or pickle: the
index is serialized into them again, and the op rebuilds its searcher.
Compiled and exported code runs no Python there, so it searches the
buffers as they were; call `searcher.sync()` after changing the index and
before running it. Calls on the raw pybind module
(`searcher.searcher.searcher`) aren't seen at all. For many small changes,
batch them: each sync serializes the whole index.

#### Memory

The native backend holds the index twice: the buffers (the serialized
index) and the searcher the op builds from them. `builder()`,
`create_searcher()` and `load_searcher()` drop their pybind searcher once
the index is in the buffers; `searcher.searcher` loads one from the buffers
when first used, and keeps it (for changes), a third copy. A `Searcher`
made around an existing pybind searcher (`Searcher(p)`,
`Searcher.from_pybind(p)`) shares it, as on the Python backend. Measured
(RSS after `malloc_trim`, Python 3.14, torch 2.14) for a 100,000 × 64 tree +
AH + reorder index:

| | Python backend | native backend |
|---|---|---|
| after `build()` | +35.4 MB (the pybind searcher) | +35.3 MB (buffers: 28.4 MB) |
| after the first search | +36.1 MB | +69.3 MB (+ the op's searcher) |
| after using `searcher.searcher` | | +101.7 MB (+ a pybind searcher) |

#### Limits of the native backend

* Checked with an op library built against torch 2.10.0's headers:
  `python_torch` (both backends), `python_torch_native` and the example
  pass with each of these, and AOTInductor packages compile and load in a
  fresh process on each device listed:

  | torch | Python | GPU | AOTInductor |
  |---|---|---|---|
  | 2.10.0 (CPU) | 3.14 | | CPU |
  | 2.11.0+cu128 | 3.13 | RTX 3060 Ti | CPU, CUDA (pip CUDA 12.8.1 headers) |
  | 2.14.0+cu132 | 3.14 | RTX 3060 Ti | CPU, CUDA (pip CUDA 13.2.1 headers; also the 13.4 toolkit) |
  | 2.14.0+rocm7.14 | 3.12 | Radeon RX 7600 XT | CPU, ROCm |

  CUDA needs a toolkit's headers (see
  [AOTInductor on NVIDIA GPUs](#aotinductor-on-nvidia-gpus)).
* The index stays in host memory; the searches run on the CPU (as on the
  Python backend). The op accepts index tensors on a GPU but copies them
  to host memory on every call.
* `torch.compile(mode="reduce-overhead")`: the ops are marked as unsafe to
  capture in CUDA graphs. LibTorch's stable ABI can't define an op with
  tags, so `scann_torch_ops` adds the tag to the op objects in Python,
  where Inductor reads it.
* Exporting captures the `Searcher`'s index at export time; later changes
  to the `Searcher` don't reach saved programs.

### Limits

* **Python backend: no `torch.export` (or AOTInductor), no pickling.** An
  exported program is meant to run without the Python objects that made
  it, but the Python backend's op finds its searcher through a handle into
  a registry of the current process, so the program would hold the handle,
  not the index. Exporting a model that searches therefore raises (strict
  and non-strict export) with a message saying so; install
  `scann-core-torch` for the native backend. A graph that still reaches
  another process holding the op (for example through `torch.jit.trace`)
  fails there with "no searcher with handle ...": handles are random,
  never reused across processes. `pickle`, `copy.deepcopy` and
  `torch.save(model)` raise `TypeError`, and `model.state_dict()` doesn't
  include the index: save it with `serialize()` and load it with
  `load_searcher()`.
* **The default k is read once**, when the `Searcher` is made (or its
  index loaded with `load_state_dict()`). After
  `searcher.searcher.rebalance(config)` with another `num_neighbors`, pass
  `final_num_neighbors`, or make a new `Searcher`.
* Deleting the `Searcher` (and whatever holds it) frees its handle (Python
  backend) or its cached searcher (native backend); the pybind searcher is
  freed once nothing else refers to it. Compiled code doesn't keep it
  alive.

### Overhead

Median time per call on an AMD Threadripper PRO 7975WX with an RTX 3060 Ti
(torch 2.14.0+cu132, Python 3.14), for a 100,000 × 64 tree + AH + reorder
index, against the pybind searcher called directly with the same data (a
numpy array for CPU rows, the CUDA tensor for GPU rows, which
`scann_ops_pybind` copies to host memory too), with the Python backend:

| | pybind | `scann.torch` | compiled (`torch.compile`) |
|---|---|---|---|
| 1 query, `search`, CPU | 9.8 µs | 20.1 µs | 44.6 µs |
| 1 query, `search_batched`, CPU | 12.3 µs | 24.1 µs | 58.1 µs |
| 1,000 queries, `search_batched`, CPU | 11.01 ms | 11.06 ms | 11.17 ms |
| 1,000 queries, `search_batched_parallel`, CPU | 0.77 ms | 0.87 ms | 0.96 ms |
| 1 query, `search`, GPU queries | 20.6 µs | 59.6 µs | 104 µs |
| 1,000 queries, `search_batched`, GPU queries | 11.24 ms | 11.25 ms | 11.43 ms |
| 1,000 queries, `search_batched_parallel`, GPU queries | 0.88 ms | 1.00 ms | 1.13 ms |

Eagerly, a call costs about 10 µs more on the CPU (the custom op's
dispatch; converting the results takes 4 µs for 1,000 queries), and about
40 µs more with GPU queries (copying the results back, and waiting for the
GPU). With 1,000 queries the parallel search took about 0.1 ms (13%) longer
through `scann.torch`, consistently, though the conversions account for only
a few µs of it; the serial one was within noise. A compiled function that
only searches adds its own per-call cost (guards, the compiled wrapper), 25
to 45 µs for one query and 0.1 to 0.2 ms for 1,000; in a real model that
cost is shared with the rest of the forward pass. When latency matters and
the model doesn't need the search inside it, call the pybind searcher on
the model's output.

The native backend, same machine and index, measured in two runs each
alongside the Python backend (the machine was shared with another build,
so the 1,000-query rows vary by about ±10% between runs):

| | Python backend | native backend |
|---|---|---|
| 1 query, `search`, CPU | 32.0 / 32.4 µs | 24.8 / 26.7 µs |
| 1 query, `search_batched`, CPU | 38.7 / 38.9 µs | 28.3 / 30.9 µs |
| 1,000 queries, `search_batched`, CPU | 11.64 / 11.74 ms | 11.56 / 12.64 ms |
| 1,000 queries, `search_batched_parallel`, CPU | 0.86 / 0.82 ms | 0.81 / 0.95 ms |
| 1 query, `search`, compiled | 64.0 / 65.7 µs | 55.2 / 54.4 µs |
| 1 query, `search`, GPU queries | 73.0 / 73.1 µs | 54.8 / 54.2 µs |
| 1,000 queries, `search_batched_parallel`, GPU queries | 1.10 / 0.91 ms | 1.08 / 1.09 ms |

A call costs about 8 to 20 µs less on the native backend (no Python in the
op, no numpy conversions); each call also fingerprints the index tensors
(included above). For 1,000 queries the search dominates and the backends
are within noise of each other.

Checked with torch 2.10.0 (CPU, Python 3.12), 2.11.0+cu128 (Python 3.13,
RTX 3060 Ti), 2.14.0+cu132 (Python 3.14, RTX 3060 Ti) and 2.14.0+rocm7.14
(Python 3.12, Radeon RX 7600 XT): `python_torch` and the example pass on
each.

## LangChain

`langchain_community.vectorstores.ScaNN` works on scann-core without
changes. It uses `builder`, `create_searcher`, `search_batched`,
`serialize` and `load_searcher`, all with upstream's signatures.

Checked with langchain-community 0.4.2 and langchain-core 1.6.5: every
operation the store supports returns the same documents and scores as the
upstream wheel (`scann==1.4.2`): building with either distance strategy
and with `normalize_L2`, a custom `scann_config`, metadata filters, score
thresholds, relevance scores, and `save_local`/`load_local`. The
`python_langchain` test checks the results against an exact search on
every CI run.

```python
from langchain_community.vectorstores import ScaNN
from langchain_community.vectorstores.utils import DistanceStrategy

store = ScaNN.from_texts(texts, embeddings, metadatas=metadatas,
                         distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
docs = store.similarity_search("query", k=5)
store.save_local("index_dir")
store = ScaNN.load_local("index_dir", embeddings,
                         allow_dangerous_deserialization=True)
```

Things to know. These are properties of LangChain's store; upstream
ScaNN behaves the same way:

* **It builds a brute-force index** unless you pass `scann_config`. That's
  exact but slow for large collections. For a tree + AH index, make the
  config with scann's builder and pass it in:

  ```python
  import numpy as np, scann
  vectors = np.array(embeddings.embed_documents(texts), dtype=np.float32)
  config = (scann.scann_ops_pybind.builder(vectors, 10, "dot_product")
            .tree(num_leaves=1000, num_leaves_to_search=50,
                  training_sample_size=len(texts))
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(100)
            .create_config())
  store = ScaNN.from_texts(texts, embeddings, scann_config=config,
                           distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
  ```

  Match the builder's distance (`dot_product` or `squared_l2`) to the
  store's `distance_strategy`. [tutorial/](tutorial/README.md) explains
  how to choose the tree and AH settings.
* **No updates.** The store's `add_texts` and `delete` raise
  `NotImplementedError`; rebuild the store to change its contents.
  (scann-core itself supports updates; see
  [tutorial part 6](tutorial/06-updating.md).)
* **Asking for more results than there are documents** returns document 0
  again for the missing ones, with a NaN score. ScaNN pads short results
  with index 0 and a NaN distance, and the store only skips index -1.
  Drop results whose score is NaN.
* **Scores are squared L2 distances** for the default
  `EUCLIDEAN_DISTANCE` strategy, and dot products for
  `MAX_INNER_PRODUCT`. LangChain's relevance scores
  (`similarity_search_with_relevance_scores`) assume distances between
  unit-length vectors; with other vectors they fall outside [0, 1] and
  LangChain warns. Use normalized embeddings, or `normalize_L2=True`.
* **`load_local` unpickles** the docstore, hence
  `allow_dangerous_deserialization=True`. Only load folders you trust.
* **langchain-community is being retired.** Its maintainers are moving
  integrations into separate packages, and ScaNN doesn't have one yet. The
  store keeps working as long as langchain-community installs.
