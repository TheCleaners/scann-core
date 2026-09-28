# Using scann-core with other libraries

scann-core installs the same `scann` Python module as the upstream
`scann` wheel, with the same `scann_ops_pybind` API. Libraries written
against upstream ScaNN work with it unchanged. This page covers inputs
from other array libraries, the libraries that have been checked, and what
to know when using them.

TensorFlow has its own page: [tensorflow.md](tensorflow.md).

## Installing in place of `scann`

Both packages install a top-level `scann` module, so only one of them can
be installed in an environment. Remove the upstream wheel first:

```sh
pip uninstall scann
pip install scann-core      # release candidates: pip install --pre scann-core
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
(`python_torch_input`, with CUDA tensors when a GPU is present).

For searching inside TensorFlow graphs, see [tensorflow.md](tensorflow.md).

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
