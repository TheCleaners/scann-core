# scann-core-torch

The native backend of [scann-core](https://github.com/TheCleaners/scann-core)'s
`scann.torch`: C++ PyTorch ops (`torch.ops.scann.search`,
`torch.ops.scann.search_batched`) that search a ScaNN index held in a
module's buffers.

```sh
pip install scann-core-torch    # installs the matching scann-core, and torch>=2.10
```

With it installed, `scann.torch.Searcher` keeps the index in its buffers
and searches with these ops, with the same API and results as without it.
What it adds:

* `torch.export` and AOTInductor programs of a model that searches carry
  the index and run in other processes;
* `state_dict()` contains the index (and `load_state_dict()` restores it);
* a model holding a `Searcher` can be pickled (`torch.save(model)`).

```python
import torch
import scann.torch as scann_torch

searcher = scann_torch.builder(embeddings, 10, "dot_product").score_ah(2).reorder(100).build()
assert scann_torch.backend() == "native"

ep = torch.export.export(model_that_searches, (features,))
torch.export.save(ep, "retriever.pt2")    # the index is inside
```

The ops are built against LibTorch's stable ABI only, so the one wheel per
architecture (`py3-none-manylinux_2_34_x86_64` / `_aarch64`) works with
any torch >= 2.10 (CPU, CUDA and ROCm builds) and any Python >= 3.10.
Searches run on the CPU; queries on a GPU are copied to host memory and
the results back.

Documentation:
[docs/integrations.md](https://github.com/TheCleaners/scann-core/blob/main/docs/integrations.md#scanntorch-searching-from-pytorch-models).
Derived from Google's [ScaNN](https://github.com/google-research/google-research/tree/master/scann);
not an official Google product. Apache-2.0.
