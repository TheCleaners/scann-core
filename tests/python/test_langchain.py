# Copyright 2026 Elias Benali (@ebenali) and TheCleaners.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LangChain's ScaNN vector store (langchain_community.vectorstores.ScaNN)
running on scann-core.

The store does `import scann` and uses scann_ops_pybind's builder,
create_searcher, search_batched, serialize and load_searcher, so scann-core
serves it unchanged. This checks its results against an exact search in
numpy, for both distance strategies, normalize_L2, a custom tree + AH
config, metadata filters, save/load and re-saving into the same folder.

Without langchain-community the test exits with 77, which ctest reports as
skipped (SKIP_RETURN_CODE), unless SCANN_TEST_REQUIRE_LANGCHAIN is set.
"""

import math
import os
import sys
import tempfile
import warnings

import numpy as np

SKIP = 77
DIM = 32


def main():
  try:
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")  # langchain-community's sunset notice
      from langchain_community.vectorstores import ScaNN
      from langchain_community.vectorstores.utils import DistanceStrategy
      from langchain_core.embeddings import DeterministicFakeEmbedding
  except ImportError as e:
    if os.environ.get("SCANN_TEST_REQUIRE_LANGCHAIN"):
      raise
    print(f"SKIPPED: langchain-community is not importable ({e})")
    return SKIP
  import scann

  emb = DeterministicFakeEmbedding(size=DIM)
  n = 300
  texts = [f"document number {i} about topic {i % 7}" for i in range(n)]
  metas = [{"topic": i % 7, "i": i} for i in range(n)]
  ids = [f"id{i}" for i in range(n)]
  vecs = np.array(emb.embed_documents(texts), dtype=np.float32)
  queries = [texts[5], texts[123], "topic 3", "something else entirely"]

  def exact(query, strategy, normalize=False, k=5, allowed=None):
    """(doc index, score) pairs as the store should return them."""
    q = np.array(emb.embed_query(query), dtype=np.float32)
    db = vecs
    if normalize:
      q = q / np.linalg.norm(q)
      db = db / np.linalg.norm(db, axis=1, keepdims=True)
    if strategy == "ip":
      # scann_ops_pybind returns dot products for dot_product searchers.
      scores = db @ q
      order = np.argsort(-scores, kind="stable")
    else:
      scores = ((db - q) ** 2).sum(axis=1)
      order = np.argsort(scores, kind="stable")
    if allowed is not None:
      order = [i for i in order if allowed(i)]
    return [(int(i), float(scores[i])) for i in order[:k]]

  def check(store, strategy, what, normalize=False):
    for q in queries:
      got = store.similarity_search_with_score(q, k=5)
      want = exact(q, strategy, normalize)
      got_ids = [d.metadata["i"] for d, _ in got]
      assert got_ids == [i for i, _ in want], (what, q, got_ids, want)
      np.testing.assert_allclose([float(s) for _, s in got],
                                 [s for _, s in want], rtol=1e-4, atol=1e-4,
                                 err_msg=f"{what}: {q}")
    # Metadata filter: the store over-fetches (fetch_k) and filters.
    got = store.similarity_search_with_score(
        queries[2], k=4, filter={"topic": 3}, fetch_k=n)
    want = exact(queries[2], strategy, normalize, k=4,
                 allowed=lambda i: i % 7 == 3)
    assert [d.metadata["i"] for d, _ in got] == [i for i, _ in want], what
    # A document's own text finds it first.
    top = store.similarity_search(texts[42], k=1)[0]
    assert top.metadata["i"] == 42 and top.page_content == texts[42], what

  s_l2 = ScaNN.from_texts(texts, emb, metadatas=metas, ids=ids)
  check(s_l2, "l2", "squared L2 (default)")
  s_ip = ScaNN.from_texts(texts, emb, metadatas=metas, ids=ids,
                          distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
  check(s_ip, "ip", "max inner product")
  s_norm = ScaNN.from_texts(texts, emb, metadatas=metas, ids=ids,
                            normalize_L2=True)
  check(s_norm, "l2", "normalize_L2", normalize=True)

  # A tree + AH + reorder config from scann's own builder, searching every
  # leaf and reordering exactly: same results as exact search.
  cfg = (scann.scann_ops_pybind.builder(vecs, 10, "dot_product")
         .tree(num_leaves=12, num_leaves_to_search=12, training_sample_size=n)
         .score_ah(2, anisotropic_quantization_threshold=0.2).reorder(n)
         .create_config())
  s_cfg = ScaNN.from_texts(texts, emb, metadatas=metas, ids=ids,
                           scann_config=cfg,
                           distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
  check(s_cfg, "ip", "tree + AH config")

  # Relevance scores (LangChain maps distances to [0, 1]-ish scores).
  # (For squared L2 on vectors that aren't unit length, LangChain's
  # relevance mapping goes below 0 and it warns; see docs/integrations.md.)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rel = s_l2.similarity_search_with_relevance_scores(texts[5], k=3)
  assert rel[0][0].metadata["i"] == 5 and math.isclose(rel[0][1], 1.0,
                                                       abs_tol=1e-4), rel[0]

  # save_local / load_local, and saving another index into the same folder
  # (scann-core's serialize replaces the previous index atomically).
  with tempfile.TemporaryDirectory() as d:
    s_l2.save_local(d)
    loaded = ScaNN.load_local(d, emb, allow_dangerous_deserialization=True)
    check(loaded, "l2", "loaded")
    s_ip.save_local(d)
    loaded = ScaNN.load_local(
        d, emb, allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    check(loaded, "ip", "re-saved into the same folder")
    leftovers = [f for f in os.listdir(os.path.join(d, "index.scann"))
                 if f.startswith(".scann_staging")]
    assert not leftovers, leftovers

  # More results than documents: ScaNN pads short rows with index 0 and a
  # NaN distance (upstream too); the store only skips index -1, so it
  # returns document 0 again with NaN scores. Pinned here so a change on
  # either side is noticed; docs/integrations.md tells users to drop NaNs.
  small = ScaNN.from_texts(texts[:3], emb, metadatas=metas[:3], ids=ids[:3])
  got = small.similarity_search_with_score(texts[0], k=6)
  real = [(d.metadata["i"], s) for d, s in got if not math.isnan(float(s))]
  assert sorted(i for i, _ in real) == [0, 1, 2], got
  padded = [(d.metadata["i"], s) for d, s in got if math.isnan(float(s))]
  assert padded == [] or all(i == 0 for i, _ in padded), got

  # The store itself doesn't support updates (LangChain raises).
  for f in (lambda: s_l2.add_texts(["new"]), lambda: s_l2.delete(["id1"])):
    try:
      f()
      raise AssertionError("LangChain's ScaNN store accepted an update")
    except NotImplementedError:
      pass

  print(f"PASSED ({len(padded)} padded results for k > n)")
  return 0


if __name__ == "__main__":
  sys.exit(main())
