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

"""builder(...).l2_as_dot_product(): squared L2 through an inner-product index.

- The builder's config: l2_as_dot_product, a dot-product tree and AH over
  dimensionality + 1 dimensions (residual AH, AVQ and SOAR allowed); its
  errors (dot_product, spherical, truncate, autopilot, bad scale).
- search / search_batched / search_batched_parallel return the ids of the
  manual recipe (docs/tuning.md: a dot_product index on
  [x, (c - |x|^2) / (2s)], searched with [q, s]), with and without SOAR,
  and |q|^2 + c - 2 q'.x' as distances, which are the exact squared L2
  distances; recall against brute force.
- serialize() / load_searcher() keep the reduction; upsert (including a
  point with |x|^2 far above c), delete and rebalance, before and after
  loading, with docids; an index grown from empty.
- scann.torch.Searcher and scann.tf on such an index (each backend that is
  available; skipped without PyTorch / TensorFlow) return the pybind
  searcher's results.

Run with scann-core's build/python on PYTHONPATH:
  PYTHONPATH=build/python python tests/python/test_l2_as_dot_product.py
"""

import importlib.util
import os
import shutil
import tempfile

import numpy as np
import scann

D, N, K = 16, 4000, 10
SCALE = 8.0  # ~0.4x the data's RMS norm, as the default
FAILURES = []


def check(cond, what):
  if not cond:
    FAILURES.append(what)
    print("FAIL:", what)


def expect_error(fn, contains, what):
  try:
    fn()
  except (ValueError, RuntimeError) as e:  # pylint: disable=broad-except
    check(contains in str(e), f"{what}: error {e!r} lacks {contains!r}")
    print(f"ok (error as intended): {what} -> {e}")
    return
  check(False, f"{what}: no error")


def data(n, seed, sigma=4.0, mean=3.0):
  # Multiples of 1/8: |x|^2 is exact in float64 in any order, so the manual
  # augmentation below is bit-identical to scann-core's.
  rng = np.random.default_rng(seed)
  return (np.round(rng.normal(mean, sigma, (n, D)) * 8) / 8).astype(np.float32)


def sq_norms(x):
  return np.einsum("ij,ij->i", x, x, dtype=np.float64)


def augment(x, c, s):
  return np.hstack([x, ((c - sq_norms(x)) / (2 * s))[:, None]]).astype(
      np.float32)


def exact_sq_l2(q, x):
  return ((q.astype(np.float64)[:, None, :] - x.astype(np.float64)[None]) ** 2
         ).sum(-1)


def recall(ids, q, x):
  d = exact_sq_l2(q, x)
  kth = np.sort(d, axis=1)[:, K - 1]
  got = np.take_along_axis(d, ids[:, :K].astype(np.int64), 1)
  return float(np.mean(got <= kth[:, None] + 1e-6))


def check_distances(ids, dist, q, x, c, what):
  d = np.take_along_axis(exact_sq_l2(q, x), ids.astype(np.int64), 1)
  tol = 1e-5 * (sq_norms(q)[:, None] + sq_norms(x)[ids.astype(np.int64)] +
                abs(c) + 1)
  check(np.all(np.abs(dist - d) <= tol),
        f"{what}: distances aren't squared L2 (max error "
        f"{np.abs(dist - d).max()})")


def configure(b, soar=False):
  return (b.tree(40, 5, random_init=False, avq=2.5, quantize_centroids=True,
                 **({"soar_lambda": 1.0} if soar else {}))
          .score_ah(3, anisotropic_quantization_threshold=8.0)
          .reorder(100))


def search_all(s, q, leaves=5, pre=40):
  single = [s.search(v, K, pre, leaves) for v in q]
  return (np.array([i for i, _ in single]), np.array([d for _, d in single]),
          *s.search_batched(q, K, pre, leaves_to_search=leaves),
          *s.search_batched_parallel(q, K, pre, leaves_to_search=leaves,
                                     batch_size=16))


def test_builder_config():
  x = data(100, 0)
  b = scann.scann_ops_pybind.builder(x, K, "squared_l2")
  text = configure(b, soar=True).l2_as_dot_product(scale=2.0).create_config()
  for want in ("l2_as_dot_product { scale: 2.0 }", "input_dim: 17",
               "use_residual_quantization: True", "avq: 2.5",
               "TWO_CENTER_ORTHOGONALITY_AMPLIFIED",
               'query_tokenization_distance_override {distance_measure: '
               '"DotProductDistance"}', '"SquaredL2Distance"}'):
    check(want in text, f"builder config lacks {want!r}:\n{text}")

  def build(distance, fn):
    return lambda: fn(scann.scann_ops_pybind.builder(x, K, distance)
                      ).create_config()
  expect_error(build("dot_product", lambda b: b.score_brute_force()
                     .l2_as_dot_product()), "squared_l2", "dot_product")
  expect_error(build("squared_l2", lambda b: b.tree(4, 1, spherical=True)
                     .score_brute_force().l2_as_dot_product()),
               "spherical", "spherical")
  expect_error(build("squared_l2", lambda b: b.truncate(8).tree(4, 1)
                     .score_brute_force().l2_as_dot_product()),
               "truncate", "truncate")
  expect_error(build("squared_l2", lambda b: b.autopilot()
                     .l2_as_dot_product()), "autopilot", "autopilot")
  expect_error(build("squared_l2", lambda b: b.score_brute_force()
                     .l2_as_dot_product(scale=0)), "scale", "scale 0")
  expect_error(build("squared_l2", lambda b: b.tree(4, 1, avq=1.0)
                     .score_brute_force()), "AVQ", "avq without it")


def test_against_manual_recipe(soar):
  x, q = data(N, 1), data(64, 2)
  c = float(sq_norms(x).mean())
  what = "SOAR" if soar else "tree"
  opt = configure(scann.scann_ops_pybind.builder(x, K, "squared_l2"),
                  soar).l2_as_dot_product(scale=SCALE, center=c).build()
  man = configure(scann.scann_ops_pybind.builder(augment(x, c, SCALE), K,
                                                 "dot_product"), soar).build()
  qa = np.hstack([q, np.full((len(q), 1), SCALE, np.float32)])
  got, want = search_all(opt, q), search_all(man, qa)
  qn = sq_norms(q)[:, None]
  for mode, gi, gd, wi, wd in zip(("search", "batched", "parallel"),
                                  got[0::2], got[1::2], want[0::2],
                                  want[1::2]):
    check(np.array_equal(gi, wi), f"{what} {mode}: ids differ from the manual "
          "recipe")
    conv = np.maximum(qn + c - 2 * wd.astype(np.float64), 0)
    check(np.allclose(gd, conv, rtol=1e-6, atol=1e-6 * (qn.max() + c)),
          f"{what} {mode}: distances aren't |q|^2 + c - 2 q'.x'")
    check_distances(gi, gd, q, x, c, f"{what} {mode}")
  ids, _ = opt.search_batched(q, K, N, leaves_to_search=40)
  check(recall(ids, q, x) == 1.0, f"{what}: exhaustive search isn't exact")
  r = recall(opt.search_batched(q, K, 300, leaves_to_search=40)[0], q, x)
  print(f"{what}: recall@{K}, all leaves, 300 candidates: {r:.4f}")
  check(r >= 0.9, f"{what}: recall {r}")
  return opt, x, q, c


def test_persistence_and_mutation():
  x, q = data(N, 3), data(64, 4)
  docids = [f"d{i}" for i in range(N)]
  s = configure(scann.scann_ops_pybind.builder(x, K, "squared_l2")
               ).l2_as_dot_product().build(docids=docids)
  c = sq_norms(x).mean()
  cfg = s.config()
  check("l2_as_dot_product" in cfg and "center" in cfg and "scale" in cfg,
        f"config() lacks the parameters:\n{cfg}")
  tmp = tempfile.mkdtemp(prefix="scann_l2_as_dot_product_")
  try:
    s.serialize(tmp, relative_path=True)
    loaded = scann.scann_ops_pybind.load_searcher(tmp)
    check(loaded.config() == cfg, "config() changed by serialize + load")
    a, b = s.search_batched(q, K, 40, 5), loaded.search_batched(q, K, 40, 5)
    check(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]),
          "loaded searcher's results differ")

    # Mutations with docids, after loading.
    store = dict(zip(docids, x))
    far = data(1, 5, sigma=60, mean=100)
    new = data(300, 6)
    loaded.upsert(["far"] + [f"n{i}" for i in range(300)],
                  np.vstack([far, new]))
    store["far"] = far[0]
    store.update({f"n{i}": v for i, v in enumerate(new)})
    upd = data(50, 7)
    upd_ids = [f"d{i}" for i in range(0, 500, 10)]
    loaded.upsert(upd_ids, upd)
    store.update(zip(upd_ids, upd))
    dels = [f"d{i}" for i in range(1, 400, 7)] + ["n3"]
    loaded.delete(dels)
    for d in dels:
      del store[d]
    check(loaded.size() == len(store), "size after mutations")

    def self_search(searcher, label, pre):
      keys = list(store)[::13] + (["far"] if "far" in store else [])
      vecs = np.array([store[k] for k in keys])
      ids, dist = searcher.search_batched(vecs, K, pre, leaves_to_search=40)
      found = sum(ids[i][0] == k for i, k in enumerate(keys))
      check(found == len(keys),
            f"{label}: {len(keys) - found} of {len(keys)} vectors aren't "
            f"their own nearest neighbour ({pre} candidates)")
      check(np.all(np.abs(dist[:, 0]) <= 1e-4 * (sq_norms(vecs) + c)),
            f"{label}: self distances not ~0")
      # With few candidates, except the far point: its extra coordinate is
      # far outside what the AH codebooks were trained on.
      normal = [i for i, k in enumerate(keys) if k != "far"]
      ids, _ = searcher.search_batched(vecs[normal], K, 50,
                                       leaves_to_search=40)
      found = sum(ids[j][0] == keys[i] for j, i in enumerate(normal))
      check(found >= 0.97 * len(normal),
            f"{label}: only {found} of {len(normal)} found with 50 "
            "candidates")
    self_search(loaded, "after mutations", len(store))
    far_ids, far_d = loaded.search(far[0], K, 1000, 40)
    print(f"far point |x|^2 = {sq_norms(far)[0]:.0f} (c = {c:.0f}): "
          f"found {far_ids[0]!r} at {far_d[0]:g}")
    # Retraining on such an outlier degrades the quantization of everything
    # (with plain squared_l2 too); delete it first.
    loaded.delete(["far"])
    del store["far"]

    loaded.rebalance()
    check("l2_as_dot_product" in loaded.config(), "rebalance dropped it")
    self_search(loaded, "after rebalance", len(store))
    loaded.rebalance(configure(scann.scann_ops_pybind.builder(
        x, K, "squared_l2")).l2_as_dot_product().create_config())
    self_search(loaded, "after rebalance with a builder config", len(store))
    expect_error(lambda: loaded.rebalance(configure(
        scann.scann_ops_pybind.builder(x, K, "squared_l2")
    ).l2_as_dot_product(scale=1.0).create_config()), "can't change",
                 "rebalance with another scale")

    loaded.serialize(tmp)
    again = scann.scann_ops_pybind.load_searcher(tmp)
    self_search(again, "reloaded after mutations", len(store))
    ids, _ = again.search_batched(q, K, len(store), leaves_to_search=40)
    keys = list(store)
    vals = np.array([store[k] for k in keys])
    idx = np.array([[keys.index(d) for d in row] for row in ids])
    check(recall(idx, q, vals) == 1.0, "reloaded: exhaustive search not exact")
  finally:
    shutil.rmtree(tmp)


def test_grow_from_empty():
  empty = np.zeros((0, D), np.float32)
  s = (scann.scann_ops_pybind.builder(empty, K, "squared_l2")
       .score_brute_force().l2_as_dot_product(scale=50).build(docids=[]))
  x, q = data(500, 8), data(32, 9)
  s.upsert([str(i) for i in range(500)], x)
  ids, dist = s.search_batched(q)
  idx = np.array([[int(d) for d in row] for row in ids])
  check(recall(idx, q, x) == 1.0, "grown from empty: brute force not exact")
  check_distances(idx, dist, q, x, 0.0, "grown from empty")
  expect_error(lambda: scann.scann_ops_pybind.builder(
      empty, K, "squared_l2").score_brute_force().l2_as_dot_product().build(),
               "needs a scale", "empty dataset without a scale")


def test_torch(searcher, x, q, c):
  if importlib.util.find_spec("torch") is None:
    print("skipped: scann.torch (no PyTorch)")
    return
  import torch  # pylint: disable=g-import-not-at-top
  import scann.torch as scann_torch  # pylint: disable=g-import-not-at-top
  want_i, want_d = searcher.search_batched(q, K, 40, leaves_to_search=5)
  tmp = tempfile.mkdtemp(prefix="scann_l2_as_dot_product_torch_")
  try:
    searcher.serialize(tmp)
    for backend in ("python", "native"):
      if backend == "native" and "native" not in _torch_backends(scann_torch):
        print("skipped: scann.torch native backend (no scann_torch_ops)")
        continue
      for label, m in (("wrapped", scann_torch.Searcher(searcher, backend)),
                       ("loaded", scann_torch.load_searcher(tmp,
                                                            backend=backend))):
        what = f"scann.torch {backend} {label}"
        i, d = m.search_batched(torch.from_numpy(q), K, 40, 5)
        check(np.array_equal(i.numpy(), want_i.astype(np.int64)) and
              np.allclose(d.numpy(), want_d), f"{what}: search_batched")
        i, d = m.search(torch.from_numpy(q[0]), K, 40, 5)
        check(np.array_equal(i.numpy(), want_i[0].astype(np.int64)),
              f"{what}: search")
        i, d = m.search_batched_parallel(torch.from_numpy(q), K, 40, 5)
        check(np.array_equal(i.numpy(), want_i.astype(np.int64)),
              f"{what}: search_batched_parallel")
        check_distances(i.numpy(), d.numpy(), q, x, c, what)
        print(f"ok: {what}")
  finally:
    shutil.rmtree(tmp)


def _torch_backends(scann_torch):
  return ("python", "native") if scann_torch.backend() == "native" or (
      scann_torch._load_native() is not None) else ("python",)  # pylint: disable=protected-access


def test_tf(searcher, x, q, c):
  if importlib.util.find_spec("tensorflow") is None:
    print("skipped: scann.tf (no TensorFlow)")
    return
  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
  import scann.tf as scann_tf  # pylint: disable=g-import-not-at-top
  want_i, want_d = searcher.search_batched(q, K, 40, leaves_to_search=5)
  tmp = tempfile.mkdtemp(prefix="scann_l2_as_dot_product_tf_")
  try:
    searcher.serialize(tmp)
    for backend in scann_tf.available_backends():
      mod = scann_tf.get_backend(backend)
      for label, s in (("from_pybind", mod.from_pybind(searcher)),
                       ("loaded", mod.load_searcher(tmp))):
        what = f"scann.tf {backend} {label}"
        i, d = s.search_batched(q, K, 40, 5)
        check(np.array_equal(i.numpy(), want_i.astype(np.int32)) and
              np.allclose(d.numpy(), want_d), f"{what}: search_batched")
        i, d = s.search(q[0], K, 40, 5)
        check(np.array_equal(i.numpy(), want_i[0].astype(np.int32)),
              f"{what}: search")
        i, d = s.search_batched_parallel(q, K, 40, 5)
        check(np.array_equal(i.numpy(), want_i.astype(np.int32)),
              f"{what}: search_batched_parallel")
        check_distances(i.numpy(), d.numpy(), q, x, c, what)
        print(f"ok: {what}")
  finally:
    shutil.rmtree(tmp)


def main():
  test_builder_config()
  searcher, x, q, c = test_against_manual_recipe(soar=False)
  test_against_manual_recipe(soar=True)
  test_persistence_and_mutation()
  test_grow_from_empty()
  test_torch(searcher, x, q, c)
  test_tf(searcher, x, q, c)
  if FAILURES:
    raise SystemExit(f"{len(FAILURES)} failure(s)")
  print("PASSED")


if __name__ == "__main__":
  main()
