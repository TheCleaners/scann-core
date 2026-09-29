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

"""builder(...).autopilot(): the tuned rules and upstream's.

- The stanza autopilot() emits for each option, and its errors.
- create_config() for given shapes (no data: unit norms assumed): the
  tuned rules' leaves, leaves to search, block size, threshold and AVQ;
  rules="upstream" gives scann-core 0.2.0's config.
- Built indexes: the tuned rules measure the data (the threshold scales
  with the norms and is recorded; squared L2 with nearly constant norms
  becomes l2_as_dot_product, unless disallowed or far from the origin);
  recall against brute force at the default settings; serialize() /
  load_searcher() and rebalance() keep the config; upsert and delete with
  an ONLINE autopilot index.

Run with scann-core's build/python on PYTHONPATH:
  PYTHONPATH=build/python python tests/python/test_autopilot.py
"""

import math
import re
import shutil
import tempfile

import numpy as np
import scann
from scann.scann_ops.py import scann_builder

FAILURES = []
RT = scann.ReorderType if hasattr(scann, "ReorderType") else scann_builder.ReorderType
IM = scann_builder.IncrementalMode


def check(cond, what):
  if not cond:
    FAILURES.append(what)
    print("FAIL:", what)
  else:
    print("ok:", what)


def expect_error(fn, contains, what):
  try:
    fn()
  except (ValueError, RuntimeError) as e:  # pylint: disable=broad-except
    check(contains in str(e), f"{what}: error {e!r} mentions {contains!r}")
    return
  check(False, f"{what}: expected an error")


def field(cfg, name):
  m = re.search(rf"\b{name}: ([-\w.+]+)", cfg)
  return m.group(1) if m else None


def preview(n, d, k, dist, **kw):
  db = np.broadcast_to(np.float32(0), (n, d))
  return scann.scann_ops_pybind.builder(db, k, dist).autopilot(**kw).create_config()


def test_stanza():
  base = scann_builder.ScannBuilder(np.zeros((10, 4), np.float32), 1, "dot_product")
  tuned = base.autopilot().create_config()
  check("rules: TUNED_V1" in tuned and "reordering_dtype: FLOAT32" in tuned,
        "tuned stanza: rules, float32 reordering")
  base = scann_builder.ScannBuilder(np.zeros((10, 4), np.float32), 1, "squared_l2")
  s = base.autopilot(mode=IM.ONLINE, quantize=RT.INT8,
                     allow_l2_as_dot_product=False).create_config()
  check("incremental_mode: ONLINE" in s and "reordering_dtype: INT8" in s and
        "allow_l2_as_dot_product: false" in s, "tuned stanza with options")
  base = scann_builder.ScannBuilder(np.zeros((10, 4), np.float32), 1, "dot_product")
  up = base.autopilot(rules="upstream").create_config()
  check("rules" not in up and "reordering_dtype: FLOAT32" in up,
        "upstream stanza: 0.2.0's (float32 reordering, no rules)")
  expect_error(lambda: scann_builder.ScannBuilder(
      np.zeros((10, 4), np.float32), 1, "dot_product").autopilot(
          rules="fast").create_config(), "rules", "unknown rules")
  expect_error(lambda: scann_builder.ScannBuilder(
      np.zeros((10, 4), np.float32), 1, "dot_product").autopilot(
          rules="upstream", allow_l2_as_dot_product=False).create_config(),
               "allow_l2_as_dot_product", "allow_l2_as_dot_product with upstream")


def test_previews():
  # (n, d, k, distance) -> leaves, leaves to search, reorder, dims per block,
  # threshold (None: none), avq.
  def t(d):  # the threshold for unit norms
    return 0.2 * min(1.0, 128 / d) ** 0.75
  cases = [
      ((1183514, 100, 10, "dot_product"), (903, 106, 317, 2, 0.2, 2.5)),
      ((1000000, 128, 10, "squared_l2"), (976, 109, 317, 2, None, None)),
      ((1344643, 768, 100, "dot_product"), (1160, 86, 1000, 4, t(768), 2.5)),
      ((1281167, 512, 100, "dot_product"), (1132, 85, 1000, 3, t(512), 2.5)),
      ((100000, 100, 10, "dot_product"), (76, 51, 317, 2, 0.2, 2.5)),
      ((5000000, 64, 10, "squared_l2"), (2236, 137, 317, 2, None, None)),
  ]
  for (n, d, k, dist), (leaves, lts, reorder, dpb, t, avq) in cases:
    cfg = preview(n, d, k, dist)
    what = f"tuned preview {n}x{d} {dist}"
    got_t = field(cfg, "noise_shaping_threshold")
    check(int(field(cfg, "num_children")) == leaves and
          int(field(cfg, "max_spill_centers")) == lts and
          int(field(cfg, "approx_num_neighbors")) == reorder and
          int(field(cfg, "num_dims_per_block")) == dpb and
          (got_t is None if t is None else abs(float(got_t) - t) < 1e-5) and
          (field(cfg, "avq") is None if avq is None else float(field(cfg, "avq")) == avq) and
          "bfloat16" not in cfg,
          what)
  cfg = preview(1183514, 100, 10, "dot_product", quantize=RT.BFLOAT16)
  check(re.search(r"bfloat16 \{\s*enabled: true", cfg) is not None,
        "quantize=BFLOAT16")
  cfg = preview(50000, 100, 10, "dot_product")
  check("brute_force" in cfg and "partitioning" not in cfg,
        "brute force below upstream's cutoff (55,020 at d=100)")
  cfg = preview(1183514, 100, 10, "dot_product", rules="upstream")
  check(field(cfg, "num_children") == "903" and field(cfg, "max_spill_centers") == "106"
        and field(cfg, "noise_shaping_threshold") == "0.2" and field(cfg, "avq") is None
        and "rules" not in cfg, "upstream preview: 0.2.0's config")


def brute_force(x, q, k, dist):
  if dist == "dot_product":
    s = -(q @ x.T)
  else:
    s = (q * q).sum(1)[:, None] - 2 * q @ x.T + (x * x).sum(1)[None, :]
  return np.argsort(s, axis=1)[:, :k]


def recall(s, x, q, k, dist):
  ids = np.stack([s.search(v)[0] for v in q])
  gt = brute_force(x, q, k, dist)
  return np.mean([len(np.intersect1d(a, b)) / k for a, b in zip(ids, gt)])


def clustered(n, d, seed, centers):
  rng = np.random.default_rng(seed)
  lab = rng.integers(0, len(centers), n)
  return (centers[lab] + 0.3 * rng.standard_normal((n, d))).astype(np.float32)


def test_built():
  d, n, k = 200, 30000, 10  # a tree from 27,510 points at d=200
  centers = np.random.default_rng(1).standard_normal((200, d)).astype(np.float32)
  x = clustered(n, d, 2, centers)
  q = clustered(50, d, 3, centers)
  # Constant norms (like SIFT's): the tuned rules use l2_as_dot_product.
  x10 = 10 * x / np.linalg.norm(x, axis=1, keepdims=True)
  q10 = 10 * q / np.linalg.norm(q, axis=1, keepdims=True)

  # squared_l2 with varying norms: plain squared L2.
  s = scann.scann_ops_pybind.builder(x, k, "squared_l2").autopilot().build()
  check(not re.search(r"\bl2_as_dot_product \{", s.config()),
        "squared_l2, varying norms: plain")

  # squared_l2, constant norms: l2_as_dot_product with a recorded threshold.
  x, q = x10, q10
  s = scann.scann_ops_pybind.builder(x, k, "squared_l2").autopilot().build()
  cfg = s.config()
  check(re.search(r"\bl2_as_dot_product \{", cfg) and field(cfg, "avq") == "2.5" and
        "rules: TUNED_V1" in cfg and
        re.search(r"tree_ah \{[^}]*noise_shaping_threshold", cfg, re.S),
        "squared_l2: l2_as_dot_product, recorded threshold")
  r = recall(s, x, q, k, "squared_l2")
  check(r >= 0.9, f"squared_l2 recall at the defaults {r:.4f} >= 0.9")
  ids, dist = s.search(q[0])
  exact = ((x[ids] - q[0]) ** 2).sum(1)
  check(np.allclose(dist, exact, rtol=1e-2, atol=0.05), "squared_l2 distances")
  tmp = tempfile.mkdtemp(dir=".")
  try:
    s.serialize(tmp)
    t = scann.scann_ops_pybind.load_searcher(tmp)
    check(t.config() == cfg, "load_searcher keeps the config")
    check(all(np.array_equal(t.search(v)[0], s.search(v)[0]) for v in q),
          "load_searcher: same results")
  finally:
    shutil.rmtree(tmp)
  s.rebalance()
  check(s.config() == cfg, "rebalance keeps the config")

  s = scann.scann_ops_pybind.builder(x, k, "squared_l2").autopilot(
      allow_l2_as_dot_product=False).build()
  cfg = s.config()
  check(not re.search(r"\bl2_as_dot_product \{", cfg) and
        "allow_l2_as_dot_product: false" in cfg,
        "allow_l2_as_dot_product=False: a plain squared L2 index")

  # dot_product at norm 5: the threshold scales with the norm.
  xn = 5 * x / np.linalg.norm(x, axis=1, keepdims=True)
  qn = q / np.linalg.norm(q, axis=1, keepdims=True)
  s = scann.scann_ops_pybind.builder(xn, k, "dot_product").autopilot().build()
  cfg = s.config()
  t = float(field(cfg, "noise_shaping_threshold"))
  want = 5 * 0.2 * (128 / 200) ** 0.75
  check(abs(t - want) < 1e-4, f"threshold {t} = {want} (norm 5, d=200)")
  r = recall(s, xn, qn, k, "dot_product")
  check(r >= 0.9, f"dot_product recall at the defaults {r:.4f} >= 0.9")

  # Far from the origin: float32 reordering, plain L2.
  xo = x + 50
  s = scann.scann_ops_pybind.builder(xo, k, "squared_l2").autopilot().build()
  cfg = s.config()
  check(not re.search(r"\bl2_as_dot_product \{", cfg) and field(cfg, "avq") is None,
        "data far from the origin: plain squared L2")

  # Upstream's rules: 0.2.0's index.
  s = scann.scann_ops_pybind.builder(xn, k, "dot_product").autopilot(
      rules="upstream").build()
  cfg = s.config()
  check("rules" not in cfg and field(cfg, "noise_shaping_threshold") == "0.2" and
        field(cfg, "num_children") == str(n // 655), "upstream rules built")

  # ONLINE (no AVQ with incremental training): upserts and deletes.
  s = scann.scann_ops_pybind.builder(x, k, "squared_l2").autopilot(
      mode=IM.ONLINE).build(docids=[str(i) for i in range(n)])
  cfg = s.config()
  check(re.search(r"\bl2_as_dot_product \{", cfg) and field(cfg, "avq") is None,
        "ONLINE: l2_as_dot_product without AVQ")
  s.upsert(docids=["new0", "new1"], database=q[:2])
  ids, _ = s.search(q[0])
  check("new0" in list(ids), "ONLINE upsert found")
  s.delete(["new0"])
  ids, _ = s.search(q[0])
  check("new0" not in list(ids), "ONLINE delete")


def main():
  test_stanza()
  test_previews()
  test_built()
  if FAILURES:
    raise SystemExit(f"{len(FAILURES)} failure(s)")
  print("PASSED")


if __name__ == "__main__":
  main()
