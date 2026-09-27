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

"""Regression test: serialize() / load_searcher() round trips and atomicity.

Upstream:
  * a tree index with every point deleted serialized without its data and
    couldn't be loaded ("dataset, hashed_dataset, ... are all null"), and a
    tree with bfloat16 brute-force leaves never serialized its data;
  * serialize() wrote into the directory in place, config first, so an
    interrupted re-serialize left a new config next to the old assets. Such
    directories crashed load_searcher() or loaded a mix of two indexes;
  * re-serializing an index without docids over one with docids left the
    old scann_docids.pkl, which load_searcher() then attached.

Run with scann-core's build/python on PYTHONPATH:
  PYTHONPATH=build/python python tests/python/test_serialization.py
"""

import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import scann
from scann.scann_ops.py.scann_builder import ReorderType

DIM, LEAVES = 8, 12
load = scann.scann_ops_pybind.load_searcher


def dataset(n, seed=0):
  return np.random.default_rng(seed).standard_normal((n, DIM)).astype(
      np.float32)


def builder(db, cfg):
  b = scann.scann_ops_pybind.builder(db, 10, "dot_product")
  tree = lambda **kw: b.tree(LEAVES, 4, training_sample_size=len(db), **kw)
  return {
      "bf": lambda: b.score_brute_force(),
      "bf_int8": lambda: b.score_brute_force(ReorderType.INT8),
      "bf_bf16": lambda: b.score_brute_force(ReorderType.BFLOAT16),
      "ah": lambda: b.score_ah(2).reorder(30),
      "tree_bf": lambda: tree().score_brute_force(),
      "tree_bf_int8": lambda: tree().score_brute_force(ReorderType.INT8),
      "tree_bf_bf16": lambda: tree().score_brute_force(ReorderType.BFLOAT16),
      "tree_ah": lambda: tree().score_ah(2).reorder(30),
      "tree_ah_no_reorder": lambda: tree().score_ah(2),
      "tree_ah_soar": lambda: tree(soar_lambda=1.5).score_ah(2).reorder(30),
  }[cfg]()


CONFIGS = ("bf", "bf_int8", "bf_bf16", "ah", "tree_bf", "tree_bf_int8",
           "tree_bf_bf16", "tree_ah", "tree_ah_no_reorder", "tree_ah_soar")
TREES = tuple(c for c in CONFIGS if c.startswith("tree"))


def expect_raises(exc, f, match):
  try:
    f()
  except exc as e:
    assert match in str(e), (match, str(e))
    return
  raise AssertionError(f"expected {exc.__name__} matching {match!r}")


def search_all_leaves(s, q):
  return s.search(q, final_num_neighbors=5, pre_reorder_num_neighbors=50,
                  leaves_to_search=LEAVES)


def check_round_trip(tmp):
  db = dataset(600)
  queries = dataset(20, seed=1)
  for cfg in CONFIGS:
    for relative in (False, True):
      s = builder(db, cfg).build(docids=[f"d{i}" for i in range(600)])
      d = os.path.join(tmp, f"{cfg}_{relative}")
      os.mkdir(d)
      s.serialize(d, relative_path=relative)
      t = load(d)
      assert t.size() == 600 and t.docids == s.docids, cfg
      got, want = t.search_batched(queries), s.search_batched(queries)
      assert got[0] == want[0], (cfg, relative)
      np.testing.assert_allclose(got[1], want[1], rtol=1e-5)
      # Mutable after loading.
      t.upsert(["new"], db[:1] * 3)
      t.delete(["d5"])
      assert t.size() == 600 and "new" in t.docid_to_id


def check_all_deleted_round_trip(tmp):
  """L2: every point deleted, serialized, loaded, then grown again."""
  db = dataset(600)
  for cfg in CONFIGS:
    s = builder(db, cfg).build(docids=[f"d{i}" for i in range(600)])
    s.delete([f"d{i}" for i in range(600)])
    assert s.size() == 0
    d = os.path.join(tmp, f"empty_{cfg}")
    os.mkdir(d)
    s.serialize(d)
    t = load(d)
    assert t.size() == 0 and t.docids == [], cfg
    assert search_all_leaves(t, db[0])[0] == [], cfg
    new = dataset(40, seed=2) * 3
    t.upsert([f"n{i}" for i in range(40)], new)
    assert t.size() == 40, cfg
    if cfg.endswith("bf") or cfg == "bf":  # exact scoring finds the point
      got, _ = search_all_leaves(t, new[7])
      assert got[0] == "n7", (cfg, got)
    # And round-trips again with its new points.
    d2 = d + "_again"
    os.mkdir(d2)
    t.serialize(d2)
    u = load(d2)
    assert u.size() == 40 and u.docids == t.docids, cfg


def list_dir(d):
  return sorted(f for f in os.listdir(d))


def check_reserialize_replaces_everything(tmp):
  """A new index over an old one: no stale files, no stale docids."""
  d = os.path.join(tmp, "replace")
  os.mkdir(d)
  old = builder(dataset(600), "tree_ah_soar").build(
      docids=[f"old{i}" for i in range(600)])
  old.serialize(d)
  assert "hashed_dataset_soar.npy" in list_dir(d)
  assert "scann_docids.pkl" in list_dir(d)
  new_db = dataset(300, seed=3)
  new = builder(new_db, "tree_bf").build()  # no docids
  new.serialize(d)
  assert list_dir(d) == [
      "datapoint_to_token.npy", "dataset.npy", "scann_assets.pbtxt",
      "scann_config.pb", "serialized_partitioner.pb"], list_dir(d)
  t = load(d)
  assert t.size() == 300 and t.docids is None
  assert t.search(new_db[3], leaves_to_search=LEAVES)[0][0] == 3


def check_failed_serialize_leaves_clean_error(tmp):
  """A serialize that fails midway: the directory fails to load, cleanly.

  A directory named dataset.npy makes renaming the new dataset.npy into
  place fail, after the manifest was replaced by the "incomplete" marker and
  other files were already replaced.
  """
  d = os.path.join(tmp, "failed")
  os.mkdir(d)
  old = builder(dataset(600), "tree_ah").build(
      docids=[f"old{i}" for i in range(600)])
  old.serialize(d)
  os.remove(os.path.join(d, "dataset.npy"))
  os.mkdir(os.path.join(d, "dataset.npy"))
  os.mkdir(os.path.join(d, "dataset.npy", "x"))
  new = builder(dataset(300, seed=3), "tree_ah").build(
      docids=[f"new{i}" for i in range(300)])
  expect_raises(RuntimeError, lambda: new.serialize(d), "Failed to serialize")
  assert not [f for f in os.listdir(d) if f.startswith(".scann_staging")]
  expect_raises(RuntimeError, lambda: load(d), "incomplete")
  shutil.rmtree(os.path.join(d, "dataset.npy"))
  new.serialize(d)
  t = load(d)
  assert t.size() == 300 and t.docids[0] == "new0"


KILL_CHILD = """
import sys, numpy as np, scann
d, n, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
db = np.random.default_rng(n).standard_normal((n, 16)).astype(np.float32)
db[:, 0] = n  # every point of this index has x0 == n
s = (scann.scann_ops_pybind.builder(db, 10, "dot_product")
     .tree(40, 4, training_sample_size=n).score_ah(2).reorder(30)
     .build(docids=[f"{tag}{i}" for i in range(n)]))
print("built", flush=True)
sys.stdin.readline()
while True:
  s.serialize(d)
"""


def check_killed_serialize(tmp):
  """kill -9 during re-serialize: the old index, the new one, or an error."""
  d = os.path.join(tmp, "killed")
  os.mkdir(d)
  script = os.path.join(tmp, "kill_child.py")
  with open(script, "w") as f:
    f.write(KILL_CHILD)
  env = dict(os.environ)
  rng = random.Random(0)
  outcomes = {}
  for trial in range(12):
    n, tag = (4000, "a") if trial % 2 == 0 else (6000, "b")
    child = subprocess.Popen([sys.executable, script, d, str(n), tag],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, env=env)
    assert child.stdout.readline().strip() == b"built"
    child.stdin.write(b"go\n")
    child.stdin.flush()
    time.sleep(rng.uniform(0.0, 0.3))
    child.send_signal(signal.SIGKILL)
    child.wait()
    try:
      t = load(d, assets_backcompat_shim=False)
    except (RuntimeError, ValueError) as e:
      msg = str(e)
      # Killed mid-commit, or before the first serialize committed anything.
      assert "incomplete" in msg or "No scann_assets.pbtxt" in msg, msg
      outcomes["error"] = outcomes.get("error", 0) + 1
      continue
    size = t.size()
    assert size in (4000, 6000), size
    tag = "a" if size == 4000 else "b"
    assert len(t.docids) == size and t.docids[0] == f"{tag}0"
    q = np.zeros(16, np.float32)
    q[0] = 1.0
    ids, _ = t.search(q, leaves_to_search=40)
    assert all(i.startswith(tag) for i in ids), (size, ids)
    outcomes[tag] = outcomes.get(tag, 0) + 1
  # A killed serialize may leave a staging directory; the next one removes
  # it.
  builder(dataset(300), "tree_bf").build().serialize(d)
  assert not [f for f in os.listdir(d) if f.startswith(".scann_staging")]
  return outcomes


def check_relative_dir(tmp):
  """serialize() into a directory given by a relative path, default
  relative_path=False: the manifest must hold absolute paths (upstream wrote
  dir/name, which the loader resolved to dir/dir/name)."""
  db = dataset(600)
  s = builder(db, "tree_ah").build(docids=[f"d{i}" for i in range(600)])
  cwd = os.getcwd()
  os.chdir(tmp)
  try:
    os.mkdir("rel_idx")
    s.serialize("rel_idx")
    t = load("rel_idx")
  finally:
    os.chdir(cwd)
  assert t.size() == 600 and t.docids == s.docids
  assert t.search(db[3])[0] == s.search(db[3])[0]
  # Still loads from elsewhere: the recorded paths are absolute.
  t = load(os.path.join(tmp, "rel_idx"))
  assert t.size() == 600


def main():
  tmp = tempfile.mkdtemp(prefix="scann_serialization_test_")
  try:
    check_round_trip(tmp)
    check_relative_dir(tmp)
    check_all_deleted_round_trip(tmp)
    check_reserialize_replaces_everything(tmp)
    check_failed_serialize_leaves_clean_error(tmp)
    outcomes = check_killed_serialize(tmp)
  finally:
    shutil.rmtree(tmp, ignore_errors=True)
  print(f"PASSED (killed re-serialize outcomes: {outcomes})")


if __name__ == "__main__":
  main()
