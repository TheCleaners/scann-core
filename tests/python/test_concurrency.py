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

"""Stress test: searching a searcher while other threads mutate it.

Searcher threads look up "stable" points (never updated or deleted) and
check that each one finds itself, by docid. Meanwhile, mutator threads
upsert, update and delete their own points, and one thread rebalances and
resizes the thread pool. Deletes move the index's last point into the
freed slot, so a search that maps result ids to docids while a delete is
in progress returns the wrong docid (or raises IndexError).

On a free-threaded Python (3.14t, ...) this also checks that importing
scann keeps the GIL disabled.

Run with scann-core's build/python on PYTHONPATH:
  PYTHONPATH=build/python python tests/python/test_concurrency.py
SCANN_STRESS_SECONDS sets the duration (default 3).
"""

import os
import sys
import sysconfig
import threading
import time
import traceback

import numpy as np
import scann

DIM = 16
N_STABLE = 2000
SECONDS = float(os.environ.get("SCANN_STRESS_SECONDS", "3"))


def main():
  free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
  if free_threaded:
    assert not sys._is_gil_enabled(), "importing scann re-enabled the GIL"

  rng = np.random.default_rng(0)
  db = rng.standard_normal((N_STABLE, DIM)).astype(np.float32)
  stable = [f"s{i}" for i in range(N_STABLE)]
  # Every leaf is searched, and scoring is exact, so each stored vector is
  # its own nearest neighbour.
  searcher = (scann.scann_ops_pybind.builder(db, 1, "squared_l2")
              .tree(20, 20, random_init=False).score_brute_force()
              .build(docids=list(stable)))

  stop = threading.Event()
  errors = []
  counts = {"searches": 0, "mutations": 0, "rebalances": 0}
  counts_lock = threading.Lock()

  def worker(fn):
    def run(seed):
      local_rng = np.random.default_rng(seed)
      n = 0
      try:
        while not stop.is_set():
          n += fn(local_rng)
      except Exception:  # pylint: disable=broad-except
        errors.append(traceback.format_exc())
        stop.set()
      with counts_lock:
        counts[fn.__name__] += n
    return run

  def searches(r):
    i = int(r.integers(N_STABLE))
    got, _ = searcher.search(db[i], final_num_neighbors=1)
    assert got[0] == stable[i], (stable[i], got)
    rows = r.integers(N_STABLE, size=32)
    got, _ = searcher.search_batched_parallel(db[rows], final_num_neighbors=1)
    for row, g in zip(rows, got):
      assert g[0] == stable[row], (stable[row], g)
    return 2

  mine = {}  # mutator thread -> {docid: vector} it currently has indexed

  def mutations(r):
    me = threading.get_ident()
    own = mine.setdefault(me, {})
    new = {f"m{me}_{r.integers(1 << 62)}": v
           for v in r.standard_normal((4, DIM)).astype(np.float32)}
    searcher.upsert(list(new), np.stack(list(new.values())))
    own.update(new)
    if own:  # update one point in place
      docid = next(iter(own))
      own[docid] = r.standard_normal(DIM).astype(np.float32)
      searcher.upsert([docid], own[docid][None])
    if len(own) > 8:  # delete a few
      gone = list(own)[:4]
      searcher.delete(gone)
      for docid in gone:
        del own[docid]
    return 1

  def rebalances(r):
    time.sleep(0.2)
    searcher.set_num_threads(int(r.integers(1, 5)))
    assert searcher.size() >= N_STABLE
    searcher.config()
    searcher.rebalance()
    return 1

  threads = ([threading.Thread(target=worker(searches), args=(i,))
              for i in range(6)] +
             [threading.Thread(target=worker(mutations), args=(100 + i,))
              for i in range(3)] +
             [threading.Thread(target=worker(rebalances), args=(200,))])
  for t in threads:
    t.start()
  stop.wait(SECONDS)
  stop.set()
  for t in threads:
    t.join()

  if errors:
    print(f"{len(errors)} thread(s) failed; first failure:\n{errors[0]}")
    sys.exit(1)

  # Bookkeeping matches the index, and every point finds itself.
  expected = dict(zip(stable, db))
  for own in mine.values():
    expected.update(own)
  assert len(searcher.docids) == searcher.size() == len(expected), (
      len(searcher.docids), searcher.size(), len(expected))
  assert all(searcher.docid_to_id[d] == i for i, d in enumerate(searcher.docids))
  for docid, vec in expected.items():
    got, _ = searcher.search(vec, final_num_neighbors=1)
    assert got[0] == docid, (docid, got)

  print(f"PASSED ({'free-threaded' if free_threaded else 'GIL'} Python "
        f"{sys.version.split()[0]}; {counts['searches']} searches, "
        f"{counts['mutations']} mutation rounds, "
        f"{counts['rebalances']} rebalances in {SECONDS:g} s)")


if __name__ == "__main__":
  main()
