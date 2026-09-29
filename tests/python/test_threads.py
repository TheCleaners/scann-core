#!/usr/bin/env python3
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

"""Thread-pool sizing (Linux; exit code 77 elsewhere).

Upstream sized every searcher's query pool and the default training pool
from the machine's online CPUs, ignoring the CPU affinity (taskset, cpusets)
and container quotas, and started the query pool in every constructor: under
taskset -c 0-3 on a 64-CPU machine, a searcher started 63 threads. Checked
here, each in a fresh process with a restricted affinity:

- threads are created only for the CPUs the process may use, and only once
  a parallel search needs them (none after build or load alone);
- SCANN_NUM_THREADS overrides the count;
- set_num_threads(n) means n workers in all (the caller plus n - 1 pool
  threads): 1 starts no thread;
- results don't depend on the thread count.

Run with scann-core's build/python on PYTHONPATH:
  PYTHONPATH=build/python python tests/python/test_threads.py
"""

import json
import os
import subprocess
import sys

CHILD = r"""
import json, os, sys, time
import numpy as np
import scann

def threads():
  # A thread that was just joined can stay listed for a moment; pool threads
  # stay. The minimum over a short window counts the ones that stay.
  n = []
  for _ in range(5):
    n.append(len(os.listdir("/proc/self/task")))
    time.sleep(0.02)
  return min(n)

rng = np.random.default_rng(0)
data = rng.standard_normal((4000, 16)).astype(np.float32)
queries = rng.standard_normal((300, 16)).astype(np.float32)
base = threads()
out = {"base": base}
s = (scann.scann_ops_pybind.builder(data, 10, "dot_product")
     .tree(num_leaves=20, num_leaves_to_search=5, training_sample_size=4000)
     .score_ah(2, anisotropic_quantization_threshold=0.2)
     .reorder(50)
     .build(docids=[str(i) for i in range(len(data))]))
out["after_build"] = threads() - base
set_threads = int(sys.argv[1])
if set_threads > 0:
  s.set_num_threads(set_threads)
idx, dist = s.search_batched_parallel(queries, batch_size=8)
out["after_search"] = threads() - base
ref, ref_dist = s.search_batched(queries)
out["same_results"] = bool(np.array_equal(np.asarray(idx), np.asarray(ref)) and
                           np.array_equal(dist, ref_dist))
s.upsert(["new0", "new1"], rng.standard_normal((2, 16)).astype(np.float32),
         batch_size=2)
out["after_upsert"] = threads() - base
print("RESULT " + json.dumps(out))
"""


def run(cpus, env_threads=None, set_threads=0):
  env = dict(os.environ)
  env.pop("SCANN_NUM_THREADS", None)
  if env_threads is not None:
    env["SCANN_NUM_THREADS"] = str(env_threads)
  p = subprocess.run(
      [sys.executable, "-c", CHILD, str(set_threads)], env=env,
      capture_output=True, text=True, check=False,
      preexec_fn=lambda: os.sched_setaffinity(0, cpus))
  if p.returncode != 0:
    raise AssertionError(f"child failed ({p.returncode}):\n{p.stderr[-4000:]}")
  line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")][-1]
  return json.loads(line[len("RESULT "):])


def main():
  if not sys.platform.startswith("linux") or not hasattr(os,
                                                          "sched_getaffinity"):
    print("skipped: Linux only")
    sys.exit(77)
  mine = sorted(os.sched_getaffinity(0))
  if len(mine) < 2:
    print("skipped: needs at least 2 CPUs")
    sys.exit(77)
  two = set(mine[:2])

  # Default: the pool has as many workers as CPUs in the affinity mask, i.e.
  # one pool thread next to the caller, started by the first parallel
  # search; none survives the build.
  r = run(two)
  print("2 CPUs, defaults:", r)
  assert r["after_build"] == 0, r
  assert r["after_search"] == 1, r
  assert r["after_upsert"] == 1, r
  assert r["same_results"], r

  # SCANN_NUM_THREADS wins over the affinity.
  r = run(two, env_threads=5)
  print("2 CPUs, SCANN_NUM_THREADS=5:", r)
  assert r["after_build"] == 0, r
  assert r["after_search"] == 4, r
  assert r["same_results"], r

  # set_num_threads(n): n workers, n - 1 pool threads; 1 starts none.
  r = run(two, set_threads=1)
  print("2 CPUs, set_num_threads(1):", r)
  assert r["after_search"] == 0, r
  assert r["after_upsert"] == 0, r
  assert r["same_results"], r
  r = run(two, set_threads=3)
  print("2 CPUs, set_num_threads(3):", r)
  assert r["after_search"] == 2, r
  assert r["same_results"], r

  # An invalid override is ignored.
  r = run(two, env_threads="lots")
  print("2 CPUs, SCANN_NUM_THREADS=lots:", r)
  assert r["after_search"] == 1, r
  print("OK")


if __name__ == "__main__":
  main()
