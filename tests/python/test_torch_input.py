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

"""PyTorch tensors as scann_ops_pybind inputs.

Every entry point that takes vectors (builder, create_searcher, search,
search_batched, search_batched_parallel, upsert) accepts tensors and gives
exactly the results of the same data as numpy arrays: float32/64/16 and
bfloat16, non-contiguous views, tensors that require grad, and (when a GPU
is present) CUDA tensors, which are copied to host memory. A contiguous
float32 CPU tensor is read without a copy.

Indexes here have no tree: tree training starts from random centres, so
two tree builds differ even from the same numpy array.

Without PyTorch the test exits with 77, which ctest reports as skipped
(SKIP_RETURN_CODE), unless SCANN_TEST_REQUIRE_TORCH is set.
"""

import os
import sys

import numpy as np

SKIP = 77
N, DIM, K = 2000, 32, 10


def main():
  try:
    import torch
  except ImportError as e:
    if os.environ.get("SCANN_TEST_REQUIRE_TORCH"):
      raise
    print(f"SKIPPED: torch is not importable ({e})")
    return SKIP
  import scann
  from scann.scann_ops.py import scann_ops_pybind as P

  rng = np.random.default_rng(0)
  db = rng.standard_normal((N, DIM)).astype(np.float32)
  q = rng.standard_normal((16, DIM)).astype(np.float32)
  db_t, q_t = torch.from_numpy(db.copy()), torch.from_numpy(q.copy())

  def build(x, docids=None):
    return (P.builder(x, K, "dot_product").score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(100).build(
            docids=docids))

  def same(a, b, what):
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0])), what
    np.testing.assert_array_equal(np.asarray(a[1]), np.asarray(b[1]), what)

  ref = build(db)
  want = ref.search_batched(q)

  # Entry points.
  same(build(db_t).search_batched(q), want, "builder(tensor)")
  cfg = P.builder(db, K, "dot_product").score_brute_force().create_config()
  same(P.create_searcher(db_t, cfg).search_batched(q),
       P.create_searcher(db, cfg).search_batched(q), "create_searcher(tensor)")
  same(ref.search(q_t[3]), ref.search(q[3]), "search(tensor)")
  same(ref.search_batched(q_t), want, "search_batched(tensor)")
  same(ref.search_batched_parallel(q_t), ref.search_batched_parallel(q),
       "search_batched_parallel(tensor)")
  a = build(db, docids=[f"d{i}" for i in range(N)])
  b = build(db, docids=[f"d{i}" for i in range(N)])
  new = rng.standard_normal((3, DIM)).astype(np.float32)
  a.upsert(["x", "y", "d7"], torch.from_numpy(new.copy()))
  b.upsert(["x", "y", "d7"], new)
  a.upsert("z", torch.from_numpy(new[0].copy()))
  b.upsert("z", new[0])
  same(a.search_batched(q), b.search_batched(q), "upsert(tensor)")

  # Dtypes and layouts, each against the same values as a numpy array.
  cases = {
      "float64": q_t.double(),
      "float16": q_t.half(),
      "bfloat16": q_t.bfloat16(),
      "transposed view": q_t.t().contiguous().t(),
      "strided slice": torch.from_numpy(np.repeat(q, 2, axis=1))[:, ::2],
      "requires_grad": q_t.clone().requires_grad_(),
  }
  for what, t in cases.items():
    as_np = t.detach().float().numpy()
    same(ref.search_batched(t), ref.search_batched(as_np), what)
  same(build(db_t.clone().requires_grad_()).search_batched(q), want,
       "builder(requires_grad tensor)")
  same(build(db_t.double()).search_batched(q),
       build(db.astype(np.float64)).search_batched(q), "builder(float64)")

  # No copy for a contiguous float32 CPU tensor; one for others.
  host = P._host_array(q_t)  # pylint: disable=protected-access
  assert np.shares_memory(host, q_t.numpy()), "float32 tensor was copied"
  assert P._host_array(q) is q  # pylint: disable=protected-access

  # Empty inputs: 0 queries, and an index built from a 0-row tensor.
  idx, dist = ref.search_batched(q_t[:0])
  assert len(idx) == 0 and np.asarray(dist).shape[0] == 0
  empty = P.create_searcher(torch.zeros((0, DIM)), cfg, docids=[])
  empty.upsert(["p"], torch.from_numpy(db[:1].copy()))
  assert empty.size() == 1

  # GPU tensors are copied to host memory.
  cuda = torch.cuda.is_available()
  if cuda:
    same(ref.search_batched(q_t.cuda()), want, "search_batched(cuda)")
    same(ref.search(q_t[3].cuda()), ref.search(q[3]), "search(cuda)")
    same(build(db_t.cuda()).search_batched(q), want, "builder(cuda)")
    c = build(db, docids=[f"d{i}" for i in range(N)])
    c.upsert(["x", "y", "d7"], torch.from_numpy(new.copy()).cuda())
    same(c.search_batched(q), b.search_batched(q), "upsert(cuda)")

  print("PASSED" + (f" (CUDA: {torch.cuda.get_device_name(0)})" if cuda
                    else " (no CUDA device: GPU cases not run)"))
  return 0


if __name__ == "__main__":
  sys.exit(main())
