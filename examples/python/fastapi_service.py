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

"""A FastAPI search service over one scann-core index.

The service loads a saved index once, at startup, and every request
shares it (a searcher is thread-safe). Three endpoints:

  POST /search        one query per request. A plain `def` endpoint, so
                      FastAPI runs it in its thread pool; the search
                      releases the GIL while it runs.
  POST /search_batch  a client-side batch: one search_batched_parallel()
                      call on the index's own thread pool.
  POST /search_micro  one query per request, but concurrent requests are
                      collected for up to 2 ms and searched as one batch
                      (server-side micro-batching, with asyncio).

The script builds and saves a small index, then runs the app in-process
with FastAPI's TestClient (no server, no port), sends concurrent requests
from threads, and checks every answer against a direct search. To serve
for real: `uvicorn fastapi_service:app` with SCANN_INDEX_DIR set.

Needs FastAPI and, for its TestClient, httpx2 (`pip install fastapi
httpx2`; Starlette 1.7 also accepts httpx, with a deprecation warning).
Exits with status 77 (reported as skipped by ctest) without them. Run with
scann-core installed, or from a CMake build tree:
  PYTHONPATH=<build>/python python examples/python/fastapi_service.py
"""

import asyncio
import contextlib
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

try:
  import fastapi
  import pydantic
  from fastapi.testclient import TestClient  # needs httpx2 (or httpx)
except ImportError as e:
  print(f"FastAPI or httpx2 is not installed ({e}); skipping "
        "(pip install fastapi httpx2)")
  sys.exit(77)

import numpy as np
import scann

DIM, K = 32, 10


# --- The service ------------------------------------------------------------
class Query(pydantic.BaseModel):
  vector: list[float]
  k: int = K


class Queries(pydantic.BaseModel):
  vectors: list[list[float]]
  k: int = K


class MicroBatcher:
  """Groups concurrent single-query searches into batched searches.

  Requests wait at most `max_wait_s` for others to join, and a batch holds
  at most `max_batch` queries. The search runs in a worker thread (it
  releases the GIL), so the event loop keeps accepting requests meanwhile.
  """

  def __init__(self, searcher, max_batch=64, max_wait_s=0.002):
    self.searcher = searcher
    self.max_batch, self.max_wait_s = max_batch, max_wait_s
    self.queue = asyncio.Queue()
    self.batches = self.queries = 0

  async def search(self, vector, k):
    future = asyncio.get_running_loop().create_future()
    await self.queue.put((vector, k, future))
    return await future

  async def run(self):
    while True:
      batch = [await self.queue.get()]
      deadline = asyncio.get_running_loop().time() + self.max_wait_s
      while len(batch) < self.max_batch:
        timeout = deadline - asyncio.get_running_loop().time()
        if timeout <= 0:
          break
        try:
          batch.append(await asyncio.wait_for(self.queue.get(), timeout))
        except asyncio.TimeoutError:
          break
      self.batches += 1
      self.queries += len(batch)
      # One search per distinct k (clients rarely mix them).
      for k in {k for _, k, _ in batch}:
        group = [(v, f) for v, kk, f in batch if kk == k]
        try:
          docids, scores = await asyncio.to_thread(
              self.searcher.search_batched_parallel,
              np.stack([v for v, _ in group]), final_num_neighbors=k)
        except Exception as e:  # pylint: disable=broad-except
          for _, f in group:
            f.set_exception(e)
          continue
        for (_, f), d, s in zip(group, docids, scores):
          f.set_result((d, s))


def vector_array(values):
  v = np.asarray(values, dtype=np.float32)
  if v.shape[-1:] != (DIM,) or not np.isfinite(v).all():
    raise fastapi.HTTPException(400, f"expected finite {DIM}-dim vectors")
  return v


@contextlib.asynccontextmanager
async def lifespan(app):
  # Load once per process. The searcher is shared by every request.
  searcher = scann.scann_ops_pybind.load_searcher(os.environ["SCANN_INDEX_DIR"])
  batcher = MicroBatcher(searcher)
  task = asyncio.create_task(batcher.run())
  app.state.searcher, app.state.batcher = searcher, batcher
  yield
  task.cancel()


app = fastapi.FastAPI(lifespan=lifespan)


def response(docids, scores):
  # Rows shorter than k (only when the index has fewer than k points) are
  # padded with docid None and a NaN score, which JSON can't carry: drop them.
  keep = [i for i, d in enumerate(docids) if d is not None]
  return {"docids": [docids[i] for i in keep],
          "scores": [float(scores[i]) for i in keep]}


@app.post("/search")
def search(query: Query):
  docids, scores = app.state.searcher.search(
      vector_array(query.vector), final_num_neighbors=query.k)
  return response(docids, scores)


@app.post("/search_batch")
def search_batch(queries: Queries):
  docids, scores = app.state.searcher.search_batched_parallel(
      vector_array(queries.vectors), final_num_neighbors=queries.k)
  return {"results": [response(d, s) for d, s in zip(docids, scores)]}


@app.post("/search_micro")
async def search_micro(query: Query):
  docids, scores = await app.state.batcher.search(
      vector_array(query.vector), query.k)
  return response(docids, scores)


# --- Build an index, run the app in-process, check the answers -------------
def main():
  rng = np.random.default_rng(0)
  centers = rng.standard_normal((50, DIM))

  def clustered(n):
    x = centers[rng.integers(0, 50, n)] + 0.3 * rng.standard_normal((n, DIM))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)

  items = clustered(20_000)
  docids = [f"item-{i}" for i in range(len(items))]
  index = (scann.scann_ops_pybind.builder(items, K, "dot_product")
           .tree(num_leaves=140, num_leaves_to_search=20, random_init=False)
           .score_ah(2, anisotropic_quantization_threshold=0.2)
           .reorder(100)
           .build(docids=docids))
  queries = clustered(400)
  expected, _ = index.search_batched(queries)

  with tempfile.TemporaryDirectory() as index_dir:
    # A deploy step: save the index where the service will find it.
    index.serialize(index_dir, relative_path=True)
    os.environ["SCANN_INDEX_DIR"] = index_dir

    with TestClient(app) as client:  # runs the lifespan: loads the index
      one = client.post("/search", json={"vector": queries[0].tolist()})
      assert one.status_code == 200, one.text
      print("POST /search       ->", one.json()["docids"][:3])
      assert one.json()["docids"] == expected[0]

      batch = client.post("/search_batch",
                          json={"vectors": queries.tolist(), "k": K}).json()
      assert [r["docids"] for r in batch["results"]] == expected
      print(f"POST /search_batch -> {len(batch['results'])} results, all "
            "equal to a direct search")

      # Concurrent single-query requests, as from many clients at once.
      def ask(endpoint, i):
        r = client.post(endpoint, json={"vector": queries[i].tolist()})
        assert r.status_code == 200, r.text
        return r.json()["docids"]

      for endpoint in ("/search", "/search_micro"):
        with ThreadPoolExecutor(16) as pool:
          got = list(pool.map(lambda i, e=endpoint: ask(e, i),
                              range(len(queries))))
        assert got == expected, endpoint
        print(f"POST {endpoint:13} -> {len(queries)} concurrent requests "
              "from 16 threads, all equal to a direct search")
      b = app.state.batcher
      print(f"  micro-batching: {b.queries} queries in {b.batches} batches")

      bad = client.post("/search", json={"vector": [0.0] * (DIM - 1)})
      assert bad.status_code == 400, bad.status_code
      print("wrong dimension    ->", bad.status_code, bad.json()["detail"])


if __name__ == "__main__":
  main()
