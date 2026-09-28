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

"""Part 8, TensorFlow: the part 3 index in tf.function and a SavedModel.

Needs TensorFlow (pip install 'scann-core[tf]'). The SavedModel section
needs scann.tf's op backend (a CMake build with -DSCANN_BUILD_TF_OP=ON; see
docs/tensorflow.md) and is skipped without it.
"""

import os
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
import tensorflow as tf
import keras
import scann
import scann.tf

from tutorial_data import Timer, load_glove, recall

dataset, queries, true_neighbors = load_glove()
print(f"TensorFlow {tf.__version__}; scann.tf backend: {scann.tf.backend()}; "
      f"available: {scann.tf.available_backends()}")

# --- 1. The part 3 index, on each backend ----------------------------------
# Built once with scann_ops_pybind; from_pybind() wraps it (Python backend)
# or copies it into TensorFlow variables (op backend).
with Timer() as t:
  pybind = (scann.scann_ops_pybind.builder(dataset, 10, "dot_product")
            .tree(num_leaves=2000, num_leaves_to_search=100,
                  training_sample_size=250000)
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(100)
            .build())
print(f"built in {t.seconds:.1f} s")
p_idx, p_dist = pybind.search_batched_parallel(queries)
print(f"pybind searcher: recall@10 {recall(p_idx, true_neighbors):.4f}")

searchers = {}
for name in scann.tf.available_backends():
  with Timer() as t:
    searchers[name] = scann.tf.get_backend(name).from_pybind(pybind)
  s = searchers[name]

  @tf.function(input_signature=[tf.TensorSpec([None, 100], tf.float32)])
  def retrieve(q, s=s):
    return s.search_batched_parallel(q, final_num_neighbors=10)

  idx, dist = retrieve(tf.constant(queries))
  print(f"{name:6} backend: from_pybind {t.seconds:.1f} s; tf.function gives "
        f"{idx.dtype.name} {idx.shape}, {dist.dtype.name}; identical to "
        f"pybind: {np.array_equal(idx, p_idx) and np.array_equal(dist, p_dist)}")

# --- 2. A model that encodes, then searches ---------------------------------
# The stand-in query tower of the PyTorch half: GloVe queries hidden behind a
# random rotation, and one Dense layer (+ normalization) that undoes it.
rotation, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((100, 100)))
features = (queries @ rotation).astype(np.float32)
encoder = keras.Sequential([
    keras.Input(shape=(100,)),
    keras.layers.Dense(100, use_bias=False),
    keras.layers.UnitNormalization(),
])
encoder.layers[0].set_weights([rotation.T.astype(np.float32)])


class Retrieval(tf.Module):
  """Features in, the 10 nearest GloVe words out."""

  def __init__(self, encoder, searcher):
    super().__init__()
    self.encoder = encoder
    # The op backend's searcher is a tf.Module holding the index; with the
    # Python backend, serialize_to_module() raises NotImplementedError.
    self.index = searcher.serialize_to_module()

  @tf.function(input_signature=[tf.TensorSpec([None, 100], tf.float32)])
  def retrieve(self, features):
    searcher = scann.tf.searcher_from_module(self.index)
    return searcher.search_batched_parallel(self.encoder(features),
                                            final_num_neighbors=10)


try:
  searchers["python"].serialize_to_module()
except NotImplementedError as e:
  print(f"\nPython backend, serialize_to_module(): NotImplementedError: "
        f"{str(e)[:60]}...")

if "op" not in searchers:
  print("op backend not built: skipping the SavedModel")
  sys.exit(0)

model = Retrieval(encoder, searchers["op"])
found, _ = model.retrieve(features)
print(f"Retrieval.retrieve: recall@10 {recall(found.numpy(), true_neighbors):.4f}"
      f"; {np.sum(found.numpy() != p_idx)} of 100,000 results differ from "
      "searching the GloVe queries directly")

with tempfile.TemporaryDirectory() as tmp:
  export_dir = os.path.join(tmp, "retrieval")
  with Timer() as t:
    tf.saved_model.save(model, export_dir,
                        signatures={"serving_default": model.retrieve})
  size = sum(os.path.getsize(os.path.join(d, f))
             for d, _, files in os.walk(export_dir) for f in files)
  print(f"\nSavedModel written in {t.seconds:.1f} s: {size / 2**20:.1f} MiB")
  np.save(os.path.join(tmp, "features.npy"), features)
  # A fresh process: import scann.tf (it registers the op), load, run. The
  # index comes from the SavedModel's variables only.
  child = """
import sys, time
import numpy as np, tensorflow as tf
import scann.tf
start = time.perf_counter()
loaded = tf.saved_model.load(sys.argv[1])
serve = loaded.signatures["serving_default"]
out = serve(features=tf.constant(np.load(sys.argv[2])))
print(f"{time.perf_counter() - start:.1f}", sorted(out))
np.save(sys.argv[3], out["indices"].numpy())
"""
  out = os.path.join(tmp, "indices.npy")
  result = subprocess.run(
      [sys.executable, "-c", child, export_dir,
       os.path.join(tmp, "features.npy"), out],
      check=True, capture_output=True, text=True).stdout.splitlines()[-1]
  seconds, outputs = result.split(" ", 1)
  loaded = np.load(out)
print(f"fresh process: loaded and searched 10,000 queries in {seconds} s; "
      f"signature outputs {outputs}")
print(f"  recall@10 {recall(loaded, true_neighbors):.4f}; identical to "
      f"before saving: {np.array_equal(loaded, found.numpy())}")

# --- 3. What a call costs ---------------------------------------------------
# Median time per call, measured round-robin so that all columns see the
# same machine load.
spec = [tf.TensorSpec([None, 100], tf.float32)]
search_fn = {name: tf.function(
    lambda q, s=s: s.search_batched_parallel(q, final_num_neighbors=10),
    input_signature=spec) for name, s in searchers.items()}
encode = tf.function(lambda f: encoder(f, training=False), input_signature=spec)
retrieve_py = tf.function(  # the Python backend, closed over (not saveable)
    lambda f: searchers["python"].search_batched_parallel(encoder(f), 10),
    input_signature=spec)


def median_us(calls, reps):
  times = {name: [] for name in calls}
  for fn in calls.values():
    for _ in range(3):
      fn()
  for _ in range(reps):
    for name, fn in calls.items():
      start = time.perf_counter()
      fn()
      times[name].append(time.perf_counter() - start)
  return {name: 1e6 * statistics.median(t) for name, t in times.items()}


def table(title, make_calls):
  print(f"\n{title}")
  for n, reps in ((1, 2000), (1000, 200)):
    us = median_us(make_calls(queries[:n], features[:n]), reps)
    if n == 1:
      print(f"  {'':15}" + "".join(f"{c:>14}" for c in us))
    label = "1 query" if n == 1 else f"{n:,} queries"
    print(f"  {label:15}" + "".join(
        f"{v:>14.0f}" if v >= 1000 else f"{v:>14.1f}" for v in us.values()))


def search_only(x, _):
  t = tf.constant(x)
  return {"pybind": lambda: pybind.search_batched_parallel(x),
          "op, eager": lambda: searchers["op"].search_batched_parallel(t, 10),
          "op, tf.fn": lambda: search_fn["op"](t),
          "python, eager": lambda: searchers["python"].search_batched_parallel(
              t, 10),
          "python, tf.fn": lambda: search_fn["python"](t)}


def encode_and_search(_, f):
  t = tf.constant(f)
  return {"Keras+pybind": lambda: pybind.search_batched_parallel(
              encoder(t, training=False).numpy()),
          "tf.fn+pybind": lambda: pybind.search_batched_parallel(
              encode(t).numpy()),
          "op, tf.fn": lambda: model.retrieve(t),
          "python, tf.fn": lambda: retrieve_py(t)}


table("search_batched_parallel, µs per call (median)", search_only)
table("encoder + search, µs per call (median)", encode_and_search)
