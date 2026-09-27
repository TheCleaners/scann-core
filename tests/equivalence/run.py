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

"""Equivalence test: scann-core's Python package vs. the upstream wheel.

Runs the same datasets, configs and queries through
  * the reference: the Bazel-built upstream wheel (`scann`, installed in the
    given interpreter's site-packages), and
  * the candidate: scann-core's package (<build dir>/python),
each in its own subprocess (both extension modules are named `scann_pybind`,
so they cannot share a process).

Deterministic configs must match exactly:
  (a) identical neighbour lists (ids and order), single and batched search;
  (b) distance deltas <= 1e-5 where ids agree (in practice: 0);
  (c) identical recall@k against exact float64 search;
  and the candidate's ScannBuilder must emit byte-identical config text.

Nondeterministic configs: upstream's k-means random initialization
(`.tree(random_init=True)`, the builder default) appends the sampled
initial centers in absl::flat_hash_set iteration order, which abseil
randomizes per process and per table; small-cluster reinitialization then
consumes the RNG in cluster-index order, so the trained partitioner depends
on hash-table state. The *wheel* gives different indexes run to run for
these configs, so bit-exact comparison is ill-defined. Each implementation
instead trains them REPEATS times in one process (different heap state per
build) and their recall distributions are compared. Every such config has a
`random_init=False` (k-means++, deterministic) twin in the strict set.

Fixtures (reference single-query results for the deterministic configs)
are written for the Rust equivalence test (rust/tests/equivalence.rs).

Usage:
  python tests/equivalence/run.py --build-dir build-native \
      --python /path/to/venv/bin/python --out build-native/equivalence
"""

import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

K = 10
SEED = 20260922
REPEATS = 8


# ---------------------------------------------------------------------------
# Data: clustered, anisotropic, L2-normalised -- closer to real embeddings
# than i.i.d. Gaussians (decaying per-dimension variance, mixture structure).
# Queries are held out: drawn from the same process, never in the database.
# ---------------------------------------------------------------------------
def make_dataset(rng, n, nq, dim, n_clusters):
  scale = 1.0 / np.sqrt(1.0 + np.arange(dim) / 8.0)
  centers = rng.standard_normal((n_clusters, dim)) * scale

  def draw(m):
    c = centers[rng.integers(0, n_clusters, size=m)]
    x = c + 0.6 * rng.standard_normal((m, dim)) * scale
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)

  return draw(n), draw(nq)


DATASETS = {
    "A": dict(n=5000, nq=200, dim=128, n_clusters=50),
    "B": dict(n=4000, nq=200, dim=768, n_clusters=40),
}


def _tree_ah(n, random_init, aq=0.2):
  def f(s, b):
    b = b.tree(num_leaves=64, num_leaves_to_search=8, training_sample_size=n,
               random_init=random_init)
    b = b.score_ah(2, anisotropic_quantization_threshold=aq) if aq else b.score_ah(2)
    return b.reorder(100)
  return f


# name -> (dataset, distance, deterministic, fn(scann module, builder))
CONFIGS = {
    "A_brute_force_dot": ("A", "dot_product", True, lambda s, b: b.score_brute_force()),
    "A_ah_int8reorder_dot": (
        "A", "dot_product", True,
        lambda s, b: b.score_ah(2, anisotropic_quantization_threshold=0.2)
        .reorder(100, quantize=s.ReorderType.INT8)),
    "A_autopilot_dot": ("A", "dot_product", True, lambda s, b: b.autopilot()),
    "A_tree_ah_reorder_dot_kmpp": ("A", "dot_product", True, _tree_ah(5000, False)),
    "A_tree_ah_reorder_l2_kmpp": ("A", "squared_l2", True, _tree_ah(5000, False, aq=None)),
    "B_brute_force_dot": ("B", "dot_product", True, lambda s, b: b.score_brute_force()),
    "B_tree_ah_reorder_dot_kmpp": ("B", "dot_product", True, _tree_ah(4000, False)),
    # builder defaults (random_init=True): nondeterministic upstream
    "A_tree_ah_reorder_dot": ("A", "dot_product", False, _tree_ah(5000, True)),
    "A_tree_ah_reorder_l2": ("A", "squared_l2", False, _tree_ah(5000, True, aq=None)),
    "B_tree_ah_reorder_dot": ("B", "dot_product", False, _tree_ah(4000, True)),
}


# ---------------------------------------------------------------------------
# Worker: runs inside one implementation's interpreter.
# ---------------------------------------------------------------------------
def worker(workdir, label, ref_configs_path, preimport):
  # Diagnostic control over process state: the wheel's scann/__init__.py
  # imports TensorFlow when it's installed, scann-core's doesn't.
  if preimport == "tensorflow":
    import tensorflow  # pylint: disable=g-import-not-at-top,unused-import
  elif preimport == "block-tensorflow":
    sys.modules["tensorflow"] = None  # `import tensorflow` -> ModuleNotFoundError
  import scann  # pylint: disable=g-import-not-at-top

  data = {name: (np.load(os.path.join(workdir, f"{name}_db.npy")),
                 np.load(os.path.join(workdir, f"{name}_q.npy")))
          for name in DATASETS}
  ref_configs = None
  if ref_configs_path:
    with open(ref_configs_path) as f:
      ref_configs = json.load(f)

  info = {"label": label, "scann_file": scann.__file__,
          "tensorflow_imported": sys.modules.get("tensorflow") is not None}
  own_configs, config_match = {}, {}
  for name, (ds, distance, deterministic, configure) in CONFIGS.items():
    dbv, qv = data[ds]
    own = configure(scann, scann.scann_ops_pybind.builder(dbv, K, distance)).create_config()
    own_configs[name] = own
    config = own
    if ref_configs is not None:
      config_match[name] = (own == ref_configs[name])
      config = ref_configs[name]
    if deterministic:
      searcher = scann.scann_ops_pybind.create_searcher(dbv, config)
      single = [searcher.search(q, final_num_neighbors=K) for q in qv]
      b_idx, b_dist = searcher.search_batched(qv, final_num_neighbors=K)
      np.savez(os.path.join(workdir, f"{label}__{name}.npz"),
               single_idx=np.stack([np.asarray(i, dtype=np.int64) for i, _ in single]),
               single_dist=np.stack([np.asarray(d, dtype=np.float32) for _, d in single]),
               batched_idx=np.asarray(b_idx, dtype=np.int64),
               batched_dist=np.asarray(b_dist, dtype=np.float32))
    else:
      runs = []
      for _ in range(REPEATS):
        idx, _ = scann.scann_ops_pybind.create_searcher(dbv, config).search_batched(
            qv, final_num_neighbors=K)
        runs.append(np.asarray(idx, dtype=np.int64))
      np.save(os.path.join(workdir, f"{label}__{name}.npy"), np.stack(runs))
  info["config_match"] = config_match
  with open(os.path.join(workdir, f"{label}__configs.json"), "w") as f:
    json.dump(own_configs, f, indent=1)
  with open(os.path.join(workdir, f"{label}__info.json"), "w") as f:
    json.dump(info, f, indent=1)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def run_worker(python, workdir, label, pythonpath, ref_configs=None, preimport=None):
  env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LD_LIBRARY_PATH")}
  env["TF_CPP_MIN_LOG_LEVEL"] = "3"
  if pythonpath:
    env["PYTHONPATH"] = pythonpath
  cmd = [python, os.path.abspath(__file__), "--worker", label, "--workdir", workdir]
  if ref_configs:
    cmd += ["--ref-configs", ref_configs]
  if preimport:
    cmd += ["--preimport", preimport]
  # cwd = workdir so a `scann/` source directory in the caller's cwd can't
  # shadow the package under test.
  r = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True)
  if r.returncode != 0:
    sys.stderr.write(r.stdout + r.stderr)
    raise SystemExit(f"worker {label} failed")
  with open(os.path.join(workdir, f"{label}__info.json")) as f:
    return json.load(f)


def exact_topk(db, q, distance):
  db64, q64 = db.astype(np.float64), q.astype(np.float64)
  if distance == "dot_product":
    return np.argsort(-(q64 @ db64.T), axis=1, kind="stable")[:, :K]
  d = (q64 ** 2).sum(1)[:, None] - 2 * q64 @ db64.T + (db64 ** 2).sum(1)[None, :]
  return np.argsort(d, axis=1, kind="stable")[:, :K]


def recall(idx, truth):
  return float(np.mean([len(set(a) & set(b)) / K for a, b in zip(idx, truth)]))


def compare(a_idx, a_dist, b_idx, b_dist):
  same_list = np.all(a_idx == b_idx, axis=1)
  mask = a_idx == b_idx
  max_delta = float(np.max(np.abs(a_dist - b_dist)[mask])) if mask.any() else float("nan")
  return dict(identical_lists=int(same_list.sum()), n=int(len(a_idx)),
              max_abs_dist_delta=max_delta)


def write_rust_fixture(out_dir, name, db, q, config, idx, dist):
  d = os.path.join(out_dir, name)
  os.makedirs(d, exist_ok=True)
  np.ascontiguousarray(db, dtype="<f4").tofile(os.path.join(d, "dataset.f32"))
  np.ascontiguousarray(q, dtype="<f4").tofile(os.path.join(d, "queries.f32"))
  np.ascontiguousarray(idx, dtype="<u4").tofile(os.path.join(d, "ref_indices.u32"))
  np.ascontiguousarray(dist, dtype="<f4").tofile(os.path.join(d, "ref_distances.f32"))
  with open(os.path.join(d, "config.pbtxt"), "w") as f:
    f.write(config)
  with open(os.path.join(d, "meta.txt"), "w") as f:
    f.write(f"{db.shape[0]} {db.shape[1]} {q.shape[0]} {K}\n")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--build-dir")
  p.add_argument("--python", default=sys.executable,
                 help="interpreter with the reference wheel installed")
  p.add_argument("--core-python",
                 help="interpreter for scann-core (default: --python); needs numpy and a "
                      "protobuf runtime at least as new as scann-core's protoc")
  p.add_argument("--out")
  p.add_argument("--worker")
  p.add_argument("--workdir")
  p.add_argument("--ref-configs")
  p.add_argument("--preimport", choices=["tensorflow", "block-tensorflow"])
  args = p.parse_args()

  if args.worker:
    worker(args.workdir, args.worker, args.ref_configs, args.preimport)
    return

  out = os.path.abspath(args.out)
  # Workers run with cwd = a temp dir, so resolve interpreters now.
  args.python = os.path.abspath(args.python)
  args.core_python = os.path.abspath(args.core_python or args.python)
  os.makedirs(out, exist_ok=True)
  workdir = tempfile.mkdtemp(prefix="scann-equiv-")
  atexit.register(shutil.rmtree, workdir, ignore_errors=True)
  rng = np.random.default_rng(SEED)
  data = {}
  for name, spec in DATASETS.items():
    data[name] = make_dataset(rng, **spec)
    np.save(os.path.join(workdir, f"{name}_db.npy"), data[name][0])
    np.save(os.path.join(workdir, f"{name}_q.npy"), data[name][1])

  candidate_path = os.path.join(os.path.abspath(args.build_dir), "python")
  infos = {
      "ref": run_worker(args.python, workdir, "ref", None),
      "ref2": run_worker(args.python, workdir, "ref2", None),
      "core": run_worker(args.core_python, workdir, "core", candidate_path,
                         ref_configs=os.path.join(workdir, "ref__configs.json")),
  }
  assert "site-packages" in infos["ref"]["scann_file"], infos["ref"]
  assert infos["core"]["scann_file"].startswith(candidate_path), infos["core"]
  with open(os.path.join(workdir, "ref__configs.json")) as f:
    ref_configs = json.load(f)

  report = {"implementations": infos, "k": K, "seed": SEED, "repeats": REPEATS,
            "deterministic": {}, "nondeterministic": {}}
  fixtures = os.path.join(out, "fixtures")
  ok = True
  print(f"reference: {infos['ref']['scann_file']} "
        f"(tensorflow imported: {infos['ref']['tensorflow_imported']})")
  print(f"candidate: {infos['core']['scann_file']} "
        f"(tensorflow imported: {infos['core']['tensorflow_imported']})\n")

  print("Deterministic configs -- must be bit-identical")
  hdr = (f"{'config':28s} {'mode':8s} {'cfg==':6s} {'ref~ref2':9s} {'core==ref':10s} "
         f"{'max|Δdist|':11s} {'recall ref':10s} {'recall core':11s}")
  print(hdr)
  print("-" * len(hdr))
  for name, (ds, distance, deterministic, _) in CONFIGS.items():
    if not deterministic:
      continue
    res = {m: np.load(os.path.join(workdir, f"{m}__{name}.npz")) for m in ("ref", "ref2", "core")}
    db, q = data[ds]
    truth = exact_topk(db, q, distance)
    entry = {"config_text_identical": infos["core"]["config_match"][name]}
    for mode in ("single", "batched"):
      i_, d_ = f"{mode}_idx", f"{mode}_dist"
      det = compare(res["ref"][i_], res["ref"][d_], res["ref2"][i_], res["ref2"][d_])
      cmp_ = compare(res["ref"][i_], res["ref"][d_], res["core"][i_], res["core"][d_])
      r_ref, r_core = recall(res["ref"][i_], truth), recall(res["core"][i_], truth)
      entry[mode] = dict(reference_self=det, core_vs_reference=cmp_,
                         recall_reference=r_ref, recall_core=r_core)
      passed = (cmp_["identical_lists"] == cmp_["n"] and cmp_["max_abs_dist_delta"] <= 1e-5
                and r_ref == r_core and entry["config_text_identical"])
      ok &= passed
      print(f"{name:28s} {mode:8s} {str(entry['config_text_identical']):6s} "
            f"{det['identical_lists']:>3d}/{det['n']:<5d} {cmp_['identical_lists']:>3d}/{cmp_['n']:<6d} "
            f"{cmp_['max_abs_dist_delta']:<11.3e} {r_ref:<10.4f} {r_core:<11.4f}"
            f"{'' if passed else '  <-- MISMATCH'}")
    report["deterministic"][name] = entry
    write_rust_fixture(fixtures, name, db, q, ref_configs[name],
                       res["ref"]["single_idx"], res["ref"]["single_dist"])

  print(f"\nNondeterministic upstream (random_init=True) -- recall@{K} over {REPEATS} "
        "trainings per implementation")
  hdr = (f"{'config':28s} {'cfg==':6s} {'distinct ref':12s} {'distinct core':13s} "
         f"{'recall ref mean±sd [min,max]':32s} {'recall core mean±sd [min,max]':32s}")
  print(hdr)
  print("-" * len(hdr))
  for name, (ds, distance, deterministic, _) in CONFIGS.items():
    if deterministic:
      continue
    db, q = data[ds]
    truth = exact_topk(db, q, distance)
    runs = {m: np.load(os.path.join(workdir, f"{m}__{name}.npy")) for m in ("ref", "core")}
    rec = {m: np.array([recall(r, truth) for r in runs[m]]) for m in runs}
    distinct = {m: len({r.tobytes() for r in runs[m]}) for m in runs}
    se = np.sqrt(rec["ref"].var(ddof=1) / REPEATS + rec["core"].var(ddof=1) / REPEATS)
    diff = abs(rec["ref"].mean() - rec["core"].mean())
    passed = diff <= max(3 * se, 1e-3) and infos["core"]["config_match"][name]
    ok &= passed
    fmt = lambda r: f"{r.mean():.4f}±{r.std(ddof=1):.4f} [{r.min():.4f},{r.max():.4f}]"
    print(f"{name:28s} {str(infos['core']['config_match'][name]):6s} "
          f"{distinct['ref']:>3d}/{REPEATS:<8d} {distinct['core']:>3d}/{REPEATS:<9d} "
          f"{fmt(rec['ref']):32s} {fmt(rec['core']):32s}{'' if passed else '  <-- DIFFERENT'}")
    report["nondeterministic"][name] = dict(
        config_text_identical=infos["core"]["config_match"][name],
        distinct_indexes={m: distinct[m] for m in runs},
        recall={m: rec[m].tolist() for m in runs},
        mean_diff=float(diff), three_se=float(3 * se))

  with open(os.path.join(out, "report.json"), "w") as f:
    json.dump(report, f, indent=1)
  print(f"\nreport: {os.path.join(out, 'report.json')}\nrust fixtures: {fixtures}")
  print("RESULT:", "EQUIVALENT" if ok else "MISMATCH")
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
