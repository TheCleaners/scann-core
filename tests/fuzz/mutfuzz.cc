// Copyright 2026 Elias Benali (@ebenali) and TheCleaners.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Mutation fuzzer: random add / update / delete / maintenance / retrain /
// serialize+reload on a ScannInterface, mirrored by a shadow copy of the
// vectors (ScaNN semantics: delete moves the last point into the freed
// slot). After each step, checks n_points; periodically checks exhaustive
// searches (all leaves, pre-reorder = all) against brute force over the
// shadow for configs with exact float reordering / scoring.
//
// Expectations that are approximations by design (not index bugs):
//  - int8 (fixed-point) scoring and reordering quantize with per-dimension
//    ranges fixed when the index is built (fixed_point_multiplier_quantile
//    = 1: the largest |x_k| of the build dataset). A vector added later with
//    a larger component is clipped to that range (c = clip(x); codes are
//    -128..127, so negative values clip at -128/127 of it). Squared L2
//    is scored as |q|^2 + |x|^2 - 2 q.c with x's float norm (stored at
//    add time), dot product as -q.c. A successful retrain (int8 reordering:
//    the float data is reconstructed from the int8 data) keeps the clipped
//    values and re-derives the ranges and norms from them; the data is then
//    quantized again, so each retrain can add one more rounding. The shadow
//    keeps that model per datapoint (`stored`, `norm2`, `requant`: retrains
//    since the datapoint was written) and allows one quantization step per
//    dimension per rounding.
//  - An upper tree searches only its own num_leaves_to_search partitions;
//    leaves_to_search at search time overrides the lower tree only, so
//    "all leaves" is exhaustive only when the upper tree searches all of its
//    partitions. tree_upper_* does; tree_upper2_* (2 of 3) is checked for
//    exact distances but not for the exact top-k.
//  - bfloat16 and AH without reordering: see the tolerances in check().
// Every checked search is also repeated and must match bit for bit.
//
// Configs with "l2mips" search squared L2 through an inner-product index
// (L2AsDotProduct): the shadow holds the vectors as given, the mutator gets
// them through ScannInterface::ToStoredDatapoints (with the extra
// coordinate), and the searches return squared L2 distances.
//
// With INJECT=1 in the environment, some adds and updates on tree + AH
// indexes get a precomputed leaf artifact of the wrong type for one of their
// leaves: the mutation must fail and leave the index exactly as it was
// (checked with exhaustive searches before and after).
//
// usage: mutfuzz <config> <seed> <steps> [batch]
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

#include "scann/scann_ops/cc/scann.h"
#include "scann/tree_x_hybrid/mutator.h"
#include "scann/tree_x_hybrid/tree_ah_hybrid_residual.h"
#include "scann/tree_x_hybrid/tree_x_hybrid_smmd.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann_core/config_builder.h"

using namespace research_scann;
using Mutator = SingleMachineSearcherBase<float>::Mutator;
using scann_core::ConfigBuilder;
namespace sc = scann_core;

static int g_fail = 0;
using TreeAhArtifacts = TreeXHybridMutator<TreeAHHybridResidual>::TreeXPrecomputedMutationArtifacts;
using SmmdArtifacts = TreeXHybridMutator<TreeXHybridSMMD<float>>::TreeXPrecomputedMutationArtifacts;
struct BogusArtifacts : UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts {};
#define FAILF(...) do { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); if (++g_fail > 5) std::exit(3); } while (0)

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  // SCANN_TEST_FORCE_AVX2=1: AVX2 kernels (and the canonical LUT16 layout) on
  // an AVX-512 CPU, as in api_exercise.
  if (const char* e = std::getenv("SCANN_TEST_FORCE_AVX2"); e && *e == '1') {
    research_scann::flags_internal::should_use_avx512 = false;
    research_scann::flags_internal::should_use_avx512_vnni = false;
    research_scann::flags_internal::should_use_amx = false;
  }
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <config> <seed> <steps> [batch]\n", argv[0]);
    return 2;
  }
  // Serialize + reload steps write here ($TMPDIR or the system temp dir);
  // removed at the end.
  const std::filesystem::path work_dir =
      std::filesystem::temp_directory_path() /
      ("scann_mutfuzz_" + std::to_string(::getpid()));
  std::string cfg = argv[1];
  int seed = std::atoi(argv[2]);
  int steps = std::atoi(argv[3]);
  int max_batch = argc > 4 ? std::atoi(argv[4]) : 1;
  const size_t dim = 8;
  size_t n0 = 600;
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd;
  // Spherical trees store unit vectors, so the shadow does too.
  const bool unit = std::string(argv[1]).find("sph") != std::string::npos;
  auto rand_vec = [&] {
    std::vector<float> v(dim); for (auto& x : v) x = nd(rng);
    if (unit) { double n2 = 0; for (float x : v) n2 += double(x) * x; for (auto& x : v) x /= std::sqrt(n2); }
    return v;
  };

  bool l2 = cfg.find("l2") != std::string::npos;
  ConfigBuilder b(10, l2 ? sc::DistanceMeasure::kSquaredL2 : sc::DistanceMeasure::kDotProduct, dim);
  bool exact = true;  // exact float distances after reorder / scoring
  bool exhaustive = true;  // "all leaves" searches every datapoint
  bool clip_int8 = false;  // int8 quantization: shadow distances to the clipped vector
  sc::TreeOptions t; t.num_leaves = 12; t.num_leaves_to_search = 4; t.training_sample_size = n0; t.random_init = false;
  t.min_partition_size = 5;
  if (cfg.find("incr") != std::string::npos) t.incremental_threshold_fraction = 0.2;
  if (cfg.find("soar") != std::string::npos) t.soar_lambda = 1.5;
  if (cfg.find("avq") != std::string::npos) t.avq = 2.5;
  if (cfg.find("sph") != std::string::npos) t.spherical = true;
  if (cfg.find("qc") != std::string::npos) t.quantize_centroids = true;
  if (cfg.find("tree") != std::string::npos) b.Tree(t);
  // l2mips: squared L2 through an inner-product index (L2AsDotProduct);
  // mutations go through ScannInterface::ToStoredDatapoints.
  const bool mips = cfg.find("l2mips") != std::string::npos;
  if (mips) b.L2AsDotProduct();
  if (cfg.find("pca") != std::string::npos) { sc::PcaOptions p; p.reduction_dim = 6; b.Pca(p); }
  if (cfg.find("trunc") != std::string::npos) b.Truncate(6);
  if (cfg.find("upper") != std::string::npos) {
    sc::UpperTreeOptions u; u.num_leaves = 3; u.num_leaves_to_search = 3;
    if (cfg.find("upper2") != std::string::npos) { u.num_leaves_to_search = 2; exhaustive = false; }
    b.UpperTree(u);
  }
  if (cfg.find("auto") != std::string::npos) { b.Autopilot(cfg.find("autoinc") != std::string::npos ? sc::IncrementalMode::kOnlineIncremental : sc::IncrementalMode::kOnline); exact = false; }
  if (cfg.find("auto") != std::string::npos) {
  } else if (cfg.find("ah") != std::string::npos) {
    sc::AhOptions a; a.dimensions_per_block = cfg.find("dpb3") != std::string::npos ? 3 : 2; a.training_sample_size = n0;
    if (cfg.find("lut256") != std::string::npos) a.hash_type = sc::HashType::kLut256;
    b.ScoreAh(a);
    if (cfg.find("noreorder") == std::string::npos) {
      sc::ReorderOptions r; r.reordering_num_neighbors = 40;
      if (cfg.find("rint8") != std::string::npos) { r.quantize = sc::Quantization::kInt8; exact = false; clip_int8 = true; }
      if (cfg.find("rbf16") != std::string::npos) { r.quantize = sc::Quantization::kBfloat16; exact = false; }
      b.Reorder(r);
    } else exact = false;
  } else {
    auto q = sc::Quantization::kFloat32;
    if (cfg.find("int8") != std::string::npos) { q = sc::Quantization::kInt8; exact = false; clip_int8 = true; }
    if (cfg.find("bf16") != std::string::npos) { q = sc::Quantization::kBfloat16; exact = false; }
    b.ScoreBruteForce(q);
  }
  auto text = b.BuildText(n0);
  if (!text.ok()) { std::printf("config: %s\n", text.status().ToString().c_str()); return 2; }

  std::vector<std::vector<float>> shadow;
  std::vector<float> flat;
  for (size_t i = 0; i < n0; ++i) { shadow.push_back(rand_vec()); flat.insert(flat.end(), shadow.back().begin(), shadow.back().end()); }
  // int8 model (see the header comment): per-dimension range, and per
  // datapoint the (clipped) vector the index holds and the norm it uses.
  std::vector<float> qmax(dim, 0);
  std::vector<std::vector<float>> stored = shadow;
  std::vector<double> norm2;
  std::vector<int> requant(shadow.size(), 0);
  auto sqnorm = [&](const std::vector<float>& v) { double n = 0; for (float x : v) n += double(x) * x; return n; };
  for (auto& v : shadow) norm2.push_back(sqnorm(v));
  auto set_qmax = [&] {  // at build and after a successful retrain
    std::fill(qmax.begin(), qmax.end(), 0.f);
    for (auto& v : stored) for (size_t k = 0; k < dim; ++k) qmax[k] = std::max(qmax[k], std::fabs(v[k]));
    for (size_t i = 0; i < stored.size(); ++i) norm2[i] = sqnorm(stored[i]);
  };
  set_qmax();
  // int8 codes span -128..127 with 127 = qmax: negative values clip at
  // -qmax * 128 / 127.
  auto clipped = [&](std::vector<float> v) { for (size_t k = 0; k < dim; ++k) v[k] = std::clamp(v[k], -qmax[k] * 128 / 127, qmax[k]); return v; };
  auto shadow_set = [&](size_t i, const std::vector<float>& v) { shadow[i] = v; stored[i] = clipped(v); norm2[i] = sqnorm(v); requant[i] = 0; };
  auto shadow_add = [&](const std::vector<float>& v) { shadow.push_back(v); stored.push_back(clipped(v)); norm2.push_back(sqnorm(v)); requant.push_back(0); };
  auto shadow_remove = [&](size_t i) {
    shadow[i] = shadow.back(); shadow.pop_back();
    stored[i] = stored.back(); stored.pop_back();
    norm2[i] = norm2.back(); norm2.pop_back();
    requant[i] = requant.back(); requant.pop_back();
  };
  auto retrained = [&] { for (int& r : requant) ++r; set_qmax(); };
  auto s = std::make_unique<ScannInterface>();
  auto st = s->Initialize(flat, n0, *text, 1);
  if (!st.ok()) { std::printf("init: %s\n", st.ToString().c_str()); return 2; }

  // The vector as the index stores it (with l2mips, with its extra
  // coordinate); a DatapointPtr to it stays valid until the next call.
  std::vector<float> stored_buf;
  auto to_stored = [&](const std::vector<float>& v) {
    if (!mips) return MakeDatapointPtr(v.data(), dim);
    s->ToStoredDatapoints(v, &stored_buf);
    return MakeDatapointPtr(stored_buf.data(), stored_buf.size());
  };
  auto dist = [&](const std::vector<float>& q, const std::vector<float>& x) {
    double d = 0;
    for (size_t k = 0; k < dim; ++k) d += l2 ? (q[k] - x[k]) * (q[k] - x[k]) : -double(q[k]) * x[k];
    return d;
  };
  auto check = [&](const char* when, int step) {
    if (s->n_points() != shadow.size()) FAILF("%s step %d: n_points %zu != shadow %zu", when, step, s->n_points(), shadow.size());
    if (shadow.empty()) return;
    for (int qi = 0; qi < 3; ++qi) {
      std::vector<float> q = (qi == 0) ? shadow[rng() % shadow.size()] : rand_vec();
      NNResultsVector res;
      auto st = s->Search(MakeDatapointPtr(q.data(), dim), &res, 5, 100000, 100000);
      if (!st.ok()) { FAILF("%s step %d: search: %s", when, step, st.ToString().c_str()); continue; }
      {  // Searches are deterministic: repeating one on an unchanged index
         // returns the same results, bit for bit.
        NNResultsVector again;
        (void)s->Search(MakeDatapointPtr(q.data(), dim), &again, 5, 100000, 100000);
        if (again != res && !std::getenv("LOOSE")) FAILF("%s step %d: repeated search differs", when, step);
      }
      for (auto& [idx, d] : res) {
        if (idx >= shadow.size()) { FAILF("%s step %d: result index %u >= n %zu", when, step, idx, shadow.size()); continue; }
        double ref = dist(q, shadow[idx]);
        // Approximate scoring: bfloat16 keeps 8 mantissa bits (observed
        // error <= 0.6% of 1 + |ref|). AH without reordering scores q.x^ for
        // x's AH reconstruction x^: the error q.(x - x^) scales with |q||x|,
        // not with |ref| (q.x can be near 0); observed <= 0.44 |q||x| over
        // 100 runs per config.
        double tol = exact ? 1e-3 * (1 + std::fabs(ref)) : 0.25 * (1 + std::fabs(ref));
        if (cfg.find("bf16") != std::string::npos) tol = 0.02 * (1 + std::fabs(ref));
        if (mips && cfg.find("bf16") != std::string::npos) {
          // Through the reduction, bfloat16 rounds q'.x' (~ (|q|^2 + |x|^2)
          // / 2), not the difference q - x: the error scales with the norms
          // even when the distance is ~0.
          double qn = 0, xn = 0;
          for (size_t k = 0; k < dim; ++k) { qn += double(q[k]) * q[k]; xn += double(shadow[idx][k]) * shadow[idx][k]; }
          tol += 0.01 * (qn + xn);
        }
        if (cfg.find("noreorder") != std::string::npos) {
          double qn = 0, xn = 0;
          for (size_t k = 0; k < dim; ++k) { qn += double(q[k]) * q[k]; xn += double(shadow[idx][k]) * shadow[idx][k]; }
          tol = 0.75 * std::sqrt(qn * xn) + 1e-3 * (1 + std::fabs(ref));
        }
        if (clip_int8) {
          double qc = 0, qq = 0, step_err = 0;
          for (size_t k = 0; k < dim; ++k) {
            qc += double(q[k]) * stored[idx][k];
            qq += double(q[k]) * q[k];
            step_err += std::fabs(q[k]) * qmax[k] / 127;
          }
          ref = l2 ? qq + norm2[idx] - 2 * qc : -qc;
          tol = (l2 ? 2 : 1) * step_err * (1 + requant[idx]) + 1e-3 * (1 + std::fabs(ref));
        }
        double got = d;
        if (std::fabs(ref - got) > tol) FAILF("%s step %d: index %u distance %g, shadow says %g", when, step, idx, got, ref);
      }
      if (exact && exhaustive) {
        std::vector<double> all;
        for (auto& x : shadow) all.push_back(dist(q, x));
        std::sort(all.begin(), all.end());
        size_t k = std::min<size_t>(5, shadow.size());
        if (res.size() != k) FAILF("%s step %d: %zu results, expected %zu", when, step, res.size(), k);
        else if (std::fabs(res[k - 1].second - all[k - 1]) > 1e-3 * (1 + std::fabs(all[k - 1])))
          FAILF("%s step %d: k-th distance %g, brute force %g", when, step, res[k - 1].second, all[k - 1]);
      }
    }
  };
  const bool inject = std::getenv("INJECT") != nullptr;
  int injected = 0;
  // Exhaustive search results for a fixed probe set: equal before and after
  // a failed mutation iff nothing observable changed.
  std::vector<std::vector<float>> probes;
  for (int i = 0; i < 8; ++i) probes.push_back(rand_vec());
  auto snapshot = [&] {
    std::vector<NNResultsVector> out;
    for (auto& q : probes) {
      NNResultsVector res;
      (void)s->Search(MakeDatapointPtr(q.data(), dim), &res, 20, 100000, 100000);
      std::sort(res.begin(), res.end());
      out.push_back(res);
    }
    return std::make_pair(s->n_points(), out);
  };
  // Exact: searches are deterministic (SOAR candidates used to be merged in
  // hash-map order; LOOSE=1 restores the old 1e-5 tolerance).
  auto same = [](const auto& a, const auto& b) {
    if (a.first != b.first || a.second.size() != b.second.size()) return false;
    for (size_t p = 0; p < a.second.size(); ++p) {
      if (a.second[p].size() != b.second[p].size()) return false;
      for (size_t j = 0; j < a.second[p].size(); ++j) {
        auto [i1, d1] = a.second[p][j]; auto [i2, d2] = b.second[p][j];
        if (i1 != i2 || (!std::getenv("LOOSE") ? d1 != d2 : std::fabs(d1 - d2) > 1e-5 * (1 + std::fabs(d1)))) return false;
      }
    }
    return true;
  };
  // Artifacts for v with leaf `bad` replaced by a bogus one; null if this
  // index's artifacts aren't TreeAHHybridResidual's.
  auto bogus_artifacts = [&](Mutator* m, const std::vector<float>& v) -> std::unique_ptr<UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts> {
    auto a = m->ComputePrecomputedMutationArtifacts(to_stored(v));
    if (auto* t = dynamic_cast<TreeAhArtifacts*>(a.get()); t && !t->tokens.empty()) {
      t->leaf_precomputed_artifacts[rng() % t->tokens.size()] = std::make_unique<BogusArtifacts>();
      return a;
    }
    if (auto* t = dynamic_cast<SmmdArtifacts*>(a.get()); t && !t->tokens.empty()) {
      t->leaf_precomputed_artifacts[rng() % t->tokens.size()] = std::make_unique<BogusArtifacts>();
      return a;
    }
    return nullptr;
  };
  check("init", 0);
  int retrains = 0, reloads = 0;
  for (int step = 1; step <= steps; ++step) {
    auto m_or = s->GetMutator();
    if (!m_or.ok()) { FAILF("GetMutator: %s", m_or.status().ToString().c_str()); break; }
    auto* m = *m_or;
    int op = rng() % 100;
    if (inject && !shadow.empty() && rng() % 4 == 0) {
      auto v = rand_vec();
      if (std::getenv("CONTROL2")) {
        auto b1 = snapshot(); auto b2 = snapshot();
        ++injected;
        if (!same(b1, b2)) {
          FAILF("step %d: two snapshots differ", step);
          if (std::getenv("DEBUG"))
            for (size_t p = 0; p < probes.size(); ++p)
              for (size_t j = 0; j < std::min(b1.second[p].size(), b2.second[p].size()); ++j)
                if (b1.second[p][j] != b2.second[p][j])
                  std::printf("  probe %zu #%zu: %u:%.9g vs %u:%.9g\n", p, j, b1.second[p][j].first, b1.second[p][j].second, b2.second[p][j].first, b2.second[p][j].second);
        }
        continue;
      }
      auto a = bogus_artifacts(m, v);
      if (a) {
        auto before = snapshot();
        UntypedSingleMachineSearcherBase::MutationOptions mo{.precomputed_mutation_artifacts = a.get()};
        const bool is_update = rng() % 2;
        DatapointIndex i = rng() % shadow.size();
        bool ok = std::getenv("CONTROL") ? false : is_update ? m->UpdateDatapoint(to_stored(v), i, mo).ok()
                            : m->AddDatapoint(to_stored(v), "", mo).ok();
        if (ok) {
          // Leaves that take no artifacts (brute force) ignore the bogus
          // one: a normal mutation.
          if (is_update) shadow_set(i, v); else shadow_add(v);
        } else if (++injected, 0) {
        } else if (auto after = snapshot(); !same(after, before)) {
          FAILF("step %d: failed %s of %u changed the index", step, is_update ? "update" : "add", i);
          if (std::getenv("DEBUG")) {
            std::printf("  n %zu -> %zu\n", before.first, after.first);
            for (size_t p = 0; p < probes.size(); ++p) {
              if (before.second[p] == after.second[p]) continue;
              std::printf("  probe %zu:\n   before:", p);
              for (auto& [x, d] : before.second[p]) std::printf(" %u:%.6g", x, d);
              std::printf("\n   after: ");
              for (auto& [x, d] : after.second[p]) std::printf(" %u:%.6g", x, d);
              std::printf("\n");
              break;
            }
          }
        }
        check("inject", step);
        continue;
      }
    }
    if (max_batch > 1) m->set_mutation_threadpool(s->parallel_query_pool());
    if (op < 45) {  // add a batch
      int bs = 1 + rng() % max_batch;
      DenseDataset<float> ds;
      std::vector<std::vector<float>> vs;
      for (int i = 0; i < bs; ++i) { vs.push_back(rand_vec()); (void)ds.Append(to_stored(vs.back()), ""); }
      auto pre = m->ComputePrecomputedMutationArtifacts(ds, s->parallel_query_pool());
      for (int i = 0; i < bs; ++i) {
        UntypedSingleMachineSearcherBase::MutationOptions mo{.precomputed_mutation_artifacts = pre[i].get()};
        auto r = m->AddDatapoint(to_stored(vs[i]), "", mo);
        if (!r.ok()) { FAILF("step %d add: %s", step, r.status().ToString().c_str()); continue; }
        if (*r != shadow.size()) FAILF("step %d add returned %u, expected %zu", step, *r, shadow.size());
        shadow_add(vs[i]);
      }
    } else if (op < 65 && !shadow.empty()) {  // update
      DatapointIndex i = rng() % shadow.size();
      auto v = rand_vec();
      auto r = m->UpdateDatapoint(to_stored(v), i, {});
      if (!r.ok()) FAILF("step %d update %u: %s", step, i, r.status().ToString().c_str());
      else { if (*r != i) FAILF("step %d update returned %u, expected %u", step, *r, i); shadow_set(i, v); }
    } else if (op < 95 && !shadow.empty()) {  // delete
      int nd_ = 1 + rng() % std::max(1, max_batch);
      for (int j = 0; j < nd_ && !shadow.empty(); ++j) {
        DatapointIndex i = (rng() % 4 == 0) ? shadow.size() - 1 : rng() % shadow.size();
        auto r = m->RemoveDatapoint(i);
        if (!r.ok()) { FAILF("step %d delete %u: %s", step, i, r.ToString().c_str()); continue; }
        shadow_remove(i);
      }
    } else if (op < 97) {
      auto r = s->RetrainAndReindex("");
      ++retrains;
      if (!r.ok()) std::printf("step %d retrain: %s (n=%zu)\n", step, r.status().ToString().c_str(), shadow.size());
      else retrained();
      check("retrain", step);
      continue;  // the mutator is stale after a retrain
    } else if (op < 99) {  // serialize + reload
      std::string dir = (work_dir / "art").string();
      std::filesystem::remove_all(dir); std::filesystem::create_directories(dir);
      auto a = s->Serialize(dir);
      if (!a.ok()) { FAILF("step %d serialize: %s", step, a.status().ToString().c_str()); continue; }
      auto art = ScannInterface::LoadArtifacts(s->config() ? *s->config() : ScannConfig(), *a);
      if (!art.ok()) { FAILF("step %d load: %s", step, art.status().ToString().c_str()); continue; }
      auto s2 = std::make_unique<ScannInterface>();
      auto st2 = s2->Initialize(*std::move(art));
      if (!st2.ok()) { FAILF("step %d reinit (n=%zu): %s", step, shadow.size(), st2.ToString().c_str()); continue; }
      s = std::move(s2);
      ++reloads;
      check("reload", step);
      continue;
    } else {
      auto hs = s->GetHealthStats();
      if (!hs.ok()) std::printf("health: %s\n", hs.status().ToString().c_str());
    }
    auto im = m->IncrementalMaintenance();
    if (!im.ok()) FAILF("step %d maintenance: %s", step, im.status().ToString().c_str());
    else if (im->has_value()) {
      auto r = s->RetrainAndReindex("");
      ++retrains;
      if (!r.ok()) std::printf("step %d maint-retrain: %s\n", step, r.status().ToString().c_str());
      else retrained();
    }
    check("step", step);
  }
  std::printf("done cfg=%s seed=%d n=%zu retrains=%d reloads=%d injected=%d failures=%d\n", cfg.c_str(), seed, shadow.size(), retrains, reloads, injected, g_fail);
  std::error_code ec;
  std::filesystem::remove_all(work_dir, ec);
  return g_fail ? 1 : 0;
}
