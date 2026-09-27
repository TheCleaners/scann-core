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

// Regression tests for loading and serializing index directories.
//
//  - Damaged or inconsistent directories must fail to load with an error.
//    Upstream trusted every file: out of 57 damaged variants of small
//    indexes, 7 segfaulted, 1 died of SIGFPE, 1 of LOG(FATAL), several threw
//    C++ exceptions out of Status-returning functions (std::terminate), and
//    20 loaded without error, most of them silently wrong. The variants here
//    are generated at run time from freshly built indexes: damaged .npy
//    headers, wrong dtypes and shapes, out-of-range tokens, files swapped
//    between indexes (as an interrupted in-place re-serialize leaves them),
//    and manifest errors.
//  - A tree with every point deleted must serialize to a directory that
//    loads (upstream wrote no data for it), and so must a tree with
//    bfloat16 brute-force leaves.
//  - SerializeToDirectory: a re-serialize replaces the whole index, and one
//    that fails midway leaves a directory that fails to load, cleanly.
//
// Run it under ASan/UBSan: a crash or a sanitizer report is a failure too.

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "google/protobuf/text_format.h"
#include "scann/data_format/datapoint.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/scann_ops/scann_assets.pb.h"
#include "scann/utils/io_npy.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/io_oss_wrapper.h"
#include "scann/utils/types.h"
#include "scann_core/config_builder.h"

namespace {

namespace fs = std::filesystem;
using research_scann::ConstSpan;
using research_scann::DatapointIndex;
using research_scann::DatapointPtr;
using research_scann::NNResultsVector;
using research_scann::ScannAsset;
using research_scann::ScannAssets;
using research_scann::ScannInterface;

int g_failures = 0;
int g_checks = 0;

void Fail(const std::string& what) {
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++g_failures;
}

bool Ok(const absl::Status& s, const std::string& what) {
  if (!s.ok()) Fail(absl::StrCat(what, ": ", s.ToString()));
  return s.ok();
}

constexpr int kLeaves = 12;
fs::path g_root;

using Vec = std::vector<float>;
DatapointPtr<float> Ptr(const Vec& v) {
  return DatapointPtr<float>(nullptr, v.data(), v.size(), v.size());
}

Vec RandomData(size_t n, size_t dim, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd;
  Vec v(n * dim);
  for (float& x : v) x = nd(rng);
  return v;
}

// --- configs -----------------------------------------------------------------

struct Spec {
  bool tree = true;
  int leaves = kLeaves;
  std::optional<double> soar;
  enum { kBf, kBfInt8, kBfBf16, kAh, kAhReorder } scoring = kAhReorder;
};

std::string Config(const Spec& spec, size_t n, size_t dim) {
  scann_core::ConfigBuilder b(10, scann_core::DistanceMeasure::kDotProduct,
                              dim);
  if (spec.tree) {
    scann_core::TreeOptions tree;
    tree.num_leaves = spec.leaves;
    tree.num_leaves_to_search = 4;
    tree.training_sample_size = static_cast<int64_t>(n);
    tree.random_init = false;
    tree.soar_lambda = spec.soar;
    b.Tree(tree);
  }
  scann_core::AhOptions ah;
  scann_core::ReorderOptions reorder;
  reorder.reordering_num_neighbors = 40;
  switch (spec.scoring) {
    case Spec::kBf:
      b.ScoreBruteForce();
      break;
    case Spec::kBfInt8:
      b.ScoreBruteForce(scann_core::Quantization::kInt8);
      break;
    case Spec::kBfBf16:
      b.ScoreBruteForce(scann_core::Quantization::kBfloat16);
      break;
    case Spec::kAh:
      b.ScoreAh(ah);
      break;
    case Spec::kAhReorder:
      b.ScoreAh(ah).Reorder(reorder);
      break;
  }
  auto text = b.BuildText(n);
  if (!Ok(text.status(), "ConfigBuilder")) return "";
  return *text;
}

bool Build(ScannInterface& s, const Spec& spec, size_t n, size_t dim,
           uint32_t seed, Vec* data_out = nullptr) {
  Vec data = RandomData(n, dim, seed);
  if (!Ok(s.Initialize(ConstSpan<float>(data), n, Config(spec, n, dim), 1),
          "Initialize"))
    return false;
  if (data_out) *data_out = std::move(data);
  return true;
}

// Builds an index and serializes it into g_root/name.
std::string BuildAndSave(const std::string& name, const Spec& spec, size_t n,
                         size_t dim, uint32_t seed) {
  ScannInterface s;
  if (!Build(s, spec, n, dim, seed)) return "";
  const std::string dir = (g_root / name).string();
  fs::create_directories(dir);
  if (!Ok(s.SerializeToDirectory(dir, /*relative_path=*/true),
          name + ": SerializeToDirectory"))
    return "";
  return dir;
}

// LoadArtifacts + Initialize, then a search and a mutation, as a user would.
absl::Status Load(const std::string& dir, size_t* n_points = nullptr) {
  auto artifacts = ScannInterface::LoadArtifacts(dir);
  if (!artifacts.ok()) return artifacts.status();
  ScannInterface s;
  absl::Status st = s.Initialize(*std::move(artifacts));
  if (!st.ok()) return st;
  if (n_points) *n_points = s.n_points();
  Vec q = RandomData(1, s.dimensionality(), 99);
  NNResultsVector res;
  st = s.Search(Ptr(q), &res, 10, -1, kLeaves);
  if (!st.ok()) return st;
  for (const auto& [idx, dist] : res)
    if (idx >= s.n_points())
      return absl::InternalError(absl::StrCat("result index ", idx,
                                              " out of range"));
  return absl::OkStatus();
}

void ExpectError(const std::string& dir, const std::string& what,
                 const std::string& expected = "") {
  ++g_checks;
  absl::Status st = Load(dir);
  if (st.ok()) {
    Fail(absl::StrCat(what, ": loaded without error"));
  } else if (!expected.empty() && !absl::StrContains(st.message(), expected)) {
    Fail(absl::StrCat(what, ": error \"", st.ToString(), "\" lacks \"",
                      expected, "\""));
  }
}

void ExpectLoads(const std::string& dir, const std::string& what,
                 std::optional<size_t> n = std::nullopt) {
  ++g_checks;
  size_t n_points = 0;
  if (Ok(Load(dir, &n_points), what + ": load") && n && *n != n_points)
    Fail(absl::StrCat(what, ": ", n_points, " points, expected ", *n));
}

// --- directory and file surgery -------------------------------------------------

std::string Copy(const std::string& src, const std::string& name) {
  const fs::path dst = g_root / "cases" / name;
  fs::remove_all(dst);
  fs::create_directories(dst.parent_path());
  fs::copy(src, dst, fs::copy_options::recursive);
  return dst.string();
}

void CopyFile(const std::string& from_dir, const std::string& to_dir,
              const std::string& file) {
  fs::copy_file(fs::path(from_dir) / file, fs::path(to_dir) / file,
                fs::copy_options::overwrite_existing);
}

void WriteFile(const std::string& path, const std::string& contents) {
  std::ofstream(path, std::ios::binary | std::ios::trunc) << contents;
}

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

ScannAssets ReadAssets(const std::string& dir) {
  ScannAssets assets;
  google::protobuf::TextFormat::ParseFromString(
      ReadFile(dir + "/scann_assets.pbtxt"), &assets);
  return assets;
}

void WriteAssets(const std::string& dir, const ScannAssets& assets) {
  std::string text;
  google::protobuf::TextFormat::PrintToString(assets, &text);
  WriteFile(dir + "/scann_assets.pbtxt", text);
}

void DropAsset(const std::string& dir, ScannAsset::AssetType type) {
  ScannAssets assets = ReadAssets(dir), kept;
  for (const auto& a : assets.assets())
    if (a.asset_type() != type) *kept.add_assets() = a;
  WriteAssets(dir, kept);
}

void AddAsset(const std::string& dir, ScannAsset::AssetType type,
              const std::string& path) {
  ScannAssets assets = ReadAssets(dir);
  ScannAsset* a = assets.add_assets();
  a->set_asset_type(type);
  a->set_asset_path(path);
  WriteAssets(dir, assets);
}

// A .npy file with the given header dict and payload; header_len_override
// writes a wrong header length.
void WriteRawNpy(const std::string& path, const std::string& dict,
                 const std::string& payload, int header_len_override = -1) {
  std::string header = dict;
  while ((10 + header.size() + 1) % 64) header += ' ';
  header += '\n';
  const int len = header_len_override >= 0 ? header_len_override
                                            : static_cast<int>(header.size());
  std::string out("\x93NUMPY\x01\x00", 8);
  out += static_cast<char>(len & 0xff);
  out += static_cast<char>((len >> 8) & 0xff);
  WriteFile(path, out + header + payload);
}

std::string Dict(const std::string& descr, const std::string& shape,
                 bool fortran = false) {
  return absl::StrCat("{'descr': '", descr, "', 'fortran_order': ",
                      fortran ? "True" : "False", ", 'shape': ", shape, ", }");
}

std::string Zeros(size_t bytes) { return std::string(bytes, '\0'); }

template <typename T>
std::pair<std::vector<T>, std::vector<size_t>> ReadNpy(const std::string& p) {
  auto r = research_scann::NumpyToVectorAndShape<T>(p);
  if (!r.ok()) {
    Fail(absl::StrCat("reading ", p, ": ", r.status().ToString()));
    return {};
  }
  return *std::move(r);
}

template <typename T>
void WriteNpy(const std::string& p, const std::vector<T>& v,
              const std::vector<size_t>& leading_dims = {}) {
  Ok(research_scann::VectorToNumpy(p, v, leading_dims), "writing " + p);
}

// --- damaged directories ---------------------------------------------------------

void DamagedNpyHeaders(const std::string& bf) {
  const std::string ds = "/dataset.npy";
  const size_t n = 200, d = 8;
  std::string c;

  c = Copy(bf, "npy_bad_magic");
  WriteFile(c + ds, "\x93NUMPX\x01\x00" + Zeros(100));
  ExpectError(c, "bad magic", "magic");

  c = Copy(bf, "npy_header_len_past_end");  // upstream read past its buffer
  WriteRawNpy(c + ds, Dict("<f4", "(200, 8)"), "", 0xffff);
  ExpectError(c, "header length past the end", "header length");

  c = Copy(bf, "npy_unterminated_header");
  WriteFile(c + ds, std::string("\x93NUMPY\x01\x00\x10\x00", 10) +
                        "{'descr': '<f4'}");
  ExpectError(c, "unterminated header");

  c = Copy(bf, "npy_version_9");
  {
    std::string f = ReadFile(c + ds);
    f[6] = 9;
    WriteFile(c + ds, f);
  }
  ExpectError(c, "unknown .npy version", "version");

  c = Copy(bf, "npy_scalar_shape");  // upstream: segfault
  WriteRawNpy(c + ds, Dict("<f4", "()"), Zeros(4));
  ExpectError(c, "shape ()", "dimensions");

  c = Copy(bf, "npy_1d");  // upstream: loaded as 4800 rows of 1 dimension
  WriteRawNpy(c + ds, Dict("<f4", "(1600,)"), Zeros(1600 * 4));
  ExpectError(c, "1-D dataset", "dimensions");

  c = Copy(bf, "npy_shape_overflow");  // upstream: SIGFPE
  WriteRawNpy(c + ds, Dict("<f4", "(65536, 65536, 65536, 65536)"), "");
  ExpectError(c, "overflowing shape", "overflows");

  c = Copy(bf, "npy_dim_above_int_max");  // upstream: std::stoi threw
  WriteRawNpy(c + ds, Dict("<f4", "(99999999999, 8)"), "");
  ExpectError(c, "dimension above INT_MAX", "truncated");

  c = Copy(bf, "npy_truncated");
  {
    std::string f = ReadFile(c + ds);
    WriteFile(c + ds, f.substr(0, f.size() / 2));
  }
  ExpectError(c, "truncated", "truncated");

  c = Copy(bf, "npy_trailing_bytes");
  WriteFile(c + ds, ReadFile(c + ds) + Zeros(12));
  ExpectError(c, "trailing bytes", "bytes of data");

  c = Copy(bf, "npy_big_endian");  // upstream: loaded byte-swapped floats
  WriteRawNpy(c + ds, Dict(">f4", "(200, 8)"), Zeros(n * d * 4));
  ExpectError(c, "big-endian", "byte order");

  c = Copy(bf, "npy_int32_as_float");  // upstream: loaded int bits as floats
  WriteRawNpy(c + ds, Dict("<i4", "(200, 8)"), Zeros(n * d * 4));
  ExpectError(c, "int32 dtype", "dtype");

  c = Copy(bf, "npy_float64");
  WriteRawNpy(c + ds, Dict("<f8", "(200, 8)"), Zeros(n * d * 8));
  ExpectError(c, "float64 dtype", "dtype");

  c = Copy(bf, "npy_fortran_order");
  WriteRawNpy(c + ds, Dict("<f4", "(200, 8)", true), Zeros(n * d * 4));
  ExpectError(c, "Fortran order", "Fortran");

  c = Copy(bf, "npy_zero_columns");  // upstream: SIGFPE
  WriteRawNpy(c + ds, Dict("<f4", "(200, 0)"), "");
  ExpectError(c, "0 columns", "0 columns");

  c = Copy(bf, "npy_missing_file");
  fs::remove(c + ds);
  ExpectError(c, "missing dataset.npy", "Failed to open");

  c = Copy(bf, "npy_unknown_header_key");
  WriteRawNpy(c + ds,
              "{'descr': '<f4', 'fortran_order': False, 'shape': (200, 8), "
              "'x': 1}",
              Zeros(n * d * 4));
  ExpectError(c, "unknown header key", "header key");

  // Valid variants numpy may write: version 2.0, "|u1" for bytes.
  c = Copy(bf, "npy_version_2");
  {
    std::string f = ReadFile(c + ds);
    const size_t len = static_cast<unsigned char>(f[8]) |
                       (static_cast<unsigned char>(f[9]) << 8);
    std::string header = f.substr(10, len);
    header.insert(header.size() - 1, "  ");  // keep 64-byte alignment
    std::string out("\x93NUMPY\x02\x00", 8);
    const uint32_t hl = header.size();
    for (int i = 0; i < 4; ++i) out += static_cast<char>((hl >> (8 * i)) & 0xff);
    WriteFile(c + ds, out + header + f.substr(10 + len));
  }
  ExpectLoads(c, "npy version 2.0", 200);
}

void DamagedTrees(const std::string& tree_ah, const std::string& tree_ah_b,
                  const std::string& tree_ah_24, const std::string& tree_d16,
                  size_t n) {
  const std::string tok = "/datapoint_to_token.npy";
  std::string c;

  auto with_tokens = [&](const std::string& name,
                         const std::function<void(std::vector<int32_t>&)>& f) {
    std::string d = Copy(tree_ah, name);
    auto [t, shape] = ReadNpy<int32_t>(d + tok);
    f(t);
    WriteNpy(d + tok, t);
    return d;
  };
  // Upstream: vector::at threw std::out_of_range (std::terminate in C++).
  c = with_tokens("tok_out_of_range", [](auto& t) { t[5] = kLeaves; });
  ExpectError(c, "token == n_leaves", "outside [0, 12)");
  c = with_tokens("tok_negative", [](auto& t) { t[5] = -5; });
  ExpectError(c, "negative token", "outside [0, 12)");
  c = with_tokens("tok_short", [&](auto& t) { t.resize(n / 2); });
  ExpectError(c, "short tokenization", "Inconsistent index files");
  c = with_tokens("tok_long", [&](auto& t) { t.push_back(0); });
  ExpectError(c, "long tokenization", "Inconsistent index files");

  c = Copy(tree_ah, "tok_float");
  {
    auto [t, shape] = ReadNpy<int32_t>(c + tok);
    WriteNpy(c + tok, std::vector<float>(t.begin(), t.end()));
  }
  ExpectError(c, "float tokenization", "dtype");

  c = Copy(tree_ah, "tok_2d");
  {
    auto [t, shape] = ReadNpy<int32_t>(c + tok);
    WriteNpy(c + tok, t, {t.size() / 2});
  }
  ExpectError(c, "2-D tokenization", "dimensions");

  c = Copy(tree_ah, "tok_missing");  // recomputed from the dataset: fine
  DropAsset(c, ScannAsset::TOKENIZATION_NPY);
  ExpectLoads(c, "missing tokenization", n);

  // Files of another index, as an interrupted in-place serialize leaves them.
  c = Copy(tree_ah, "partitioner_more_leaves");
  CopyFile(tree_ah_24, c, "serialized_partitioner.pb");
  ExpectError(c, "partitioner with more leaves than num_children",
              "at most 12");
  c = Copy(tree_ah_24, "partitioner_fewer_leaves");
  CopyFile(tree_ah, c, "serialized_partitioner.pb");
  ExpectError(c, "partitioner with fewer leaves than the tokens", "outside");
  c = Copy(tree_ah, "partitioner_other_dims");
  CopyFile(tree_d16, c, "serialized_partitioner.pb");
  ExpectError(c, "partitioner of another dimensionality", "dimensionality");
  c = Copy(tree_ah, "codebook_other_dims");  // upstream: BadStatusOrAccess
  CopyFile(tree_d16, c, "ah_codebook.pb");
  ExpectError(c, "codebook of another dimensionality", "blocks");
  c = Copy(tree_ah, "config_other_dims");
  CopyFile(tree_d16, c, "scann_config.pb");
  ExpectError(c, "config of another dimensionality");
  c = Copy(tree_ah, "dataset_other_rows");  // upstream: abort
  CopyFile(tree_ah_b, c, "dataset.npy");
  ExpectError(c, "dataset of another size", "Inconsistent index files");
  c = Copy(tree_ah, "hashed_other_rows");  // upstream: loaded
  CopyFile(tree_ah_b, c, "hashed_dataset.npy");
  ExpectError(c, "hashed dataset of another size", "Inconsistent index files");

  c = Copy(tree_ah, "hashed_codes_255");  // upstream: loaded
  {
    auto [h, shape] = ReadNpy<uint8_t>(c + "/hashed_dataset.npy");
    std::fill(h.begin(), h.end(), 255);
    WriteNpy(c + "/hashed_dataset.npy", h, {shape[0]});
  }
  ExpectError(c, "hashed codes out of range", "centers");

  c = Copy(tree_ah, "hashed_extra_blocks");  // upstream: loaded
  {
    auto [h, shape] = ReadNpy<uint8_t>(c + "/hashed_dataset.npy");
    std::vector<uint8_t> wide;
    for (size_t i = 0; i < shape[0]; ++i)
      for (int k = 0; k < 2; ++k)
        wide.insert(wide.end(), h.begin() + i * shape[1],
                    h.begin() + (i + 1) * shape[1]);
    WriteNpy(c + "/hashed_dataset.npy", wide, {shape[0]});
  }
  ExpectError(c, "hashed dataset with extra blocks", "blocks");

  c = Copy(tree_ah, "codebook_garbage");
  WriteFile(c + "/ah_codebook.pb", "\x0a\xff\xff\xff\x0f");
  ExpectError(c, "garbage codebook");

  c = Copy(tree_ah, "assets_duplicate");
  AddAsset(c, ScannAsset::DATASET_NPY, "dataset.npy");
  ExpectError(c, "an asset listed twice", "more than one DATASET_NPY");

  c = Copy(tree_ah, "assets_empty");
  WriteFile(c + "/scann_assets.pbtxt", "");
  ExpectError(c, "empty manifest");

  c = Copy(tree_ah, "assets_incomplete_marker");
  WriteFile(c + "/scann_assets.pbtxt",
            "scann_core_incomplete_serialization: true\n");
  ExpectError(c, "interrupted SerializeToDirectory", "incomplete");
}

void DamagedSoar(const std::string& soar, const std::string& tree_ah) {
  std::string c;
  // The upstream report's reproduction: re-saving a non-SOAR index over a
  // SOAR one, interrupted after scann_config.pb (upstream: segfault).
  c = Copy(soar, "soar_files_non_soar_config");
  CopyFile(tree_ah, c, "scann_config.pb");
  ExpectError(c, "SOAR assets, non-SOAR config", "SOAR");

  c = Copy(tree_ah, "soar_config_non_soar_files");
  CopyFile(soar, c, "scann_config.pb");
  ExpectError(c, "non-SOAR assets, SOAR config", "SOAR tokenization");

  c = Copy(soar, "soar_no_tokenization");  // upstream: segfault
  DropAsset(c, ScannAsset::TOKENIZATION_NPY);
  ExpectError(c, "SOAR hashed dataset without tokenization", "tokenization");

  c = Copy(soar, "soar_asset_twice");  // upstream: segfault
  AddAsset(c, ScannAsset::AH_DATASET_SOAR_NPY, "hashed_dataset_soar.npy");
  ExpectError(c, "SOAR asset listed twice", "more than one");

  c = Copy(soar, "soar_tokenization_one_entry");  // upstream: SIGFPE
  WriteNpy(c + "/datapoint_to_token.npy", std::vector<int32_t>{0});
  ExpectError(c, "1-entry SOAR tokenization", "two per datapoint");

  c = Copy(soar, "soar_same_leaf_twice");
  {
    auto [t, shape] = ReadNpy<int32_t>(c + "/datapoint_to_token.npy");
    t[1] = t[0];
    WriteNpy(c + "/datapoint_to_token.npy", t);
  }
  ExpectError(c, "datapoint in the same leaf twice", "twice");

  c = Copy(soar, "soar_short_soar_hashed");
  {
    auto [h, shape] = ReadNpy<uint8_t>(c + "/hashed_dataset_soar.npy");
    h.resize(h.size() / 2);
    WriteNpy(c + "/hashed_dataset_soar.npy", h, {shape[0] / 2});
  }
  ExpectError(c, "short SOAR hashed dataset", "Inconsistent index files");

  c = Copy(soar, "soar_no_dataset");  // upstream: LOG(FATAL)
  DropAsset(c, ScannAsset::DATASET_NPY);
  ExpectError(c, "exact reordering without dataset.npy", "dataset");
}

void DamagedInt8(const std::string& bf_int8) {
  std::string c = Copy(bf_int8, "int8_no_multipliers");  // upstream: segfault
  DropAsset(c, ScannAsset::INT8_MULTIPLIERS_NPY);
  ExpectError(c, "int8 without multipliers", "multipliers");

  c = Copy(bf_int8, "int8_short_multipliers");  // upstream: loaded
  WriteNpy(c + "/int8_multipliers.npy", std::vector<float>(4, 1.0f));
  ExpectError(c, "short int8 multipliers");

  c = Copy(bf_int8, "int8_short_norms");
  WriteNpy(c + "/dp_norms.npy", std::vector<float>(10, 1.0f));
  AddAsset(c, ScannAsset::INT8_NORMS_NPY, "dp_norms.npy");
  ExpectError(c, "int8 norms of the wrong length");
}

// --- serialize ---------------------------------------------------------------------

// L2: every point deleted, serialized, loaded, grown again, reloaded.
void AllDeletedRoundTrip(const std::string& name, const Spec& spec) {
  ++g_checks;
  const size_t n = 600, dim = 8;
  ScannInterface s;
  if (!Build(s, spec, n, dim, 11)) return;
  auto m = s.GetMutator();
  if (!Ok(m.status(), name + ": GetMutator")) return;
  while (s.n_points() > 0)
    if (!Ok((*m)->RemoveDatapoint(s.n_points() - 1), name + ": Remove"))
      return;
  // Searching an index whose searched leaves are all empty: the AVX2 LUT16
  // kernel divided by num_blocks = 0 (SIGFPE) for tree + AH.
  const auto expect_no_results = [&](const ScannInterface& idx,
                                     const std::string& what) {
    Vec q = RandomData(1, dim, 13);
    NNResultsVector res;
    if (Ok(idx.Search(Ptr(q), &res, 5, -1, kLeaves), name + ": " + what) &&
        !res.empty())
      Fail(absl::StrCat(name, ": ", what, " returned ", res.size(),
                        " results from an empty index"));
  };
  expect_no_results(s, "search after deleting everything");
  const std::string dir = (g_root / ("empty_" + name)).string();
  fs::create_directories(dir);
  if (!Ok(s.SerializeToDirectory(dir), name + ": serialize empty")) return;

  auto artifacts = ScannInterface::LoadArtifacts(dir);
  if (!Ok(artifacts.status(), name + ": LoadArtifacts (empty)")) return;
  ScannInterface t;
  if (!Ok(t.Initialize(*std::move(artifacts)), name + ": Initialize (empty)"))
    return;
  if (t.n_points() != 0) Fail(name + ": reloaded empty index isn't empty");
  expect_no_results(t, "search of the reloaded empty index");
  auto tm = t.GetMutator();
  if (!Ok(tm.status(), name + ": GetMutator (reloaded)")) return;
  Vec added = RandomData(5, dim, 12);
  for (auto& x : added) x *= 3;
  for (size_t i = 0; i < 5; ++i) {
    Vec v(added.begin() + i * dim, added.begin() + (i + 1) * dim);
    if (!Ok((*tm)->AddDatapoint(Ptr(v), "", {}).status(), name + ": Add"))
      return;
  }
  if (spec.scoring == Spec::kBf || spec.scoring == Spec::kBfBf16 ||
      spec.scoring == Spec::kBfInt8) {
    for (size_t i = 0; i < 5; ++i) {
      Vec v(added.begin() + i * dim, added.begin() + (i + 1) * dim);
      NNResultsVector res;
      if (!Ok(t.Search(Ptr(v), &res, 1, -1, kLeaves), name + ": Search"))
        return;
      // Dot product: a point needn't be its own nearest neighbour, but it
      // must be found among 5 points.
      if (res.empty() || res[0].first >= 5)
        Fail(absl::StrCat(name, ": search after re-adding found nothing"));
    }
  }
  const std::string dir2 = dir + "_again";
  fs::create_directories(dir2);
  if (!Ok(t.SerializeToDirectory(dir2), name + ": serialize again")) return;
  ExpectLoads(dir2, name + ": regrown", 5);
}

void Bfloat16TreeRoundTrip() {
  ++g_checks;
  Spec spec;
  spec.scoring = Spec::kBfBf16;
  const size_t n = 600, dim = 8;
  ScannInterface s;
  Vec data;
  if (!Build(s, spec, n, dim, 13, &data)) return;
  const std::string dir = (g_root / "tree_bf16").string();
  fs::create_directories(dir);
  if (!Ok(s.SerializeToDirectory(dir), "tree_bf16: serialize")) return;
  auto artifacts = ScannInterface::LoadArtifacts(dir);
  if (!Ok(artifacts.status(), "tree_bf16: LoadArtifacts")) return;
  ScannInterface t;
  if (!Ok(t.Initialize(*std::move(artifacts)), "tree_bf16: Initialize")) return;
  if (t.n_points() != n) Fail("tree_bf16: wrong size after reload");
  size_t mismatches = 0;
  for (size_t i = 0; i < n; i += 37) {
    Vec q(data.begin() + i * dim, data.begin() + (i + 1) * dim);
    NNResultsVector a, b;
    if (!Ok(s.Search(Ptr(q), &a, 5, -1, kLeaves), "tree_bf16: search") ||
        !Ok(t.Search(Ptr(q), &b, 5, -1, kLeaves), "tree_bf16: search reloaded"))
      return;
    mismatches += a != b;
  }
  if (mismatches) Fail("tree_bf16: reloaded index gives other results");
  auto m = t.GetMutator();
  if (Ok(m.status(), "tree_bf16: GetMutator")) {
    Vec v = RandomData(1, dim, 14);
    Ok((*m)->AddDatapoint(Ptr(v), "", {}).status(), "tree_bf16: add");
    Ok((*m)->RemoveDatapoint(3), "tree_bf16: remove");
  }
}

std::vector<std::string> Files(const std::string& dir) {
  std::vector<std::string> names;
  for (const auto& e : fs::directory_iterator(dir))
    names.push_back(e.path().filename().string());
  std::sort(names.begin(), names.end());
  return names;
}

void AtomicReserialize() {
  ++g_checks;
  const std::string dir = (g_root / "reserialize").string();
  fs::create_directories(dir);
  if (!Ok(ScannInterface().SerializeToDirectory(dir + "/missing").ok()
              ? absl::InternalError("serialized into a missing directory")
              : absl::OkStatus(),
          "missing directory"))
    return;

  Spec soar;
  soar.soar = 1.5;
  ScannInterface a;
  if (!Build(a, soar, 1200, 8, 21)) return;
  if (!Ok(a.SerializeToDirectory(dir, false, {{"scann_docids.pkl", "old"}}),
          "serialize SOAR"))
    return;
  WriteFile(dir + "/unrelated.txt", "kept");
  fs::create_directories(dir + "/.scann_staging_leftover");

  // A different index over it: no SOAR file, no docids, no leftover.
  Spec bf;
  bf.scoring = Spec::kBf;
  ScannInterface b;
  if (!Build(b, bf, 500, 8, 22)) return;
  if (!Ok(b.SerializeToDirectory(dir), "serialize tree_bf over SOAR")) return;
  const std::vector<std::string> want = {
      "datapoint_to_token.npy", "dataset.npy",     "scann_assets.pbtxt",
      "scann_config.pb",        "serialized_partitioner.pb", "unrelated.txt"};
  if (Files(dir) != want)
    Fail(absl::StrCat("files after re-serialize: ",
                      absl::StrJoin(Files(dir), " ")));
  ExpectLoads(dir, "re-serialized", 500);

  // A serialize that fails midway: a directory named dataset.npy blocks the
  // rename of the new dataset.npy, after the manifest was replaced by the
  // marker and other files were replaced.
  fs::remove(dir + "/dataset.npy");
  fs::create_directories(dir + "/dataset.npy/blocker");
  if (a.SerializeToDirectory(dir).ok()) Fail("serialize past a blocker");
  for (const auto& f : Files(dir))
    if (absl::StartsWith(f, ".scann_staging_"))
      Fail("failed serialize left " + f);
  ExpectError(dir, "failed serialize", "incomplete");
  fs::remove_all(dir + "/dataset.npy");
  if (!Ok(a.SerializeToDirectory(dir), "serialize after the failure")) return;
  ExpectLoads(dir, "after a failed serialize", 1200);
}

}  // namespace

int main() {
  // SCANN_TEST_FORCE_AVX2=1: the AVX2 kernels on a CPU that has AVX-512 too,
  // as in api_exercise.
  if (const char* e = std::getenv("SCANN_TEST_FORCE_AVX2"); e && *e == '1') {
    research_scann::flags_internal::should_use_avx512 = false;
    research_scann::flags_internal::should_use_avx512_vnni = false;
    research_scann::flags_internal::should_use_amx = false;
    std::printf("SCANN_TEST_FORCE_AVX2: AVX-512 kernels disabled\n");
  }
  g_root = fs::temp_directory_path() /
           absl::StrCat("scann_artifact_loading_", ::getpid());
  fs::remove_all(g_root);
  fs::create_directories(g_root);

  Spec bf_spec;
  bf_spec.tree = false;
  bf_spec.scoring = Spec::kBf;
  Spec bf_int8_spec = bf_spec;
  bf_int8_spec.scoring = Spec::kBfInt8;
  Spec tree_ah_spec;
  Spec tree_ah_24_spec;
  tree_ah_24_spec.leaves = 24;
  Spec soar_spec;
  soar_spec.soar = 1.5;

  const size_t n = 1200;
  const std::string bf = BuildAndSave("bf", bf_spec, 200, 8, 1);
  const std::string bf_int8 = BuildAndSave("bf_int8", bf_int8_spec, 200, 8, 2);
  const std::string tree_ah = BuildAndSave("tree_ah", tree_ah_spec, n, 8, 3);
  const std::string tree_ah_b =
      BuildAndSave("tree_ah_b", tree_ah_spec, 900, 8, 4);
  const std::string tree_ah_24 =
      BuildAndSave("tree_ah_24", tree_ah_24_spec, n, 8, 5);
  const std::string tree_d16 = BuildAndSave("tree_d16", tree_ah_spec, n, 16, 6);
  const std::string soar = BuildAndSave("soar", soar_spec, n, 8, 7);
  if (g_failures) return 1;

  for (const std::string& d : {bf, bf_int8, tree_ah, soar})
    ExpectLoads(d, "undamaged " + d);
  DamagedNpyHeaders(bf);
  DamagedTrees(tree_ah, tree_ah_b, tree_ah_24, tree_d16, n);
  DamagedSoar(soar, tree_ah);
  DamagedInt8(bf_int8);

  Spec s;
  s.scoring = Spec::kBf;
  AllDeletedRoundTrip("tree_bf", s);
  s.scoring = Spec::kBfInt8;
  AllDeletedRoundTrip("tree_bf_int8", s);
  s.scoring = Spec::kBfBf16;
  AllDeletedRoundTrip("tree_bf_bf16", s);
  s.scoring = Spec::kAh;
  AllDeletedRoundTrip("tree_ah_no_reorder", s);
  AllDeletedRoundTrip("tree_ah_soar", soar_spec);
  AllDeletedRoundTrip("bf", bf_spec);
  Bfloat16TreeRoundTrip();
  AtomicReserialize();

  fs::remove_all(g_root);
  std::printf("%s: %d checks, %d failure(s)\n",
              g_failures ? "FAILED" : "PASSED", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
