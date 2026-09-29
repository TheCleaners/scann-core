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

// The AVX-512 LUT16 layout and kernel (0.3):
//  - the layout helpers agree with upstream's Avx512PlatformSpecificSwizzle,
//    round-trip, and address every code; partial conversions, resizes and
//    per-datapoint writes keep a dataset equal to its canonical twin;
//  - on AVX-512 CPUs, every LUT16 entry point (distances, int16 and float
//    top-N with restricts, predicates, int32 accumulators, prefetching)
//    returns bit for bit what the AVX2 kernel returns on the canonical
//    layout.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "scann/data_format/datapoint.h"
#include "scann/hashes/asymmetric_hashing2/querying.h"
#include "scann/hashes/internal/lut16_avx512_swizzle.h"
#include "scann/hashes/internal/lut16_interface.h"
#include "scann/restricts/restrict_allowlist.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/intrinsics/flags.h"

namespace {

using research_scann::Datapoint;
using research_scann::DatapointIndex;
using research_scann::FastTopNeighbors;
using research_scann::RestrictAllowlist;
using research_scann::RestrictAllowlistConstView;
using research_scann::asymmetric_hashing2::PackedDataset;
namespace ai = research_scann::asymmetric_hashing_internal;
namespace ah2 = research_scann::asymmetric_hashing2;

int g_failures = 0;

void Fail(const std::string& what) {
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++g_failures;
}

std::vector<uint8_t> RandomBytes(size_t n, std::mt19937& rng) {
  std::vector<uint8_t> v(n);
  for (auto& b : v) b = rng();
  return v;
}

uint8_t Nibble(const std::vector<uint8_t>& v, size_t byte, bool high) {
  return high ? v[byte] >> 4 : v[byte] & 0x0F;
}

const size_t kBlocks[] = {1, 2, 3, 4, 5, 7, 8, 13, 25, 43, 50, 64, 192};
const size_t kSizes[] = {1,   2,   15,  16,  31,  32,  33,  63,  64,  96,
                         97,  127, 128, 129, 160, 200, 224, 225, 255, 256,
                         257, 288, 383, 384, 385, 480, 511, 512, 600, 1000};

void TestLayout() {
  const int failures = g_failures;
  std::mt19937 rng(1);
  int checked = 0;
  for (size_t nb : kBlocks) {
    for (size_t n : kSizes) {
      const size_t n32 = (n + 31) / 32;
      const std::vector<uint8_t> canon = RandomBytes(n32 * 16 * nb, rng);
      std::vector<uint8_t> swz = canon;
      ai::Lut16Avx512Swizzle(swz.data(), n32, nb);
      const std::string name = absl::StrCat("layout nb=", nb, " n=", n);
#ifdef __x86_64__
      if (research_scann::RuntimeSupportsAvx512()) {
        std::vector<uint8_t> up = canon;
        ai::Avx512PlatformSpecificSwizzle(up.data(), n, nb);
        if (up != swz) Fail(name + ": differs from upstream's swizzle");
      }
#endif
      for (size_t dp = 0; dp < 32 * n32; ++dp) {
        for (size_t b = 0; b < nb; ++b) {
          const auto c = ai::Lut16CanonicalNibbleAddress(nb, dp, b);
          const auto a = ai::Lut16Avx512NibbleAddress(n32, nb, dp, b);
          if (Nibble(canon, c.byte, c.high) != Nibble(swz, a.byte, a.high)) {
            Fail(absl::StrCat(name, ": address of dp ", dp, " block ", b));
            dp = 32 * n32;
            break;
          }
        }
      }
      for (size_t first = 0; first <= 8 * (n32 / 8); first += 8) {
        std::vector<uint8_t> part = swz;
        ai::Lut16Avx512Unswizzle(part.data(), n32, nb, first);
        const size_t cut = first * 16 * nb;
        if (!std::equal(part.begin(), part.begin() + cut, swz.begin()) ||
            !std::equal(part.begin() + cut, part.end(), canon.begin() + cut)) {
          Fail(absl::StrCat(name, ": partial unswizzle from group ", first));
        }
        ai::Lut16Avx512Swizzle(part.data(), n32, nb, first);
        if (part != swz) {
          Fail(absl::StrCat(name, ": partial swizzle from group ", first));
        }
      }
      ++checked;
    }
  }
  if (g_failures == failures) {
    std::printf("ok: layout helpers, %d shapes\n", checked);
  }
}

// Grows and shrinks two packed datasets, one per layout, the way
// asymmetric_hashing2's mutator does; they must stay equal.
void TestMutation() {
  std::mt19937 rng(2);
  for (size_t nb : {3, 25, 50}) {
    PackedDataset canon, swz;
    canon.num_blocks = swz.num_blocks = nb;
    swz.avx512_layout = true;
    std::vector<std::vector<uint8_t>> rows;
    auto add = [&](PackedDataset* p, const Datapoint<uint8_t>& h) {
      const size_t i = p->num_datapoints++;
      if (!(i & 31)) {
        ah2::ResizeLUT16PackedData(p, p->bit_packed_data.size() / (16 * nb) + 1);
      }
      return ah2::SetLUT16Hash(h.ToPtr(), i, p);
    };
    auto remove = [&](PackedDataset* p, size_t index) {
      const size_t new_size = --p->num_datapoints;
      Datapoint<uint8_t> last = ah2::GetLUT16Hash(new_size, *p);
      auto s = ah2::SetLUT16Hash(last.ToPtr(), index, p);
      if (!(new_size & 31)) {
        ah2::ResizeLUT16PackedData(p, p->bit_packed_data.size() / (16 * nb) - 1);
      }
      return s;
    };
    const std::string name = absl::StrCat("mutation nb=", nb);
    for (int step = 0; step < 3000; ++step) {
      const bool grow = rows.size() < 20 || (step / 700) % 2 == 0
                            ? rng() % 4 != 0
                            : rng() % 4 == 0;
      if (grow || rows.empty()) {
        Datapoint<uint8_t> h;
        for (size_t b = 0; b < nb; ++b) h.mutable_values()->push_back(rng() % 16);
        rows.push_back(*h.mutable_values());
        if (!add(&canon, h).ok() || !add(&swz, h).ok()) Fail(name + ": add");
      } else if (rng() % 3 == 0) {
        const size_t i = rng() % rows.size();
        Datapoint<uint8_t> h;
        for (size_t b = 0; b < nb; ++b) h.mutable_values()->push_back(rng() % 16);
        rows[i] = *h.mutable_values();
        if (!ah2::SetLUT16Hash(h.ToPtr(), i, &canon).ok() ||
            !ah2::SetLUT16Hash(h.ToPtr(), i, &swz).ok()) {
          Fail(name + ": update");
        }
      } else {
        const size_t i = rng() % rows.size();
        rows[i] = rows.back();
        rows.pop_back();
        if (!remove(&canon, i).ok() || !remove(&swz, i).ok()) {
          Fail(name + ": remove");
        }
      }
      std::vector<uint8_t> back = swz.bit_packed_data;
      if (nb && !back.empty()) {
        ai::Lut16Avx512Unswizzle(back.data(), back.size() / (16 * nb), nb);
      }
      if (back != canon.bit_packed_data) {
        Fail(absl::StrCat(name, ": layouts diverge at step ", step));
        break;
      }
      if (step % 97 == 0) {
        auto u = ah2::UnpackDataset(ah2::CreatePackedDatasetView(swz));
        for (size_t i = 0; i < rows.size(); ++i) {
          if (!std::equal(rows[i].begin(), rows[i].end(), u[i].values())) {
            Fail(absl::StrCat(name, ": unpacked row ", i, " at step ", step));
            break;
          }
        }
        ah2::PackedDataset copy = swz;
        ah2::SetLUT16Layout(&copy, false);
        if (copy.bit_packed_data != canon.bit_packed_data) {
          Fail(name + ": SetLUT16Layout(false)");
        }
      }
    }
    std::printf("done: %s, %zu datapoints left\n", name.c_str(), rows.size());
  }
}

#ifdef __x86_64__

struct Case {
  size_t n, nb, nq;
  std::vector<uint8_t> canon, swz, canon_next, swz_next;
  std::vector<std::vector<uint8_t>> luts;
};

Case MakeCase(size_t n, size_t nb, size_t nq, std::mt19937& rng) {
  Case c{n, nb, nq};
  const size_t n32 = (n + 31) / 32;
  c.canon = RandomBytes(n32 * 16 * nb, rng);
  c.swz = c.canon;
  ai::Lut16Avx512Swizzle(c.swz.data(), n32, nb);
  c.canon_next = RandomBytes(4 * 16 * nb, rng);
  c.swz_next = c.canon_next;
  ai::Lut16Avx512Swizzle(c.swz_next.data(), 4, nb);
  for (size_t q = 0; q < nq; ++q) {
    c.luts.push_back(RandomBytes(16 * nb, rng));
    // Skewed tables too: values near 0 or 255 stress the int16 sums.
    if (q % 3 == 1) {
      for (auto& v : c.luts.back()) v = v % 8;
    } else if (q % 3 == 2) {
      for (auto& v : c.luts.back()) v = 255 - v % 8;
    }
  }
  return c;
}

template <typename DistT>
void CompareDistances(const Case& c, const std::string& name) {
  const size_t n32 = (c.n + 31) / 32;
  std::vector<const uint8_t*> luts;
  for (const auto& l : c.luts) luts.push_back(l.data());
  std::vector<std::vector<DistT>> out[2];
  std::vector<float> mults;
  for (size_t q = 0; q < c.nq; ++q) mults.push_back(0.01f * (q + 1));
  for (int avx512 : {0, 1}) {
    out[avx512].assign(c.nq, std::vector<DistT>(32 * n32));
    std::vector<DistT*> ptrs;
    for (auto& o : out[avx512]) ptrs.push_back(o.data());
    ai::LUT16Args<DistT> args;
    args.packed_dataset = avx512 ? c.swz.data() : c.canon.data();
    args.next_partition = avx512 ? c.swz_next.data() : c.canon_next.data();
    args.enable_avx512_codepath = avx512;
    args.num_32dp_simd_iters = n32;
    args.num_blocks = c.nb;
    args.lookups = {luts.data(), luts.size()};
    args.distances = {ptrs.data(), ptrs.size()};
    args.prefetch_strategy = (c.n % 2) ? ai::PrefetchStrategy::kSmart
                                       : ai::PrefetchStrategy::kSeq;
    if constexpr (std::is_same_v<DistT, float>) {
      ai::LUT16Interface::GetFloatDistances(args, mults);
    } else {
      ai::LUT16Interface::GetDistances(args);
    }
  }
  if (out[0] != out[1]) Fail(name);
}

using Results = std::vector<std::vector<std::pair<DatapointIndex, float>>>;

// mode: 0 plain, 1 restricts, 2 predicates, 3 tight epsilon, 4 epsilon below
// every distance (thresholds clamp).
Results TopFloat(const Case& c, int avx512, int mode, size_t k) {
  std::vector<const uint8_t*> luts;
  for (const auto& l : c.luts) luts.push_back(l.data());
  std::vector<FastTopNeighbors<float>> tops;
  std::vector<float> biases, mults;
  std::vector<RestrictAllowlist> allow;
  std::vector<RestrictAllowlistConstView> views;
  std::mt19937 rng(c.n * 131 + c.nb);
  for (size_t q = 0; q < c.nq; ++q) {
    const float eps = mode == 3 ? 5.0f : mode == 4 ? -1e30f
                                                   : std::numeric_limits<float>::infinity();
    tops.emplace_back(k, eps);
    biases.push_back(q % 2 ? -3.5f : 0.25f * q);
    mults.push_back(q % 3 == 0 ? 1.0f : 7.3f / (q + 1));
    if (mode == 1) {
      allow.emplace_back(c.n, false);
      for (size_t i = 0; i < c.n; ++i)
        if (rng() % 3) allow.back().AddToAllowlist(i);
    }
  }
  for (auto& a : allow) views.emplace_back(a);
  std::vector<FastTopNeighbors<float>*> ptrs;
  for (auto& t : tops) ptrs.push_back(&t);
  ai::LUT16ArgsTopN<float> args;
  args.packed_dataset = avx512 ? c.swz.data() : c.canon.data();
  args.next_partition = avx512 ? c.swz_next.data() : c.canon_next.data();
  args.enable_avx512_codepath = avx512;
  args.num_32dp_simd_iters = (c.n + 31) / 32;
  args.num_blocks = c.nb;
  args.lookups = {luts.data(), luts.size()};
  args.first_dp_index = 1000;
  args.num_datapoints = c.n;
  args.fast_topns = {ptrs.data(), ptrs.size()};
  args.biases = biases;
  args.fixed_point_multipliers = mults;
  args.prefetch_strategy = (c.n % 2) ? ai::PrefetchStrategy::kSmart
                                     : ai::PrefetchStrategy::kSmartT0;
  if (mode == 1) args.restrict_whitelists = views;
  if (mode == 2) {
    args.datapoint_translation_predicate = [](DatapointIndex i) {
      return 3 * i + 1;
    };
    args.batch_filter_predicate =
        [](DatapointIndex first, uint32_t mask,
           const std::function<DatapointIndex(DatapointIndex)>&) {
          return mask & (first % 64 ? 0x5555AAAAu : 0xF0F00F0Fu);
        };
  }
  ai::LUT16Interface::GetTopFloatDistances(args);
  Results r;
  for (auto& t : tops) {
    r.emplace_back();
    t.FinishSorted(&r.back());
  }
  return r;
}

Results TopInt16(const Case& c, int avx512, bool restricts, size_t k) {
  std::vector<const uint8_t*> luts;
  for (const auto& l : c.luts) luts.push_back(l.data());
  std::vector<FastTopNeighbors<int16_t>> tops;
  std::vector<RestrictAllowlist> allow;
  std::vector<RestrictAllowlistConstView> views;
  std::mt19937 rng(c.n * 7 + c.nb);
  for (size_t q = 0; q < c.nq; ++q) {
    tops.emplace_back(k, q % 2 ? int16_t{32767} : int16_t{-50});
    if (restricts) {
      allow.emplace_back(c.n, false);
      for (size_t i = 0; i < c.n; ++i)
        if (rng() % 2) allow.back().AddToAllowlist(i);
    }
  }
  for (auto& a : allow) views.emplace_back(a);
  std::vector<FastTopNeighbors<int16_t>*> ptrs;
  for (auto& t : tops) ptrs.push_back(&t);
  ai::LUT16ArgsTopN<int16_t> args;
  args.packed_dataset = avx512 ? c.swz.data() : c.canon.data();
  args.enable_avx512_codepath = avx512;
  args.num_32dp_simd_iters = (c.n + 31) / 32;
  args.num_blocks = c.nb;
  args.lookups = {luts.data(), luts.size()};
  args.num_datapoints = c.n;
  args.fast_topns = {ptrs.data(), ptrs.size()};
  if (restricts) args.restrict_whitelists = views;
  ai::LUT16Interface::GetTopDistances(args);
  Results r;
  for (auto& t : tops) {
    std::vector<std::pair<DatapointIndex, int16_t>> v;
    t.FinishSorted(&v);
    r.emplace_back();
    for (auto [i, d] : v) r.back().emplace_back(i, d);
  }
  return r;
}

bool SameBits(const Results& a, const Results& b) {
  if (a.size() != b.size()) return false;
  for (size_t q = 0; q < a.size(); ++q) {
    if (a[q].size() != b[q].size()) return false;
    for (size_t i = 0; i < a[q].size(); ++i) {
      if (a[q][i].first != b[q][i].first ||
          std::memcmp(&a[q][i].second, &b[q][i].second, sizeof(float)) != 0) {
        return false;
      }
    }
  }
  return true;
}

void TestKernels() {
  if (!research_scann::RuntimeSupportsAvx512()) {
    std::printf("skipped: kernel comparison (no AVX-512)\n");
    return;
  }
  const int failures = g_failures;
  std::mt19937 rng(3);
  int cases = 0;
  const size_t sizes[] = {1, 17, 32, 45, 100, 128, 150, 255, 256, 300, 480, 700};
  for (size_t nb : {1, 2, 3, 5, 8, 43, 50, 64, 255, 256, 300}) {
    for (size_t n : sizes) {
      for (size_t nq : {1, 2, 3, 4, 5, 7, 9}) {
        if (nb >= 255 && nq > 3 && n > 300) continue;
        Case c = MakeCase(n, nb, nq, rng);
        const std::string name =
            absl::StrCat("kernel nb=", nb, " n=", n, " nq=", nq);
        CompareDistances<int16_t>(c, name + " int16 distances");
        CompareDistances<int32_t>(c, name + " int32 distances");
        CompareDistances<float>(c, name + " float distances");
        for (int mode = 0; mode < 5; ++mode) {
          for (size_t k : {1, 10, 100}) {
            if (!SameBits(TopFloat(c, 0, mode, k), TopFloat(c, 1, mode, k))) {
              Fail(absl::StrCat(name, " top float mode ", mode, " k ", k));
            }
          }
        }
        for (bool r : {false, true}) {
          if (!SameBits(TopInt16(c, 0, r, 20), TopInt16(c, 1, r, 20))) {
            Fail(absl::StrCat(name, " top int16 restricts=", r));
          }
        }
        ++cases;
      }
    }
  }
  if (g_failures == failures) {
    std::printf("ok: AVX-512 kernel == AVX2 kernel, %d cases\n", cases);
  }
}

#endif

}  // namespace

int main() {
  TestLayout();
  TestMutation();
#ifdef __x86_64__
  TestKernels();
#endif
  if (g_failures) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("all passed\n");
  return 0;
}
