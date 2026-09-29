// Modified in 2026 by Elias Benali (@ebenali) and TheCleaners for
// scann-core (a derived work of ScaNN, not an official Google product);
// see NOTICE.
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "scann/hashes/internal/lut16_avx512.h"
#include "scann/oss_wrappers/scann_bits.h"
#include "scann/utils/common.h"

#ifdef __x86_64__

#include "absl/base/prefetch.h"
#include "scann/hashes/internal/lut16_avx512_swizzle.h"
#include "scann/utils/bits.h"
#include "scann/utils/intrinsics/avx512.h"

namespace research_scann {
namespace avx512 {
namespace lut16 {

using asymmetric_hashing_internal::LUT16Args;
using asymmetric_hashing_internal::LUT16ArgsTopN;
using asymmetric_hashing_internal::PrefetchStrategy;

// scann-core: unaligned by default (the codes are in std::vectors).
template <PrefetchStrategy kPrefetch0, bool kAlignedData0 = false>
struct LUT16Tuning {
  static constexpr PrefetchStrategy kPrefetch = kPrefetch0;
  static constexpr bool kAlignedData = kAlignedData0;

  using WithoutDataAlignment = LUT16Tuning<kPrefetch, false>;
};

inline bool IsCacheAligned(const uint8_t* data_start) {
  return IsDivisibleBy(reinterpret_cast<uintptr_t>(data_start), 64);
}

// scann-core: kSmart prefetching continues into the next partition, as the
// AVX2 kernel's does (upstream's AVX-512 one ran past this partition's end).
struct PrefetchCtx {
  const uint8_t* end = nullptr;   // end of this partition's codes
  const uint8_t* next = nullptr;  // the next partition's, or null (last)
};

template <typename Tuning>
SCANN_AVX512_INLINE void PrefetchCodes(const uint8_t* data_start,
                                       const PrefetchCtx& ctx) {
  constexpr size_t kAhead = asymmetric_hashing_internal::kPrefetchBytesAhead;
  if constexpr (Tuning::kPrefetch == PrefetchStrategy::kSmart ||
                Tuning::kPrefetch == PrefetchStrategy::kSmartT0) {
    // Same distance ahead; past this partition's end, into the next one.
    const size_t left = static_cast<size_t>(ctx.end - data_start);
    if (left > kAhead) {
      absl::PrefetchToLocalCache(data_start + kAhead);
    } else if (ctx.next != nullptr) {
      absl::PrefetchToLocalCacheNta(ctx.next + (kAhead - left));  // as AVX2
    }
  } else if constexpr (Tuning::kPrefetch != PrefetchStrategy::kOff) {
    absl::PrefetchToLocalCache(data_start + kAhead);
  }
}

SCANN_AVX512_INLINE Avx512<uint8_t, 2> SplitNibbles(
    Avx512<uint8_t> interleaved_codes) {
  Avx512<uint8_t, 2> codes;
  codes[0] = interleaved_codes & Avx512<uint8_t>(0x0F);
  codes[1] = Avx512<uint8_t>(Avx512<uint16_t>(interleaved_codes) >> 4) &
             Avx512<uint8_t>(0x0F);
  return codes;
}

template <typename Tuning>
SCANN_AVX512_INLINE Avx512<uint8_t, 2> LoadDatabaseCodes(
    const uint8_t* data_start, const PrefetchCtx& ctx) {
  if constexpr (Tuning::kAlignedData) {
    DCHECK(IsCacheAligned(data_start));
  }
  PrefetchCodes<Tuning>(data_start, ctx);
  return SplitNibbles(Avx512<uint8_t>::Load<Tuning::kAlignedData>(data_start));
}

template <size_t kNumCodesPerIter, typename Tuning>
SCANN_AVX512_INLINE Avx512<uint8_t> LoadLUT(const uint8_t* lookup_ptr) {
  if constexpr (kNumCodesPerIter == 4) {
    return Avx512<uint8_t>::Load(lookup_ptr);
  }
  if constexpr (kNumCodesPerIter == 2) {
    return _mm512_broadcast_i64x4(*Avx2<uint8_t>::Load(lookup_ptr));
  }
  if constexpr (kNumCodesPerIter == 1) {
    return _mm512_broadcast_i64x2(*Sse4<uint8_t>::Load(lookup_ptr));
  }
}

SCANN_AVX512_INLINE Avx512<int16_t, 4> LUT16Core(Avx512<uint8_t, 2> codes,
                                                 Avx512<uint8_t> lut) {
  const Avx512<uint8_t> vals00 = _mm512_shuffle_epi8(*lut, *codes[0]);
  const Avx512<uint8_t> vals64 = _mm512_shuffle_epi8(*lut, *codes[1]);

  Avx512<int16_t, 4> extr;
  extr[0] = Avx512<int16_t>(Avx512<uint16_t>(vals00));
  extr[1] = Avx512<int16_t>(Avx512<uint16_t>(vals00) >> 8);
  extr[2] = Avx512<int16_t>(Avx512<uint16_t>(vals64));
  extr[3] = Avx512<int16_t>(Avx512<uint16_t>(vals64) >> 8);

  return extr;
}

template <size_t kNumQueries, size_t kNumRegisters>
SCANN_AVX512_INLINE Avx512<int16_t, kNumQueries, kNumRegisters>
SubtractBiasTerm(
    Avx512<int16_t, kNumQueries, kNumRegisters> uint16_accumulators,
    size_t num_codes_per_dp) {
  Avx512<int16_t, kNumQueries, kNumRegisters> signed_results;
  constexpr size_t kBiasTerm = 128;
  for (size_t j : Seq(kNumQueries)) {
    signed_results[j] = Avx512<int16_t, kNumRegisters>(uint16_accumulators[j]);
    signed_results[j] -= Avx512<int16_t>(num_codes_per_dp * kBiasTerm);
  }
  return signed_results;
}

template <size_t kNumQueries, typename Tuning>
SCANN_AVX512_INLINE Avx512<int16_t, kNumQueries, 8> BottomLoop256(
    const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookup_starts,
    const size_t num_codes_per_dp, const PrefetchCtx& ctx) {
  constexpr size_t kNumDatapoints = 256;
  constexpr size_t kNumCodesPerIter = 1;

  Avx512<int16_t, kNumQueries, 8> int16_accums = avx512::Zeros();
  for (auto _ : Seq(num_codes_per_dp)) {
    Avx512<uint8_t, 2> codes0 = LoadDatabaseCodes<Tuning>(data_start, ctx);
    Avx512<uint8_t, 2> codes1 =
        LoadDatabaseCodes<Tuning>(data_start + 64, ctx);
    data_start += 128;

    for (size_t j : Seq(kNumQueries)) {
      Avx512<uint8_t> lut = LoadLUT<kNumCodesPerIter, Tuning>(lookup_starts[j]);
      lookup_starts[j] += 16 * kNumCodesPerIter;

      int16_accums[j] +=
          Avx512Concat(LUT16Core(codes0, lut), LUT16Core(codes1, lut));
    }
  }

  static_assert(Avx512For<int16_t, kNumDatapoints>::kNumRegisters == 8);
  for (size_t j : Seq(kNumQueries)) {
    int16_accums[j][0] -= (int16_accums[j][1] << 8);
    int16_accums[j][2] -= (int16_accums[j][3] << 8);
    int16_accums[j][4] -= (int16_accums[j][5] << 8);
    int16_accums[j][6] -= (int16_accums[j][7] << 8);
  }

  return SubtractBiasTerm(int16_accums, num_codes_per_dp);
}

template <size_t kNumQueries, typename Tuning>
SCANN_AVX512_INLINE Avx512<int16_t, kNumQueries, 4> BottomLoop128(
    const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookup_starts,
    const size_t num_codes_per_dp, const PrefetchCtx& ctx) {
  constexpr size_t kNumDatapoints = 128;
  constexpr size_t kNumCodesPerIter = 1;

  Avx512<int16_t, kNumQueries, 4> int16_accums = avx512::Zeros();
  for (auto _ : Seq(num_codes_per_dp)) {
    Avx512<uint8_t, 2> codes = LoadDatabaseCodes<Tuning>(data_start, ctx);
    data_start += 64;

    for (size_t j : Seq(kNumQueries)) {
      Avx512<uint8_t> lut = LoadLUT<kNumCodesPerIter, Tuning>(lookup_starts[j]);
      lookup_starts[j] += 16 * kNumCodesPerIter;

      int16_accums[j] += LUT16Core(codes, lut);
    }
  }

  static_assert(Avx512For<int16_t, kNumDatapoints>::kNumRegisters == 4);
  for (size_t j : Seq(kNumQueries)) {
    int16_accums[j][0] -= (int16_accums[j][1] << 8);
    int16_accums[j][2] -= (int16_accums[j][3] << 8);
  }

  return SubtractBiasTerm(int16_accums, num_codes_per_dp);
}

template <typename T, size_t kNumRegisters>
SCANN_AVX512_INLINE Avx512<T, kNumRegisters / 2> HorizontalSum2to1(
    Avx512<T, kNumRegisters> int16_accums) {
  static_assert(IsDivisibleBy(kNumRegisters, 2));
  constexpr size_t kAA = 0;
  constexpr size_t kBB = 4;
  Avx512<T, kNumRegisters / 2> reduced;
  for (size_t j : Seq(kNumRegisters / 2)) {
    Avx512<T> shuffled0 = fake_mm512_permutex2var_epi128(
        *int16_accums[2 * j + 0], *int16_accums[2 * j + 1],
        {kAA + 0, kAA + 2, kBB + 0, kBB + 2});
    Avx512<T> shuffled1 = fake_mm512_permutex2var_epi128(
        *int16_accums[2 * j + 0], *int16_accums[2 * j + 1],
        {kAA + 1, kAA + 3, kBB + 1, kBB + 3});
    reduced[j] = shuffled0 + shuffled1;
  }

  return reduced;
}

SCANN_AVX512_INLINE Avx512<int16_t, 1> PostProcess32(
    Avx512<int16_t, 4> int16_accums) {
  Avx512<int16_t, 4> resequenced = Avx512Concat(
      int16_accums[0], int16_accums[2], int16_accums[1], int16_accums[3]);

  return HorizontalSum2to1(HorizontalSum2to1(resequenced));
}

template <size_t kNumQueries, typename OriginalTuning>
SCANN_AVX512_INLINE Avx512<int16_t, kNumQueries, 1> BottomLoop32(
    const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookup_starts,
    const size_t num_codes_per_dp, const PrefetchCtx& ctx) {
  constexpr size_t kNumDatapoints = 32;
  constexpr size_t kNumCodesPerIter = 4;

  using Tuning = typename OriginalTuning::WithoutDataAlignment;

  Avx512<int16_t, kNumQueries, 4> int16_accums = avx512::Zeros();

  const size_t num_iter = (num_codes_per_dp - 1) / kNumCodesPerIter;
  for (auto _ : Seq((num_iter))) {
    Avx512<uint8_t, 2> codes = LoadDatabaseCodes<Tuning>(data_start, ctx);
    data_start += 64;

    for (size_t j : Seq(kNumQueries)) {
      Avx512<uint8_t> lut = LoadLUT<kNumCodesPerIter, Tuning>(lookup_starts[j]);
      lookup_starts[j] += 16 * kNumCodesPerIter;

      int16_accums[j] += LUT16Core(codes, lut);
    }
  }

  {
    constexpr uint16_t kDisableCases[] = {
        0x0000,
        0xFFF0,
        0xFF00,
        0xF000,
    };
    const uint32_t kDisableBits = 0x80'80'80'80;
    const size_t codes_rem = num_codes_per_dp % kNumCodesPerIter;
    const __mmask16 disable_mask = _cvtu32_mask16(kDisableCases[codes_rem]);

    // scann-core: load only the remaining blocks' bytes, 16 per block
    // (upstream loaded all 64, up to 48 past the codes' and table's end).
    const __mmask64 load_mask =
        codes_rem == 0 ? ~uint64_t{0} : (uint64_t{1} << (16 * codes_rem)) - 1;
    PrefetchCodes<Tuning>(data_start, ctx);
    Avx512<uint8_t, 2> codes =
        SplitNibbles(_mm512_maskz_loadu_epi8(load_mask, data_start));
    // Missing blocks' lanes: 0x80 codes, which pshufb turns into 0.
    codes[0] = _mm512_mask_set1_epi32(*codes[0], disable_mask, kDisableBits);
    codes[1] = _mm512_mask_set1_epi32(*codes[1], disable_mask, kDisableBits);

    for (size_t j : Seq(kNumQueries)) {
      Avx512<uint8_t> lut =  // scann-core: masked, as the codes
          _mm512_maskz_loadu_epi8(load_mask, lookup_starts[j]);
      lookup_starts[j] += 16 * kNumCodesPerIter;

      int16_accums[j] += LUT16Core(codes, lut);
    }
  }

  Avx512For<int16_t, kNumQueries, kNumDatapoints> results =
      avx512::Uninitialized();
  static_assert(Avx512For<int16_t, kNumDatapoints>::kNumRegisters == 1);
  for (size_t j : Seq(kNumQueries)) {
    int16_accums[j][0] -= (int16_accums[j][1] << 8);
    int16_accums[j][2] -= (int16_accums[j][3] << 8);

    results[j] = PostProcess32(int16_accums[j]);
  }

  return SubtractBiasTerm(results, num_codes_per_dp);
}

template <size_t kNumDatapoints, size_t kNumQueries, typename Tuning>
SCANN_AVX512_INLINE Avx512For<int16_t, kNumQueries, kNumDatapoints> BottomLoop(
    const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookup_starts,
    const size_t num_codes_per_dp, const PrefetchCtx& ctx) {
  if constexpr (kNumDatapoints == 256) {
    return BottomLoop256<kNumQueries, Tuning>(data_start, lookup_starts,
                                              num_codes_per_dp, ctx);
  }
  if constexpr (kNumDatapoints == 128) {
    return BottomLoop128<kNumQueries, Tuning>(data_start, lookup_starts,
                                              num_codes_per_dp, ctx);
  }
  if constexpr (kNumDatapoints == 32) {
    return BottomLoop32<kNumQueries, Tuning>(data_start, lookup_starts,
                                             num_codes_per_dp, ctx);
  }
  static_assert(kNumDatapoints == 256 || kNumDatapoints == 128 ||
                kNumDatapoints == 32);

  LOG(FATAL) << "Unhandled.";
}

template <size_t kBottomLevelBatchSize, size_t kNumQueries>
SCANN_AVX512_INLINE array<const uint8_t*, kBottomLevelBatchSize>
MakeBottomLevelBatchLookupArray(
    array<const uint8_t*, kNumQueries> mid_level_lookups, size_t start) {
  DCHECK_LE(start + kBottomLevelBatchSize, kNumQueries);
  array<const uint8_t*, kBottomLevelBatchSize> result;
  for (size_t j : Seq(kBottomLevelBatchSize)) {
    result[j] = mid_level_lookups[start + j];
  }
  return result;
}

template <size_t kNumDatapoints, size_t kNumQueries, typename Tuning>
SCANN_AVX512_INLINE Avx512For<int16_t, kNumQueries, kNumDatapoints>
Int16MiddleLoop(const uint8_t* data_start,
                array<const uint8_t*, kNumQueries> lookup_starts,
                const size_t num_codes_per_dp, const PrefetchCtx& ctx) {
  if constexpr (kNumQueries <= 3) {
    return BottomLoop<kNumDatapoints, kNumQueries, Tuning>(
        data_start, lookup_starts, num_codes_per_dp, ctx);
  }

  constexpr size_t kSizeB = (kNumQueries == 1) ? 1 : 2;
  constexpr size_t kNumBCases[] = {0, 2, 1};
  constexpr size_t kNumB = (kNumQueries == 1) ? 1 : kNumBCases[kNumQueries % 3];

  constexpr size_t kRemaining = kNumQueries - kNumB * kSizeB;
  static_assert(kRemaining % 3 == 0, "");

  constexpr size_t kSizeA = 3;
  constexpr size_t kNumA = kRemaining / 3;

  Avx512For<int16_t, kNumQueries, kNumDatapoints> result;
  for (size_t j : Seq(kNumA)) {
    const size_t start = j * kSizeA;

    auto bottom_level_lookups =
        MakeBottomLevelBatchLookupArray<kSizeA>(lookup_starts, start);

    Avx512For<int16_t, kSizeA, kNumDatapoints> acc =
        BottomLoop<kNumDatapoints, kSizeA, Tuning>(
            data_start, bottom_level_lookups, num_codes_per_dp, ctx);

    for (size_t jj : Seq(kSizeA)) {
      result[start + jj] = acc[jj];
    }
  }

  for (size_t j : Seq(kNumB)) {
    const size_t start = kNumA * kSizeA + j * kSizeB;
    auto bottom_level_lookups =
        MakeBottomLevelBatchLookupArray<kSizeB>(lookup_starts, start);

    Avx512For<int16_t, kSizeB, kNumDatapoints> acc =
        BottomLoop<kNumDatapoints, kSizeB, Tuning>(
            data_start, bottom_level_lookups, num_codes_per_dp, ctx);

    for (size_t jj : Seq(kSizeB)) {
      result[start + jj] = acc[jj];
    }
  }

  return result;
}

template <size_t kNumDatapoints, size_t kNumQueries, typename Tuning>
SCANN_AVX512_INLINE Avx512For<int32_t, kNumQueries, kNumDatapoints>
Int32MiddleLoop(const uint8_t* data_start,
                array<const uint8_t*, kNumQueries> lookup_starts,
                const size_t num_codes_per_dp, const PrefetchCtx& ctx) {
  Avx512For<int32_t, kNumQueries, kNumDatapoints> int32_accums =
      avx512::Zeros();
  constexpr size_t kMaxCodesPerIter = 256;

  for (size_t k : SeqWithStride<kMaxCodesPerIter>(num_codes_per_dp)) {
    const size_t num_codes_this_iter =
        std::min<size_t>(kMaxCodesPerIter, num_codes_per_dp - k);

    Avx512For<int16_t, kNumQueries, kNumDatapoints> int16_accums =
        Int16MiddleLoop<kNumDatapoints, kNumQueries, Tuning>(
            data_start, lookup_starts, num_codes_this_iter, ctx);

    data_start += kNumDatapoints * num_codes_this_iter / 2;
    for (size_t j : Seq(kNumQueries)) {
      lookup_starts[j] += 16 * num_codes_this_iter;
      int32_accums[j] += int16_accums[j].template ExpandTo<int32_t>();
    }
  }
  return int32_accums;
}

template <size_t kNumDatapoints, size_t kNumQueries, typename Tuning>
SCANN_AVX512_INLINE Avx512For<float, kNumQueries, kNumDatapoints>
Float32MiddleLoop(const uint8_t* data_start,
                  array<const uint8_t*, kNumQueries> lookup_starts,
                  const size_t num_codes_per_dp,
                  array<float, kNumQueries> mults, const PrefetchCtx& ctx) {
  Avx512For<int32_t, kNumQueries, kNumDatapoints> int32_dists =
      Int32MiddleLoop<kNumDatapoints, kNumQueries, Tuning>(
          data_start, lookup_starts, num_codes_per_dp, ctx);

  Avx512For<float, kNumQueries, kNumDatapoints> float_dists;
  for (size_t j : Seq(kNumQueries)) {
    float_dists[j] =
        Avx512<float>(mults[j]) * int32_dists[j].template ConvertTo<float>();
  }

  return float_dists;
}

template <typename T, size_t kNumDatapoints, size_t kNumQueries,
          typename Tuning>
SCANN_AVX512_INLINE Avx512For<T, kNumQueries, kNumDatapoints> MiddleLoop(
    const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookup_starts,
    const size_t num_codes_per_dp, array<float, kNumQueries> mults,
    const PrefetchCtx& ctx) {
  if constexpr (IsSame<T, int16_t>()) {
    return Int16MiddleLoop<kNumDatapoints, kNumQueries, Tuning>(
        data_start, lookup_starts, num_codes_per_dp, ctx);
  } else if constexpr (IsSame<T, int32_t>()) {
    return Int32MiddleLoop<kNumDatapoints, kNumQueries, Tuning>(
        data_start, lookup_starts, num_codes_per_dp, ctx);
  } else if constexpr (IsSame<T, float>()) {
    return Float32MiddleLoop<kNumDatapoints, kNumQueries, Tuning>(
        data_start, lookup_starts, num_codes_per_dp, mults, ctx);
  } else {
    static_assert(IsSameAny<T, int16_t, int32_t, float>());
    LOG(FATAL) << "Unhandled.";
  }
}

template <size_t size, typename T>
SCANN_INLINE array<T, size> ToLocalArray(ConstSpan<T> span) {
  DCHECK_EQ(span.size(), size);
  array<T, size> result;
  std::copy(span.begin(), span.begin() + size, result.begin());
  return result;
}

template <typename T, size_t kNumQueries, typename Tuning>
SCANN_AVX512_INLINE void GetDistancesImpl(
    LUT16Args<T> args, ConstSpan<float> inv_fp_multipliers = {}) {
  const uint8_t* data_start = args.packed_dataset;
  const size_t num_codes_per_dp = args.num_blocks;
  array<const uint8_t*, kNumQueries> lookups =
      ToLocalArray<kNumQueries>(args.lookups);
  array<T*, kNumQueries> distances_ptrs =
      ToLocalArray<kNumQueries>(args.distances);
  array<float, kNumQueries> mults;
  if constexpr (IsSame<T, float>()) {
    mults = ToLocalArray<kNumQueries>(inv_fp_multipliers);
  }

  if (args.num_32dp_simd_iters == 0) return;
  DCHECK(data_start);
  for (size_t j : Seq(kNumQueries)) {
    DCHECK(lookups[j]);
  }
  const PrefetchCtx ctx{
      data_start + args.num_32dp_simd_iters * 16 * num_codes_per_dp,  // end
      args.next_partition};

  const size_t num_256dp_simd_iters = args.num_32dp_simd_iters / 8;
  for (auto _ : Seq(num_256dp_simd_iters)) {
    Avx512For<T, kNumQueries, 256> dists =
        MiddleLoop<T, 256, kNumQueries, Tuning>(data_start, lookups,
                                                num_codes_per_dp, mults, ctx);
    data_start += 256 * num_codes_per_dp / 2;
    for (size_t j : Seq(kNumQueries)) {
      dists[j].Store(distances_ptrs[j]);
      distances_ptrs[j] += 256;
    }
  }
  args.num_32dp_simd_iters %= 8;

  const size_t num_128dp_simd_iters = args.num_32dp_simd_iters / 4;
  for (auto _ : Seq(num_128dp_simd_iters)) {
    Avx512For<T, kNumQueries, 128> dists =
        MiddleLoop<T, 128, kNumQueries, Tuning>(data_start, lookups,
                                                num_codes_per_dp, mults, ctx);
    data_start += 128 * num_codes_per_dp / 2;
    for (size_t j : Seq(kNumQueries)) {
      dists[j].Store(distances_ptrs[j]);
      distances_ptrs[j] += 128;
    }
  }
  args.num_32dp_simd_iters %= 4;

  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  for (auto _ : Seq(num_32dp_simd_iters)) {
    Avx512For<T, kNumQueries, 32> dists =
        MiddleLoop<T, 32, kNumQueries, Tuning>(data_start, lookups,
                                               num_codes_per_dp, mults, ctx);
    data_start += 32 * num_codes_per_dp / 2;
    for (size_t j : Seq(kNumQueries)) {
      dists[j].Store(distances_ptrs[j]);
      distances_ptrs[j] += 32;
    }
  }
}

}  // namespace lut16
}  // namespace avx512

namespace asymmetric_hashing_internal {

using avx512::lut16::GetDistancesImpl;
using avx512::lut16::LUT16Tuning;

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_AVX512_OUTLINE void
LUT16Avx512<kNumQueries, kPrefetch>::GetInt16Distances(
    LUT16Args<int16_t> args) {
  using Tuning = LUT16Tuning<kPrefetch>;
  GetDistancesImpl<int16_t, kNumQueries, Tuning>(std::move(args), {});
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_AVX512_OUTLINE void
LUT16Avx512<kNumQueries, kPrefetch>::GetInt32Distances(
    LUT16Args<int32_t> args) {
  using Tuning = LUT16Tuning<kPrefetch>;
  GetDistancesImpl<int32_t, kNumQueries, Tuning>(std::move(args), {});
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_AVX512_OUTLINE void
LUT16Avx512<kNumQueries, kPrefetch>::GetFloatDistances(
    LUT16Args<float> args, ConstSpan<float> inv_fp_multipliers) {
  using Tuning = LUT16Tuning<kPrefetch>;
  GetDistancesImpl<float, kNumQueries, Tuning>(std::move(args),
                                               inv_fp_multipliers);
}

}  // namespace asymmetric_hashing_internal

namespace avx512 {
namespace lut16 {

// scann-core: the top-N functions, rewritten. They compute the distances
// with the loops above, then do per group of 32 datapoints exactly what
// LUT16Avx2's top-N functions do, so both kernels return the same results.
// Upstream's differed: pushes from 64-datapoint masks, a float threshold
// below the int16 range converted to int16 out of range, int16
// accumulators only, predicates ignored.

// As in lut16_avx2.inc.
SCANN_AVX512_INLINE int16_t GetInt16Threshold(float float_threshold) {
  constexpr float kMinValue = std::numeric_limits<int16_t>::min();
  constexpr float kMaxValue = std::numeric_limits<int16_t>::max();
  return std::max(kMinValue, std::min(kMaxValue, float_threshold));
}

SCANN_AVX512_INLINE int32_t GetInt32Threshold(float float_threshold) {
  constexpr float kMinValue = std::numeric_limits<int32_t>::min();

  static const float kMaxValue =
      std::nextafterf(std::numeric_limits<int32_t>::max(), 0);
  return std::max(kMinValue, std::min(kMaxValue, float_threshold));
}

template <typename T, size_t kNumQueries, bool kHasAnyPredicate,
          bool kUseInt32Accumulator, typename TopN>
class TopNPusher {
 public:
  static_assert(IsSameAny<T, int16_t, float>());
  static_assert(!kHasAnyPredicate || IsSame<T, float>());
  static_assert(!kUseInt32Accumulator || IsSame<T, float>());
  using ThresholdT =
      std::conditional_t<kUseInt32Accumulator, int32_t, int16_t>;

  SCANN_AVX512_INLINE explicit TopNPusher(LUT16ArgsTopN<T, TopN>& args)
      : args_(args) {
    for (size_t j : Seq(kNumQueries)) {
      if constexpr (IsSame<T, float>()) {
        biases_[j] = args.biases[j];
        mults_[j] = args.fixed_point_multipliers[j];
        simd_biases_[j] = _mm512_set1_ps(biases_[j]);
        simd_inv_mults_[j] = _mm512_set1_ps(1.0 / mults_[j]);  // double 1/x
      }
      thresholds_[j] = Threshold(j, args.fast_topns[j]->epsilon());
    }
    for (size_t j : Seq(kNumQueries)) {
      args.fast_topns[j]->AcquireMutator(&mutators_[j]);
    }
    whitelists_ = args.template GetRestrictWhitelistPtrs<kNumQueries>();
    final_mask_ = GetFinalMask32(args.num_datapoints);
    last_group_ = args.num_32dp_simd_iters - 1;
  }

  // Group k for query j; acc1 (datapoints 16..31) only with int32.
  SCANN_AVX512_INLINE void Push(size_t j, size_t k, __m512i acc0,
                                __m512i acc1) {
    // Bit i: datapoint i's accumulator is below the threshold (signed).
    auto compute_push_mask = [&]() SCANN_AVX512_INLINE_LAMBDA -> uint32_t {
      if constexpr (kUseInt32Accumulator) {
        const __m512i t = _mm512_set1_epi32(thresholds_[j]);
        return static_cast<uint32_t>(_mm512_cmplt_epi32_mask(acc0, t)) |
               (static_cast<uint32_t>(_mm512_cmplt_epi32_mask(acc1, t))
                << 16);
      } else {
        return _mm512_cmplt_epi16_mask(acc0,
                                       _mm512_set1_epi16(thresholds_[j]));
      }
    };
    uint32_t push_mask = compute_push_mask();
    if (!push_mask) return;

    if constexpr (IsSame<T, float>()) {
      __m512i lo = acc0, hi = acc1;
      if constexpr (!kUseInt32Accumulator) {
        lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(acc0));  // 0..15
        hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(acc0, 1));
      }
      // float(acc) * inv_mult + bias, unfused as in the AVX2 kernel.
      _mm512_storeu_ps(
          distances_buffer_,
          _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(lo),
                                      simd_inv_mults_[j]),
                        simd_biases_[j]));
      _mm512_storeu_ps(
          distances_buffer_ + 16,
          _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(hi),
                                      simd_inv_mults_[j]),
                        simd_biases_[j]));
    } else {
      _mm512_storeu_si512(distances_buffer_, acc0);
    }

    if (k == last_group_) {
      push_mask &= final_mask_;  // padding datapoints of the last group
    }
    if (whitelists_[j]) {
      push_mask &= whitelists_[j][k];  // restricts: 32 bits per group
    }
    const DatapointIndex first_dp_index = args_.first_dp_index + 32 * k;
    if constexpr (kHasAnyPredicate) {
      if (args_.batch_filter_predicate) {
        push_mask = args_.batch_filter_predicate(
            first_dp_index, push_mask, args_.datapoint_translation_predicate);
      }
    }

    while (push_mask) {  // in datapoint order, as the AVX2 kernel
      const int offset = bits::FindLSBSetNonZero(push_mask);
      push_mask &= (push_mask - 1);
      DatapointIndex dp_idx = first_dp_index + offset;
      DCHECK(!whitelists_[j] || args_.restrict_whitelists[j].IsWhitelisted(
                                    dp_idx - args_.first_dp_index))
          << dp_idx;
      if constexpr (kHasAnyPredicate) {
        if (args_.datapoint_translation_predicate) {
          dp_idx = args_.datapoint_translation_predicate(dp_idx);
        }
      }
      bool needs_gc;
      if constexpr (IsSame<T, float>()) {
        needs_gc = mutators_[j].PushNoEpsilonCheck(dp_idx,
                                                   distances_buffer_[offset]);
      } else {
        needs_gc = mutators_[j].Push(dp_idx, distances_buffer_[offset]);
      }
      if (ABSL_PREDICT_FALSE(needs_gc)) {
        mutators_[j].GarbageCollect();
        thresholds_[j] = Threshold(j, mutators_[j].epsilon());
        push_mask &= compute_push_mask();  // lower threshold, rest of group
      }
    }
  }

 private:
  SCANN_AVX512_INLINE ThresholdT Threshold(size_t j, T epsilon) const {
    if constexpr (IsSame<T, int16_t>()) {
      return epsilon;
    } else {
      const float float_threshold = (epsilon - biases_[j]) * mults_[j];
      if constexpr (kUseInt32Accumulator) {
        return GetInt32Threshold(float_threshold);
      } else {
        return GetInt16Threshold(float_threshold);
      }
    }
  }

  static constexpr size_t kNumQueriesFloat = IsSame<T, float>() ? kNumQueries
                                                                : 0;
  LUT16ArgsTopN<T, TopN>& args_;
  typename TopN::Mutator mutators_[kNumQueries];
  ThresholdT thresholds_[kNumQueries];
  array<float, kNumQueriesFloat> biases_;
  array<float, kNumQueriesFloat> mults_;
  array<__m512, kNumQueriesFloat> simd_biases_;
  array<__m512, kNumQueriesFloat> simd_inv_mults_;
  array<const uint32_t*, kNumQueries> whitelists_;
  uint32_t final_mask_;
  size_t last_group_;
  T distances_buffer_[32];
};

template <typename T, size_t kNumQueries, typename Tuning,
          bool kHasAnyPredicate, bool kUseInt32Accumulator, typename TopN>
SCANN_AVX512_OUTLINE void GetTopDistancesImpl(LUT16ArgsTopN<T, TopN>& args) {
  const uint8_t* data_start = args.packed_dataset;
  const size_t num_codes_per_dp = args.num_blocks;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  DCHECK_EQ(num_32dp_simd_iters, DivRoundUp(args.num_datapoints, 32));
  if (num_32dp_simd_iters == 0) return;
  DCHECK(data_start);
  auto lookups = ToLocalArray<kNumQueries>(args.lookups);
  const PrefetchCtx ctx{data_start + num_32dp_simd_iters * 16 * num_codes_per_dp,
                        args.next_partition};

  TopNPusher<T, kNumQueries, kHasAnyPredicate, kUseInt32Accumulator, TopN>
      pusher(args);

  size_t k = 0;
  auto super_group = [&](auto num_datapoints_tag) SCANN_AVX512_INLINE_LAMBDA {
    constexpr size_t kNumDatapoints = decltype(num_datapoints_tag)::value;
    constexpr size_t kNumGroups = kNumDatapoints / 32;
    // Group g's accumulators: register g (32 x int16), or registers 2g and
  // 2g + 1 (16 x int32 each); all in datapoint order.
    if constexpr (kUseInt32Accumulator) {
      Avx512For<int32_t, kNumQueries, kNumDatapoints> dists =
          Int32MiddleLoop<kNumDatapoints, kNumQueries, Tuning>(
              data_start, lookups, num_codes_per_dp, ctx);
      for (size_t j : Seq(kNumQueries)) {
        for (size_t g : Seq(kNumGroups)) {
          pusher.Push(j, k + g, *dists[j][2 * g], *dists[j][2 * g + 1]);
        }
      }
    } else {
      Avx512For<int16_t, kNumQueries, kNumDatapoints> dists =
          Int16MiddleLoop<kNumDatapoints, kNumQueries, Tuning>(
              data_start, lookups, num_codes_per_dp, ctx);
      for (size_t j : Seq(kNumQueries)) {
        for (size_t g : Seq(kNumGroups)) {
          pusher.Push(j, k + g, *dists[j][g], _mm512_setzero_si512());
        }
      }
    }
    data_start += kNumDatapoints * num_codes_per_dp / 2;  // 4 bits per code
    k += kNumGroups;
  };

  // The super-groups of the layout (lut16_avx512_swizzle.h), in order.
  for (auto _ : Seq(num_32dp_simd_iters / 8)) {
    super_group(std::integral_constant<size_t, 256>());
  }
  if (num_32dp_simd_iters % 8 >= 4) {
    super_group(std::integral_constant<size_t, 128>());
  }
  for (auto _ : Seq(num_32dp_simd_iters % 4)) {
    super_group(std::integral_constant<size_t, 32>());
  }
}

}  // namespace lut16
}  // namespace avx512

namespace asymmetric_hashing_internal {

using avx512::lut16::GetTopDistancesImpl;

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
void LUT16Avx512<kNumQueries, kPrefetch>::GetTopInt16Distances(
    LUT16ArgsTopN<int16_t> args) {
  using Tuning = LUT16Tuning<kPrefetch>;
  return GetTopDistancesImpl<int16_t, kNumQueries, Tuning, false, false>(
      args);
}

// scann-core: accumulator and predicates chosen as in LUT16Avx2.
template <size_t kNumQueries, PrefetchStrategy kPrefetch>
void LUT16Avx512<kNumQueries, kPrefetch>::GetTopFloatDistances(
    LUT16ArgsTopN<float> args) {
  using Tuning = LUT16Tuning<kPrefetch>;
  const bool has_predicate =
      args.batch_filter_predicate || args.datapoint_translation_predicate;
  const bool use_int32 = args.num_blocks >= 256;  // as AVX2's ChooseAccum
  if (has_predicate) {
    if (use_int32) {
      return GetTopDistancesImpl<float, kNumQueries, Tuning, true, true>(args);
    }
    return GetTopDistancesImpl<float, kNumQueries, Tuning, true, false>(args);
  }
  if (use_int32) {
    return GetTopDistancesImpl<float, kNumQueries, Tuning, false, true>(args);
  }
  return GetTopDistancesImpl<float, kNumQueries, Tuning, false, false>(args);
}

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#endif
