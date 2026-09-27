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

# Global compile flags, applied to scann-core *and* every FetchContent
# dependency (the equivalent of Bazel's --copt, which reaches external
# repositories too). Each flag below is carried over from the Bazel build
# the reference wheel was produced with; see README.md "Compile flags" for
# the ones that were deliberately not carried over.
#
# Include this before cmake/Dependencies.cmake: directory-scoped options
# only reach subdirectories (i.e. FetchContent deps) added afterwards.

# --- Instruction set -------------------------------------------------------
# Upstream's documented x86 build uses -mavx -mfma (and its PyPI wheels
# require AVX+FMA); the ARM build uses -march=armv8-a+simd. That is the
# portable default here. The ISA baseline changes which code paths are
# compiled in (and FMA availability changes floating-point contraction), so
# bit-for-bit comparisons (tests/equivalence against the upstream wheel) are
# only meaningful between builds with the same value: the default.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(_scann_default_arch "-mavx;-mfma")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  set(_scann_default_arch "-march=armv8-a+simd")
else()
  set(_scann_default_arch "")
endif()
set(SCANN_ARCH_FLAGS "${_scann_default_arch}" CACHE STRING
  "Instruction-set flags applied to scann-core and all dependencies (;-separated)")

# --- Highway dispatch targets --------------------------------------------
# The upstream Bazel build used to pass
# HWY_DISABLED_TARGETS=(HWY_AVX3_SPR|HWY_AVX10_2) because highway 1.3.0's
# vqsort float16 kernels fail to compile for those targets with clang 23.
# With highway 1.4.0 (see Dependencies.cmake) that is no longer needed, so
# the default is empty -- Highway's own target set -- in both builds. If you
# set it, it is applied globally, to highway's own library and to every
# ScaNN translation unit including hwy headers; a per-TU value would make
# the dispatch tables disagree across TUs.
set(SCANN_HWY_DISABLED_TARGETS "" CACHE STRING
  "Value of HWY_DISABLED_TARGETS (empty: Highway's defaults)")

set(SCANN_GLOBAL_COMPILE_OPTIONS
  ${SCANN_ARCH_FLAGS}
  # Carried over from upstream's documented build. Default-on for C++14+
  # in GCC and in clang >= 19, so a no-op with the toolchains this was
  # validated on, but it keeps older clang ABI-compatible with the rest.
  -fsized-deallocation
)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  # Bazel passes --copt=-w: third-party warning noise only.
  list(APPEND SCANN_GLOBAL_COMPILE_OPTIONS -w)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # GCC >= 14 type-checks template bodies before instantiation; some ScaNN
  # templates are only ever instantiated under clang (e.g. AMX tile code).
  # This defers those diagnostics to instantiation, where they still apply.
  # (Plain flag, not a $<COMPILE_LANGUAGE:CXX> genex: rust/CMakeLists.txt
  # writes these options to a file, which must be language-independent. GCC
  # accepts it for C files too.)
  list(APPEND SCANN_GLOBAL_COMPILE_OPTIONS -Wno-template-body)
endif()

set(SCANN_GLOBAL_COMPILE_DEFINITIONS "")
# Builds without NDEBUG (Debug, or no build type): Highway makes Lanes()
# non-constexpr in its debug mode, so that code can't come to depend on it,
# and ScaNN's Highway one-to-many kernels only exist when it is constexpr
# (#if HWY_HAVE_CONSTEXPR_LANES in one_to_many_impl_highway.inc) while
# one_to_many_asymmetric.h calls them unconditionally: ScaNN doesn't compile
# ("use of undeclared identifier 'highway'"). Upstream only builds with
# NDEBUG (Bazel -c opt). Turning Highway's debug mode off there disables
# only Highway's internal assertions; release and sanitizer builds are
# unaffected. scann-core's public headers don't include Highway, so this
# stays inside scann-core's build.
list(APPEND SCANN_GLOBAL_COMPILE_DEFINITIONS
  "$<$<NOT:$<CONFIG:Release,RelWithDebInfo,MinSizeRel>>:HWY_IS_DEBUG_BUILD=0>")
if(NOT SCANN_HWY_DISABLED_TARGETS STREQUAL "")
  list(APPEND SCANN_GLOBAL_COMPILE_DEFINITIONS "HWY_DISABLED_TARGETS=${SCANN_HWY_DISABLED_TARGETS}")
endif()

# --- Sanitizers ------------------------------------------------------------
# e.g. -DSCANN_SANITIZE=address,undefined or -DSCANN_SANITIZE=thread.
# Applied globally: dependencies are instrumented too (required for TSan,
# and ASan's container-overflow checks need consistent instrumentation).
set(SCANN_SANITIZE "" CACHE STRING "Comma-separated -fsanitize= value (empty: none)")
if(SCANN_SANITIZE)
  list(APPEND SCANN_GLOBAL_COMPILE_OPTIONS
    "-fsanitize=${SCANN_SANITIZE}" -fno-omit-frame-pointer -g)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # One report per check site instead of merged ones (clang only).
    list(APPEND SCANN_GLOBAL_COMPILE_OPTIONS -fno-sanitize-merge)
  endif()
  add_link_options("-fsanitize=${SCANN_SANITIZE}")
endif()

add_compile_options(${SCANN_GLOBAL_COMPILE_OPTIONS})
add_compile_definitions(${SCANN_GLOBAL_COMPILE_DEFINITIONS})

message(STATUS "scann-core: arch flags: ${SCANN_ARCH_FLAGS}")
message(STATUS "scann-core: HWY_DISABLED_TARGETS=${SCANN_HWY_DISABLED_TARGETS}")
