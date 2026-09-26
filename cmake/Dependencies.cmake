# Copyright 2026 ebenali and TheCleaners.
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

# Vendors scann-core's dependencies via FetchContent.
#
# Each dependency is pinned to an exact release archive and its SHA-256:
# the latest release of each as of 2026-09-22. (The initial import commit
# pinned instead the exact archives upstream's Bazel build resolved, to
# prove the extraction equivalent to the upstream wheel; see git history
# and tests/equivalence/.) cnpy has no releases; it is pinned to a commit,
# which is also its current HEAD.
#
# Offline / reproducible builds -- see README.md "Dependencies":
#   * -DFETCHCONTENT_SOURCE_DIR_<NAME>=/path   use an existing source tree
#     (NAME = ABSL, ZLIB, PROTOBUF, HIGHWAY, EIGEN, CNPY, PYBIND11)
#   * -DFETCHCONTENT_FULLY_DISCONNECTED=ON     never touch the network
#     (requires every dependency to be populated already)
#   * -DSCANN_USE_SYSTEM_DEPS=ON               find_package() first; a found
#     package is only accepted if its version matches the pin exactly.

include(FetchContent)

# abseil and protobuf link Threads::Threads, but imported targets are
# directory-scoped: import it here too so the top level can resolve it (see
# BundleStatic.cmake, which walks the link graph from this directory).
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if(SCANN_USE_SYSTEM_DEPS)
  set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE ALWAYS)
else()
  set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
endif()

# --- Dependency build options ----------------------------------------------
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

set(ZLIB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZLIB_INSTALL OFF CACHE BOOL "" FORCE)

set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_CONFORMANCE OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_PROTOC_BINARIES ON CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(utf8_range_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
# ScaNN never uses protobuf's gzip streams. With this ON, protobuf's
# find_package(ZLIB) picks up the *system* libz, giving the build a second,
# unpinned zlib alongside the vendored one cnpy uses.
set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
set(protobuf_ABSL_PROVIDER "package" CACHE STRING "" FORCE)

set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB ON CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)

# --- Declarations ----------------------------------------------------------
FetchContent_Declare(absl
  URL https://github.com/abseil/abseil-cpp/releases/download/20260817.0/abseil-cpp-20260817.0.tar.gz
  URL_HASH SHA256=f7e05179df39c45434cad433f5783840bb3788ef322976f9138bc6b72b3a107d
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  FIND_PACKAGE_ARGS CONFIG)

FetchContent_Declare(zlib
  URL https://github.com/madler/zlib/releases/download/v1.3.2/zlib-1.3.2.tar.gz
  URL_HASH SHA256=bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  FIND_PACKAGE_ARGS NAMES ZLIB)

FetchContent_Declare(protobuf
  URL https://github.com/protocolbuffers/protobuf/releases/download/v36.2/protobuf-36.2.tar.gz
  URL_HASH SHA256=3d9642a662d10e68ebae5e53f14dcce5105684212d5078f8e0d47d1ab3ae6b64
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  FIND_PACKAGE_ARGS CONFIG)

# Upstream used 1.3.0, which clang 23 cannot compile for any SIMD target
# above the baseline ISA (always_inline intrinsics rejected inside its
# per-target pragma regions); 1.4.0 compiles with any baseline and needs no
# HWY_DISABLED_TARGETS workaround.
FetchContent_Declare(highway
  URL https://github.com/google/highway/releases/download/1.4.0/highway-1.4.0.tar.gz
  URL_HASH SHA256=36f672ab48ddb3c8555e9e89e16fe400cd7d16c6eb455a1a3d0c146a63ababdc
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  FIND_PACKAGE_ARGS NAMES hwy CONFIG)

FetchContent_Declare(eigen
  URL https://gitlab.com/libeigen/eigen/-/package_files/246179253/download
      https://github.com/eigen-mirror/eigen/archive/refs/tags/5.0.1.tar.gz
      https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
  URL_HASH SHA256=e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec
  DOWNLOAD_NAME eigen-5.0.1.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  FIND_PACKAGE_ARGS NAMES Eigen3 CONFIG)

FetchContent_Declare(cnpy
  GIT_REPOSITORY https://github.com/sammymax/cnpy.git
  GIT_TAG 57184ee0db37cac383fc29175950747a46a8b512
  # cnpy ships an ancient CMakeLists.txt (cmake_minimum_required 2.x, which
  # CMake >= 4 rejects outright). Pointing SOURCE_SUBDIR at a path that
  # doesn't exist makes FetchContent populate the source without
  # add_subdirectory()-ing it; the `cnpy` target is defined below instead.
  SOURCE_SUBDIR "no-such-dir-skip-cnpy-cmakelists")

FetchContent_MakeAvailable(absl zlib protobuf highway eigen cnpy)

if(SCANN_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)
  set(PYBIND11_INSTALL OFF CACHE BOOL "" FORCE)
  set(PYBIND11_TEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v3.1.0.tar.gz
    URL_HASH SHA256=ef712655692a2e9bf7bb7874c022564a45f91d847ddee987e720cd9e28849665
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    FIND_PACKAGE_ARGS CONFIG)
  FetchContent_MakeAvailable(pybind11)
endif()

# --- Exact-version enforcement for find_package() results -----------------
# FIND_PACKAGE_ARGS only engages with SCANN_USE_SYSTEM_DEPS=ON. A system
# package that doesn't match the pin exactly is rejected rather than used.
function(_scann_require_exact name found_version)
  FetchContent_GetProperties(${name})
  if(${name}_POPULATED)
    return()  # vendored: the URL_HASH already pins it exactly
  endif()
  set(ok FALSE)
  foreach(want ${ARGN})
    if(found_version VERSION_EQUAL want)
      set(ok TRUE)
    endif()
  endforeach()
  if(NOT ok)
    message(FATAL_ERROR
      "SCANN_USE_SYSTEM_DEPS: system ${name} is version '${found_version}', "
      "but scann-core is pinned to '${ARGN}'. Install the pinned version, "
      "point FETCHCONTENT_SOURCE_DIR_<NAME> at it, or turn SCANN_USE_SYSTEM_DEPS off.")
  endif()
  message(STATUS "scann-core: using system ${name} ${found_version}")
endfunction()

if(SCANN_USE_SYSTEM_DEPS)
  _scann_require_exact(absl "${absl_VERSION}" 20260817)
  _scann_require_exact(zlib "${ZLIB_VERSION}${ZLIB_VERSION_STRING}" 1.3.2)
  # protobuf's package version is the release number; its C++ runtime
  # reports 7.36.2 for the same release.
  _scann_require_exact(protobuf "${protobuf_VERSION}" 36.2 7.36.2)
  _scann_require_exact(highway "${hwy_VERSION}" 1.4.0)
  _scann_require_exact(eigen "${Eigen3_VERSION}" 5.0.1)
  if(SCANN_BUILD_PYTHON)
    _scann_require_exact(pybind11 "${pybind11_VERSION}" 3.1.0)
  endif()
endif()

# --- Normalize target names between vendored and system packages ---------
if(TARGET hwy AND NOT TARGET hwy::hwy)
  add_library(hwy::hwy ALIAS hwy)
endif()
if(TARGET hwy_contrib AND NOT TARGET hwy::hwy_contrib)
  add_library(hwy::hwy_contrib ALIAS hwy_contrib)
endif()

if(TARGET ZLIB::ZLIBSTATIC)
  set(SCANN_ZLIB_TARGET ZLIB::ZLIBSTATIC)  # vendored (static only)
else()
  set(SCANN_ZLIB_TARGET ZLIB::ZLIB)        # system
endif()

# Directory holding google/protobuf/*.proto, for the protoc import path.
FetchContent_GetProperties(protobuf)
if(protobuf_POPULATED)
  set(SCANN_PROTOBUF_WKT_DIR "${protobuf_SOURCE_DIR}/src")
else()
  get_target_property(_pb_incs protobuf::libprotobuf INTERFACE_INCLUDE_DIRECTORIES)
  foreach(_dir ${_pb_incs})
    if(EXISTS "${_dir}/google/protobuf/timestamp.proto")
      set(SCANN_PROTOBUF_WKT_DIR "${_dir}")
    endif()
  endforeach()
  if(NOT SCANN_PROTOBUF_WKT_DIR)
    message(FATAL_ERROR "Could not locate google/protobuf/timestamp.proto in the system protobuf's include dirs")
  endif()
endif()

# --- cnpy --------------------------------------------------------------------
# The one source file ScaNN needs, with the flags cnpy requires.
FetchContent_GetProperties(cnpy)
add_library(cnpy STATIC "${cnpy_SOURCE_DIR}/cnpy/cnpy.cpp")
target_include_directories(cnpy PUBLIC "${cnpy_SOURCE_DIR}")
target_link_libraries(cnpy PUBLIC ${SCANN_ZLIB_TARGET})
target_compile_options(cnpy PRIVATE -Wno-unused-variable -fexceptions)
