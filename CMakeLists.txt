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

cmake_minimum_required(VERSION 3.27)

# Bazel's `-c opt` compiles everything (ScaNN and its dependencies) at -O2,
# with -O3 only on the specific targets that ask for it (see
# cmake/SourceFlags.cmake). Mirror that instead of CMake's -O3 Release
# default. Must be set before project() to take effect as the default.
if(NOT DEFINED CMAKE_CXX_FLAGS_RELEASE)
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "Flags used by the CXX compiler during RELEASE builds.")
endif()
if(NOT DEFINED CMAKE_C_FLAGS_RELEASE)
  set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "Flags used by the C compiler during RELEASE builds.")
endif()

project(scann_core VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Bazel passes -std=c++17, not -std=gnu++17. Besides dialect, this matters
# numerically under GCC: gnu++17 defaults to -ffp-contract=fast, c++17 to
# -ffp-contract=off, which changes FMA contraction and therefore distances.
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(SCANN_BUILD_STATIC "Build the static (.a) scann_core library" ON)
option(SCANN_BUILD_SHARED "Build the shared (.so/.dylib) scann_core library" ON)
option(SCANN_BUILD_PYTHON "Build the pybind11 Python module and package" ON)
option(SCANN_BUILD_RUST_BINDINGS "Build the Rust bindings (via cxx)" ON)
option(SCANN_BUILD_TESTS "Build the C++ tests" ON)
option(SCANN_BUILD_EXAMPLES "Build the C++ examples" ON)
option(SCANN_USE_SYSTEM_DEPS
  "Try find_package() for each dependency before fetching it (versions are still required to match the pins exactly)" OFF)
option(SCANN_ENABLE_LTO "Build scann_core with link-time optimization (see README before enabling)" OFF)

if(NOT SCANN_BUILD_STATIC AND NOT SCANN_BUILD_SHARED)
  message(FATAL_ERROR "At least one of SCANN_BUILD_STATIC / SCANN_BUILD_SHARED must be ON")
endif()
if(SCANN_BUILD_RUST_BINDINGS AND NOT SCANN_BUILD_STATIC)
  message(FATAL_ERROR "SCANN_BUILD_RUST_BINDINGS=ON requires SCANN_BUILD_STATIC=ON (the crate links the static archive)")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Global flags first: like Bazel's --copt, they must also reach the
# dependencies (highway in particular), so they are set before FetchContent.
include(Flags)
include(Dependencies)
include(Proto)
include(SourceFlags)

set(SCANN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(SCANN_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

# -----------------------------------------------------------------------
# Generated sources
# -----------------------------------------------------------------------
file(GLOB_RECURSE SCANN_PROTO_FILES CONFIGURE_DEPENDS "${SCANN_SRC_DIR}/*.proto")
scann_generate_cpp_protos(SCANN_PROTO_SRCS SCANN_PROTO_HDRS
  PROTO_ROOT "${SCANN_SRC_DIR}"
  OUT_DIR "${SCANN_GEN_DIR}"
  PROTOS ${SCANN_PROTO_FILES})

# Equivalent of hashes/internal/template_sharding.bzl's batch_size_sharder.
scann_shard_lut16_templates(SCANN_LUT16_SRCS
  TEMPLATE_DIR "${SCANN_SRC_DIR}/scann/hashes/internal/bazel_templates"
  OUT_DIR "${SCANN_GEN_DIR}/scann/hashes/internal")

# -----------------------------------------------------------------------
# Core library
# -----------------------------------------------------------------------
file(GLOB_RECURSE SCANN_CORE_SRCS CONFIGURE_DEPENDS "${SCANN_SRC_DIR}/*.cc")
# Templates are expanded above, never compiled directly.
list(FILTER SCANN_CORE_SRCS EXCLUDE REGEX "\\.tpl\\.cc$")
# The pybind11-typed facade and module entry point are consumers of the
# core, not part of it -- built by python/CMakeLists.txt.
list(FILTER SCANN_CORE_SRCS EXCLUDE REGEX "/scann_ops/cc/python/scann_pybind\\.cc$")
list(FILTER SCANN_CORE_SRCS EXCLUDE REGEX "/scann_ops/cc/scann_npy\\.cc$")

# scann-core's own additions (the C++ config builder), kept outside src/ so
# src/ stays a faithful copy of upstream plus documented fixes.
set(SCANN_EXTRA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/core")
file(GLOB_RECURSE SCANN_EXTRA_SRCS CONFIGURE_DEPENDS "${SCANN_EXTRA_DIR}/*.cc")

scann_apply_bazel_source_flags("${SCANN_SRC_DIR}" ${SCANN_LUT16_SRCS})

# Usage requirements, shared by every target built from the core objects.
add_library(scann_core_deps INTERFACE)
target_include_directories(scann_core_deps INTERFACE
  "$<BUILD_INTERFACE:${SCANN_SRC_DIR}>"
  "$<BUILD_INTERFACE:${SCANN_EXTRA_DIR}>"
  "$<BUILD_INTERFACE:${SCANN_GEN_DIR}>"
)
# The abseil list is exactly the set of @com_google_absl labels referenced
# by the Bazel packages this tree was extracted from, mapped to abseil's
# CMake target names.
target_link_libraries(scann_core_deps INTERFACE
  protobuf::libprotobuf
  absl::algorithm_container
  absl::any_invocable
  absl::base
  absl::bits
  absl::btree
  absl::check
  absl::core_headers
  absl::endian
  absl::flags
  absl::flat_hash_map
  absl::flat_hash_set
  absl::hash
  absl::inlined_vector
  absl::int128
  absl::log
  absl::memory
  absl::no_destructor
  absl::node_hash_map
  absl::node_hash_set
  absl::nullability
  absl::optional
  absl::prefetch
  absl::random_distributions
  absl::random_random
  absl::span
  absl::status
  absl::statusor
  absl::str_format
  absl::string_view
  absl::strings
  absl::synchronization
  absl::time
  absl::variant
  hwy::hwy
  hwy::hwy_contrib
  Eigen3::Eigen
  cnpy
)

# Compiled exactly once; the static and shared libraries both relink these
# objects, so building both is not twice the compile time.
add_library(scann_core_objects OBJECT ${SCANN_CORE_SRCS} ${SCANN_EXTRA_SRCS} ${SCANN_LUT16_SRCS} ${SCANN_PROTO_SRCS})
target_link_libraries(scann_core_objects PUBLIC scann_core_deps)

if(SCANN_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _scann_ipo_ok OUTPUT _scann_ipo_msg LANGUAGES CXX)
  if(NOT _scann_ipo_ok)
    message(FATAL_ERROR "SCANN_ENABLE_LTO=ON but IPO is not supported: ${_scann_ipo_msg}")
  endif()
  set_target_properties(scann_core_objects PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# Several distance measures register themselves by name from static
# initializers (SCANN_REGISTER_DISTANCE_MEASURE); Bazel marks those
# libraries alwayslink = 1. Any consumer of the static archive therefore
# has to link scann_core's own objects whole-archive, or the linker drops
# the registrations and config lookups like "DotProductDistance" fail at
# runtime. scann::core_static does that for CMake consumers.
if(SCANN_BUILD_STATIC)
  add_library(scann_core_static STATIC $<TARGET_OBJECTS:scann_core_objects>)
  target_link_libraries(scann_core_static PUBLIC scann_core_deps)
  set_target_properties(scann_core_static PROPERTIES OUTPUT_NAME scann_core)

  add_library(scann_core_static_whole INTERFACE)
  target_link_libraries(scann_core_static_whole INTERFACE
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,scann_core_static>")
  add_library(scann::core_static ALIAS scann_core_static_whole)

  # For consumers outside CMake (the Rust crate, hand-written Makefiles):
  # every transitive static dependency archive merged into one, so linking
  # needs just libscann_core.a (whole-archive) + libscann_core_deps.a.
  include(BundleStatic)
  scann_bundle_static_deps(scann_core_deps_bundle
    ROOT scann_core_deps
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libscann_core_deps.a"
    SYSTEM_LIBS_VAR SCANN_CORE_SYSTEM_LIBS)
endif()

if(SCANN_BUILD_SHARED)
  add_library(scann_core_shared SHARED $<TARGET_OBJECTS:scann_core_objects>)
  # Dependencies are linked into the .so; consumers only need their headers,
  # not a second copy of abseil/protobuf.
  target_link_libraries(scann_core_shared
    PRIVATE scann_core_deps
    INTERFACE "$<COMPILE_ONLY:scann_core_deps>")
  set_target_properties(scann_core_shared PROPERTIES
    OUTPUT_NAME scann_core
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
  if(SCANN_ENABLE_LTO)
    set_target_properties(scann_core_shared PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
  endif()
  add_library(scann::core_shared ALIAS scann_core_shared)
endif()

if(SCANN_BUILD_STATIC)
  add_library(scann::core ALIAS scann_core_static_whole)
else()
  add_library(scann::core ALIAS scann_core_shared)
endif()

if(SCANN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(SCANN_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(SCANN_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(SCANN_BUILD_RUST_BINDINGS)
  add_subdirectory(rust)
endif()
