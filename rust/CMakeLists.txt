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

# Builds the scann-core Rust crate by invoking cargo. CMake compiles no Rust
# and cargo compiles none of scann_core: build.rs only compiles the small
# cxx-bridge shim (src/bridge.rs + src/shim.cc) and links it against
#   libscann_core.a       scann_core's own objects, linked whole-archive
#                         (static-initializer distance measure registrations)
#   libscann_core_deps.a  every transitive static dependency, merged
#
# Everything build.rs needs is written to scann_core_rust_build.env below,
# derived from the real CMake targets so the shim is compiled with the same
# include paths, defines and ISA flags as the core (the shim includes
# scann/abseil/highway headers; mismatched flags there would be an ODR
# violation). For standalone `cargo build`/`cargo check`/rust-analyzer,
# export SCANN_CORE_BUILD_ENV=<build dir>/rust/scann_core_rust_build.env.

set(SCANN_RUST_ENV_FILE "${CMAKE_CURRENT_BINARY_DIR}/scann_core_rust_build.env")
set(SCANN_RUST_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/target")
set(SCANN_EQUIV_FIXTURES "${PROJECT_BINARY_DIR}/equivalence/fixtures" CACHE PATH
  "Fixture directory produced by tests/equivalence/run.py, consumed by the Rust equivalence test")

set(_objs scann_core_objects)
set(_system_links "")
foreach(item ${SCANN_CORE_SYSTEM_LIBS})
  string(APPEND _system_links "link=${item}\n")
endforeach()

file(GENERATE OUTPUT "${SCANN_RUST_ENV_FILE}" CONTENT
"cxx=${CMAKE_CXX_COMPILER}
lib_dir=$<TARGET_FILE_DIR:scann_core_static>
deps_archive=${PROJECT_BINARY_DIR}/libscann_core_deps.a
include=$<JOIN:$<REMOVE_DUPLICATES:$<TARGET_PROPERTY:${_objs},INCLUDE_DIRECTORIES>>,
include=>
flag=$<JOIN:$<REMOVE_DUPLICATES:$<TARGET_PROPERTY:${_objs},COMPILE_OPTIONS>>,
flag=>
define=$<JOIN:$<REMOVE_DUPLICATES:$<TARGET_PROPERTY:${_objs},COMPILE_DEFINITIONS>>,
define=>
define=$<$<CONFIG:Release,RelWithDebInfo,MinSizeRel>:NDEBUG>
${_system_links}")

# Everything build.rs needs to compile the shim and link the static libraries.
add_custom_target(scann_core_rust_inputs)
add_dependencies(scann_core_rust_inputs scann_core_static scann_core_deps_bundle)

if(NOT SCANN_BUILD_RUST_BINDINGS)
  return()  # SCANN_RUST_BUILD_ENV_ONLY: build.rs drives cargo itself
endif()

find_program(CARGO_EXECUTABLE cargo)
if(NOT CARGO_EXECUTABLE)
  message(FATAL_ERROR
    "SCANN_BUILD_RUST_BINDINGS is ON but `cargo` was not found on PATH. "
    "Install Rust (https://rustup.rs) or configure with -DSCANN_BUILD_RUST_BINDINGS=OFF.")
endif()

set(_cargo_env
  "${CMAKE_COMMAND}" -E env
  "SCANN_CORE_BUILD_ENV=${SCANN_RUST_ENV_FILE}"
  "CXX=${CMAKE_CXX_COMPILER}")
# --locked: never rewrite Cargo.lock in the source tree from a CMake build.
set(_cargo_args
  --manifest-path "${PROJECT_SOURCE_DIR}/Cargo.toml"
  --target-dir "${SCANN_RUST_TARGET_DIR}"
  --release --locked)

add_custom_target(scann_core_rust ALL
  COMMAND ${_cargo_env} "${CARGO_EXECUTABLE}" build --lib --examples ${_cargo_args}
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  COMMENT "cargo build (scann-core Rust bindings)"
  VERBATIM)
add_dependencies(scann_core_rust scann_core_static scann_core_deps_bundle)

# cargo test: the API tests are self-contained; the equivalence test runs
# against SCANN_EQUIV_FIXTURES when that directory exists, and skips otherwise.
add_custom_target(scann_core_rust_test
  COMMAND ${_cargo_env} "SCANN_CORE_EQUIV_FIXTURES=${SCANN_EQUIV_FIXTURES}"
          "${CARGO_EXECUTABLE}" test ${_cargo_args} -- --nocapture
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  COMMENT "cargo test (scann-core Rust bindings)"
  VERBATIM)
add_dependencies(scann_core_rust_test scann_core_static scann_core_deps_bundle)

if(SCANN_BUILD_TESTS)
  add_test(NAME rust
           COMMAND ${_cargo_env} "SCANN_CORE_EQUIV_FIXTURES=${SCANN_EQUIV_FIXTURES}"
                   "${CARGO_EXECUTABLE}" test ${_cargo_args}
           WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  # The examples check their own results (built by scann_core_rust above).
  foreach(example quickstart updating)
    add_test(NAME example_rust_${example}
             COMMAND ${_cargo_env} "${CARGO_EXECUTABLE}" run --quiet
                     --example ${example} ${_cargo_args}
             WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    set_tests_properties(example_rust_${example} PROPERTIES TIMEOUT 600)
  endforeach()
endif()
