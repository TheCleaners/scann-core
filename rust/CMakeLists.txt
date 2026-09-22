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

find_program(CARGO_EXECUTABLE cargo)
if(NOT CARGO_EXECUTABLE)
  message(FATAL_ERROR
    "SCANN_BUILD_RUST_BINDINGS is ON but `cargo` was not found on PATH. "
    "Install Rust (https://rustup.rs) or configure with -DSCANN_BUILD_RUST_BINDINGS=OFF.")
endif()

set(SCANN_RUST_ENV_FILE "${CMAKE_CURRENT_BINARY_DIR}/scann_core_rust_build.env")
set(SCANN_RUST_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/target")
set(SCANN_EQUIV_FIXTURES "${CMAKE_BINARY_DIR}/equivalence/fixtures" CACHE PATH
  "Fixture directory produced by tests/equivalence/run.py, consumed by the Rust equivalence test")

set(_objs scann_core_objects)
set(_system_links "")
foreach(item ${SCANN_CORE_SYSTEM_LIBS})
  string(APPEND _system_links "link=${item}\n")
endforeach()

file(GENERATE OUTPUT "${SCANN_RUST_ENV_FILE}" CONTENT
"cxx=${CMAKE_CXX_COMPILER}
lib_dir=$<TARGET_FILE_DIR:scann_core_static>
deps_archive=${CMAKE_BINARY_DIR}/libscann_core_deps.a
include=$<JOIN:$<REMOVE_DUPLICATES:$<TARGET_PROPERTY:${_objs},INCLUDE_DIRECTORIES>>,
include=>
flag=$<JOIN:$<REMOVE_DUPLICATES:$<TARGET_PROPERTY:${_objs},COMPILE_OPTIONS>>,
flag=>
define=$<JOIN:$<REMOVE_DUPLICATES:$<TARGET_PROPERTY:${_objs},COMPILE_DEFINITIONS>>,
define=>
define=$<$<CONFIG:Release,RelWithDebInfo,MinSizeRel>:NDEBUG>
${_system_links}")

set(_cargo_env
  "${CMAKE_COMMAND}" -E env
  "SCANN_CORE_BUILD_ENV=${SCANN_RUST_ENV_FILE}"
  "CXX=${CMAKE_CXX_COMPILER}")
set(_cargo_args
  --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
  --target-dir "${SCANN_RUST_TARGET_DIR}"
  --release)

add_custom_target(scann_core_rust ALL
  COMMAND ${_cargo_env} "${CARGO_EXECUTABLE}" build --lib --examples ${_cargo_args}
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT "cargo build (scann-core Rust bindings)"
  VERBATIM)
add_dependencies(scann_core_rust scann_core_static scann_core_deps_bundle)

# Not part of ALL: needs the fixtures written by tests/equivalence/run.py.
add_custom_target(scann_core_rust_test
  COMMAND ${_cargo_env} "SCANN_CORE_EQUIV_FIXTURES=${SCANN_EQUIV_FIXTURES}"
          "${CARGO_EXECUTABLE}" test ${_cargo_args} -- --nocapture
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT "cargo test (scann-core Rust bindings)"
  VERBATIM)
add_dependencies(scann_core_rust_test scann_core_static scann_core_deps_bundle)
