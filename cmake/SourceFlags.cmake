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

# Per-target `copts` from the upstream Bazel BUILD files, carried over as
# per-source COMPILE_OPTIONS. Source-level options come after global and
# target-level ones on the command line, preserving Bazel's ordering (its
# target copts also follow --copt), so e.g. -mtune=generic below overrides
# a global -mtune=native exactly as it does under Bazel.

# scann_apply_bazel_source_flags(<src dir> <generated lut16 shards...>)
function(scann_apply_bazel_source_flags src_dir)
  set(lut16_generated ${ARGN})
  set(clang_only "")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(clang_only -Wno-pass-failed)
  endif()

  # hashes/internal:asymmetric_hashing_impl_omit_frame_pointer
  set_source_files_properties(
    "${src_dir}/scann/hashes/internal/asymmetric_hashing_impl_omit_frame_pointer.cc"
    PROPERTIES COMPILE_OPTIONS "-fomit-frame-pointer")

  # hashes/internal:lut16_{sse4,avx2,avx512,highway} -- "built at -O3 even
  # for fastbuild and dbg builds" (large inlined frames otherwise overflow
  # the stack in unoptimized builds).
  set_source_files_properties(
    ${lut16_generated}
    "${src_dir}/scann/hashes/internal/lut16_avx512_swizzle.cc"
    PROPERTIES COMPILE_OPTIONS "-O3")

  # distance_measures/many_to_many:many_to_many_floating_point
  set(m2m_fp_opts -O3 ${clang_only})
  set_source_files_properties(
    "${src_dir}/scann/distance_measures/many_to_many/many_to_many_double.cc"
    "${src_dir}/scann/distance_measures/many_to_many/many_to_many_float.cc"
    PROPERTIES COMPILE_OPTIONS "${m2m_fp_opts}")

  # distance_measures/many_to_many:many_to_many -- -mtune=generic is an
  # upstream workaround ("TODO: b/455596112 - Remove once AMX bug is fixed").
  set(m2m_opts -O3 ${clang_only} -mtune=generic)
  set_source_files_properties(
    "${src_dir}/scann/distance_measures/many_to_many/many_to_many_fixed8.cc"
    "${src_dir}/scann/distance_measures/many_to_many/many_to_many_orthogonality_amplification.cc"
    "${src_dir}/scann/distance_measures/many_to_many/many_to_many_sfp8.cc"
    PROPERTIES COMPILE_OPTIONS "${m2m_opts}")

  # distance_measures/one_to_one:limited_inner_product
  set_source_files_properties(
    "${src_dir}/scann/distance_measures/one_to_one/limited_inner_product.cc"
    PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize")
endfunction()

# scann_shard_lut16_templates(<out var> TEMPLATE_DIR <dir> OUT_DIR <dir>)
#
# CMake equivalent of hashes/internal/template_sharding.bzl: each
# bazel_templates/<name>.tpl.cc becomes <rule>_1.cc .. <rule>_9.cc with
# {BATCH_SIZE} substituted, exactly as the six batch_size_sharder rules in
# hashes/internal/BUILD.bazel do (all use max_batch_size = 9).
function(scann_shard_lut16_templates out_var)
  cmake_parse_arguments(ARG "" "TEMPLATE_DIR;OUT_DIR" "" ${ARGN})
  # template stem : Bazel rule name (determines the generated file names)
  set(shards
    "lut16_sse4:lut16_sse4_batches"
    "lut16_avx2:lut16_avx2_batches"
    "lut16_avx512_noprefetch:lut16_avx512_noprefetch"
    "lut16_avx512_prefetch:lut16_avx512_prefetch"
    "lut16_avx512_smart:lut16_avx512_smart"
    "lut16_highway:lut16_highway_batches")
  set(max_batch_size 9)
  file(MAKE_DIRECTORY "${ARG_OUT_DIR}")

  set(generated "")
  foreach(shard ${shards})
    string(REPLACE ":" ";" parts "${shard}")
    list(GET parts 0 stem)
    list(GET parts 1 rule)
    set(template "${ARG_TEMPLATE_DIR}/${stem}.tpl.cc")
    if(NOT EXISTS "${template}")
      message(FATAL_ERROR "LUT16 template not found: ${template}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${template}")
    file(READ "${template}" content)
    foreach(batch RANGE 1 ${max_batch_size})
      string(REPLACE "{BATCH_SIZE}" "${batch}" expanded "${content}")
      set(out "${ARG_OUT_DIR}/${rule}_${batch}.cc")
      # Only rewrite on change so reconfiguring doesn't force recompiles.
      set(existing "")
      if(EXISTS "${out}")
        file(READ "${out}" existing)
      endif()
      if(NOT existing STREQUAL expanded)
        file(WRITE "${out}" "${expanded}")
      endif()
      list(APPEND generated "${out}")
    endforeach()
  endforeach()
  set(${out_var} ${generated} PARENT_SCOPE)
endfunction()
