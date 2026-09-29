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

# C++ tests. Everything registered with ctest here is self-contained;
# the equivalence runs against the upstream wheel (tests/equivalence/run.py)
# are opt-in via SCANN_TEST_FIXTURES.

# End-to-end API exercise: builds indexes for a set of configs, checks the
# search modes agree, serializes and reloads, mutates, retrains, and feeds
# it bad input. With "-" it generates its own synthetic data and configs.
add_executable(scann_core_api_exercise cpp/api_exercise.cc)
target_link_libraries(scann_core_api_exercise PRIVATE scann::core)
add_test(NAME api_exercise COMMAND scann_core_api_exercise - 1)
add_test(NAME api_exercise_threaded COMMAND scann_core_api_exercise - 0)
# Timeouts well above the normal run time (seconds natively; under QEMU and
# sanitizers, a few minutes), so a hang fails instead of stalling ctest for
# its default 25 minutes.
set_tests_properties(api_exercise api_exercise_threaded PROPERTIES TIMEOUT 900)
# The same with the AVX2 kernels on an AVX-512 machine (see the test's main()).
# x86-64 only: elsewhere it would only repeat api_exercise.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(SCANN_TEST_X86 ON)
  add_test(NAME api_exercise_avx2 COMMAND scann_core_api_exercise - 0)
  set_tests_properties(api_exercise_avx2 PROPERTIES TIMEOUT 900
    ENVIRONMENT SCANN_TEST_FORCE_AVX2=1)
endif()

# Regressions for mutation defects: a failed RetrainAndReindex must leave the
# index usable (upstream left a dangling docid collection), a tree with
# bfloat16 brute-force leaves must support add / update / delete, and a
# failed update in a SOAR tree must change nothing.
add_executable(scann_core_mutation_regressions cpp/mutation_regressions.cc)
target_link_libraries(scann_core_mutation_regressions PRIVATE scann::core)
add_test(NAME mutation_regressions COMMAND scann_core_mutation_regressions)
set_tests_properties(mutation_regressions PROPERTIES TIMEOUT 900)

# Mutation fuzzer (fuzz/mutfuzz.cc): random adds, updates, deletes, retrains
# and save/reload on small indexes, checked against a shadow copy with
# exhaustive searches. Every config for 600 steps here (well under a second
# each, a few seconds under sanitizers), two tree + AH configs also with
# injected leaf failures; scripts/fuzz.sh runs longer campaigns.
add_executable(scann_core_mutfuzz fuzz/mutfuzz.cc)
target_link_libraries(scann_core_mutfuzz PRIVATE scann::core)
set(SCANN_MUTFUZZ_CONFIGS
  bf_l2 bf_dot int8_dot int8_l2 bf16_dot ah_dot ah_noreorder_dot auto_dot
  autoinc_dot tree_bf_l2 tree_bf_dot tree_int8_l2 tree_bf16_l2 tree_bf16_dot
  tree_ah_l2 tree_ah_dot tree_ah_noreorder_dot tree_ah_rint8_dot
  tree_ah_rbf16_dot tree_ah_lut256_dot tree_ah_dpb3_l2 tree_qc_ah_dot
  tree_pca_ah_dot tree_trunc_ah_dot tree_upper_ah_dot tree_upper2_ah_dot
  tree_sph_ah_dot tree_sph_bf_l2 tree_sph_bf16_dot tree_incr_ah_dot
  tree_incr_bf_l2 tree_soar_ah_dot tree_soar_ah_rint8_dot tree_soar_bf_dot
  tree_soar_bf16_dot tree_soar_int8_dot tree_incr_soar_ah_dot
  l2mips_bf tree_l2mips_ah tree_l2mips_ah_dpb3 tree_soar_l2mips_ah
  tree_l2mips_ah_rbf16 tree_incr_l2mips_ah)
foreach(cfg IN LISTS SCANN_MUTFUZZ_CONFIGS)
  add_test(NAME mutfuzz_${cfg} COMMAND scann_core_mutfuzz ${cfg} 1 600 4)
  set_tests_properties(mutfuzz_${cfg} PROPERTIES TIMEOUT 900 LABELS mutfuzz)
endforeach()
foreach(cfg tree_ah_dot tree_soar_ah_dot tree_l2mips_ah)
  add_test(NAME mutfuzz_${cfg}_inject COMMAND scann_core_mutfuzz ${cfg} 2 600 4)
  set_tests_properties(mutfuzz_${cfg}_inject PROPERTIES TIMEOUT 900
    LABELS mutfuzz ENVIRONMENT INJECT=1)
endforeach()

# Damaged or inconsistent index directories (generated at run time) fail to
# load with an error instead of crashing or loading garbage; all-deleted
# and bfloat16-leaf trees round-trip; SerializeToDirectory replaces an index
# in one commit. Every directory is also loaded from memory
# (LoadArtifactsFromMemory, used by the TensorFlow op), with the same result.
add_executable(scann_core_artifact_loading cpp/artifact_loading.cc)
target_link_libraries(scann_core_artifact_loading PRIVATE scann::core)
add_test(NAME artifact_loading COMMAND scann_core_artifact_loading)
set_tests_properties(artifact_loading PROPERTIES TIMEOUT 900)
if(SCANN_TEST_X86)
  add_test(NAME artifact_loading_avx2 COMMAND scann_core_artifact_loading)
  set_tests_properties(artifact_loading_avx2 PROPERTIES TIMEOUT 900
    ENVIRONMENT SCANN_TEST_FORCE_AVX2=1)
endif()

# Regressions for raw config values that aborted the process, overflowed the
# heap or hit undefined behavior (each must be an error now), and for valid
# configs upstream failed to build (tree + PCA/TRUNCATE + non-residual AH).
add_executable(scann_core_config_regressions cpp/config_regressions.cc)
target_link_libraries(scann_core_config_regressions PRIVATE scann::core)
add_test(NAME config_regressions COMMAND scann_core_config_regressions)
set_tests_properties(config_regressions PROPERTIES TIMEOUT 900)

# The exact L2 -> inner-product reduction (l2_as_dot_product): results equal
# to a manually augmented dot-product index's, exact squared L2 distances,
# persistence (directory and in memory), mutations, retraining, config
# errors, and a clean load failure where the reduction is unknown.
add_executable(scann_core_l2_as_dot_product_test cpp/l2_as_dot_product_test.cc)
target_link_libraries(scann_core_l2_as_dot_product_test PRIVATE scann::core)
add_test(NAME l2_as_dot_product COMMAND scann_core_l2_as_dot_product_test)
set_tests_properties(l2_as_dot_product PROPERTIES TIMEOUT 900)

# The default thread count (scann_core::AvailableCPUs): cgroup quota parsing,
# the SCANN_NUM_THREADS override, the affinity mask.
add_executable(scann_core_available_cpus_test cpp/available_cpus_test.cc)
target_link_libraries(scann_core_available_cpus_test PRIVATE scann::core)
add_test(NAME available_cpus COMMAND scann_core_available_cpus_test)
set_tests_properties(available_cpus PROPERTIES TIMEOUT 60)

# A Python test run with this build's package on PYTHONPATH.
function(scann_add_python_test name script)
  add_test(NAME ${name}
           COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${PROJECT_BINARY_DIR}/python"
                   "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/${script}" ${ARGN})
  set_tests_properties(${name} PROPERTIES TIMEOUT 300)
endfunction()

set(SCANN_TEST_FIXTURES "" CACHE PATH
  "Optional: fixtures from tests/equivalence/run.py, also run through scann_core_api_exercise")
if(SCANN_TEST_FIXTURES)
  add_test(NAME api_exercise_equivalence_fixtures
           COMMAND scann_core_api_exercise "${SCANN_TEST_FIXTURES}" 0)
endif()

# ConfigBuilder vs. upstream's Python ScannBuilder. The expected configs come
# from tests/config_builder/make_expected.py, run on this build's Python
# package (its builder is upstream's Python code, unchanged), or from
# SCANN_CONFIG_BUILDER_EXPECTED if set.
add_executable(scann_core_config_builder_test cpp/config_builder_test.cc)
target_link_libraries(scann_core_config_builder_test PRIVATE scann::core)
set(SCANN_CONFIG_BUILDER_EXPECTED "" CACHE PATH
  "Optional: a directory written by tests/config_builder/make_expected.py")
if(SCANN_CONFIG_BUILDER_EXPECTED)
  add_test(NAME config_builder
           COMMAND scann_core_config_builder_test "${SCANN_CONFIG_BUILDER_EXPECTED}")
elseif(SCANN_BUILD_PYTHON AND SCANN_PYTHON_RUNTIME_OK)
  set(_expected "${CMAKE_CURRENT_BINARY_DIR}/config_builder_expected")
  scann_add_python_test(config_builder_expected config_builder/make_expected.py
                        --out "${_expected}")
  set_tests_properties(config_builder_expected PROPERTIES
    FIXTURES_SETUP config_builder_expected)
  add_test(NAME config_builder COMMAND scann_core_config_builder_test "${_expected}")
  set_tests_properties(config_builder PROPERTIES
    FIXTURES_REQUIRED config_builder_expected)
else()
  message(STATUS "scann-core: the config_builder test needs the Python "
                 "package (with numpy) or SCANN_CONFIG_BUILDER_EXPECTED; skipped")
endif()

if(SCANN_BUILD_PYTHON AND SCANN_PYTHON_RUNTIME_OK)
  scann_add_python_test(python_docid_bookkeeping python/test_docid_bookkeeping.py)
  # NaN/infinity data, queries and upserts, and tree options on non-tree
  # indexes: errors, not crashes.
  scann_add_python_test(python_input_validation python/test_input_validation.py)
  # Repeated docids in one upsert, zero-query batches, too many rows,
  # spherical partitioning of non-normalized data, rebalance without float
  # data.
  scann_add_python_test(python_wrapper_edge_cases python/test_wrapper_edge_cases.py)
  # A deadlock between searches and updates would show up here as a hang.
  scann_add_python_test(python_concurrency python/test_concurrency.py)
  # Grow from empty and retrain into a SOAR tree (google-research#2712).
  scann_add_python_test(python_rebalance_flow python/test_rebalance_flow.py)
  # Mutating and rebalancing a tree with a PCA/TRUNCATE projection.
  scann_add_python_test(python_projection_mutation python/test_projection_mutation.py)
  # serialize()/load_searcher() round trips, including an all-deleted tree,
  # and re-serializing (also killed midway) over an existing index.
  scann_add_python_test(python_serialization python/test_serialization.py)
  # Raw config values that crashed the process; builder checks.
  scann_add_python_test(python_config_validation python/test_config_validation.py)
  # Thread pools sized from the CPU affinity (and SCANN_NUM_THREADS), started
  # lazily; set_num_threads(n) = n workers. Linux only (77 elsewhere).
  scann_add_python_test(python_threads python/test_threads.py)
  set_tests_properties(python_threads PROPERTIES SKIP_RETURN_CODE 77)
  # scann.tf against the pybind searcher, eagerly, in tf.function and
  # tf.data, with each of its backends (the op one only when scann_tf_ops
  # is built); backend selection; SavedModel with the op backend. Skipped
  # (exit code 77) when TensorFlow isn't installed.
  scann_add_python_test(python_tf python/test_tf.py)
  set_tests_properties(python_tf PROPERTIES SKIP_RETURN_CODE 77)
  scann_add_python_test(python_langchain python/test_langchain.py)
  set_tests_properties(python_langchain PROPERTIES SKIP_RETURN_CODE 77)
  scann_add_python_test(python_torch_input python/test_torch_input.py)
  set_tests_properties(python_torch_input PROPERTIES SKIP_RETURN_CODE 77)
  # scann.torch against the pybind searcher, eagerly and compiled
  # (torch.compile with Inductor), CPU and CUDA. Skipped without PyTorch.
  # Inductor's and Triton's caches go to the build tree, not /tmp (TMPDIR
  # too: Inductor keeps precompiled headers there whatever its cache dir).
  set(_torch_cache "${CMAKE_CURRENT_BINARY_DIR}/torch_cache")
  file(MAKE_DIRECTORY "${_torch_cache}/tmp")
  set(SCANN_TORCH_TEST_ENV
    "TORCHINDUCTOR_CACHE_DIR=${_torch_cache}/inductor"
    "TRITON_CACHE_DIR=${_torch_cache}/triton"
    "TMPDIR=${_torch_cache}/tmp")
  # With the native op built (SCANN_BUILD_TORCH_OP), its backend must load
  # whenever PyTorch does: the tests fail instead of skipping it.
  if(SCANN_BUILD_TORCH_OP)
    list(APPEND SCANN_TORCH_TEST_ENV "SCANN_TEST_REQUIRE_TORCH_NATIVE=1")
  endif()
  # Every check once per backend: Python, and native when scann_torch_ops
  # is importable.
  scann_add_python_test(python_torch python/test_torch.py)
  set_tests_properties(python_torch PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1200
                       ENVIRONMENT "${SCANN_TORCH_TEST_ENV}")
  # The native backend (torch_op/) in depth: parity for every index type,
  # compile, export and AOTInductor in fresh processes, state_dict and
  # pickling, stale state, errors, concurrency. Skipped (77) without
  # PyTorch or without the op.
  scann_add_python_test(python_torch_native python/test_torch_native.py)
  set_tests_properties(python_torch_native PROPERTIES SKIP_RETURN_CODE 77
                       TIMEOUT 1800 ENVIRONMENT "${SCANN_TORCH_TEST_ENV}")
  # builder(...).l2_as_dot_product(): the manual recipe's ids, squared L2
  # distances, persistence, mutations, rebalance, and scann.torch /
  # scann.tf on such an index when PyTorch / TensorFlow are installed.
  scann_add_python_test(python_l2_as_dot_product python/test_l2_as_dot_product.py)
  set_tests_properties(python_l2_as_dot_product PROPERTIES TIMEOUT 900
                       ENVIRONMENT "${SCANN_TORCH_TEST_ENV}")

  # The Python examples assert their results, so they run as tests too.
  # The TensorFlow ones exit with 77 (skipped) without TensorFlow.
  foreach(example quickstart updating serving_threads tensorflow_wrapper
                  tensorflow_serving)
    scann_add_python_test(example_py_${example}
                          ../examples/python/${example}.py)
  endforeach()
  set_tests_properties(example_py_tensorflow_wrapper
                       example_py_tensorflow_serving
                       PROPERTIES SKIP_RETURN_CODE 77)
  # The PyTorch example: 77 without PyTorch; compiles with Inductor.
  scann_add_python_test(example_py_torch_retrieval
                        ../examples/python/torch_retrieval.py)
  set_tests_properties(example_py_torch_retrieval PROPERTIES
                       SKIP_RETURN_CODE 77 TIMEOUT 900
                       ENVIRONMENT "${SCANN_TORCH_TEST_ENV}")
  # docs/frameworks.md's examples: a FastAPI service run in-process (77
  # without FastAPI and httpx2), and batch retrieval with multiprocessing
  # and, when Ray is installed, Ray Data (77 without pyarrow).
  foreach(example fastapi_service batch_retrieval)
    scann_add_python_test(example_py_${example}
                          ../examples/python/${example}.py)
  endforeach()
  set_tests_properties(example_py_fastapi_service example_py_batch_retrieval
                       PROPERTIES SKIP_RETURN_CODE 77)
endif()
