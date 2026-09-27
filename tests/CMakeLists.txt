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
  # A deadlock between searches and updates would show up here as a hang.
  scann_add_python_test(python_concurrency python/test_concurrency.py)
endif()
