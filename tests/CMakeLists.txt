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

# C++ tests. api_exercise needs the fixtures written by
# tests/equivalence/run.py (reference results from the upstream wheel);
# point SCANN_TEST_FIXTURES at them to register it with ctest.
add_executable(scann_core_api_exercise cpp/api_exercise.cc)
target_link_libraries(scann_core_api_exercise PRIVATE scann::core)

set(SCANN_TEST_FIXTURES "" CACHE PATH
  "Fixture directory for scann_core_api_exercise (see tests/equivalence/run.py)")
if(SCANN_TEST_FIXTURES)
  add_test(NAME api_exercise COMMAND scann_core_api_exercise "${SCANN_TEST_FIXTURES}" 1)
  add_test(NAME api_exercise_threaded COMMAND scann_core_api_exercise "${SCANN_TEST_FIXTURES}" 0)
endif()

# ConfigBuilder vs. upstream's Python ScannBuilder: needs the expected
# configs written by tests/config_builder/make_expected.py.
add_executable(scann_core_config_builder_test cpp/config_builder_test.cc)
target_link_libraries(scann_core_config_builder_test PRIVATE scann::core)
set(SCANN_CONFIG_BUILDER_EXPECTED "" CACHE PATH
  "Directory written by tests/config_builder/make_expected.py")
if(SCANN_CONFIG_BUILDER_EXPECTED)
  add_test(NAME config_builder COMMAND scann_core_config_builder_test "${SCANN_CONFIG_BUILDER_EXPECTED}")
endif()
