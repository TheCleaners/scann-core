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

foreach(example quickstart updating)
  add_executable(scann_core_example_${example} cpp/${example}.cc)
  target_link_libraries(scann_core_example_${example} PRIVATE scann::core)
  set_target_properties(scann_core_example_${example} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  # The examples check their own results, so they double as tests. (The
  # Python examples are registered in tests/CMakeLists.txt, the Rust ones
  # in rust/CMakeLists.txt.)
  if(SCANN_BUILD_TESTS)
    add_test(NAME example_cpp_${example} COMMAND scann_core_example_${example})
    set_tests_properties(example_cpp_${example} PROPERTIES TIMEOUT 300)
  endif()
endforeach()

# The tutorial's C++ program (docs/tutorial/07-cpp-and-rust.md).
add_executable(scann_core_tutorial_glove ../docs/tutorial/code/cpp/glove.cc)
target_link_libraries(scann_core_tutorial_glove PRIVATE scann::core)
set_target_properties(scann_core_tutorial_glove PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
