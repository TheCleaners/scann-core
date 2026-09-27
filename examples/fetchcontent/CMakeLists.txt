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

# A separate CMake project using scann-core, pulled in with FetchContent.
# Copy this as a starting point. To build it against a local checkout
# instead of GitHub (as scann-core's CI does):
#   cmake -S examples/fetchcontent -B build-consumer \
#         -DFETCHCONTENT_SOURCE_DIR_SCANN_CORE=$PWD
cmake_minimum_required(VERSION 3.27)
project(scann_core_consumer LANGUAGES C CXX)

include(FetchContent)
FetchContent_Declare(scann_core
  GIT_REPOSITORY https://github.com/TheCleaners/scann-core.git
  GIT_TAG v0.2.0-rc.1)
# Pulled in as a subproject, scann-core builds only its libraries (no
# Python/Rust bindings, tests or examples) unless asked to.
FetchContent_MakeAvailable(scann_core)

add_executable(consumer main.cc)
target_link_libraries(consumer PRIVATE scann::core)
