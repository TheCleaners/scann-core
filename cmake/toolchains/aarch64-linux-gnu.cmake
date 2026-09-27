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

# Cross-compiling for 64-bit Arm Linux with clang, on a Debian/Ubuntu host
# with the aarch64 cross libraries installed (package g++-aarch64-linux-gnu,
# which clang finds on its own). If qemu-aarch64 is available, it becomes
# the emulator: CMake runs build-time tools (protoc) and ctest runs the tests
# through it, so the whole build and test suite work from an x86 machine.
#
#   cmake -S . -B build-aarch64 -G Ninja \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-linux-gnu.cmake
#
# Set SCANN_CLANG_SUFFIX (e.g. -19) to pick a versioned clang.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(_triple aarch64-linux-gnu)
set(_suffix "$ENV{SCANN_CLANG_SUFFIX}")
set(CMAKE_C_COMPILER "clang${_suffix}")
set(CMAKE_CXX_COMPILER "clang++${_suffix}")
set(CMAKE_C_COMPILER_TARGET ${_triple})
set(CMAKE_CXX_COMPILER_TARGET ${_triple})
set(CMAKE_ASM_COMPILER_TARGET ${_triple})
# lld links for any target without a separate cross binutils.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-fuse-ld=lld")

set(CMAKE_FIND_ROOT_PATH /usr/${_triple})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

find_program(_scann_qemu qemu-aarch64 qemu-aarch64-static)
if(_scann_qemu)
  set(CMAKE_CROSSCOMPILING_EMULATOR "${_scann_qemu};-L;/usr/${_triple}")
endif()
