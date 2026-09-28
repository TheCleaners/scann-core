#!/usr/bin/env bash
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

# Cross-compiles scann-core for aarch64 and runs its C++ tests under QEMU
# user-mode emulation, on several emulated CPUs (Neon only, Neon + dot
# product, SVE, SVE2, and SVE at 512 and 2048 bits). Each CPU makes the
# runtime feature detection select different kernels. Checks correctness
# only: emulated timings mean nothing.
#
# Run it in a throwaway x86-64 container (it installs packages):
#   docker run --rm --platform linux/amd64 -v $PWD:/src -w /src ubuntu:26.04 \
#     scripts/cross-aarch64.sh
#
# Environment: CLANG_VERSION (default 19), BUILD_DIR (default build-aarch64).
# With CMAKE_C_COMPILER_LAUNCHER/CMAKE_CXX_COMPILER_LAUNCHER=ccache (and
# CCACHE_DIR pointing at a mounted directory), the build goes through ccache,
# which is installed along with the toolchain; CI does this.
set -euo pipefail
cd "$(dirname "$0")/.."
CLANG_VERSION="${CLANG_VERSION:-19}"
BUILD_DIR="${BUILD_DIR:-build-aarch64}"

if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  echo 'Acquire::Retries "10"; Acquire::http::Timeout "60";' > /etc/apt/apt.conf.d/80retries
  apt-get update -qq >/dev/null
  apt-get install -y -qq "clang-$CLANG_VERSION" "lld-$CLANG_VERSION" \
    g++-aarch64-linux-gnu qemu-user cmake ninja-build git ca-certificates file \
    ccache >/dev/null
fi

echo "::group::configure + build (aarch64, clang-$CLANG_VERSION)"
SCANN_CLANG_SUFFIX="-$CLANG_VERSION" cmake -S . -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-linux-gnu.cmake \
  -DSCANN_BUILD_PYTHON=OFF -DSCANN_BUILD_RUST_BINDINGS=OFF \
  -DSCANN_BUILD_SHARED=OFF ${SCANN_CMAKE_ARGS:-}
cmake --build "$BUILD_DIR"
file "$BUILD_DIR/tests/scann_core_api_exercise" | cut -d, -f1-2
if [ "${CMAKE_CXX_COMPILER_LAUNCHER:-}" = ccache ]; then ccache --show-stats; fi
echo "::endgroup::"

# QEMU_CPU selects the emulated CPU, and with it the kernels ScaNN's runtime
# detection picks: cortex-a57 = Neon only; neoverse-n1 = + dot product;
# neoverse-v1 = + i8mm, SVE (256-bit); neoverse-n2 = + SVE2 (128-bit). Then
# SVE at other vector lengths (sve-default-vector-length is in bytes).
for cpu in cortex-a57 neoverse-n1 neoverse-v1 neoverse-n2 \
           "max,sve-default-vector-length=64" "max,sve-default-vector-length=256"; do
  echo "::group::ctest, QEMU_CPU=$cpu"
  QEMU_CPU="$cpu" ctest --test-dir "$BUILD_DIR" --output-on-failure -j "$(nproc)"
  echo "::endgroup::"
done
