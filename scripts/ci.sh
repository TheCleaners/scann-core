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

# What CI runs (.github/workflows/ci.yml), runnable locally too, e.g. in a
# container: docker run --rm -v $PWD:/src -w /src ubuntu:26.04 scripts/ci.sh
#
#   1. configure, build and run every test (C++, Python, Rust);
#   2. build examples/fetchcontent against this checkout;
#   3. `pip install .` into a fresh venv and use the package.
#
# Environment: CC/CXX (default: clang/clang++), BUILD_DIR (default: build-ci).
set -euo pipefail
cd "$(dirname "$0")/.."

export CC="${CC:-clang}" CXX="${CXX:-clang++}"
BUILD_DIR="${BUILD_DIR:-build-ci}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "::group::toolchain"
"$CXX" --version | head -1
cmake --version | head -1
python3 --version
cargo --version
echo "::endgroup::"

echo "::group::python venv"
python3 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install --quiet numpy "protobuf>=7.36.2"
echo "::endgroup::"

echo "::group::configure + build"
cmake -S . -B "$BUILD_DIR" -G Ninja -DPython_EXECUTABLE="$WORK/venv/bin/python"
cmake --build "$BUILD_DIR"
echo "::endgroup::"

echo "::group::ctest"
ctest --test-dir "$BUILD_DIR" --output-on-failure
echo "::endgroup::"

echo "::group::examples/fetchcontent against this checkout"
deps=()
for d in absl zlib protobuf highway eigen; do
  deps+=("-DFETCHCONTENT_SOURCE_DIR_${d^^}=$PWD/$BUILD_DIR/_deps/$d-src")
done
cmake -S examples/fetchcontent -B "$WORK/consumer" -G Ninja \
  -DFETCHCONTENT_SOURCE_DIR_SCANN_CORE="$PWD" "${deps[@]}"
cmake --build "$WORK/consumer"
"$WORK/consumer/consumer"
echo "::endgroup::"

echo "::group::pip install ."
python3 -m venv "$WORK/pip"
"$WORK/pip/bin/pip" install --quiet .
(cd "$WORK" && "$WORK/pip/bin/python" -c '
import numpy as np, scann
db = np.random.default_rng(0).standard_normal((2000, 16)).astype(np.float32)
s = scann.scann_ops_pybind.builder(db, 5, "squared_l2").tree(20, 5).score_ah(2).reorder(50).build()
assert s.search(db[3])[0][0] == 3
print("scann", scann.__version__, "installed and working")')
echo "::endgroup::"
