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

# Builds the Python package for each supported CPython, including the
# free-threaded builds, and runs the Python tests (and the Python examples,
# example_py_*) with each. The C++ library is compiled once: for each
# interpreter the same build tree is reconfigured, and only the pybind
# module and the generated protobuf modules are rebuilt.
#
# Interpreters come from uv (https://docs.astral.sh/uv/), which downloads
# them when needed:
#
#   scripts/python-versions.sh
#   PYTHON_VERSIONS="3.12 3.14t" scripts/python-versions.sh
#
# Environment: CC/CXX (default: clang/clang++), BUILD_DIR (default:
# build-pyversions), PYTHON_VERSIONS (default: every supported version),
# TF_PYTHON_VERSIONS (default: 3.12): the versions that also get
# tensorflow-cpu, so that python_tf runs (and must not skip) there; it is
# skipped on the others. Empty to install TensorFlow nowhere.
set -euo pipefail
cd "$(dirname "$0")/.."

export CC="${CC:-clang}" CXX="${CXX:-clang++}"
BUILD_DIR="${BUILD_DIR:-build-pyversions}"
PYTHON_VERSIONS="${PYTHON_VERSIONS:-3.10 3.11 3.12 3.13 3.14 3.14t 3.15 3.15t}"
TF_PYTHON_VERSIONS="${TF_PYTHON_VERSIONS-3.12}"
command -v uv >/dev/null || { echo "scripts/python-versions.sh needs uv" >&2; exit 1; }
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

passed=() failed=()
for v in $PYTHON_VERSIONS; do
  echo "::group::Python $v"
  uv venv -q --python "$v" "$WORK/$v"
  VIRTUAL_ENV="$WORK/$v" uv pip install -q numpy "protobuf>=7.36.2"
  require_tf=
  if [[ " $TF_PYTHON_VERSIONS " == *" $v "* ]]; then
    VIRTUAL_ENV="$WORK/$v" uv pip install -q "protobuf>=7.36.2" "tensorflow-cpu>=2.21"
    require_tf=1
  fi
  py="$WORK/$v/bin/python"
  "$py" -c 'import sys, sysconfig; print(sys.version, "(free-threaded)" if sysconfig.get_config_var("Py_GIL_DISABLED") else "")'
  # -U drops the previous interpreter's cached FindPython results.
  cmake -S . -B "$BUILD_DIR" -G Ninja -U 'Python_*' -U '_Python_*' \
    -DPython_EXECUTABLE="$py" -DSCANN_BUILD_RUST_BINDINGS=OFF \
    -DSCANN_BUILD_SHARED=OFF -DSCANN_BUILD_EXAMPLES=OFF >/dev/null
  cmake --build "$BUILD_DIR"
  if SCANN_TEST_REQUIRE_TF=$require_tf \
     ctest --test-dir "$BUILD_DIR" -R '^(python_|config_builder|example_py_)' --output-on-failure; then
    passed+=("$v")
  else
    failed+=("$v")
  fi
  echo "::endgroup::"
done

echo "passed: ${passed[*]:-none}"
if ((${#failed[@]})); then
  echo "FAILED: ${failed[*]}"
  exit 1
fi
