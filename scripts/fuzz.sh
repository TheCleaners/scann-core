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

# A mutation-fuzzer campaign (tests/fuzz/mutfuzz.cc): every config, seeds
# SEEDS, batch sizes 1 and 4, plain and with injected leaf failures
# (INJECT=1, tree + AH configs), PAR runs at a time; then a summary. Best
# with a sanitizer build:
#   cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#     -DSCANN_SANITIZE=address,undefined -DSCANN_BUILD_PYTHON=OFF \
#     -DSCANN_BUILD_RUST_BINDINGS=OFF
#   cmake --build build-asan --target scann_core_mutfuzz
#   BIN=build-asan/tests/scann_core_mutfuzz scripts/fuzz.sh
#
# Environment: BIN (default build/tests/scann_core_mutfuzz), LOG (default
# fuzz-logs), SEEDS (default "1 2 3"), STEPS (default 300), PAR (default
# nproc/4), ONLY (a subset of configs), MODES (default "plain inject"),
# TMO (seconds per run, default 1500).
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="${BIN:-build/tests/scann_core_mutfuzz}"
LOG="${LOG:-fuzz-logs}"
SEEDS="${SEEDS:-1 2 3}"
STEPS="${STEPS:-300}"
PAR="${PAR:-$(( $(nproc) / 4 > 0 ? $(nproc) / 4 : 1 ))}"
MODES="${MODES:-plain inject}"
export TMO="${TMO:-1500}"
CONFIGS="bf_l2 bf_dot int8_dot int8_l2 bf16_dot ah_dot ah_noreorder_dot auto_dot
autoinc_dot tree_bf_l2 tree_bf_dot tree_int8_l2 tree_bf16_l2 tree_bf16_dot
tree_ah_l2 tree_ah_dot tree_ah_noreorder_dot tree_ah_rint8_dot tree_ah_rbf16_dot
tree_ah_lut256_dot tree_ah_dpb3_l2 tree_qc_ah_dot tree_pca_ah_dot
tree_trunc_ah_dot tree_upper_ah_dot tree_upper2_ah_dot tree_sph_ah_dot
tree_sph_bf_l2 tree_sph_bf16_dot tree_incr_ah_dot tree_incr_bf_l2
tree_soar_ah_dot tree_soar_ah_rint8_dot tree_soar_bf_dot tree_soar_bf16_dot
tree_soar_int8_dot tree_incr_soar_ah_dot"
[ -x "$BIN" ] || { echo "no fuzzer binary at $BIN (see the header)" >&2; exit 1; }
mkdir -p "$LOG"
export BIN LOG ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0}" \
  UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1}"

run_one() {  # config seed batch mode
  local log="$LOG/$1_$2_$3${4/plain/}.log"
  [ "$4" = inject ] && log="$LOG/$1_$2_$3_inj.log"
  if [ "$4" = inject ]; then
    INJECT=1 timeout "$TMO" "$BIN" "$1" "$2" "$STEPS" "$3" > "$log" 2>&1 && rc=0 || rc=$?
  else
    timeout "$TMO" "$BIN" "$1" "$2" "$STEPS" "$3" > "$log" 2>&1 && rc=0 || rc=$?
  fi
  echo "rc=$rc" >> "$log"
}
export -f run_one
export STEPS

start=$(date +%s)
for mode in $MODES; do
  for c in ${ONLY:-$CONFIGS}; do
    [ "$mode" = inject ] && [[ "$c" != tree_*ah* ]] && continue
    for seed in $SEEDS; do
      for b in 1 4; do echo "$c $seed $b $mode"; done
    done
  done
done | xargs -P "$PAR" -L 1 bash -c 'run_one "$0" "$1" "$2" "$3"'
echo "elapsed $(( $(date +%s) - start ))s"
python3 tests/fuzz/summarize.py "$LOG"
