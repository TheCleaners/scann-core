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

"""Checks the TensorFlow op library's dynamic symbols and dependencies.

Usage: check_symbols.py <_scann_tf_ops.so> <nm> <readelf>

- It exports no symbols (nm -D --defined-only is empty): nothing of
  scann-core, abseil or protobuf can interpose on TensorFlow's copies.
- It imports nothing of TensorFlow's C++ API or of TensorFlow's abseil,
  protobuf, Eigen or tsl: every undefined symbol is a TF_* C function or
  comes from the C/C++ runtime. (A library linked with the TensorFlow
  library before scann-core's dependency archives silently imports
  TensorFlow's protobuf and abseil instead.)
- It needs libtensorflow_framework.so.2 (not libtensorflow_cc) and has no
  RPATH/RUNPATH.
"""

import re
import subprocess
import sys

so, nm, readelf = sys.argv[1:4]


def run(*args):
  return subprocess.run(args, check=True, capture_output=True,
                        text=True).stdout


defined = run(nm, "-D", "--defined-only", so).split("\n")
defined = [l for l in defined if l.strip()]
assert not defined, "exported symbols:\n" + "\n".join(defined[:50])

# (type, mangled name) of every undefined symbol.
undefined = [
    tuple(l.split()[-2:])
    for l in run(nm, "-D", "--undefined-only", so).split("\n")
    if l.strip()
]
tf_c = [n for _, n in undefined if n.startswith("TF_")]
assert tf_c, "no TF_* imports: is this the op library?"
# Namespaces as they appear in mangled names (length-prefixed).
forbidden = re.compile(r"(?<![0-9])(10tensorflow|3tsl|3xla|4absl|6google|"
                       r"5Eigen|14research_scann|10scann_core)")
# GCC references the TLS init function (_ZTH) of an extern thread_local
# (abseil's cordz_next_sample) weakly; the variable itself is defined
# locally, and without a dynamic initializer the reference stays null.
# (The name carries abseil's inline namespace, so it can't bind to
# TensorFlow's abseil either.)
bad = [
    n for t, n in undefined
    if forbidden.search(n) and not (t in "wv" and n.startswith("_ZTH"))
]
assert not bad, "imports C++ symbols it must define itself:\n" + "\n".join(
    bad[:50])

dynamic = run(readelf, "-d", so)
needed = re.findall(r"\(NEEDED\)\s+Shared library: \[(.*)\]", dynamic)
assert "libtensorflow_framework.so.2" in needed, needed
assert not [n for n in needed if "tensorflow_cc" in n], needed
assert "RPATH" not in dynamic and "RUNPATH" not in dynamic, dynamic
print(f"OK: 0 exported symbols; imports {len(tf_c)} TF_* C functions and "
      f"{len(undefined) - len(tf_c)} other symbols (C/C++ runtime); NEEDED "
      f"{', '.join(needed)}; no rpath")
