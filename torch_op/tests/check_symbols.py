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

"""Checks the PyTorch op library's dynamic symbols and dependencies.

Usage: check_symbols.py <_scann_torch_ops.so> <nm> <readelf> [torch lib dir]

- It exports no symbols (nm -D --defined-only is empty): nothing of
  scann-core, abseil or protobuf can interpose on PyTorch's copies (or on
  the scann pybind module's).
- It imports only LibTorch's stable C shim (aoti_torch_*, torch_*) and the
  C/C++ runtime: nothing of torch's, c10's or ATen's C++ API (which would
  tie it to one torch version), nothing of Python's C API, and none of
  abseil, protobuf, Eigen, highway or scann-core, which it must define
  itself. (libtorch_cpu.so exports protobuf symbols of its own: a library
  linked with it before scann-core's dependency archives would silently
  bind to them.)
- Every C/C++ runtime import is versioned (e.g. @GLIBCXX_3.4.26), i.e.
  bound to libstdc++/libc at link time: torch's libraries also export some
  libstdc++ symbols (std::filesystem, std::string members), unversioned,
  and a library linked with them before libstdc++ binds to those.
- It needs libtorch_cpu.so and no other torch library, no libpython, and
  has no RPATH/RUNPATH.
- Given torch's lib directory, it reports how many protobuf, abseil and
  std:: symbols torch's libraries export (the reason for the link order).
"""

import os
import re
import subprocess
import sys

so, nm, readelf = sys.argv[1:4]
torch_lib = sys.argv[4] if len(sys.argv) > 4 else None


def run(*args):
  return subprocess.run(args, check=True, capture_output=True,
                        text=True).stdout


defined = [l for l in run(nm, "-D", "--defined-only", so).split("\n")
           if l.strip()]
assert not defined, "exported symbols:\n" + "\n".join(defined[:50])

# (type, name[@version]) of every undefined symbol.
undefined = [
    tuple(l.split()[-2:])
    for l in run(nm, "-D", "--undefined-only", so).split("\n")
    if l.strip()
]
shim = [n for _, n in undefined
        if n.startswith("aoti_torch_") or n.startswith("torch_")]
assert shim, "no aoti_torch_* / torch_* imports: is this the op library?"
# Namespaces as they appear in mangled names (length-prefixed), and
# Python's C API.
forbidden = re.compile(r"(?<![0-9])(5torch|3c10|2at|4absl|6google|5Eigen|"
                       r"14research_scann|10scann_core|3hwy|8pybind11)|^_?Py")
# GCC references the TLS init function (_ZTH) of an extern thread_local
# (abseil's cordz_next_sample) weakly; the variable itself is defined
# locally (see tf_op/tests/check_symbols.py).
bad = [
    n for t, n in undefined
    if forbidden.search(n) and not (t in "wv" and n.startswith("_ZTH"))
]
assert not bad, "imports symbols it must not:\n" + "\n".join(bad[:50])
other = [n for _, n in undefined if n not in shim]
unversioned = [n for t, n in undefined
               if n not in shim and "@" not in n and t not in "wv"]
assert not unversioned, ("unversioned C/C++ runtime imports (bound to "
                         "torch's copies?):\n" + "\n".join(unversioned[:50]))

dynamic = run(readelf, "-d", so)
needed = re.findall(r"\(NEEDED\)\s+Shared library: \[(.*)\]", dynamic)
assert "libtorch_cpu.so" in needed, needed
unexpected = [n for n in needed
              if n != "libtorch_cpu.so" and not re.match(
                  r"lib(c|m|pthread|dl|rt|gcc_s|stdc\+\+|c\+\+|c\+\+abi|"
                  r"unwind|atomic)\.so", n) and not n.startswith("ld-linux")]
assert not unexpected, f"unexpected NEEDED libraries: {unexpected}"
assert "RPATH" not in dynamic and "RUNPATH" not in dynamic, dynamic

print(f"OK: 0 exported symbols; imports {len(shim)} stable C shim functions "
      f"and {len(other)} other symbols (C/C++ runtime, all versioned); "
      f"NEEDED {', '.join(needed)}; no rpath, no libpython")

if torch_lib:
  # Why the link order matters: torch's own libraries export symbols of
  # libraries scann-core links statically, and of libstdc++.
  for lib in ("libtorch_cpu.so", "libc10.so"):
    path = os.path.join(torch_lib, lib)
    if not os.path.exists(path):
      continue
    syms = run(nm, "-D", "--defined-only", path).split("\n")
    counts = {
        ns: sum(1 for l in syms if re.search(pat, l))
        for ns, pat in (("protobuf", r"6google8protobuf"),
                        ("abseil", r"4absl"),
                        ("std", r" _ZN?K?St(?!4hash)|_ZSt"))
    }
    print(f"  {lib} exports {counts['protobuf']} protobuf, "
          f"{counts['abseil']} abseil and {counts['std']} std:: symbols; the "
          "op imports none of them from it")
