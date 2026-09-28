# Copyright 2026 Elias Benali (@ebenali) and TheCleaners.
# SPDX-License-Identifier: Apache-2.0
"""Summarizes a scripts/fuzz.sh log directory: runs, injected failures,
sanitizer reports and failing runs. Usage: summarize.py <log dir>"""
import glob, os, re, collections
import sys
D = sys.argv[1]
tot = collections.Counter()
bad = []
san = []
for f in sorted(glob.glob(D + "/*.log")):
    t = open(f, errors="replace").read()
    name = os.path.basename(f)
    mode = "inj" if name.endswith("_inj.log") else "plain"
    tot[mode + "_runs"] += 1
    if "ERROR: AddressSanitizer" in t or "runtime error" in t or "LeakSanitizer" in t:
        san.append(name)
    m = re.search(r"^done .*injected=(\d+) failures=(\d+)", t, re.M)
    rc = re.search(r"^rc=(\d+)", t, re.M)
    if m:
        tot[mode + "_injected"] += int(m.group(1))
    if not m or m.group(2) != "0" or not rc or rc.group(1) != "0":
        fails = [l for l in t.splitlines() if l.startswith("FAIL")]
        kinds = collections.Counter(re.sub(r"\d+(\.\d+)?(e[-+]\d+)?", "#", l)[:90] for l in fails)
        bad.append((name, rc.group(1) if rc else "?", dict(kinds)))
print(dict(tot))
print("sanitizer reports:", san)
for b in bad:
    print(b)
