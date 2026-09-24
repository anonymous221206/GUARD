#!/usr/bin/env python3
"""LaTeX rows for the harm-tolerance table, from results/delta/*.csv (exp_delta.py)."""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/delta"
data = {b: list(csv.DictReader(open(RES / f"{b}.csv"))) for b in ("mosei", "drugban")}
deltas = sorted({float(r["delta"]) for r in data["mosei"]})
m = lambda rs, k: float(np.mean([float(r[k]) for r in rs]))
for d in deltas:
    cells = []
    for b in ("mosei", "drugban"):
        rs = [r for r in data[b] if float(r["delta"]) == d]
        cells += [f"${m(rs, 'gain'):+.3f}$", f"${m(rs, 'joint_harm'):.3f}$",
                  f"${m(rs, 'flip_harm'):.3f}$", f"${m(rs, 'flip_help'):.3f}$"]
    print(f"${d:g}$ & " + " & ".join(cells) + r" \\")
for b in ("mosei", "drugban"):
    over = {d: sum(float(r["joint_harm"]) > 0.2 for r in data[b] if float(r["delta"]) == d)
            for d in deltas}
    n = sum(1 for r in data[b] if float(r["delta"]) == deltas[0])
    print(f"% {b}: {n} cells per delta; above alpha: {over}")
