#!/usr/bin/env python3
"""Correlation between gain and true headroom on the Figure 3 sweep, with a seed bootstrap."""
import collections
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rows = list(csv.DictReader(open(ROOT / "results/synthetic_sweep/sweep.csv")))
by = collections.defaultdict(list)
for r in rows:
    by[r["seed"]].append(r)
seeds = sorted(by)
# each seed bisects its own grid, so points are matched by their position along it
arr = lambda k: np.array([[float(r[k]) for r in sorted(by[s], key=lambda r: float(r["tau"]))]
                          for s in seeds])
ph, gain = arr("ph_true"), arr("d_acc")
corr = lambda i: np.corrcoef(ph[i].mean(0), gain[i].mean(0))[0, 1]
rng = np.random.default_rng(0)
boot = [corr(rng.integers(0, len(seeds), len(seeds))) for _ in range(5000)]
print(f"r = {corr(np.arange(len(seeds))):.3f} over {ph.shape[1]} points, "
      f"95% seed bootstrap [{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]; "
      f"{ph.size} individual runs r = {np.corrcoef(ph.ravel(), gain.ravel())[0, 1]:.3f}")
