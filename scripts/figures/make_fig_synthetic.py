#!/usr/bin/env python3
"""The Bayes-gap prediction on a dense synthetic sweep.

Raw error is held fixed along the sweep and only its source moves, so the two
panels answer one question each: does correction gain follow the gap, and does
the fitted blend weight find the gap without being told it is there.
Data: ablation_data/SYNTH/sweep.csv, from experiments/exp_synthetic_sweep.py.
"""
import csv, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle

# same path convention as the other figure scripts in this directory: results
# ships with the repo, artifacts is what download_artifacts.sh fills in
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
A = Path(os.environ.get("GUARD_ARTIFACTS", ROOT / "artifacts"))
SRC = next((p for p in (A / "synthetic_sweep/sweep.csv",
                        ROOT / "results/synthetic_sweep/sweep.csv") if p.exists()),
           ROOT / "results/synthetic_sweep/sweep.csv")
OUT = Path(os.environ.get("GUARD_FIGOUT", HERE / "figures")) / "fig_synthetic.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(open(SRC)))
seeds = sorted({int(r["seed"]) for r in rows})
npt = len(rows) // len(seeds)

# rows are seed-major, so grid point i is every npt-th row
def at(i, key):
    return np.array([float(r[key]) for j, r in enumerate(rows) if j % npt == i])

ph   = np.array([at(i, "ph_true").mean() for i in range(npt)])
acc  = np.array([at(i, "base_acc").mean() for i in range(npt)])
acc_s= np.array([at(i, "base_acc").std(ddof=1) / np.sqrt(len(seeds)) for i in range(npt)])
gain = np.array([at(i, "d_acc").mean() for i in range(npt)])
gain_s=np.array([at(i, "d_acc").std(ddof=1) / np.sqrt(len(seeds)) for i in range(npt)])
beta = np.array([at(i, "beta").mean() for i in range(npt)])
beta_s=np.array([at(i, "beta").std(ddof=1) / np.sqrt(len(seeds)) for i in range(npt)])

figstyle.apply()
fig, ax = plt.subplots(1, 2, figsize=(figstyle.FULL_IN, 1.70))

a = ax[0]
a.axhline(0.0, color=figstyle.GREY, lw=0.6, zorder=1)
a.errorbar(ph, gain, yerr=gain_s, color=figstyle.OURS, marker="o",
           ms=figstyle.MS_OURS, lw=figstyle.LW_OURS, capsize=1.4, zorder=3,
           label="accuracy gain")
a.errorbar(ph, acc - acc.mean(), yerr=acc_s, color=figstyle.GREY, marker="s",
           ms=figstyle.MS, lw=figstyle.LW, capsize=1.4, ls="--", zorder=2,
           label="base accuracy, centred")
a.set_xlabel("potential headroom $\\mathrm{PH}_{\\mathrm{true}}$")
a.set_ylabel("change in accuracy")
# the legend sits where the curve is lowest, but the panel is short, so lift the
# top of the axis until the two rows clear the rising curve
a.set_ylim(top=a.get_ylim()[1] + 0.22 * (a.get_ylim()[1] - a.get_ylim()[0]))
a.legend(frameon=False, fontsize=figstyle.LEG, loc="upper left", handletextpad=0.5,
         labelspacing=0.3, borderaxespad=0.2)

b = ax[1]
b.errorbar(ph, beta, yerr=beta_s, color=figstyle.OURS, marker="o",
           ms=figstyle.MS_OURS, lw=figstyle.LW_OURS, capsize=1.4)
b.set_xlabel("potential headroom $\\mathrm{PH}_{\\mathrm{true}}$")
b.set_ylabel("fitted $\\widehat\\beta$")
b.set_ylim(-0.03, 0.85)

for a_ in ax:
    a_.spines["top"].set_visible(False)
    a_.spines["right"].set_visible(False)

fig.tight_layout(pad=0.4)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(str(OUT).replace(".pdf", "_preview.png"), bbox_inches="tight", dpi=300)
print("wrote", OUT)
print(f"base acc {acc.mean():.4f} +- {acc.std():.4f} across the sweep")
print(f"PH {ph.max():.4f} -> {ph.min():.4f}; gain {gain.max():+.4f} -> {gain.min():+.4f}; "
      f"beta {beta.max():.2f} -> {beta.min():.2f}")
r = np.corrcoef(ph, gain)[0, 1]
print(f"corr(PH, gain) = {r:.4f}")
