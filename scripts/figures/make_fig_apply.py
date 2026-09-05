#!/usr/bin/env python3
"""Intervention rate against the budget, for the appendix.

A looser budget does not always buy more intervention: on benchmarks with many
classes the conformal threshold tightens until no label is plausible, the
implementation falls back to the full label set, and the gate refuses.
"""
import csv, collections, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# figures land beside the script unless the caller says otherwise
OUTDIR = os.environ.get("GUARD_FIGOUT", os.path.join(HERE, "figures"))
os.makedirs(OUTDIR, exist_ok=True)
NEW = os.path.join(HERE, "ablation_data/ALPHA_NEW")
OUT = os.path.join(OUTDIR, "fig_apply.pdf")

rows = []
for f in ("affective", "drugban", "opportunity", "ptbxl", "ninapro", "ave"):
    rows += list(csv.DictReader(open(os.path.join(NEW, f + ".csv"))))
ex = [r for r in rows if str(r["exchangeable"]) in ("True", "true")]

FAM = [("affective", "CMU-MOSEI / IEMOCAP", "#4c6faa", "^"),
       ("drugban", "DrugBAN", "#8a6d3b", "o"),
       ("opportunity", "OPPORTUNITY", "#e08b00", "s"),
       ("ptbxl","PTB-XL","#b5179e","D"),
       ("ninapro", "NinaPro DB5", "#8b5fbf", "v"),
       ("ave", "AVE", "#2a9d8f", "P")]

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, ax = plt.subplots(figsize=(3.1, 1.9))
for fam, name, col, mk in FAM:
    d = collections.defaultdict(list)
    for r in ex:
        if r["family"] == fam:
            d[float(r["alpha"])].append(float(r["apply_rate"]))
    a = sorted(d)
    ax.plot(a, [np.mean(d[x]) for x in a], color=col, lw=1.0, marker=mk, ms=2.4, label=name)
ax.set_xlabel(r"budget $\alpha$", fontsize=7)
ax.set_ylabel("intervention rate", fontsize=7)
ax.tick_params(labelsize=7)
ax.margins(x=0.06, y=0.08)
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
           fontsize=6.6, bbox_to_anchor=(0.5, -0.30), handletextpad=0.5, columnspacing=1.2)
fig.tight_layout(pad=0.4, rect=(0, 0.16, 1, 1))
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
print("ok ->", OUT)
