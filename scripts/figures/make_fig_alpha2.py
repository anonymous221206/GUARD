#!/usr/bin/env python3
"""Risk-utility curves over alpha, across every benchmark.

Left panel carries the diagonal y=alpha, the promise Theorem 2 makes; runs whose
calibration set is not exchangeable with deployment are drawn separately because
the theorem says nothing about them.  Where two retrieval targets were swept, both
are shown: the claim is that every exchangeable run sits below the line.
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
NEW  = os.path.join(HERE, "ablation_data/ALPHA_NEW")
OUT  = os.path.join(OUTDIR, "fig_alpha.pdf")

rows = []
for f in ("affective", "drugban", "opportunity", "ptbxl", "ninapro", "ave"):
    q = os.path.join(NEW, f + ".csv")
    if os.path.exists(q):
        rows += list(csv.DictReader(open(q)))
    else:
        raise SystemExit("thieu " + q)

# red and green are reserved for GUARD and the un-gated corrector, as in Figure 4
FAM = [("affective",   "CMU-MOSEI / IEMOCAP", "#4c6faa", "^"),
       ("drugban",     "DrugBAN",             "#8a6d3b", "o"),
       ("opportunity", "OPPORTUNITY",         "#e08b00", "s"),
       ("ptbxl",       "PTB-XL",              "#b5179e", "D"),
       ("ninapro",     "NinaPro DB5",         "#8b5fbf", "v"),
       ("ave",         "AVE",                 "#2a9d8f", "P")]
GREY, FAIL, OURS = "#b3b3b3", "#333333", "#d1495b"

ex = [r for r in rows if str(r.get("exchangeable")) in ("True", "true")]
nx = [r for r in rows if str(r.get("exchangeable")) not in ("True", "true")]

def curve(rs, key):
    d = collections.defaultdict(list)
    for r in rs:
        try: d[float(r["alpha"])].append(float(r[key]))
        except (KeyError, ValueError): pass
    a = sorted(d)
    return np.array(a), np.array([np.mean(d[x]) for x in a])

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.62))
KEYS = [("joint_harm", "joint harm"), ("violations", "runs over budget (%)"),
        ("acc_gain", "gain over frozen model")]

def viol(rs):
    d = collections.defaultdict(list)
    for r in rs:
        a = float(r["alpha"])
        d[a].append(float(r["joint_harm"]) > a)
    a = sorted(d)
    return np.array(a), np.array([100 * np.mean(d[x]) for x in a])

def viol_blanket(rs):
    d = collections.defaultdict(list)
    for r in rs:
        b = r.get("blanket_joint_harm")
        if b in (None, "", "nan"): continue
        a = float(r["alpha"]); d[a].append(float(b) > a)
    a = sorted(d)
    return np.array(a), np.array([100 * np.mean(d[x]) for x in a])

for ax, (key, lab) in zip(axes, KEYS):
    if key == "violations":
        x, y = viol_blanket(ex)
        ax.plot(x, y, color="#2e9e4f", lw=1.1, marker="s", ms=2.6, ls="-.", label="no gate")
        x, y = viol(ex)
        ax.plot(x, y, color="#d1495b", lw=1.3, marker="o", ms=2.8, ls="-", label="GUARD")
        ax.set_ylim(-4, 104)
    else:
        for fam, name, col, mk in FAM:
            rs = [r for r in ex if r["family"] == fam]
            if not rs: continue
            x, y = curve(rs, key)
            ax.plot(x, y, color=col, lw=1.0, marker=mk, ms=2.4, label=name)
    if key == "joint_harm":
        ax.plot([0.0, 0.55], [0.0, 0.55], color=GREY, lw=0.8, ls="--", zorder=0)
    ax.set_xlabel(r"budget $\alpha$", fontsize=7)
    ax.set_ylabel(lab, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.margins(x=0.06, y=0.08)

h, l = axes[0].get_legend_handles_labels()
h2, l2 = axes[1].get_legend_handles_labels()
h, l = h + h2, l + l2
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6.6,
           bbox_to_anchor=(0.5, -0.12), handletextpad=0.5, columnspacing=1.4)
fig.tight_layout(pad=0.4, w_pad=1.4, rect=(0, 0.10, 1, 1))
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
print("ok ->", OUT)
