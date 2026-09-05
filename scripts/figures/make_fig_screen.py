#!/usr/bin/env python3
"""The cross-mask accuracy screen, from ablation_data/METRICS/crossmask_precondition.csv.

Two panels sharing the retained-gain axis. The screen asks whether the richer-mask
predictor is more accurate than the masked one on deployment data; it is a necessary
condition for Proposition 4, never a sufficient one. The left panel plots the outcome
against the richer predictor's absolute accuracy, the right against its margin over the
masked predictor, because only the first of the two separates the outcomes here.

Cells whose hard-label certified gain falls below HEADROOM are dropped: with no gain to
retain, the ratio gain_cross/gain_hard is noise. The conclusion is unchanged for any cut
between 0.005 and 0.03.
"""
import csv, itertools, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# figures land beside the script unless the caller says otherwise
OUTDIR = os.environ.get("GUARD_FIGOUT", os.path.join(HERE, "figures"))
os.makedirs(OUTDIR, exist_ok=True)
SRC = os.path.join(HERE, "ablation_data/METRICS/crossmask_precondition.csv")
OUT = os.path.join(OUTDIR, "fig_screen.pdf")
HEADROOM = 0.02
GREEN, RED, GREY = "#4c7a34", "#b85450", "#999999"

import figstyle
figstyle.apply()

def load(th=HEADROOM):
    out = []
    for r in csv.DictReader(open(SRC)):
        gh, gc = float(r["gain_hard"]), float(r["gain_cross"])
        if gh < th:
            continue
        ra, pa = float(r["richer_acc"]), float(r["poorer_acc"])
        out.append(dict(ra=ra, mar=ra - pa, ret=gc / gh,
                        met=r["precondition_met"] == "True"))
    return out

def auroc(P, key):
    pos = [r for r in P if r["ret"] >= .5]
    neg = [r for r in P if r["ret"] < .5]
    w = sum(1. if a[key] > b[key] else (.5 if a[key] == b[key] else 0.)
            for a, b in itertools.product(pos, neg))
    return w / (len(pos) * len(neg)), len(pos), len(neg)

if __name__ == "__main__":
    D = load()
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.0), sharey=True)
    for a, (key, lab) in zip(ax, [("ra", "richer-mask predictor accuracy"),
                                  ("mar", "margin over masked predictor")]):
        for met, col, mk, nm in [(True, GREEN, "o", "screen passes"),
                                 (False, RED, "X", "screen fails")]:
            p = [d for d in D if d["met"] == met]
            a.scatter([d[key] for d in p], [d["ret"] for d in p], s=22, c=col,
                      marker=mk, edgecolors="none", alpha=.85, label=nm, zorder=3)
        a.axhline(.5, color=GREY, lw=.8, ls="--", zorder=1)
        a.set_xlabel(lab)
        a.set_ylim(-.15, 1.25)
        a.grid(alpha=.25, lw=.5)
        for s in ("top", "right"):
            pass
    # to nen khoang trong cua panel trai: khong cell nao roi vao day
    P = [d for d in D if d["met"]]
    lo = max(d["ra"] for d in P if d["ret"] < .5)
    hi = min(d["ra"] for d in P if d["ret"] >= .5)
    ax[0].axvspan(lo, hi, color="#dddddd", alpha=.45, zorder=0)
    ax[0].set_ylabel("fraction of hard-label\ncertified gain retained")
    ax[0].legend(loc="center right", fontsize=7, handletextpad=.3)
    fig.tight_layout(pad=.4)
    fig.savefig(OUT)
    a1, k, l = auroc(P, "ra")
    a2, _, _ = auroc(P, "mar")
    print("%d cell (fail %d, pass %d: giu %d, mat %d)"
          % (len(D), len(D) - len(P), len(P), k, l))
    print("khoang trong richer_acc: %.3f den %.3f" % (lo, hi))
    print("AUROC richer_acc=%.3f margin=%.3f -> %s" % (a1, a2, OUT))
