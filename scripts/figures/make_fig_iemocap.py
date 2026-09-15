#!/usr/bin/env python3
"""Figure 3: GUARD and SIEVE on IEMOCAP, against the same backbone.

Left panel is four-class accuracy against the deployment missing rate; right
panel is each method's gain over its own backbone, which is the only column the
two share, because SIEVE retrains the weights and GUARD does not.

Our two series come from ``experiments/iemocap_eta.py``. The published series
are quoted, not recomputed: MoMKE and SIEVE as reported in Table 2 of Gao et al.
(2026), which is also where our Table 8 quotes them from. They are literals here
for that reason, and the docstring is the only place they are edited.
"""
import json, os, sys, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get("GUARD_FIGOUT", HERE / "figures"))
sys.path.insert(0, str(HERE))
import figstyle

ETA = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# Gao et al. (2026), Table 2: the SIEVE paper's own MoMKE row and SIEVE row
MOMKE_PUB = [.776, .751, .736, .715, .695, .684, .665, .644]
SIEVE_PUB = [.793, .768, .748, .725, .711, .691, .681, .656]


def ours():
    rows = json.load(open(ROOT / "results/iemocap_eta.json"))
    g = collections.defaultdict(list)
    for r in rows:
        g[r["eta"]].append(r)
    etas = sorted(g)
    base = np.array([np.mean([r["base"] for r in g[e]]) for e in etas])
    gain = np.array([np.mean([r["GUARD"][0] for r in g[e]]) for e in etas])
    se = np.array([np.std([r["GUARD"][0] for r in g[e]], ddof=1)
                   / np.sqrt(len(g[e])) for e in etas])
    return etas, base, base + gain, gain, se


def main():
    figstyle.apply()
    etas, frozen, guard, gain, se = ours()
    if list(etas) != ETA:
        print("canh bao: luoi eta khac bang da cong bo", etas, file=sys.stderr)
    pub_eta = ETA[:len(MOMKE_PUB)]
    sieve_gain = np.array(SIEVE_PUB) - np.array(MOMKE_PUB)

    fig, (l, r) = plt.subplots(1, 2, figsize=(figstyle.FULL_IN, 2.05))

    l.plot(pub_eta, MOMKE_PUB, color=figstyle.PALETTE[0], marker="^",
           ms=figstyle.MS, lw=figstyle.LW, ls="--", label="MoMKE (published)")
    l.plot(pub_eta, SIEVE_PUB, color=figstyle.PALETTE[1], marker="s",
           ms=figstyle.MS, lw=figstyle.LW, ls="--", label="$+$ SIEVE (published)")
    l.plot(etas, frozen, color=figstyle.PALETTE[0], marker="^",
           ms=figstyle.MS, lw=figstyle.LW, label="MoMKE, our reproduction")
    l.plot(etas, guard, color=figstyle.OURS, marker="o",
           ms=figstyle.MS_OURS, lw=figstyle.LW_OURS, label="$+$ GUARD")
    l.set_xlabel(r"missing rate $\eta$", fontsize=figstyle.LABEL, labelpad=1.5)
    l.set_ylabel("four-class accuracy", fontsize=figstyle.LABEL, labelpad=1.5)

    r.axhline(0, color=figstyle.GREY, lw=0.6)
    r.plot(pub_eta, sieve_gain, color=figstyle.PALETTE[1], marker="s",
           ms=figstyle.MS, lw=figstyle.LW, ls="--", label="$+$ SIEVE (published)")
    r.errorbar(etas, gain, yerr=se, color=figstyle.OURS, marker="o",
               ms=figstyle.MS_OURS, lw=figstyle.LW_OURS, elinewidth=0.6,
               capsize=1.5, label="$+$ GUARD")
    r.set_xlabel(r"missing rate $\eta$", fontsize=figstyle.LABEL, labelpad=1.5)
    r.set_ylabel("gain over own backbone", fontsize=figstyle.LABEL, labelpad=1.5)

    for ax in (l, r):
        ax.tick_params(labelsize=figstyle.TICK, pad=1.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    figstyle.legend_below(fig, l, 4, y=-0.05, fontsize=figstyle.LEG)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.6)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_iemocap.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "fig_iemocap_preview.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("da ve", OUT / "fig_iemocap.pdf")


if __name__ == "__main__":
    main()
