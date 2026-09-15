#!/usr/bin/env python3
"""Figure 4: what the correction buys, and what the gate buys, as degradation deepens.

Each benchmark gets two stacked axes on one degradation axis. The upper one is the
benchmark's own metric: the frozen model, the same corrector applied to every
sample, and GUARD. Those last two nearly coincide, which is the point -- the gate
costs almost no accuracy. The lower strip is what it costs instead: joint harm,
against the budget it was calibrated to. Reading only the upper axis would make
the gate look like it does nothing.

Sources, all produced by code in this repository:
  PTB-XL       experiments/ptbxl_sev_dense.py
  NinaPro DB5  experiments/nina_sev_dense.py
  OPPORTUNITY  experiments/opp_dcl.py
  DrugBAN      scripts/drugban_protladder_guard.py
  IEMOCAP      experiments/iemocap_eta.py
  AVE          experiments/ave_eta.py
"""
import json, os, sys, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R = ROOT / "results"
A = Path(os.environ.get("GUARD_ARTIFACTS", ROOT / "artifacts"))
OUT = Path(os.environ.get("GUARD_FIGOUT", HERE / "figures"))
sys.path.insert(0, str(HERE))
import figstyle

FROZEN, BLANKET = figstyle.PALETTE[0], figstyle.PALETTE[1]
OURS, GREY = figstyle.OURS, figstyle.GREY
ALPHA = 0.2
# GUARD lands on top of the ungated curve wherever the gate admits everything.
# Every line is the same weight, so the ungated one is dashed and drawn last:
# green dashes over a red line mean the two agree, not that one is missing.
LW, MS = 0.85, 1.8
DASH_UNGATED = (0, (3.2, 2.0))
DOT_BUDGET = (0, (1, 1.8))


def _read(path):
    return json.load(open(path)) if Path(path).exists() else None


def ptbxl():
    d = _read(A / "ptbxl_dropladder/severity_dense_auc.json") or \
        _read(R / "ptbxl_dropladder/severity_dense_auc.json")
    if d is None:
        return None
    d = sorted(d, key=lambda r: -r["leads"])
    return dict(title="PTB-XL", xlabel="ECG leads kept", ylabel="macro AUC",
                x=[r["leads"] for r in d], ticks=None, invert=True,
                frozen=[r["frozen"] for r in d], blanket=[r["blanket"] for r in d],
                guard=[r["guard"] for r in d],
                bharm=[r["bharm"] for r in d], harm=[r["harm"] for r in d])


def ninapro():
    d = _read(R / "ninapro/severity_dense.json")
    if d is None:
        return None
    d = sorted(d, key=lambda r: -int(r["level"]))
    return dict(title="NinaPro DB5", xlabel="sEMG electrodes kept", ylabel="accuracy",
                x=[int(r["level"]) for r in d], ticks=None, invert=True,
                frozen=[r["frozen"] for r in d], blanket=[r["blanket"] for r in d],
                guard=[r["guard"] for r in d],
                bharm=[r["bharm"] for r in d], harm=[r["harm"] for r in d])


def opportunity():
    d = _read(R / "gates/opportunity_dcl_full.json")
    if d is None or "severity" not in d:
        return None
    s = d["severity"]
    order = sorted(s, key=lambda c: -s[c]["frozen"])
    short = {"no_shoes": "no shoes", "severe_three_sensors": "3 sensors",
             "low_cost_accels_only": "accels only", "no_imu_family": "no IMU"}
    return dict(title="OPPORTUNITY", xlabel="sensor subset", ylabel="weighted F1",
                x=list(range(len(order))), ticks=[short.get(c, c) for c in order],
                invert=False,
                frozen=[s[c]["frozen"] for c in order],
                blanket=[s[c]["blanket"] for c in order],
                guard=[s[c]["guard"] for c in order],
                bharm=[s[c]["bharm"] for c in order],
                harm=[s[c]["harm"] for c in order])


def drugban():
    d = _read(A / "drugban_protladder_v2/guard_results.json") or \
        _read(R / "drugban_protladder_v2/guard_results.json")
    if d is None:
        return None
    pcts = sorted((int(k) for k in d["per_fraction"]), reverse=True)
    m = [d["per_fraction"][str(p)]["mean"] for p in pcts]
    return dict(title="DrugBAN", xlabel="protein sequence kept", ylabel="accuracy",
                x=pcts, ticks=None, invert=True, percent=True,
                frozen=[r["frozen_metric"] for r in m],
                blanket=[r["guard_without_certify_metric"] for r in m],
                guard=[r["guard_metric"] for r in m],
                bharm=[r["guard_without_certify_joint_harm"] for r in m],
                harm=[r["joint_harm"] for r in m])


def _eta_panel(path, title, ylabel):
    rows = _read(path)
    if rows is None:
        return None
    g = collections.defaultdict(list)
    for r in rows:
        g[r["eta"]].append(r)
    etas = sorted(g)
    base = [float(np.mean([r["base"] for r in g[e]])) for e in etas]
    add = lambda key: [b + float(np.mean([r[key][0] for r in g[e]]))
                       for b, e in zip(base, etas)]
    hm = lambda key: [float(np.mean([r[key][1] for r in g[e]])) for e in etas]
    return dict(title=title, xlabel=r"modalities missing, rate $\eta$", ylabel=ylabel,
                x=etas, ticks=None, invert=False, frozen=base,
                blanket=add("blanket"), guard=add("GUARD"),
                bharm=hm("blanket"), harm=hm("GUARD"))


def iemocap():
    return _eta_panel(R / "iemocap_eta.json", "IEMOCAP", "4-class accuracy")


def ave():
    return _eta_panel(R / "ave_eta.json", "AVE", "$29$-way acc.")


PANELS = [ptbxl, ninapro, opportunity, drugban, iemocap, ave]
NCOL = 3


# twelve axes at one column width, and the figure is read for the shape of the
# curves rather than for its labels, so the axes sit below the caption size and
# the space goes to the plots; the legend stays larger, being read as text
LAB, TCK, LEG = 7.2, 6.4, 7.0


def main():
    figstyle.apply()
    live, missing = [], []
    for fn in PANELS:
        p = fn()
        (live if p else missing).append(p or fn.__name__)
    if missing:
        print("thieu du lieu, bo qua panel:", missing, file=sys.stderr)
    if not live:
        sys.exit("khong co panel nao co du lieu")

    nblock = (len(live) + NCOL - 1) // NCOL
    fig = plt.figure(figsize=(figstyle.FULL_IN, 1.52 * nblock))
    # one cell per benchmark, split inside into metric and harm: nesting keeps the
    # two axes of a benchmark touching without gluing separate benchmarks together
    outer = fig.add_gridspec(nblock, NCOL, hspace=0.55, wspace=0.34,
                             left=0.085, right=0.995, top=0.950, bottom=0.125)

    for i, p in enumerate(live):
        r, c = divmod(i, NCOL)
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[2.0, 0.88], hspace=0.09)
        top = fig.add_subplot(inner[0])
        bot = fig.add_subplot(inner[1], sharex=top)
        x = p["x"]
        top.plot(x, p["frozen"], label="frozen model", color=FROZEN,
                 marker="^", ms=MS, lw=LW, zorder=2)
        top.plot(x, p["guard"], label="+ GUARD", color=OURS,
                 marker="o", ms=MS, lw=LW, zorder=3)
        top.plot(x, p["blanket"], label="GUARD w/o Certify", color=BLANKET,
                 marker="s", ms=MS, lw=LW, ls=DASH_UNGATED, zorder=4)
        top.set_title(p["title"], fontsize=LAB, pad=3)
        top.set_ylabel(p["ylabel"], fontsize=LAB, labelpad=1.5)
        top.tick_params(labelsize=TCK, pad=1.5, labelbottom=False)

        bot.axhline(ALPHA, color=GREY, lw=0.7, ls=DOT_BUDGET)
        bot.plot(x, p["harm"], color=OURS, marker="o", ms=MS, lw=LW, zorder=3)
        bot.plot(x, p["bharm"], color=BLANKET, marker="s", ms=MS, lw=LW,
                 ls=DASH_UNGATED, zorder=4)
        bot.set_ylabel("joint harm", fontsize=LAB, labelpad=1.5)
        bot.set_xlabel(p["xlabel"], fontsize=LAB, labelpad=1.5)
        bot.tick_params(labelsize=TCK, pad=1.5)
        hi = max(max(p["bharm"]), ALPHA)
        bot.set_ylim(0, hi * 1.18)
        bot.set_yticks([0, ALPHA] if hi < 0.32 else [0, ALPHA, round(hi, 1)])

        if p["ticks"] is not None:
            bot.set_xticks(x)
            bot.set_xticklabels(p["ticks"], rotation=20, ha="right", fontsize=TCK)
        elif p.get("percent"):
            bot.set_xticks([100, 70, 50, 30, 10])
            bot.set_xticklabels(["all", "70%", "50%", "30%", "10%"], fontsize=TCK)
        if p["invert"]:
            top.invert_xaxis()
        for ax in (top, bot):
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

    h, l = fig.axes[0].get_legend_handles_labels()
    order = [l.index("frozen model"), l.index("GUARD w/o Certify"), l.index("+ GUARD")]
    h, l = [h[i] for i in order], [l[i] for i in order]
    h.append(plt.Line2D([], [], color=GREY, lw=0.7, ls=DOT_BUDGET))
    l.append(r"budget $\alpha=0.2$")
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=LEG,
               bbox_to_anchor=(0.5, -0.045), handletextpad=0.5, columnspacing=1.3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_severity.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "fig_severity_preview.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("da ve", OUT / "fig_severity.pdf", f"({len(live)} panel, co dai harm)")


if __name__ == "__main__":
    main()
