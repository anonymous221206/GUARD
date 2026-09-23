#!/usr/bin/env python3
"""LaTeX rows for the retrieval-pool table, from results/source_pool/*.csv.

Written by ``experiments/exp_source_pool.py``. Every number in the table and in
the paragraph around it is printed here, so the two cannot drift apart.
"""
import csv
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/source_pool"
ALPHA = 0.2
NUM = ("pool_frac", "noise", "n_pool", "n_conf", "rep", "gate_metric_delta",
       "joint_harm", "gate_accuracy", "base_accuracy")

rows = []
for f in sorted(glob.glob(str(RES / "*.csv"))):
    for r in csv.DictReader(open(f)):
        for k in NUM:
            r[k] = float(r[k]) if r.get(k, "") != "" else None
        r["acc_gain"] = r["gate_accuracy"] - r["base_accuracy"]
        rows.append(r)


def sel(**kw):
    return [r for r in rows if all(
        (v(r[k]) if callable(v) else r[k] == v) for k, v in kw.items())]


def mean(rs, key):
    return float(np.mean([r[key] for r in rs]))


def f3(v, sign=True):
    return f"${v:+.3f}$" if sign else f"${v:.3f}$"


def row(label, cells):
    print(f"{label:<34} & " + " & ".join(cells) + r" \\")


is_full = lambda x: x == 1.0
is_sub = lambda x: x is not None and x < 1.0

# ---- (a) CMU-MOSEI: pool size, and label flips at full size
def mosei_size(policy, N):
    if N == "full":
        return sel(bench="mosei", study="size", policy=policy, pool_frac=is_full)
    return sel(bench="mosei", study="size", policy=policy, pool_frac=is_sub, n_pool=N)


sizes = [50.0, 150.0, 400.0, "full"]
print("% (a) MOSEI pool size: 50 150 400 full")
for p, lab in (("hard", "Hard-label gain"), ("cross_mask", "Cross-mask gain"),
               ("selected", "GUARD gain")):
    row(lab, [f3(mean(mosei_size(p, N), "gate_metric_delta")) for N in sizes])
row("GUARD joint harm", [f3(mean(mosei_size("selected", N), "joint_harm"), False)
                         for N in sizes])

rates = (0.1, 0.2, 0.3, 0.4)
noisy = lambda p, x: sel(bench="mosei", study="noise", policy=p, noise=x)
picks = lambda rs: f"${100 * np.mean([r['chosen'] == 'cross_mask' for r in rs]):.0f}$"
print("% (a') MOSEI label flips 0 .1 .2 .3 .4")
row("Hard-label gain", [f3(mean(mosei_size("hard", "full"), "gate_metric_delta"))] +
    [f3(mean(noisy("hard", x), "gate_metric_delta")) for x in rates])
row("GUARD gain", [f3(mean(mosei_size("selected", "full"), "gate_metric_delta"))] +
    [f3(mean(noisy("selected", x), "gate_metric_delta")) for x in rates])
row("GUARD picks cross-mask (\\%)", [picks(mosei_size("selected", "full"))] +
    [picks(noisy("selected", x)) for x in rates])

# ---- (b) DrugBAN random, hard-label, pool size (AUROC)
def drug_size(N):
    if N == "full":
        return sel(bench="drugban", study="size", policy="hard", pool_frac=is_full)
    return sel(bench="drugban", study="size", policy="hard", pool_frac=is_sub, n_pool=N)


dsz = [100.0, 300.0, 1000.0, 3000.0, "full"]
print("% (b) DrugBAN hard-label pool size: 100 300 1000 3000 full")
row("Hard-label AUROC gain", [f3(mean(drug_size(N), "gate_metric_delta")) for N in dsz])
row("Joint harm", [f3(mean(drug_size(N), "joint_harm"), False) for N in dsz])

# ---- (c) D_conf size; "full" is each setting's own untouched D_conf (rep 0)
print("% (c) D_conf size 50 100 200 full")
for bench in ("mosei", "drugban"):
    c = sel(bench=bench, study="conf")
    nmax = {}
    for r in c:
        key = (r["name"], r["seed"], r["cond"])
        nmax[key] = max(nmax.get(key, 0), r["n_conf"])
    arms = [[r for r in c if r["n_conf"] == N] for N in (50, 100, 200)]
    arms.append([r for r in c if r["n_conf"] == nmax[(r["name"], r["seed"], r["cond"])]])
    row(f"{bench} gain", [f3(mean(a, "gate_metric_delta")) for a in arms])
    row(f"{bench} mean joint harm", [f3(mean(a, "joint_harm"), False) for a in arms])
    print(f"%   {bench} draws above alpha (%):",
          [round(100 * np.mean([r["joint_harm"] > ALPHA for r in a])) for a in arms],
          "max harm:", [round(max(r["joint_harm"] for r in a), 3) for a in arms],
          "n:", [len(a) for a in arms])

# ---- (d) cross-domain origin, GUARD as deployed (selected), AUROC gain
arms = ("deployment", "source_matched", "source_all", "source_all+deployment")
print("% (d) origin:", arms)
for ds in ("biosnap", "bindingdb"):
    got = [sel(study="origin", policy="selected", name=ds, arm=a) for a in arms]
    row(f"{ds} cluster AUROC gain", [f3(mean(g, "gate_metric_delta")) for g in got])
    print("%   pool sizes", [int(g[0]["n_pool"]) for g in got],
          "harm", [round(mean(g, "joint_harm"), 3) for g in got])

# ---- (e) random splits with a pool the host never trained on
print("% (e) clean pool, accuracy gain hard vs cross-mask")
for ds in ("human", "bindingdb", "biosnap"):
    s = sel(study="clean", name=ds)
    print(f"%   {ds}: n_pool {int(s[0]['n_pool'])}  hard "
          f"{mean([r for r in s if r['policy'] == 'hard'], 'acc_gain'):+.3f}  cross "
          f"{mean([r for r in s if r['policy'] == 'cross_mask'], 'acc_gain'):+.3f}  "
          f"harm max {max(r['joint_harm'] for r in s):.3f}")

# ---- why (e) exists: the all-inputs model on its own training pool
dumps = sorted(glob.glob(str(ROOT / "data/processed/drugban_*_random_s*")))
if dumps:
    acc = lambda d, part: float((d[f"{part}_probs"].argmax(1) == d[f"{part}_labels"]).mean())
    got = [(acc(z, "pool"), acc(z, "test")) for z in
           (np.load(Path(p) / "full.npz") for p in dumps)]
    print(f"%   richer accuracy on the pool {min(g[0] for g in got):.3f}--"
          f"{max(g[0] for g in got):.3f}, on test {min(g[1] for g in got):.3f}--"
          f"{max(g[1] for g in got):.3f} over {len(got)} random-split hosts")
