#!/usr/bin/env python3
"""Sensitivity to the harm tolerance delta, with a decision-level harm beside it.

The certificate counts an intervention as harmful when it raises the loss by more
than delta. Here delta is swept with the retrieval settings chosen once on D_fit,
so only the harm event moves. Beside the joint loss harm we report the joint rate
of *decision flips*: applied corrections that turn a correct frozen decision into
a wrong one, which delta does not see.

    python experiments/exp_delta.py --bench mosei
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_source_pool import (GRID, POLICIES, ROOT, drugban_host, drugban_split,  # noqa: E402
                             mosei_host, mosei_split)
from guard import run                                                          # noqa: E402
from guard.pipeline import select_on_fit                                       # noqa: E402
from guard.splits import Split                                                 # noqa: E402

DELTAS = (0.0, 0.02, 0.05, 0.1, 0.2)
ALPHA = 0.2


def settings(bench):
    if bench == "mosei":
        for name in ("CMAD", "TMDC", "MoMKE"):
            for mask in ("a", "v", "av"):
                host = mosei_host(name, mask)
                for seed in range(5):
                    yield name, seed, mask, host, mosei_split(len(host.labels), seed), "accuracy_nonzero"
    else:
        for ds, pool, seeds in (("biosnap_random", "source", ("s42", "s1", "s2")),
                                ("bindingdb_random", "source", ("s42", "s1", "s2")),
                                ("human_random", "source", ("s42", "s1", "s2")),
                                ("biosnap_cluster", "deployment", ("s42", "s1", "s2")),
                                ("bindingdb_cluster", "deployment", ("s42",))):
            for s in seeds:
                for cond in ("prot25", "scaffold_prot50"):
                    host, n = drugban_host(f"{ds}_{s}", cond)
                    yield ds, s, cond, host, drugban_split(n, pool), "auroc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["mosei", "drugban"], required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results/delta")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, seed, cond, host, roles, metric in settings(a.bench):
        sp = Split(roles["pool"], roles["fit"], roles["conf"], roles["test"],
                   origin={k: "deployment" for k in ("pool", "fit", "conf", "test")})
        cfg = select_on_fit(host, sp, target_grid=POLICIES["selected"], metric=metric, **GRID)
        cfg = {k: cfg[k] for k in ("k", "target", "space", "weighting", "temperature")}
        for d in DELTAS:
            r = run(host, sp, alpha=ALPHA, delta=d, metric=metric, **cfg)
            ta = r.test_arrays
            y = ta["labels"]
            ok0 = ta["base_probs"].argmax(1) == y
            ok1 = ta["gated_probs"].argmax(1) == y
            rows.append(dict(bench=a.bench, name=name, seed=seed, cond=cond, delta=d,
                             target=cfg["target"], gain=r.gate_metric_delta,
                             acc_gain=float(ok1.mean() - ok0.mean()),
                             joint_harm=r.joint_harm, apply=r.apply_rate,
                             flip_harm=float((ta["applied"] & ok0 & ~ok1).mean()),
                             flip_help=float((ta["applied"] & ~ok0 & ok1).mean())))
            x = rows[-1]
            print(f"{name} {seed} {cond} d={d:.2f} gain {x['gain']:+.4f} harm {x['joint_harm']:.3f} "
                  f"apply {x['apply']:.2f} flips -{x['flip_harm']:.3f}/+{x['flip_help']:.3f}", flush=True)
    with open(a.out / f"{a.bench}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("DA GHI")


if __name__ == "__main__":
    main()
