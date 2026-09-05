#!/usr/bin/env python3
"""Exact split-order A/D rerun for the saved DeepConvLSTM deployment dump."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("$WORKSPACE")
OUT = ROOT / "results" / "rule_ab_compare_20260828_opportunity_exact_r2"
sys.path[:0] = [str(ROOT / "src")]

from guard import HostOutputs, run
from guard.splits import Split


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    dump = ROOT / "artifacts" / "opportunity_deepconvlstm" / "dumps"
    labels = np.load(dump / "deploy_y.npy")
    configs = json.loads(str(np.load(ROOT / "data" / "processed" / "opportunity.npz",
                                     allow_pickle=True)["configs"]))
    rows, times = [], []
    dep_u = np.arange(len(labels))[::2]
    for objective in ("loss", "crossfit"):
        all_start = time.perf_counter()
        for seed in (0, 1, 2):
            # This RNG is deliberately outside the config loop: guard_dcl2.py
            # consumes one permutation per config, in this order.
            rng = np.random.default_rng(700 + seed)
            for condition in configs:
                perm = dep_u[rng.permutation(len(dep_u))]
                available = perm[:3 * (len(dep_u) // 4)]
                test = perm[3 * (len(dep_u) // 4):]
                t3 = len(available) // 3
                split = Split(available[:t3], available[t3:2*t3], available[2*t3:3*t3], test,
                              origin={k: "subject 4" for k in ("pool", "fit", "conf", "test")})
                host = HostOutputs(
                    probs=np.load(dump / f"probs_condition_specialist_{condition}_deploy_s{seed}.npy"),
                    features=np.load(dump / f"retfeat_{condition}_deploy.npy"), labels=labels)
                start = time.perf_counter()
                r = run(host, split, condition=condition, target="hard", alpha=0.2,
                        delta=0.05, k=50, beta_objective=objective)
                rows.append(dict(condition=condition, seed=seed, beta_objective=objective,
                                 beta=r.beta, base_accuracy=r.base_metric,
                                 guard_accuracy=r.base_metric + r.gate_metric_delta,
                                 joint_harm=r.joint_harm, apply_rate=r.apply_rate,
                                 seconds=time.perf_counter() - start))
                # The paper entrypoint also evaluates the full-masked host in
                # this loop.  It consumes a second permutation before moving to
                # the next configuration; preserve that split chronology while
                # measuring the condition-specialist row only.
                rng.permutation(len(dep_u))
        times.append(dict(beta_objective=objective, seconds=time.perf_counter() - all_start,
                          n_runs=len(configs) * 3))
        print(f"{objective} done", flush=True)
    with (OUT / "guard.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (OUT / "timings.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(times[0])); w.writeheader(); w.writerows(times)
    print(f"DONE {OUT}", flush=True)


if __name__ == "__main__":
    main()
