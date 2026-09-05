#!/usr/bin/env python3
"""NinaPro DB5: GUARD on frozen, per-rung CNN dumps.

    python experiments/exp_ninapro.py --artifact artifacts/ninapro_cnn
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guard import HostOutputs, run  # noqa: E402
from guard.splits import Split  # noqa: E402
from hosts.ninapro import RUNGS, build  # noqa: E402


def deployment_split(n_pool: int, n_deployment: int, seed: int, pool: str) -> Split:
    deployment = np.random.default_rng(seed).permutation(n_deployment) + n_pool
    if pool == "train":
        fit, conf, test = np.array_split(deployment, 3)
        return Split(np.arange(n_pool), fit, conf, test, origin={
            "pool": "NinaPro training repetitions", "fit": "NinaPro deployment repetitions",
            "conf": "NinaPro deployment repetitions", "test": "NinaPro deployment repetitions",
        })
    pool_part, fit, conf, test = np.array_split(deployment, 4)
    return Split(pool_part, fit, conf, test, origin={
        role: "NinaPro deployment repetitions" for role in ("pool", "fit", "conf", "test")
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--pool", choices=["train", "deployment"], default="train")
    ap.add_argument("--rungs", type=int, nargs="+", default=[12, 8, 6, 4])
    ap.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--seeds", type=int, nargs="+", default=[7], help="deployment split seeds")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=50)
    a = ap.parse_args()
    if any(rung not in RUNGS for rung in a.rungs):
        raise ValueError(f"rungs must be chosen from {RUNGS}")

    out_dir = a.out / f"ninapro_cnn_pool-{a.pool}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    print(f"{'model':>5s} {'subject':>7s} {'rung':>4s} {'target':11s} {'split':>5s} "
          f"{'base':>7s} {'tgt':>7s} {'beta':>5s} {'gate acc':>9s} {'apply':>6s} {'joint':>6s}")
    for model_seed in a.model_seeds:
        for subject in a.subjects:
            subject_dir = a.artifact / f"seed{model_seed}" / f"subject{subject:02d}"
            for rung in a.rungs:
                probs, feats, labels, richer, n_pool, n_dep = build(subject_dir, rung, a.pool)
                host = HostOutputs(probs=probs, features=feats, labels=labels, richer_probs=richer)
                for split_seed in a.seeds:
                    split = deployment_split(n_pool, n_dep, split_seed, a.pool)
                    for target in ("hard", "cross_mask"):
                        result = run(host, split, condition=str(rung), target=target,
                                     alpha=a.alpha, delta=a.delta, k=a.k,
                                     beta_objective="crossfit")
                        rows.append({**result.as_row(), "model_seed": model_seed,
                                     "subject": subject, "seed": split_seed})
                        print(f"{model_seed:5d} {subject:7d} {rung:4d} {target:11s} {split_seed:5d} "
                              f"{result.base_metric:7.4f} {result.target_accuracy:7.3f} "
                              f"{result.beta:5.2f} {result.gate_metric_delta:+9.4f} "
                              f"{result.apply_rate:6.2f} {result.joint_harm:6.3f}")
    with open(out_dir / "guard.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[key for key in rows[0] if key != "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key != "notes"})
    (out_dir / "manifest.json").write_text(json.dumps({
        "artifact": str(a.artifact), "pool": a.pool, "rungs": a.rungs,
        "model_seeds": a.model_seeds, "subjects": a.subjects, "seeds": a.seeds,
        "alpha": a.alpha, "delta": a.delta, "k": a.k, "beta_objective": "crossfit",
    }, indent=2) + "\n")
    for rung in a.rungs:
        for target in ("hard", "cross_mask"):
            selected = [row for row in rows if row["condition"] == str(rung) and row["target"] == target]
            print(f"aggregate rung {rung:2d} {target:11s}: base "
                  f"{np.mean([row['base_metric'] for row in selected]):.4f}, target "
                  f"{np.mean([row['target_accuracy'] for row in selected]):.4f}, GUARD "
                  f"{np.mean([row['base_metric'] + row['gate_metric_delta'] for row in selected]):.4f}")
    print(f"wrote {out_dir / 'guard.csv'}")


if __name__ == "__main__":
    main()
