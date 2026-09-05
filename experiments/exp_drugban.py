#!/usr/bin/env python3
"""DrugBAN: GUARD on a published drug--target host, on its own released splits.

Reads the per-condition dumps from ``hosts/drugban.py`` and runs the method.
Two split protocols, both from the host's own repository:

``random``   in-domain.  Retrieval pool from train; calibration split out of val.
``cluster``  cross-domain, defined by the host authors.  The pool can come from
             the source domain or from the deployment domain; ``--pool`` selects
             which, and the difference is one of the paper's findings.

    python experiments/exp_drugban.py --dumps data/processed/drugban_biosnap_random_s42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guard import HostOutputs, run                       # noqa: E402
from guard.splits import Split                           # noqa: E402
from guard.targets import richer_is_richer               # noqa: E402

CONDITIONS = ("full", "prot50", "prot25", "scaffold", "scaffold_prot50")


def build(dumps: Path, cond: str, pool_from: str, seed: int):
    d = np.load(dumps / f"{cond}.npz")
    rich = np.load(dumps / "full.npz")                   # the all-inputs mask
    n_pool, n_cal, n_test = (len(d["pool_labels"]), len(d["calib_labels"]),
                             len(d["test_labels"]))
    probs = np.concatenate([d["pool_probs"], d["calib_probs"], d["test_probs"]])
    feats = np.concatenate([d["pool_feats"], d["calib_feats"], d["test_feats"]])
    labels = np.concatenate([d["pool_labels"], d["calib_labels"], d["test_labels"]])
    richer = np.concatenate([rich["pool_probs"], rich["calib_probs"], rich["test_probs"]])
    rng = np.random.default_rng(seed)

    if pool_from == "source":
        perm = rng.permutation(n_cal) + n_pool
        split = Split(pool=np.arange(n_pool), fit=perm[: n_cal // 2],
                      conf=perm[n_cal // 2:], test=np.arange(n_test) + n_pool + n_cal,
                      origin={"pool": "source population", "fit": "deployment",
                              "conf": "deployment", "test": "deployment"})
    else:                                   # pool carved out of the deployment data
        perm = rng.permutation(n_cal) + n_pool
        t = n_cal // 3
        split = Split(pool=perm[:t], fit=perm[t:2 * t], conf=perm[2 * t:3 * t],
                      test=np.arange(n_test) + n_pool + n_cal,
                      origin={k: "deployment" for k in ("pool", "fit", "conf", "test")})
    host = HostOutputs(probs=probs.astype(np.float64), features=feats,
                       labels=labels, richer_probs=richer.astype(np.float64))
    return host, split


def _auroc(score, true) -> float:
    """Binary AUROC by the rank identity, ties averaged.

    DrugBAN is published under AUROC, so the table needs it alongside accuracy;
    the pipeline itself scores whatever metric it is asked for and does not carry
    this one.
    """
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):                       # average ranks within ties
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    pos = true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", type=Path, required=True)
    ap.add_argument("--pool", default="source", choices=["source", "deployment"])
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta-objective", default="crossfit",
                    choices=["loss", "metric", "crossfit"],
                    help="beta selection rule: 'loss' is rule A, 'crossfit' is "
                         "rule D, the cross-fit rule the paper reports.")
    a = ap.parse_args()

    name = f"{a.dumps.name}_pool-{a.pool}"
    out_dir = a.out / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'condition':17s} {'target':11s} {'base':>7s} {'tgt':>7s} {'beta':>5s} "
          f"{'blanket':>8s} {'gate':>8s} {'apply':>6s} {'joint':>6s} {'cond':>6s}")
    for cond in CONDITIONS:
        host, split = build(a.dumps, cond, a.pool, a.seed)
        if cond != "full":
            # Trong dieu kien "full", build() nap full.npz cho ca hai ve nen richer
            # va poorer la cung mot model: phep so tro nen vo nghia va luon tra False.
            # Phep kiem chi co nghia o cac dieu kien BI CHE.
            # Do tren split fit: cung phan phoi trien khai, nhung khong dung nhan test.
            check = richer_is_richer(host.richer_probs[split.fit],
                                     host.probs[split.fit], host.labels[split.fit])
            print(f"  pre-flight [{cond}]: richer host {check['richer_accuracy']:.4f} vs "
                  f"poorer {check['poorer_accuracy']:.4f} -> "
                  f"cross-mask {'applicable' if check['precondition_met'] else 'NOT applicable'}")
        for target in ("hard", "cross_mask"):
            r = run(host, split, condition=cond, target=target,
                    alpha=a.alpha, delta=a.delta, k=a.k,
                    beta_objective=a.beta_objective)
            A = r.test_arrays
            yt = A["labels"]
            extra = {}
            for pol, key in (("base", "base_probs"), ("blanket", "blanket_probs"),
                             ("gated", "gated_probs")):
                pp = A[key]
                extra[f"{pol}_acc"] = float((pp.argmax(1) == yt).mean())
                extra[f"{pol}_auroc"] = _auroc(pp[:, 1], yt)
            rows.append({**r.as_row(), **extra})
            print(f"{cond:17s} {target:11s} {r.base_metric:7.4f} {r.target_accuracy:7.3f} "
                  f"{r.beta:5.2f} {r.blanket_metric_delta:+8.4f} {r.gate_metric_delta:+8.4f} "
                  f"{r.apply_rate:6.2f} {r.joint_harm:6.3f} {r.cond_harm:6.3f}")
    with open(out_dir / "guard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "notes"])
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != "notes"})
    (out_dir / "manifest.json").write_text(json.dumps(
        {"dumps": str(a.dumps), "pool": a.pool, "alpha": a.alpha,
         "delta": a.delta, "k": a.k, "seed": a.seed}, indent=2))
    viol = sum(1 for r in rows if r["joint_harm"] > a.alpha)
    print(f"\n{viol}/{len(rows)} cells exceed the harm budget -> {out_dir}/guard.csv")


if __name__ == "__main__":
    main()
