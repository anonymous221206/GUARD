#!/usr/bin/env python3
"""Affective computing: GUARD on published emotion-recognition hosts.

These hosts are already strong (0.54--0.82 accuracy), so the interesting
question is not how much accuracy GUARD adds -- it adds very little -- but
whether the harm budget holds while the calibration improves.  Both numbers are
reported for every cell; picking whichever looks better per benchmark would be
metric shopping.

    python experiments/exp_affective.py --host data/raw/hosts/momke_iemocap4

Masks are subsets of {a, t, v}; ``atv`` is the full mask and doubles as the
richer host for cross-mask retrieval, so it is reported for completeness but is
degenerate for that target.
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

from guard import HostOutputs, run                       # noqa: E402
from guard.splits import Split                           # noqa: E402
from hosts.dumps import MASKS, build                     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=Path, required=True,
                    help="directory with raw_features.npz and preds.npz")
    ap.add_argument("--pool", default="train", choices=["train", "deployment"])
    ap.add_argument("--masks", nargs="+", default=list(MASKS))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--beta-objective", default="crossfit",
                    choices=["loss", "metric", "crossfit"],
                    help="beta selection rule: 'loss' is rule A, 'crossfit' is rule D.")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="added to each seed before it is used.  The reported MOSEI "
                         "rows were produced with offset 1000 over seeds 0..19; the "
                         "split a seed selects depends on it, so it must match to "
                         "reproduce them exactly.")
    ap.add_argument("--non0", action="store_true",
                    help="drop samples whose label is exactly zero: the CMU-MOSEI "
                         "binary convention, required to match the paper's MOSEI "
                         "rows.  Leave off for IEMOCAP, where 0 is a real class.")
    a = ap.parse_args()

    tag = (f"affective_{a.host.name}_pool-{a.pool}"
           + ("_non0" if a.non0 else "")
           + ("" if a.beta_objective == "crossfit" else f"_{a.beta_objective}"))
    out_dir = a.out / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'mask':5s} {'target':11s} {'seed':>4s} {'base':>7s} {'tgt':>7s} "
          f"{'beta':>5s} {'gate acc':>9s} {'loss gain':>10s} {'apply':>6s} {'joint':>6s}")

    for mask in a.masks:
        probs, feats, labels, richer, n_pool, n_dep = build(a.host, mask, a.pool,
                                                           non0=a.non0)
        host = HostOutputs(probs=probs, features=feats, labels=labels, richer_probs=richer)
        for seed in a.seeds:
            perm = np.random.default_rng(a.seed_offset + seed).permutation(n_dep) + n_pool
            thirds = np.array_split(perm, 3)
            if a.pool == "deployment":
                # the pool is part of the same population: carve it out of the
                # permutation so no index is used for two roles at once
                quarters = np.array_split(perm, 4)
                split = Split(pool=quarters[0], fit=quarters[1], conf=quarters[2],
                              test=quarters[3],
                              origin={k: "deployment session" for k in
                                      ("pool", "fit", "conf", "test")})
            else:
                split = Split(pool=np.arange(n_pool), fit=thirds[0], conf=thirds[1],
                              test=thirds[2],
                              origin={"pool": "training session",
                                      "fit": "deployment session",
                                      "conf": "deployment session",
                                      "test": "deployment session"})
            for target in ("hard", "cross_mask"):
                r = run(host, split, condition=mask, target=target,
                    beta_objective=a.beta_objective,
                        alpha=a.alpha, delta=a.delta, k=a.k)
                rows.append({**r.as_row(), "seed": seed, "host": a.host.name})
                print(f"{mask:5s} {target:11s} {seed:4d} {r.base_metric:7.4f} "
                      f"{r.target_accuracy:7.3f} {r.beta:5.2f} {r.gate_metric_delta:+9.4f} "
                      f"{r.gate_loss_gain:+10.4f} {r.apply_rate:6.2f} {r.joint_harm:6.3f}")
                for note in r.notes:
                    print(f"      note: {note}")

    with open(out_dir / "guard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "notes"])
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != "notes"})
    (out_dir / "manifest.json").write_text(json.dumps(
        {"host": str(a.host), "pool": a.pool, "alpha": a.alpha, "delta": a.delta,
         "k": a.k, "seeds": a.seeds, "masks": a.masks}, indent=2))
    viol = sum(1 for r in rows if r["joint_harm"] > a.alpha)
    acc = np.array([r["gate_metric_delta"] for r in rows])
    los = np.array([r["gate_loss_gain"] for r in rows])
    print(f"\n{viol}/{len(rows)} cells exceed the harm budget")
    print(f"accuracy change {acc.mean():+.4f} on average; loss gain {los.mean():+.4f}. "
          "On hosts this strong the loss moves and the decisions mostly do not.")


if __name__ == "__main__":
    main()
