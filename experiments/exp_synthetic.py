#!/usr/bin/env python3
"""Synthetic study: the gain follows the calibration gap, not the raw error.

Two hosts are built with the *same* expected error but different error sources.
In ``gap`` the error is a calibration gap -- the host's probabilities are
systematically skewed away from ``pi(X)`` -- and a post-hoc correction can
remove it.  In ``noise`` the same error comes from label ambiguity, which no
output edit can touch.  GUARD must help in the first case and decline in the
second; the certificate has to hold in both.

Runs in seconds and needs no download, so it doubles as an end-to-end check
that the pipeline is installed correctly::

    python experiments/exp_synthetic.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guard import HostOutputs, random_split, run          # noqa: E402


def make(kind: str, n: int, d: int, strength: float, seed: int):
    """Build a host whose error comes from a gap or from label noise."""
    rng = np.random.default_rng(seed)
    # every feature matters: retrieval is meant to be *possible* here, so that a
    # failure to gain says something about the error source, not about k-NN
    x = rng.normal(size=(n, d))
    logits = x * 1.5

    if kind == "gap":
        # true pi is sharp; the host is a smoothed, miscalibrated version of it
        p_true = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        skew = np.exp(logits * (1 - strength))
        probs = skew / skew.sum(1, keepdims=True)
    elif kind == "noise":
        # the host matches pi exactly; pi itself is close to uniform
        flat = logits * (1 - strength)
        p_true = np.exp(flat) / np.exp(flat).sum(1, keepdims=True)
        probs = p_true.copy()
    else:
        raise ValueError(kind)

    y = np.array([rng.choice(d, p=row) for row in p_true])
    return HostOutputs(probs=probs, features=x, labels=y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("results/synthetic"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"{'source':8s} {'strength':>9s} {'base acc':>9s} {'PH':>9s} "
          f"{'loss gain':>10s} {'acc gain':>9s} {'apply':>6s} {'joint':>6s}")
    for kind in ("gap", "noise"):
        for s in a.strengths:
            per_seed = []
            for seed in a.seeds:
                host = make(kind, a.n, a.classes, s, seed)
                split = random_split(np.arange(a.n), seed=seed)
                r = run(host, split, condition=f"{kind}@{s}",
                        alpha=a.alpha, delta=a.delta)
                rows.append({**r.as_row(), "source": kind, "strength": s, "seed": seed})
                per_seed.append(r)
            m = lambda f: float(np.mean([f(r) for r in per_seed]))
            print(f"{kind:8s} {s:9.2f} {m(lambda r: r.base_metric):9.4f} "
                  f"{m(lambda r: r.ph):+9.4f} {m(lambda r: r.gate_loss_gain):+10.4f} "
                  f"{m(lambda r: r.gate_metric_delta):+9.4f} "
                  f"{m(lambda r: r.apply_rate):6.2f} {m(lambda r: r.joint_harm):6.3f}")

    with open(a.out / "guard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "notes"])
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != "notes"})
    viol = sum(1 for r in rows if r["joint_harm"] > a.alpha)
    lg = lambda src: np.mean([r["gate_loss_gain"] for r in rows if r["source"] == src])
    ph = lambda src: np.mean([r["ph"] for r in rows if r["source"] == src])
    print(f"\n{viol}/{len(rows)} cells exceed the harm budget")
    print(f"gap-sourced error : measured gap {ph('gap'):+.4f}, loss recovered {lg('gap'):+.4f}")
    print(f"noise-sourced     : measured gap {ph('noise'):+.4f}, loss recovered {lg('noise'):+.4f}")
    print("\nThe skew used here preserves the arg-max, so the loss moves and the")
    print("decisions do not -- the same loss/metric split the paper reports on real data.")


if __name__ == "__main__":
    main()
