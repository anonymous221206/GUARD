#!/usr/bin/env python3
"""Measure x Certify: what each stage of the method is actually load-bearing for.

GUARD is Measure -> Recalibrate -> Certify.  Recalibrate is the correction
itself and cannot be switched off without deleting the method, so the ablation
that carries information is over the two stages that wrap it:

    Measure on   beta is fitted on D_fit, so the size of the correction is set
                 by the calibration gap measured on this deployment
    Measure off  beta = 1: the host's output is replaced by the retrieval target
                 outright, correcting without first asking how much correction
                 is warranted

    Certify on   the split-conformal LAC gate
    Certify off  blanket -- correct every sample

The cells that decide the question are the *null* conditions: full-input
DrugBAN, Hateful Memes, Food-101.  There the correction has nothing to add, so
Measure-off walks into damage that Measure-on avoids, and Certify catches what
survives.  A layered design earns its complexity only if each layer covers a
failure the other lets through, and this table is where that shows.

    python experiments/exp_modules.py

Built from the same primitives as ``exp_ablations.py``, so the numbers sit on
the same code path as the rest of the tables.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from exp_ablations import load_dumps                      # noqa: E402
from guard import losses as L                             # noqa: E402
from guard.certify import certify, harm_accounting        # noqa: E402
from guard.pipeline import BETA_GRID                      # noqa: E402
from guard.targets import hard_label_values, knn_average, standardise  # noqa: E402

#: (dump directory, adapter kind, conditions, which of them are nulls)
JOBS = (
    ("drugban_biosnap_random_s42", "drugban",
     ("full", "prot50", "prot25", "scaffold", "scaffold_prot50"), {"full"}),
    ("drugban_bindingdb_random_s42", "drugban",
     ("full", "prot50", "prot25", "scaffold", "scaffold_prot50"), {"full"}),
    ("hateful_memes", "vilt", ("complete", "textmiss", "imgmiss"),
     {"complete", "textmiss", "imgmiss"}),
    ("food101", "vilt", ("complete", "textmiss", "imgmiss"),
     {"complete", "textmiss", "imgmiss"}),
)


def retrieval_targets(host, split, k):
    """k-NN target on fit/conf/test, standardised on the pool as the pipeline does."""
    z = standardise(host.features[split.pool])
    f_pool = z(host.features[split.pool])
    values = hard_label_values(host.labels[split.pool], host.probs.shape[1], True)
    k_eff = min(k, len(split.pool) - 1)
    est = lambda q: knn_average(z(q), f_pool, values, k_eff)
    return (est(host.features[split.fit]), est(host.features[split.conf]),
            est(host.features[split.test]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed")
    ap.add_argument("--out", type=Path, default=ROOT / "results/modules")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()

    loss = L.CROSS_ENTROPY
    a.out.mkdir(parents=True, exist_ok=True)
    fh = open(a.out / "guard.csv", "w", newline="")
    cols = ["dataset", "condition", "is_null", "seed", "measure", "certify", "beta",
            "base_acc", "d_acc", "loss_gain", "apply_rate", "joint_harm", "cond_harm"]
    wr = csv.DictWriter(fh, fieldnames=cols)
    wr.writeheader()
    rows = []

    for tag, kind, conds, nulls in JOBS:
        dumps = a.data / tag
        probe = dumps / ("full.npz" if kind == "drugban" else "extract_complete.npz")
        if not probe.exists():
            print(f"  skip {tag}: no dump")
            continue
        for cond in conds:
            for seed in a.seeds:
                host, split = load_dumps(dumps, kind, cond, seed)
                y_fit, y_conf, y_test = (host.labels[split.fit], host.labels[split.conf],
                                         host.labels[split.test])
                m_fit, m_conf, m_test = (host.probs[split.fit], host.probs[split.conf],
                                         host.probs[split.test])
                t_fit, t_conf, t_test = retrieval_targets(host, split, a.k)
                base_loss = loss(m_test, y_test)
                base_acc = L.accuracy(m_test, y_test, loss)

                for measure in ("on", "off"):
                    if measure == "on":
                        beta = float(min(BETA_GRID, key=lambda b: loss(
                            (1 - b) * m_fit + b * t_fit, y_fit).mean()))
                    else:
                        beta = 1.0          # correct without measuring how much
                    pc_conf = (1 - beta) * m_conf + beta * t_conf
                    pc_test = (1 - beta) * m_test + beta * t_test

                    for cert in ("on", "off"):
                        if cert == "on":
                            apply = certify(pc_conf, y_conf, pc_test, m_test,
                                            loss, a.alpha, a.delta)["apply"]
                        else:
                            apply = np.ones(len(y_test), dtype=bool)
                        gated = np.where(apply[:, None], pc_test, m_test)
                        dl = np.where(apply, loss(pc_test, y_test), base_loss) - base_loss
                        r = dict(dataset=tag, condition=cond,
                                 is_null=int(cond in nulls), seed=seed,
                                 measure=measure, certify=cert,
                                 beta=round(beta, 4), base_acc=round(base_acc, 6),
                                 d_acc=round(L.accuracy(gated, y_test, loss) - base_acc, 6),
                                 loss_gain=round(float(-dl.mean()), 6),
                                 # supplies apply_rate, joint_harm and cond_harm
                                 **{k: round(v, 6) for k, v in
                                    harm_accounting(dl, apply, a.delta).items()})
                        wr.writerow(r)
                        rows.append(r)
            fh.flush()
            print(f"  {tag}/{cond}", flush=True)
    fh.close()

    def mean(sel, key):
        v = [r[key] for r in rows if sel(r)]
        return float(np.mean(v)) if v else float("nan")

    for scope, keep in (("all conditions", lambda r: True),
                        ("null conditions only", lambda r: r["is_null"] == 1),
                        ("conditions with a gain", lambda r: r["is_null"] == 0)):
        print(f"\n=== {scope} ===")
        print(f"{'Measure':>8s} {'Certify':>8s} {'d_acc':>9s} {'loss gain':>10s} "
              f"{'apply':>7s} {'joint harm':>11s}")
        for meas in ("on", "off"):
            for cert in ("on", "off"):
                s = lambda r: keep(r) and r["measure"] == meas and r["certify"] == cert
                print(f"{meas:>8s} {cert:>8s} {mean(s,'d_acc'):+9.4f} "
                      f"{mean(s,'loss_gain'):+10.4f} {mean(s,'apply_rate'):7.3f} "
                      f"{mean(s,'joint_harm'):11.4f}")
    print(f"\n-> {a.out / 'guard.csv'}")


if __name__ == "__main__":
    main()
