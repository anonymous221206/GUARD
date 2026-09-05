#!/usr/bin/env python3
"""Ablations: what the certificate buys, and what it does not depend on.

Three studies, all on dumps produced by the host adapters, all reusing the same
pipeline:

``selectors``  confidence thresholding matched to the gate's apply rate.  At
               equal coverage the gains are comparable; the difference is that
               a selector cannot be given a harm level in advance.
``plugins``    swap k-NN for a Nadaraya--Watson kernel or a ridge probe.  The
               framework does not depend on k-NN.
``labels``     sweep the labelled calibration budget.

    python experiments/exp_ablations.py --study selectors --dumps <dir> --kind drugban
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guard import HostOutputs                              # noqa: E402
from guard import losses as L                              # noqa: E402
from guard.baselines import PLUGINS, SELECTORS, selector_mask   # noqa: E402
from guard.certify import certify, harm_accounting         # noqa: E402
from guard.pipeline import BETA_GRID, run                  # noqa: E402
from guard.splits import Split                             # noqa: E402
from guard.targets import hard_label_values, knn_average, standardise  # noqa: E402


def load_dumps(dumps: Path, kind: str, condition: str, seed: int):
    """Return (HostOutputs, Split) for one condition of a dumped host."""
    if kind == "drugban":
        d = np.load(dumps / f"{condition}.npz")
        keys = ("pool", "calib", "test")
    else:
        d = np.load(dumps / f"extract_{condition}.npz")
        keys = ("src", "val", "test")
    p = [d[f"{k}_probs"].astype(np.float64) for k in keys]
    f = [d[f"{k}_feats"] for k in keys]
    y = [d[f"{k}_labels"] for k in keys]
    n_pool, n_cal = len(y[0]), len(y[1])
    perm = np.random.default_rng(seed).permutation(n_cal) + n_pool
    split = Split(pool=np.arange(n_pool), fit=perm[: n_cal // 2], conf=perm[n_cal // 2:],
                  test=np.arange(len(y[2])) + n_pool + n_cal,
                  origin={"pool": "training population", "fit": "deployment",
                          "conf": "deployment", "test": "deployment"})
    host = HostOutputs(probs=np.concatenate(p), features=np.concatenate(f),
                       labels=np.concatenate(y))
    return host, split


def _corrected(host, split, k, plugin="knn"):
    """Blend once; the ablations differ only in what they do afterwards."""
    loss = L.CROSS_ENTROPY
    z = standardise(host.features[split.pool])
    fp, ff = z(host.features[split.pool]), z(host.features[split.fit])
    fc, ft = z(host.features[split.conf]), z(host.features[split.test])
    vals = hard_label_values(host.labels[split.pool], host.probs.shape[1], True)
    if plugin == "knn":
        est = lambda q: knn_average(q, fp, vals, min(k, len(split.pool) - 1))
    else:
        fn = PLUGINS[plugin]
        est = lambda q: fn(q, fp, vals)
    t_fit, t_conf, t_test = est(ff), est(fc), est(ft)
    y_fit = host.labels[split.fit]
    beta = float(min(BETA_GRID, key=lambda b: loss(
        (1 - b) * host.probs[split.fit] + b * t_fit, y_fit).mean()))
    return (beta,
            (1 - beta) * host.probs[split.conf] + beta * t_conf,
            (1 - beta) * host.probs[split.test] + beta * t_test)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=["selectors", "plugins", "labels"])
    ap.add_argument("--dumps", type=Path, required=True)
    ap.add_argument("--kind", default="drugban", choices=["drugban", "vilt"])
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()

    conds = a.conditions or (["full", "prot50", "prot25", "scaffold", "scaffold_prot50"]
                             if a.kind == "drugban" else
                             ["complete", "textmiss", "imgmiss"])
    out_dir = a.out / f"ablation_{a.study}_{a.dumps.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    loss = L.CROSS_ENTROPY
    rows = []

    for cond in conds:
        for seed in a.seeds:
            host, split = load_dumps(a.dumps, a.kind, cond, seed)
            y_test = host.labels[split.test]
            m_test = host.probs[split.test]
            base_loss = loss(m_test, y_test)
            base_acc = L.accuracy(m_test, y_test, loss)

            if a.study == "plugins":
                for plugin in PLUGINS:
                    beta, pc_conf, pc_test = _corrected(host, split, a.k, plugin)
                    g = certify(pc_conf, host.labels[split.conf], pc_test, m_test,
                                loss, a.alpha, a.delta)
                    gated = np.where(g["apply"][:, None], pc_test, m_test)
                    dl = np.where(g["apply"], loss(pc_test, y_test), base_loss) - base_loss
                    rows.append(dict(condition=cond, seed=seed, arm=plugin,
                                     base_acc=base_acc,
                                     d_acc=L.accuracy(gated, y_test, loss) - base_acc,
                                     loss_gain=float(-dl.mean()),
                                     **harm_accounting(dl, g["apply"], a.delta)))
                continue

            beta, pc_conf, pc_test = _corrected(host, split, a.k)
            g = certify(pc_conf, host.labels[split.conf], pc_test, m_test,
                        loss, a.alpha, a.delta)
            coverage = float(g["apply"].mean())
            gated = np.where(g["apply"][:, None], pc_test, m_test)
            dl = np.where(g["apply"], loss(pc_test, y_test), base_loss) - base_loss
            rows.append(dict(condition=cond, seed=seed, arm="conformal gate",
                             base_acc=base_acc,
                             d_acc=L.accuracy(gated, y_test, loss) - base_acc,
                             loss_gain=float(-dl.mean()),
                             **harm_accounting(dl, g["apply"], a.delta)))

            if a.study == "selectors":
                for name in SELECTORS:
                    mask = selector_mask(name, host.probs[split.conf], pc_conf,
                                         m_test, pc_test, coverage)
                    gated = np.where(mask[:, None], pc_test, m_test)
                    dl = np.where(mask, loss(pc_test, y_test), base_loss) - base_loss
                    rows.append(dict(condition=cond, seed=seed, arm=f"selector:{name}",
                                     base_acc=base_acc,
                                     d_acc=L.accuracy(gated, y_test, loss) - base_acc,
                                     loss_gain=float(-dl.mean()),
                                     **harm_accounting(dl, mask, a.delta)))

    with open(out_dir / "guard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    arms = sorted({r["arm"] for r in rows})
    print(f"{'arm':22s} {'d_acc':>9s} {'loss gain':>10s} {'apply':>6s} "
          f"{'joint':>7s} {'cond':>7s}  cells over budget")
    for arm in arms:
        g = [r for r in rows if r["arm"] == arm]
        col = lambda kk: np.array([r[kk] for r in g])
        over = int((col("joint_harm") > a.alpha).sum())
        print(f"{arm:22s} {col('d_acc').mean():+9.4f} {col('loss_gain').mean():+10.4f} "
              f"{col('apply_rate').mean():6.2f} {col('joint_harm').mean():7.3f} "
              f"{col('cond_harm').mean():7.3f}  {over}/{len(g)}")
    print(f"\n-> {out_dir}/guard.csv")


if __name__ == "__main__":
    main()
