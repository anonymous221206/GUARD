#!/usr/bin/env python3
"""Reproduce the audio-only IEMOCAP table route (5 folds x 5 dialogue splits)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path("$WORKSPACE")
OUT = ROOT / "results" / "rule_ab_compare_20260828_iemocap_exact"
sys.path[:0] = [str(ROOT / "src")]

from guard import losses
from guard.certify import certify, harm_accounting


GRID = np.round(np.arange(0.0, 1.001, 0.025), 3)


def knn(q, pool, labels, n_class, k=50):
    out = np.empty((len(q), n_class)); sq = (pool * pool).sum(1)
    values = np.eye(n_class)[labels]
    for i in range(0, len(q), 1024):
        x = q[i:i + 1024]
        d = (x * x).sum(1)[:, None] + sq[None] - 2 * x @ pool.T
        idx = np.argpartition(d, k - 1, axis=1)[:, :k]
        out[i:i + len(x)] = values[idx].mean(1)
    return out


def gated(p, target, y, conf, test, beta, loss):
    corrected_conf = (1 - beta) * p[conf] + beta * target[conf]
    corrected_test = (1 - beta) * p[test] + beta * target[test]
    gate = certify(corrected_conf, y[conf], corrected_test, p[test], loss, 0.2, 0.05)
    applied = gate["apply"]
    out = np.where(applied[:, None], corrected_test, p[test])
    base_loss, corrected_loss = loss(p[test], y[test]), loss(corrected_test, y[test])
    accounting = harm_accounting(np.where(applied, corrected_loss, base_loss) - base_loss,
                                 applied, 0.05)
    return out, accounting


def select_pooled_beta(p, target, y, loss):
    parts = np.array_split(np.random.default_rng(11).permutation(len(y)), 4)
    best, beta = -np.inf, 0.0
    base = p.argmax(1) == y
    for b in GRID:
        corrected = (1 - b) * p + b * target
        scores = []
        for i in range(4):
            conf = parts[i]
            test = np.concatenate([parts[j] for j in range(4) if j != i])
            gate = certify(corrected[conf], y[conf], corrected[test], p[test], loss, 0.2, 0.05)
            out = np.where(gate["apply"][:, None], corrected[test], p[test])
            scores.append((out.argmax(1) == y[test]).mean() - base[test].mean())
        if np.mean(scores) > best:
            best, beta = float(np.mean(scores)), float(b)
    return beta


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    data = []
    for fold in range(5):
        z = np.load(ROOT / "artifacts" / "iemocap_momke" / "folds" / f"fold{fold}_a.npz",
                    allow_pickle=True)
        fp, fq = z["pool_feats"].astype(float), z["feats"].astype(float)
        p, y = z["probs"].astype(float), z["labels"]
        mu, sd = fp.mean(0), fp.std(0) + 1e-8
        target = knn((fq - mu) / sd, (fp - mu) / sd, z["pool_labels"], p.shape[1])
        data.append((p, target, y, z["vid"]))
    loss = losses.get("cross_entropy")
    rows = []
    for split_seed in range(5):
        splits = []
        for fold, (_, _, _, vid) in enumerate(data):
            dialogs = np.unique(vid)
            dialogs = dialogs[np.random.default_rng(1000 * split_seed + fold).permutation(len(dialogs))]
            third = len(dialogs) // 3
            pick = lambda ids: np.flatnonzero(np.isin(vid, ids))
            splits.append((pick(dialogs[:third]), pick(dialogs[third:2 * third]),
                           pick(dialogs[2 * third:])))
        pf = np.concatenate([data[f][0][splits[f][0]] for f in range(5)])
        tf = np.concatenate([data[f][1][splits[f][0]] for f in range(5)])
        yf = np.concatenate([data[f][2][splits[f][0]] for f in range(5)])
        beta = select_pooled_beta(pf, tf, yf, loss)
        for fold, (p, target, y, _) in enumerate(data):
            _, conf, test = splits[fold]
            out, a = gated(p, target, y, conf, test, beta, loss)
            rows.append(dict(split_seed=split_seed, fold=fold, beta=beta,
                             base_accuracy=float((p[test].argmax(1) == y[test]).mean()),
                             guard_accuracy=float((out.argmax(1) == y[test]).mean()),
                             joint_harm=a["joint_harm"], apply_rate=a["apply_rate"]))
        print(f"split {split_seed} done", flush=True)
    with (OUT / "guard.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"DONE {OUT}", flush=True)


if __name__ == "__main__":
    main()
