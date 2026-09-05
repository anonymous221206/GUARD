#!/usr/bin/env python3
"""PTB-XL: GUARD on frozen lead-masked ResNet1D dumps.

    python experiments/exp_ptbxl.py --host artifacts/ptbxl_resnet1d_wang
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

from guard import HostOutputs, auroc, run  # noqa: E402
from guard import pipeline as _pipeline  # noqa: E402
from guard.splits import Split  # noqa: E402
from hosts.ptbxl import MASKS, build, load  # noqa: E402


def macro_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    values = [auroc(probs[:, column], labels[:, column]) for column in range(labels.shape[1])]
    return float(np.nanmean(values))


def enable_bernoulli_crossfit() -> None:
    """Extend rule D's accuracy objective to independent Bernoulli labels.

    The package implementation's cross-fit objective predates multi-label
    hosts and compares ``argmax`` to a vector label.  PTB-XL instead uses the
    same elementwise binary accuracy that :func:`guard.losses.accuracy` uses
    for a Bernoulli host.  This patch is process-local: it changes no package
    file and leaves simplex experiments on the package implementation.
    """
    original = _pipeline._crossfit_beta

    def crossfit(m_fit, t_fit, y_fit, loss, alpha, delta, *,
                 folds=_pipeline.CROSSFIT_FOLDS, seed=_pipeline.CROSSFIT_SEED):
        if loss.simplex:
            return original(m_fit, t_fit, y_fit, loss, alpha, delta, folds, seed)
        rng = np.random.default_rng(seed)
        parts = np.array_split(rng.permutation(len(m_fit)), folds)
        best_score, best_beta = -np.inf, float(_pipeline.BETA_GRID[0])
        for beta in _pipeline.BETA_GRID:
            corrected = (1 - beta) * m_fit + beta * t_fit
            gains = []
            for fold in range(folds):
                cal = parts[fold]
                ev = np.concatenate([parts[j] for j in range(folds) if j != fold])
                gate = _pipeline._certify.certify(corrected[cal], y_fit[cal], corrected[ev],
                                                   m_fit[ev], loss, alpha, delta)
                gated = np.where(gate["apply"][:, None], corrected[ev], m_fit[ev])
                gain = ((gated > 0.5).astype(int) == np.asarray(y_fit[ev], dtype=int)).mean()
                base_ev = ((m_fit[ev] > 0.5).astype(int)
                           == np.asarray(y_fit[ev], dtype=int)).mean()
                gains.append(float(gain - base_ev))
            score = float(np.mean(gains))
            if score > best_score:
                best_score, best_beta = score, float(beta)
        return best_beta

    _pipeline._crossfit_beta = crossfit


def deployment_split(n_pool: int, n_deployment: int, seed: int, pool: str) -> Split:
    deployment = np.random.default_rng(seed).permutation(n_deployment) + n_pool
    if pool == "train":
        fit, conf, test = np.array_split(deployment, 3)
        return Split(np.arange(n_pool), fit, conf, test, origin={
            "pool": "PTB-XL folds 1--8", "fit": "PTB-XL fold 10",
            "conf": "PTB-XL fold 10", "test": "PTB-XL fold 10",
        })
    pool_part, fit, conf, test = np.array_split(deployment, 4)
    return Split(pool_part, fit, conf, test, origin={
        role: "PTB-XL fold 10" for role in ("pool", "fit", "conf", "test")
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=Path, required=True)
    ap.add_argument("--pool", choices=["train", "deployment"], default="train")
    ap.add_argument("--masks", nargs="+", default=list(MASKS))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    if any(mask not in MASKS for mask in a.masks):
        raise ValueError(f"masks must be chosen from {MASKS}")

    enable_bernoulli_crossfit()
    data = load(a.host)
    full_auc = macro_auc(data["pred"]["sess1_atv"], data["raw"]["sess1_y"])
    print(f"full-mask fold-10 macro AUC {full_auc:.10f}")
    out_dir = a.out / f"ptbxl_resnet1d_wang_pool-{a.pool}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    print(f"{'mask':5s} {'target':11s} {'seed':>4s} {'f1':>7s} {'auc':>7s} "
          f"{'tgt':>7s} {'beta':>5s} {'gate f1':>9s} {'gate auc':>9s} {'joint':>6s}")
    for mask in a.masks:
        probs, feats, labels, richer, n_pool, n_dep = build(a.host, mask, a.pool)
        host = HostOutputs(probs=probs, features=feats, labels=labels, richer_probs=richer)
        for seed in a.seeds:
            split = deployment_split(n_pool, n_dep, seed, a.pool)
            for target in ("hard", "cross_mask"):
                result = run(host, split, condition=mask, target=target,
                             loss_name="bernoulli", metric="f1_macro",
                             alpha=a.alpha, delta=a.delta, k=a.k,
                             beta_objective="crossfit")
                arrays = result.test_arrays
                base_auc = macro_auc(arrays["base_probs"], arrays["labels"])
                gate_auc = macro_auc(arrays["gated_probs"], arrays["labels"])
                rows.append({**result.as_row(), "seed": seed, "base_macro_auc": base_auc,
                             "gate_macro_auc": gate_auc,
                             "blanket_macro_auc": macro_auc(arrays["blanket_probs"], arrays["labels"])})
                print(f"{mask:5s} {target:11s} {seed:4d} {result.base_metric:7.4f} "
                      f"{base_auc:7.4f} {result.target_accuracy:7.3f} {result.beta:5.2f} "
                      f"{result.gate_metric_delta:+9.4f} {gate_auc - base_auc:+9.4f} "
                      f"{result.joint_harm:6.3f}")
    with open(out_dir / "guard.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[key for key in rows[0] if key != "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key != "notes"})
    (out_dir / "manifest.json").write_text(json.dumps({
        "host": str(a.host), "pool": a.pool, "masks": a.masks, "alpha": a.alpha,
        "delta": a.delta, "k": a.k, "seeds": a.seeds, "loss": "bernoulli",
        "reported_metric": "macro_auc (guard.csv also records macro F1)",
        "full_mask_fold10_macro_auc": full_auc, "beta_objective": "crossfit",
    }, indent=2) + "\n")
    print(f"wrote {out_dir / 'guard.csv'}")


if __name__ == "__main__":
    main()
