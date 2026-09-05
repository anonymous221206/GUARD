#!/usr/bin/env python3
"""Isolate rule D while retaining the old fit-pool and validation-selected k."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REBUTTAL = "$WORKSPACE/scripts/rebuttal"
GUARD_RELEASE = Path("$WORKSPACE")
RUNGS = (12, 8, 6, 4)
N_CLS = 41
ALPHA = 0.20


def knn_target(query, pool, pool_y, k):
    squared = (pool ** 2).sum(1)
    values = np.eye(N_CLS)[pool_y]
    output = []
    for start in range(0, len(query), 2048):
        batch = query[start:start + 2048]
        distance = ((batch ** 2).sum(1)[:, None] + squared[None, :]
                    - 2 * batch @ pool.T)
        nearest = np.argpartition(distance, k - 1, axis=1)[:, :k]
        output.append(values[nearest].mean(1))
    return np.concatenate(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)

    sys.path.insert(0, str(GUARD_RELEASE))
    sys.path.insert(0, str(GUARD_RELEASE / "src"))
    sys.path.insert(0, REBUTTAL)
    from guard import losses as guard_losses
    from ninapro41_gate import pick_D, run_gate
    import torch

    loss = guard_losses.get("cross_entropy")
    records = []
    for seed in (0, 1):
        for rung in RUNGS:
            base_values, raw_values, rule_d_values, beta_values = [], [], [], []
            for subject in range(1, 11):
                subject_dir = args.artifact / f"seed{seed}" / f"subject{subject:02d}"
                predictions = np.load(subject_dir / "preds.npz")
                embeddings = np.load(subject_dir / "masked_embeddings.npz")
                checkpoint = torch.load(
                    args.artifact / "checkpoints" / f"seed{seed}"
                    / f"subject{subject:02d}_rung{rung:02d}.pt",
                    map_location="cpu", weights_only=False,
                )
                train_global = np.asarray(checkpoint["train_index"])
                fit_global = np.asarray(checkpoint["fit_index"])
                fit_position = np.searchsorted(train_global, fit_global)
                train_embedding = embeddings[f"train_{rung}"]
                query_embedding = embeddings[f"sess1_{rung}"]
                fit_embedding = train_embedding[fit_position]
                mean = fit_embedding.mean(0)
                std = fit_embedding.std(0) + 1e-8
                pool = (fit_embedding - mean) / std
                query = (query_embedding - mean) / std
                train_y = predictions["train_y"]
                query_y = predictions["sess1_y"]
                target = knn_target(
                    query, pool, train_y[fit_position], int(checkpoint["best_k"])
                )
                poorer = predictions[f"sess1_{rung}"].astype(np.float64)
                permutation = np.random.default_rng(7).permutation(len(query_y))
                third = len(permutation) // 3
                fit = permutation[:third]
                conf = permutation[third:2 * third]
                test = permutation[2 * third:]
                beta = pick_D(poorer, target, query_y, fit, loss, ALPHA)
                output, _ = run_gate(
                    poorer, target, query_y, conf, test, beta, loss, ALPHA
                )
                base_values.append(float((poorer[test].argmax(1) == query_y[test]).mean()))
                raw_values.append(float((target[test].argmax(1) == query_y[test]).mean()))
                rule_d_values.append(float((output.argmax(1) == query_y[test]).mean()))
                beta_values.append(beta)
                print(f"seed={seed} rung={rung} subject={subject}", flush=True)
            records.append({
                "seed": seed,
                "rung": rung,
                "alpha": ALPHA,
                "base": float(np.mean(base_values)),
                "raw": float(np.mean(raw_values)),
                "D": float(np.mean(rule_d_values)),
                "beta_D": float(np.mean(beta_values)),
            })
    (args.out / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    print("NINAPRO_OLDTARGET_RULED_DONE", flush=True)


if __name__ == "__main__":
    main()
