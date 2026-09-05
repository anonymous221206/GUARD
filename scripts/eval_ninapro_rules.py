#!/usr/bin/env python3
"""Re-evaluate retrained NinaPro dumps through rule A and paper rule D."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REBUTTAL = "$WORKSPACE/scripts/rebuttal"
GUARD_RELEASE = Path("$WORKSPACE")
RUNGS = (12, 8, 6, 4)
ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.40)
N_CLS = 41
K = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def knn_target(query, pool, pool_y):
    squared = (pool ** 2).sum(1)
    values = np.eye(N_CLS)[pool_y]
    output = []
    for start in range(0, len(query), 2048):
        batch = query[start:start + 2048]
        distance = ((batch ** 2).sum(1)[:, None] + squared[None, :]
                    - 2 * batch @ pool.T)
        nearest = np.argpartition(distance, K - 1, axis=1)[:, :K]
        output.append(values[nearest].mean(1))
    return np.concatenate(output)


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)

    sys.path.insert(0, str(GUARD_RELEASE))
    sys.path.insert(0, str(GUARD_RELEASE / "src"))
    sys.path.insert(0, REBUTTAL)
    from guard import losses as guard_losses
    from guard.pipeline import _select_beta
    from ninapro41_gate import DELTA, pick_D, run_gate

    import torch

    loss = guard_losses.get("cross_entropy")
    records = []
    for seed in (0, 1):
        for rung in RUNGS:
            subject_results = {alpha: {"base": [], "raw": [], "A": [], "D": [],
                                       "beta_A": [], "beta_D": []}
                               for alpha in ALPHAS}
            for subject in range(1, 11):
                subject_dir = args.artifact / f"seed{seed}" / f"subject{subject:02d}"
                predictions = np.load(subject_dir / "preds.npz")
                embeddings = np.load(subject_dir / "masked_embeddings.npz")
                checkpoint = torch.load(
                    args.artifact / "checkpoints" / f"seed{seed}"
                    / f"subject{subject:02d}_rung{rung:02d}.pt",
                    map_location="cpu", weights_only=False,
                )
                train_y = predictions["train_y"]
                query_y = predictions["sess1_y"]
                train_embedding = embeddings[f"train_{rung}"]
                query_embedding = embeddings[f"sess1_{rung}"]
                train_global = np.asarray(checkpoint["train_index"])
                fit_global = np.asarray(checkpoint["fit_index"])
                fit_position = np.searchsorted(train_global, fit_global)
                if not np.array_equal(train_global[fit_position], fit_global):
                    raise ValueError("fit indices are not a subset of the retrieval pool")
                mean = train_embedding[fit_position].mean(0)
                std = train_embedding[fit_position].std(0) + 1e-8
                pool = (train_embedding - mean) / std
                query = (query_embedding - mean) / std
                target = knn_target(query, pool, train_y)
                poorer = predictions[f"sess1_{rung}"].astype(np.float64)

                permutation = np.random.default_rng(7).permutation(len(query_y))
                third = len(permutation) // 3
                fit = permutation[:third]
                conf = permutation[third:2 * third]
                test = permutation[2 * third:]
                base_accuracy = float((poorer[test].argmax(1) == query_y[test]).mean())
                raw_accuracy = float((target[test].argmax(1) == query_y[test]).mean())
                for alpha in ALPHAS:
                    beta_a = float(_select_beta(
                        poorer[fit], target[fit], query_y[fit], loss, "loss",
                        alpha=alpha, delta=DELTA
                    ))
                    output_a, _ = run_gate(
                        poorer, target, query_y, conf, test, beta_a, loss, alpha
                    )
                    beta_d = pick_D(poorer, target, query_y, fit, loss, alpha)
                    output_d, _ = run_gate(
                        poorer, target, query_y, conf, test, beta_d, loss, alpha
                    )
                    result = subject_results[alpha]
                    result["base"].append(base_accuracy)
                    result["raw"].append(raw_accuracy)
                    result["A"].append(float((output_a.argmax(1) == query_y[test]).mean()))
                    result["D"].append(float((output_d.argmax(1) == query_y[test]).mean()))
                    result["beta_A"].append(beta_a)
                    result["beta_D"].append(beta_d)
                print(f"seed={seed} rung={rung} subject={subject}", flush=True)

            for alpha, values in subject_results.items():
                records.append({
                    "seed": seed,
                    "rung": rung,
                    "alpha": alpha,
                    **{name: float(np.mean(series)) for name, series in values.items()},
                    **{f"{name}_subjects": series for name, series in values.items()},
                })

    (args.out / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    print("NINAPRO_RULE_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
