#!/usr/bin/env python3
"""Export exact NinaPro ladder dumps from newly trained subject checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REBUTTAL = "$WORKSPACE/scripts/rebuttal"
RUNGS = (16, 12, 8, 6, 4)
N_CH = 16
N_CLS = 41


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1))
    return parser.parse_args()


def build_model(device):
    import torch
    from torch import nn

    net = nn.Sequential(
        nn.Conv1d(N_CH, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
        nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(),
    ).to(device)
    head = nn.Linear(128, N_CLS).to(device)
    return net, head


def forward(net, head, x, device):
    import torch

    logits, embeddings = [], []
    with torch.no_grad():
        for start in range(0, len(x), 2048):
            batch = torch.tensor(x[start:start + 2048], device=device)
            embedding = net(batch)
            logits.append(head(embedding).cpu().numpy())
            embeddings.append(embedding.cpu().numpy())
    logits = np.concatenate(logits)
    probabilities = np.exp(logits - logits.max(1, keepdims=True))
    probabilities /= probabilities.sum(1, keepdims=True)
    return probabilities.astype(np.float32), np.concatenate(embeddings).astype(np.float32)


def knn_predictions(embeddings, y, fit, query, k):
    fit_embedding = embeddings[fit]
    mean = fit_embedding.mean(0)
    std = fit_embedding.std(0) + 1e-8
    fit_embedding = (fit_embedding - mean) / std
    query_embedding = (embeddings[query] - mean) / std
    squared = (fit_embedding ** 2).sum(1)
    onehot = np.eye(N_CLS)[y[fit]]
    output = []
    for start in range(0, len(query_embedding), 2048):
        batch = query_embedding[start:start + 2048]
        distance = ((batch ** 2).sum(1)[:, None] + squared[None, :]
                    - 2 * batch @ fit_embedding.T)
        nearest = np.argpartition(distance, k - 1, axis=1)[:, :k]
        output.append(onehot[nearest].mean(1).argmax(1))
    return np.concatenate(output).astype(np.int16)


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.out}")
    args.out.mkdir(parents=True)

    import torch

    sys.path.insert(0, REBUTTAL)
    from ninapro41 import POOL_REPS, QUERY_REPS, load_subject

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_summaries = {}
    for seed in args.seeds:
        checkpoint_root = args.checkpoints / f"seed{seed}"
        seed_out = args.out / f"seed{seed}"
        seed_out.mkdir()
        base_by_rung = {rung: [] for rung in RUNGS}
        target_by_rung = {rung: [] for rung in RUNGS}
        full_accuracy = []

        for subject in range(1, 11):
            x, y, repetitions = load_subject(subject)
            train_index = np.flatnonzero(np.isin(repetitions, POOL_REPS))
            query_index = np.flatnonzero(np.isin(repetitions, QUERY_REPS))
            subject_out = seed_out / f"subject{subject:02d}"
            subject_out.mkdir()
            probabilities_dump = {
                "train_y": y[train_index].astype(np.int16),
                "sess1_y": y[query_index].astype(np.int16),
            }
            embeddings_dump = {}
            targets_dump = {}

            for rung in RUNGS:
                checkpoint_path = checkpoint_root / f"subject{subject:02d}_rung{rung:02d}.pt"
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if checkpoint["artifact_status"] != "retrained_checkpoint_not_original_paper_checkpoint":
                    raise ValueError(f"unmarked checkpoint: {checkpoint_path}")
                if int(checkpoint["seed"]) != seed or int(checkpoint["subject"]) != subject:
                    raise ValueError(f"checkpoint identity mismatch: {checkpoint_path}")

                normalized = ((x - checkpoint["normalization_mean"])
                              / checkpoint["normalization_std"]).transpose(0, 2, 1)
                masked = np.zeros_like(normalized)
                channels = np.asarray(checkpoint["channels"], dtype=int)
                masked[:, channels] = normalized[:, channels]
                net, head = build_model(device)
                net.load_state_dict(checkpoint["net_state_dict"])
                head.load_state_dict(checkpoint["head_state_dict"])
                net.eval()
                head.eval()
                probabilities, embeddings = forward(net, head, masked, device)
                target_predictions = knn_predictions(
                    embeddings, y, np.asarray(checkpoint["fit_index"]), query_index,
                    int(checkpoint["best_k"]),
                )
                base_accuracy = float(
                    (probabilities[query_index].argmax(1) == y[query_index]).mean()
                )
                target_accuracy = float((target_predictions == y[query_index]).mean())
                if abs(base_accuracy - float(checkpoint["base_accuracy"])) > 1e-12:
                    raise ValueError(f"base mismatch after reload: {checkpoint_path}")
                if abs(target_accuracy - float(checkpoint["target_accuracy"])) > 1e-12:
                    raise ValueError(f"target mismatch after reload: {checkpoint_path}")

                probabilities_dump[f"train_{rung}"] = probabilities[train_index]
                probabilities_dump[f"sess1_{rung}"] = probabilities[query_index]
                embeddings_dump[f"train_{rung}"] = embeddings[train_index]
                embeddings_dump[f"sess1_{rung}"] = embeddings[query_index]
                targets_dump[f"sess1_{rung}"] = target_predictions
                base_by_rung[rung].append(base_accuracy)
                target_by_rung[rung].append(target_accuracy)
                if rung == 16:
                    full_accuracy.append(float(checkpoint["full_accuracy"]))

            np.savez_compressed(subject_out / "preds.npz", **probabilities_dump)
            np.savez_compressed(subject_out / "masked_embeddings.npz", **embeddings_dump)
            np.savez_compressed(subject_out / "target_predictions.npz", **targets_dump)
            (subject_out / "adapter.json").write_text(json.dumps({
                "artifact_status": "retrained_checkpoint_outputs_not_original_paper_checkpoint",
                "seed": seed,
                "subject": subject,
                "rungs": RUNGS,
                "train_split": "repetitions 1,3,4,6",
                "sess1_split": "repetitions 2,5",
                "retrieval_features": "exact masked CNN embeddings",
            }, indent=2) + "\n")
            print(f"exported seed {seed} subject {subject:02d}", flush=True)

        summary = {
            "seed": seed,
            "artifact_status": "retrained_not_original_paper_checkpoint",
            "full41": float(np.mean(full_accuracy)),
            "base": {str(rung): float(np.mean(base_by_rung[rung])) for rung in RUNGS},
            "target": {str(rung): float(np.mean(target_by_rung[rung])) for rung in RUNGS},
            "base_subjects": {str(rung): base_by_rung[rung] for rung in RUNGS},
            "target_subjects": {str(rung): target_by_rung[rung] for rung in RUNGS},
        }
        (seed_out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        all_summaries[f"seed{seed}"] = summary

    (args.out / "summary.json").write_text(json.dumps(all_summaries, indent=2) + "\n")
    (args.out / "adapter.json").write_text(json.dumps({
        "host": "ninapro_db5_41class_subject_specific_cnn",
        "artifact_status": "two_retrained_seeds_not_original_paper_checkpoint",
        "seeds": args.seeds,
        "rungs": RUNGS,
        "adapter": "GUARD/hosts/ninapro.py",
    }, indent=2) + "\n")
    print("NINAPRO_EXPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
