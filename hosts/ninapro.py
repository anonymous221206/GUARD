"""Adapter for retrained NinaPro DB5 41-class ladder artifacts.

NinaPro has sixteen synchronous electrodes, not three independent a/t/v
modalities.  More importantly, the paper's target retrieves with the exact
embedding produced by the frozen CNN after the complete electrode mask is
applied.  ``hosts.dumps`` reconstructs features by concatenating modality
arrays and therefore cannot preserve that experiment.  This adapter consumes
the exported per-rung masked embeddings directly and keeps subjects separate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

RUNGS = (16, 12, 8, 6, 4)
FULL_RUNG = 16


def load(subject_dir: Path) -> dict:
    """Open and validate one seed/subject artifact directory."""
    predictions = np.load(subject_dir / "preds.npz")
    embeddings = np.load(subject_dir / "masked_embeddings.npz")
    targets = np.load(subject_dir / "target_predictions.npz")
    needed = [f"{split}_{rung}" for split in ("train", "sess1") for rung in RUNGS]
    missing = [key for key in needed
               if key not in predictions.files or key not in embeddings.files]
    missing += [f"sess1_{rung}" for rung in RUNGS
                if f"sess1_{rung}" not in targets.files]
    if missing or "train_y" not in predictions.files or "sess1_y" not in predictions.files:
        raise KeyError(f"{subject_dir}: missing NinaPro arrays {missing}")
    return {"pred": predictions, "embedding": embeddings, "target": targets}


def build(subject_dir: Path, rung: int, pool_from: str = "train"):
    """Return GUARD host arrays using exact frozen masked-CNN embeddings."""
    if rung not in RUNGS:
        raise ValueError(f"unknown NinaPro rung {rung!r}")
    data = load(subject_dir)
    pred, embedding = data["pred"], data["embedding"]
    train_y, sess1_y = pred["train_y"], pred["sess1_y"]
    if pool_from == "train":
        pool_probs = pred[f"train_{rung}"]
        pool_features = embedding[f"train_{rung}"]
        pool_richer = pred[f"train_{FULL_RUNG}"]
        pool_y = train_y
    elif pool_from == "deployment":
        pool_probs = pred[f"sess1_{rung}"]
        pool_features = embedding[f"sess1_{rung}"]
        pool_richer = pred[f"sess1_{FULL_RUNG}"]
        pool_y = sess1_y
    else:
        raise ValueError(f"unknown pool source {pool_from!r}")
    deploy_probs = pred[f"sess1_{rung}"]
    deploy_features = embedding[f"sess1_{rung}"]
    deploy_richer = pred[f"sess1_{FULL_RUNG}"]
    return (
        np.concatenate([pool_probs, deploy_probs]).astype(np.float64),
        np.concatenate([pool_features, deploy_features]).astype(np.float64),
        np.concatenate([pool_y, sess1_y]),
        np.concatenate([pool_richer, deploy_richer]).astype(np.float64),
        len(pool_y),
        len(sess1_y),
    )
