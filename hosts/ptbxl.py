"""PTB-XL adapter for frozen lead-masked ``resnet1d_wang`` dumps.

Unlike ``hosts.dumps``, this module cannot reconstruct retrieval features by
concatenating ``a/t/v`` arrays.  PTB-XL supplies twelve synchronous ECG leads,
which are only grouped into ``a`` (limb leads), ``t`` (V1--V3), and ``v``
(V4--V6) as a documented missing-lead adapter.  GUARD retrieves in the frozen
network's penultimate embedding after the *entire requested lead mask* was
applied.  Therefore ``masked_embeddings.npz`` is required and used directly.

Labels and probabilities are five-dimensional multi-label vectors; callers must
use GUARD's Bernoulli loss, not the simplex cross-entropy default.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MASKS = ("a", "v", "t", "av", "at", "tv", "atv")
FULL_MASK = "atv"


def load(host_dir: Path) -> dict:
    """Open and validate an artifact produced by ``export_ptbxl_dumps.py``."""
    raw = np.load(host_dir / "raw_features.npz")
    pred = np.load(host_dir / "preds.npz")
    embedding = np.load(host_dir / "masked_embeddings.npz")
    needed = [f"{split}_{mask}" for split in ("train", "sess1") for mask in MASKS]
    missing = [key for key in needed if key not in pred.files or key not in embedding.files]
    labels = [f"{split}_y" for split in ("train", "sess1")]
    if missing or any(key not in raw.files for key in labels):
        raise KeyError(f"{host_dir}: missing predictions/embeddings={missing}")
    return {"raw": raw, "pred": pred, "embedding": embedding}


def build(host_dir: Path, mask: str, pool_from: str = "train"):
    """Return the same tuple as ``hosts.dumps.build`` using exact mask embeddings."""
    if mask not in MASKS:
        raise ValueError(f"unknown PTB-XL mask {mask!r}")
    data = load(host_dir)
    raw, pred, embedding = data["raw"], data["pred"], data["embedding"]
    train_y, sess1_y = raw["train_y"], raw["sess1_y"]
    if pool_from == "train":
        pool_probs, pool_feats, pool_rich, pool_y = (
            pred[f"train_{mask}"], embedding[f"train_{mask}"], pred[f"train_{FULL_MASK}"], train_y)
    elif pool_from == "deployment":
        pool_probs, pool_feats, pool_rich, pool_y = (
            pred[f"sess1_{mask}"], embedding[f"sess1_{mask}"], pred[f"sess1_{FULL_MASK}"], sess1_y)
    else:
        raise ValueError(f"unknown pool source {pool_from!r}")
    deploy_probs, deploy_feats, deploy_rich = (
        pred[f"sess1_{mask}"], embedding[f"sess1_{mask}"], pred[f"sess1_{FULL_MASK}"])
    return (np.concatenate([pool_probs, deploy_probs]).astype(np.float64),
            np.concatenate([pool_feats, deploy_feats]).astype(np.float64),
            np.concatenate([pool_y, sess1_y]),
            np.concatenate([pool_rich, deploy_rich]).astype(np.float64),
            len(pool_y), len(sess1_y))
