#!/usr/bin/env python3
"""Adapter for hosts we consume through saved predictions rather than code.

The affective-computing hosts (MoMKE, TMDC, CMAD, IMDer, LNLN) are used exactly
as their authors released them.  We run each once, save its outputs under every
modality mask, and never touch it again -- which is the deployment situation the
method is written for.

Expected layout, one directory per host::

    <host>/
      raw_features.npz   per-modality features: train_a, train_t, train_v,
                         sess1_a, sess1_t, sess1_v, train_y, sess1_y
      preds.npz          host outputs per mask: train_<mask>, sess1_<mask>
                         for every mask in {a,v,t,av,at,tv,atv}, plus *_y

``sess1`` is the deployment population: calibration and evaluation are both
drawn from it, so the certificate's exchangeability assumption holds.  The
retrieval pool may come from either population; ``--pool`` selects which, and
the difference is reported rather than hidden.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MASKS = ("a", "v", "t", "av", "at", "tv", "atv")
FULL_MASK = "atv"


def load(host_dir: Path) -> dict:
    raw = np.load(host_dir / "raw_features.npz")
    pred = np.load(host_dir / "preds.npz")
    missing = [f"{s}_{m}" for s in ("train", "sess1") for m in MASKS
               if f"{s}_{m}" not in pred.files]
    if missing:
        raise KeyError(f"{host_dir}: preds.npz is missing {missing}")
    return {"raw": raw, "pred": pred}


def features_for_mask(raw, split: str, mask: str) -> np.ndarray:
    """Concatenate only the modalities the mask leaves observed, z-scored on train."""
    out = []
    for m in mask:
        f = raw[f"{split}_{m}"].astype(np.float64)
        mu = raw[f"train_{m}"].mean(0)
        sd = raw[f"train_{m}"].std(0) + 1e-9
        out.append((f - mu) / sd)
    return np.concatenate(out, 1)


def build(host_dir: Path, mask: str, pool_from: str = "train"):
    """Return ``(probs, features, labels, richer_probs, pool_slice, deploy_slice)``.

    Everything is concatenated into one index space ``[pool | deployment]`` so a
    :class:`guard.splits.Split` can address it directly.
    """
    d = load(host_dir)
    raw, pred = d["raw"], d["pred"]
    y_train, y_dep = raw["train_y"], raw["sess1_y"]

    if pool_from == "train":
        pool_probs = pred[f"train_{mask}"].astype(np.float64)
        pool_feats = features_for_mask(raw, "train", mask)
        pool_rich = pred[f"train_{FULL_MASK}"].astype(np.float64)
        pool_y = y_train
    elif pool_from == "deployment":
        pool_probs = pred[f"sess1_{mask}"].astype(np.float64)
        pool_feats = features_for_mask(raw, "sess1", mask)
        pool_rich = pred[f"sess1_{FULL_MASK}"].astype(np.float64)
        pool_y = y_dep
    else:
        raise ValueError(f"unknown pool source {pool_from!r}")

    dep_probs = pred[f"sess1_{mask}"].astype(np.float64)
    dep_feats = features_for_mask(raw, "sess1", mask)
    dep_rich = pred[f"sess1_{FULL_MASK}"].astype(np.float64)

    probs = np.concatenate([pool_probs, dep_probs])
    feats = np.concatenate([pool_feats, dep_feats])
    labels = np.concatenate([pool_y, y_dep])
    richer = np.concatenate([pool_rich, dep_rich])
    return probs, feats, labels, richer, len(pool_y), len(y_dep)
