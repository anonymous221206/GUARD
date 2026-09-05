#!/usr/bin/env python3
"""Adapter for hosts we consume through saved predictions rather than code.

The affective-computing hosts (MoMKE, TMDC, CMAD, IMDer, LNLN) are used exactly
as their authors released them.  We run each once, save its outputs under every
modality mask, and never touch it again -- which is the deployment situation the
method is written for.

Expected layout, one directory per host::

    <host>/
      raw_features.npz   per-modality features: train_a, train_t, train_v,
                         <dep>_a, <dep>_t, <dep>_v, train_y, <dep>_y
      preds.npz          host outputs per mask: train_<mask>, <dep>_<mask>
                         for every mask in {a,v,t,av,at,tv,atv}, plus *_y

``<dep>`` is the deployment split, named either ``sess1`` or ``test`` depending on
the upstream project; the file may also be called ``student_preds.npz`` when the
outputs come from a student we distilled, and mask letters may appear in any order
(``tav`` for ``atv``).  All three are resolved on load.

The deployment population is where calibration and evaluation are both
drawn from it, so the certificate's exchangeability assumption holds.  The
retrieval pool may come from either population; ``--pool`` selects which, and
the difference is reported rather than hidden.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MASKS = ("a", "v", "t", "av", "at", "tv", "atv")
FULL_MASK = "atv"


PRED_NAMES = ("preds.npz", "student_preds.npz")
DEPLOY_SPLITS = ("sess1", "test")


def _canonical(pred_files, split: str, mask: str) -> str:
    """Find the key for one (split, mask), tolerating a different letter order.

    Dumps written by different hosts spell the same mask differently -- ``tav``
    against ``atv``, ``ta`` against ``at`` -- because each project ordered the
    modalities its own way.  The mask is a set, so we match on the sorted letters
    rather than forcing every dump to be rewritten.
    """
    want = "".join(sorted(mask))
    for key in pred_files:
        if not key.startswith(split + "_"):
            continue
        tail = key[len(split) + 1:]
        if "".join(sorted(tail)) == want:
            return key
    raise KeyError(f"no key for split {split!r} mask {mask!r}")


def load(host_dir: Path) -> dict:
    raw = np.load(host_dir / "raw_features.npz")
    for name in PRED_NAMES:
        if (host_dir / name).exists():
            pred = np.load(host_dir / name)
            break
    else:
        raise FileNotFoundError(f"{host_dir}: expected one of {PRED_NAMES}")

    dep = next((d for d in DEPLOY_SPLITS
                if any(k.startswith(d + "_") for k in pred.files)), None)
    if dep is None:
        raise KeyError(f"{host_dir}: no deployment split among {DEPLOY_SPLITS}")

    missing = []
    for split in ("train", dep):
        for m in MASKS:
            try:
                _canonical(pred.files, split, m)
            except KeyError:
                missing.append(f"{split}_{m}")
    if missing:
        raise KeyError(f"{host_dir}: predictions missing {missing}")
    return {"raw": raw, "pred": pred, "dep": dep}


#: Upstream projects name the same modality differently.  First key present wins.
MODALITY_ALIASES = {"a": ("a", "ac", "audio"), "v": ("v", "vis", "video"),
                    "t": ("t", "txt", "text")}


def features_for_mask(raw, split: str, mask: str, report: list | None = None):
    """Concatenate only the modalities the mask leaves observed, z-scored on train.

    A modality in the mask can have predictions but no stored representation: the
    released CMU-MOSEI dump carries audio and visual features only, because text is
    the modality that goes missing there and the retrieval space is built from what
    remains.  Such a modality is skipped and **named in** ``report`` -- never
    dropped silently, because a silent substitution here changes the retrieval
    space and therefore every number downstream.
    """
    out = []
    for m in mask:
        key = next((f"{split}_{a}" for a in MODALITY_ALIASES.get(m, (m,))
                    if f"{split}_{a}" in raw.files), None)
        if key is None:
            if report is not None and m not in report:
                report.append(m)
            continue
        base = key[len(split) + 1:]
        f = raw[key].astype(np.float64)
        mu = raw[f"train_{base}"].mean(0)
        sd = raw[f"train_{base}"].std(0) + 1e-9
        out.append((f - mu) / sd)
    if not out:
        raise KeyError(f"mask {mask!r} has no stored features in split {split!r}")
    return np.concatenate(out, 1)


def build(host_dir: Path, mask: str, pool_from: str = "train", non0: bool = False):
    """Return ``(probs, features, labels, richer_probs, pool_slice, deploy_slice)``.

    Everything is concatenated into one index space ``[pool | deployment]`` so a
    :class:`guard.splits.Split` can address it directly.

    ``non0`` drops the samples whose sentiment label is exactly zero.  This is the
    standard CMU-MOSEI binary convention and it is what the paper's MOSEI rows use:
    a zero score is neutral, so forcing it to one side of a binary decision scores
    a genuinely ambiguous sample as if it had a side.  On the released CMAD dump
    that is 1023 of 4643 test samples, and leaving them in costs about 0.14
    accuracy.  It must stay **off** for label sets where 0 is a real class, such as
    the four-way IEMOCAP task -- there it would delete a class outright.
    """
    d = load(host_dir)
    raw, pred, dep = d["raw"], d["pred"], d["dep"]
    y_train, y_dep = raw["train_y"], raw[f"{dep}_y"]

    skipped: list = []
    #: A regression host stores one scalar per sample; its labels are the raw
    #: sentiment scores and must be binarised the same way its outputs are.
    scalar_host = pred[_canonical(pred.files, dep, mask)].ndim == 1

    def P(split, mask):
        """Predictions for one (split, mask), as a distribution over classes.

        A regression host stores one scalar per sample -- the released CMU-MOSEI
        student is of this kind.  Binary accuracy on that task is the sign test, so
        the scalar is mapped through a logistic to the two-class simplex, matching
        the convention the reported numbers were produced under.  Hosts that already
        store a distribution are passed through untouched.
        """
        a = pred[_canonical(pred.files, split, mask)].astype(np.float64)
        if a.ndim == 1:
            q = 1.0 / (1.0 + np.exp(-a))
            return np.stack([1.0 - q, q], 1)
        return a

    if pool_from == "train":
        pool_probs = P("train", mask)
        pool_feats = features_for_mask(raw, "train", mask, skipped)
        pool_rich = P("train", FULL_MASK)
        pool_y = y_train
    elif pool_from == "deployment":
        pool_probs = P(dep, mask)
        pool_feats = features_for_mask(raw, dep, mask, skipped)
        pool_rich = P(dep, FULL_MASK)
        pool_y = y_dep
    else:
        raise ValueError(f"unknown pool source {pool_from!r}")

    dep_probs = P(dep, mask)
    dep_feats = features_for_mask(raw, dep, mask, skipped)
    dep_rich = P(dep, FULL_MASK)

    if non0:
        keep_pool, keep_dep = pool_y != 0, y_dep != 0
        pool_probs, pool_feats = pool_probs[keep_pool], pool_feats[keep_pool]
        pool_rich, pool_y = pool_rich[keep_pool], pool_y[keep_pool]
        dep_probs, dep_feats = dep_probs[keep_dep], dep_feats[keep_dep]
        dep_rich, y_dep = dep_rich[keep_dep], y_dep[keep_dep]

    if skipped:
        print(f"    note: mask {mask!r} has no stored features for {sorted(skipped)}; "
              f"retrieval uses the remaining modalities")

    if scalar_host:
        pool_y = (pool_y > 0).astype(np.int64)
        y_dep = (y_dep > 0).astype(np.int64)

    probs = np.concatenate([pool_probs, dep_probs])
    feats = np.concatenate([pool_feats, dep_feats])
    labels = np.concatenate([pool_y, y_dep])
    richer = np.concatenate([pool_rich, dep_rich])
    return probs, feats, labels, richer, len(pool_y), len(y_dep)
