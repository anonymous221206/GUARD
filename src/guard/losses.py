"""Canonical losses and their matching link functions.

A loss is *canonical* for a link ``m`` when the gradient of the expected loss
with respect to the output ``z`` is exactly the residual ``m(z) - t(y)``.
Every result in the paper rests on that identity, so the three losses we
support are defined here together with the link they pair with, and nothing
else in the codebase is allowed to hard-code a loss.

============  =========================  =====================
loss          link ``m(z)``              target ``t(y)``
============  =========================  =====================
cross_entropy softmax(z)                 one-hot(y)
bernoulli     sigmoid(z), per coordinate  y in {0,1} per coordinate
squared       identity                   y
============  =========================  =====================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class Loss:
    """A canonical loss together with everything the pipeline needs from it."""

    name: str
    #: per-sample loss given predicted probabilities ``p`` and labels ``y``
    pointwise: Callable[[np.ndarray, np.ndarray], np.ndarray]
    #: ``t(y)``: the target the residual is measured against
    target: Callable[[np.ndarray, int], np.ndarray]
    #: whether outputs are a simplex per sample (softmax) or independent (sigmoid)
    simplex: bool

    def __call__(self, p: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.pointwise(p, y)


def _ce_pointwise(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    # float64 for the same reason as ``_bernoulli_pointwise``
    q = np.asarray(p[np.arange(len(y)), y], dtype=np.float64)
    return -np.log(np.clip(q, EPS, None))


def _ce_target(y: np.ndarray, n_out: int) -> np.ndarray:
    return np.eye(n_out)[y]


def _bernoulli_pointwise(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    # float64 first: np.clip keeps the input dtype, and in float32 ``1 - EPS``
    # rounds to exactly 1.0, so the clip is a no-op and log(1 - q) is -inf.
    # A saturated probability then gives 0 * -inf = nan, which silently drops
    # out of the harm count because ``nan > delta`` is False.
    q = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean(axis=1)


def _bernoulli_target(y: np.ndarray, n_out: int) -> np.ndarray:
    return np.asarray(y, dtype=np.float64)


def _squared_pointwise(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = p - np.asarray(y, dtype=np.float64).reshape(p.shape)
    return 0.5 * (d ** 2).sum(axis=1)


def _brier_pointwise(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = p - np.eye(p.shape[1])[np.asarray(y, dtype=int)]
    return 0.5 * (d ** 2).sum(axis=1)


CROSS_ENTROPY = Loss("cross_entropy", _ce_pointwise, _ce_target, simplex=True)
BERNOULLI = Loss("bernoulli", _bernoulli_pointwise, _bernoulli_target, simplex=False)
SQUARED = Loss("squared", _squared_pointwise, _bernoulli_target, simplex=False)
#: the squared loss against a one-hot target, i.e. Brier.  Same family as
#: cross-entropy and the same decision rule, but a bounded loss rather than an
#: unbounded one -- which is the point of varying it.
BRIER = Loss("brier", _brier_pointwise, _ce_target, simplex=True)

REGISTRY = {loss.name: loss for loss in (CROSS_ENTROPY, BERNOULLI, SQUARED, BRIER)}


def get(name: str) -> Loss:
    """Look a loss up by name, failing loudly on a typo."""
    if name not in REGISTRY:
        raise KeyError(f"unknown loss {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]


def accuracy(p: np.ndarray, y: np.ndarray, loss: Loss) -> float:
    """Decision accuracy under the loss's decision rule."""
    if loss.simplex:
        return float((p.argmax(1) == y).mean())
    return float(((p > 0.5).astype(int) == np.asarray(y, dtype=int)).mean())


def accuracy_nonzero(p: np.ndarray, y: np.ndarray, loss: Loss,
                     raw_labels: np.ndarray) -> float:
    """Accuracy restricted to non-neutral samples.

    CMU-MOSEI reports binary accuracy over samples whose continuous sentiment
    is non-zero; neutral clips are excluded.  ``raw_labels`` carries those
    continuous values for the same rows as ``y``, so the caller keeps them
    alongside the binarised labels the pipeline uses.
    """
    keep = np.asarray(raw_labels).reshape(-1) != 0
    if not keep.any():
        return float("nan")
    if loss.simplex:
        return float((p[keep].argmax(1) == np.asarray(y)[keep]).mean())
    return float(((p[keep] > 0.5).astype(int)
                  == np.asarray(y, dtype=int)[keep]).mean())


def f1_macro(p: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> float:
    """Macro-averaged F1 for multi-label outputs."""
    pred = (p > threshold).astype(float)
    y = np.asarray(y, dtype=float)
    tp = (pred * y).sum(0)
    fp = (pred * (1 - y)).sum(0)
    fn = ((1 - pred) * y).sum(0)
    denom = 2 * tp + fp + fn
    return float(np.where(denom > 0, 2 * tp / np.maximum(denom, EPS), 0.0).mean())


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUROC; returns NaN when one class is absent."""
    order = np.argsort(score)
    rank = np.empty(len(score), dtype=float)
    rank[order] = np.arange(1, len(score) + 1)
    n_pos = float((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((rank[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
