"""Certify: apply the correction only where the loss increase is provably small.

The gate is split-conformal least-ambiguous-set (LAC).  On the calibration set
we take the ``1-alpha`` quantile of ``1 - p_corrected[true label]``; at test
time the plausible label set is everything above that threshold, and we apply
the correction only when *every* plausible label would suffer at most ``delta``
extra loss.

What is guaranteed
------------------
``P(Delta_loss > delta  AND  applied) <= alpha`` -- a **joint** statement over
the draw of the test point.  It is *not* a statement conditional on applying:
``cond_harm`` below is routinely several times ``alpha`` and that is not a
violation.  Report both columns.

The guarantee is on the **loss**.  On a thresholded downstream metric such as
macro-F1, a certified-safe loss change can still move the metric the wrong way;
:func:`certify` therefore reports the loss and the metric separately.
"""

from __future__ import annotations

import numpy as np

from .losses import EPS, Loss


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile with the standard (n+1) correction.

    Returns ``+inf`` when ``n`` is too small to support the level, i.e. when
    ``ceil((n+1)(1-alpha)) > n``.  That is the honest answer -- no finite
    threshold certifies anything at this level with this many points -- and it
    makes every label plausible, so the gate refuses.  Clamping to the largest
    observed score instead would shrink the plausible set and *loosen* the gate.
    """
    n = len(scores)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(np.quantile(scores, k / n))


def _plausible_simplex(corrected: np.ndarray, q_hat: float) -> np.ndarray:
    keep = (1 - corrected) <= q_hat
    keep[~keep.any(1)] = True          # never return an empty set
    return keep


def certify(
    corrected_conf: np.ndarray,
    labels_conf: np.ndarray,
    corrected_test: np.ndarray,
    base_test: np.ndarray,
    loss: Loss,
    alpha: float,
    delta: float,
) -> dict:
    """Decide where to apply, and account for the harm actually incurred.

    Returns the apply mask plus the calibrated quantile; the caller combines it
    with labels to compute realised metrics (see :mod:`guard.pipeline`).
    """
    if loss.simplex:
        p_true = corrected_conf[np.arange(len(labels_conf)), np.asarray(labels_conf, int)]
        q_hat = conformal_quantile(1 - p_true, alpha)
        plausible = _plausible_simplex(corrected_test, q_hat)
        # worst-case extra loss over the plausible labels
        extra = (-np.log(np.clip(np.asarray(corrected_test, dtype=np.float64), EPS, None))
                 + np.log(np.clip(np.asarray(base_test, dtype=np.float64), EPS, None)))
        worst = np.where(plausible, extra, -np.inf).max(1)
    else:
        # independent coordinates: score is the worst coordinate, so the product
        # set covers the whole label vector with probability >= 1 - alpha, and
        # because the loss is additive the worst case is the sum of per-
        # coordinate worst cases -- exact, no union bound needed.
        y = np.asarray(labels_conf, dtype=np.float64)
        p_true = y * corrected_conf + (1 - y) * (1 - corrected_conf)
        q_hat = conformal_quantile((1 - p_true).max(1), alpha)
        ok_one = (1 - corrected_test) <= q_hat
        ok_zero = corrected_test <= q_hat
        neither = ~(ok_one | ok_zero)
        ok_one |= neither & (corrected_test >= 0.5)
        ok_zero |= neither & (corrected_test < 0.5)
        # float64 first: in float32 ``1 - EPS`` rounds to 1.0, the clip becomes a
        # no-op and log(1 - c) is -inf, which poisons the worst-case bound.
        c = np.clip(np.asarray(corrected_test, dtype=np.float64), EPS, 1 - EPS)
        b = np.clip(np.asarray(base_test, dtype=np.float64), EPS, 1 - EPS)
        d_one = -np.log(c) + np.log(b)
        d_zero = -np.log(1 - c) + np.log(1 - b)
        worst = np.maximum(np.where(ok_one, d_one, -np.inf),
                           np.where(ok_zero, d_zero, -np.inf)).sum(1) / c.shape[1]
    return {"apply": worst <= delta, "q_hat": float(q_hat),
            "worst_case_extra_loss": worst}


def harm_accounting(
    delta_loss: np.ndarray, apply: np.ndarray, delta: float
) -> dict:
    """Joint and conditional harm, reported side by side on purpose."""
    return {
        "joint_harm": float((delta_loss > delta).mean()),
        "cond_harm": float((delta_loss[apply] > delta).mean()) if apply.any() else 0.0,
        "apply_rate": float(apply.mean()),
    }
