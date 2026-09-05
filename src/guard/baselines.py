"""Alternatives the paper compares the certificate against.

Two families, both intentionally reasonable so that the comparison is fair:

**Selectors** decide *where* to apply using a confidence score rather than a
certificate.  Given a target coverage they threshold the score on the
calibration set.  They match the gate's apply rate by construction; the
question is whether they also control harm, and the answer is that they do not
control it at a level you can fix in advance.

**Plug-in estimators** replace the k-NN retrieval target with another estimate
of ``pi(X)``.  The framework does not depend on k-NN; swapping the estimator
changes magnitudes, not conclusions.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------- selectors ---

def score_maxprob(probs: np.ndarray) -> np.ndarray:
    return probs.max(1)


def score_margin(probs: np.ndarray) -> np.ndarray:
    s = np.sort(probs, axis=1)
    return s[:, -1] - s[:, -2]


def score_negentropy(probs: np.ndarray) -> np.ndarray:
    q = np.clip(probs, 1e-12, None)
    return (q * np.log(q)).sum(1)


def score_agreement(base: np.ndarray, corrected: np.ndarray) -> np.ndarray:
    """How much the correction agrees with the host, as a confidence proxy."""
    return -np.abs(base - corrected).sum(1)


SELECTORS = {
    "maxprob": lambda base, corr: score_maxprob(base),
    "margin": lambda base, corr: score_margin(base),
    "negentropy": lambda base, corr: score_negentropy(base),
    "agreement": score_agreement,
    "random": lambda base, corr: np.random.default_rng(0).random(len(base)),
}


def selector_mask(
    name: str,
    base_conf: np.ndarray, corrected_conf: np.ndarray,
    base_test: np.ndarray, corrected_test: np.ndarray,
    coverage: float,
) -> np.ndarray:
    """Apply to the ``coverage`` fraction with the highest score.

    The threshold is set on the calibration split, exactly as the gate's
    quantile is, so the two see the same information.
    """
    fn = SELECTORS[name]
    s_conf = fn(base_conf, corrected_conf)
    s_test = fn(base_test, corrected_test)
    thresh = np.quantile(s_conf, 1.0 - coverage)
    return s_test >= thresh


# -------------------------------------------------------- plug-in targets ---

def kernel_target(query, pool, values, bandwidth: float | None = None,
                  chunk: int = 512) -> np.ndarray:
    """Nadaraya--Watson estimate with a Gaussian kernel.

    ``bandwidth=None`` uses the median pairwise distance of a pool subsample,
    the standard heuristic.  A fixed bandwidth makes this a straw man on data
    whose scale you have not checked.
    """
    pool_sq = (pool ** 2).sum(1)
    if bandwidth is None:
        rng = np.random.default_rng(0)
        sub = pool[rng.permutation(len(pool))[:500]]
        d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
        bandwidth = float(np.sqrt(np.median(d2[d2 > 0])) / 2.0) or 1.0
    out = np.empty((len(query), values.shape[1]))
    for i in range(0, len(query), chunk):
        q = query[i:i + chunk]
        d2 = (q ** 2).sum(1)[:, None] + pool_sq[None, :] - 2.0 * (q @ pool.T)
        w = np.exp(-d2 / (2 * bandwidth ** 2))
        w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
        out[i:i + chunk] = w @ values
    return out


def ridge_target(query, pool, values, ridge: float = 1.0) -> np.ndarray:
    """Linear probe: least squares from features to the target, then predict."""
    x = np.hstack([pool, np.ones((len(pool), 1))])
    a = x.T @ x + ridge * np.eye(x.shape[1])
    w = np.linalg.solve(a, x.T @ values)
    z = np.hstack([query, np.ones((len(query), 1))]) @ w
    z = np.clip(z, 1e-9, None)
    return z / z.sum(1, keepdims=True)


PLUGINS = {"knn": None, "kernel": kernel_target, "ridge": ridge_target}
