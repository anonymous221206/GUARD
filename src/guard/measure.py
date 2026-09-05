"""Measure: how much of the host's error is a calibration gap at all.

The quantity is the M-weighted energy of the conditional calibration gap

    PH = 0.5 * E[ (m(z) - pi(X))^T M(X) (m(z) - pi(X)) ]

which is unobservable because ``pi`` is.  The neighbour-pair estimator removes
the label-noise term instead of estimating it: with ``u = m - t(y_self)`` and
``v = m - t(y_neighbour)``, and the two labels conditionally independent given
X with the same pi,

    E[u^T M v] = gap^T M gap + O(gap * d_pi)

so no Sigma_t plug-in is needed and there is no large cancellation.

Two practical notes, both learned the hard way:

* ``PH`` is a **point estimate on the loss scale**, not an upper bound and not
  an accuracy.  Do not compare it directly against an accuracy change.
* ``PH`` is provably non-negative, so a **significantly negative** estimate is
  a useful diagnostic: it means neighbours carry systematically *opposite*
  labels, which breaks the smoothness that any retrieval-based correction
  needs.  We saw this on a benchmark built with adversarial near-duplicates.
"""

from __future__ import annotations

import numpy as np


def nearest_neighbour_index(features: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Index of each row's nearest *other* row."""
    features = np.asarray(features, dtype=np.float64)
    sq = (features ** 2).sum(1)
    idx = np.empty(len(features), dtype=np.int64)
    for i in range(0, len(features), chunk):
        j = min(i + chunk, len(features))
        d2 = sq[i:j, None] + sq[None, :] - 2.0 * (features[i:j] @ features.T)
        d2[np.arange(j - i), np.arange(i, j)] = np.inf
        idx[i:j] = d2.argmin(1)
    return idx


def potential_headroom(
    probs: np.ndarray,
    targets_self: np.ndarray,
    targets_pair: np.ndarray,
    scale: float = 1.0,
) -> dict:
    """Neighbour-pair estimate of ``PH`` with its standard error.

    ``scale`` divides the per-sample statistic; pass the number of output
    coordinates for multi-label losses so the result is on the same
    per-coordinate scale as the loss.
    """
    a = probs - targets_self
    b = probs - targets_pair
    per_sample = 0.5 * (a * b).sum(1) / scale
    n = len(per_sample)
    return {
        "ph": float(per_sample.mean()),
        "ph_se": float(per_sample.std(ddof=1) / np.sqrt(n)),
        "n": int(n),
        "negative_and_significant": bool(
            per_sample.mean() < -2 * per_sample.std(ddof=1) / np.sqrt(n)
        ),
    }
