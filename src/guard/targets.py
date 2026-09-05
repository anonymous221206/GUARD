"""Retrieval targets: the two ways of estimating pi_S(x) = E[t(Y) | X_S = x].

Both average something over the same K nearest neighbours, found using only the
*observed* part of the input.  They differ in what they average:

``hard``       the neighbours' one-hot labels.  Needs a label for every pool
               element, and inherits label noise.
``cross_mask`` the frozen host's own outputs under a *richer* mask, evaluated on
               the neighbours.  Needs no labels at all.  Proposition 4 shows it
               dominates ``hard`` when the richer host is conditionally correct.

The precondition matters in practice: when the richer host is *worse* than the
poorer one on the deployment distribution, ``cross_mask`` cannot help, and
:func:`richer_is_richer` says so before any correction is attempted.
"""

from __future__ import annotations

import numpy as np


def knn_average(
    query: np.ndarray,
    pool: np.ndarray,
    values: np.ndarray,
    k: int,
    chunk: int = 1024,
    weighting: str = "uniform",
) -> np.ndarray:
    """Average ``values`` over the ``k`` nearest pool rows for each query row.

    Distances are Euclidean in whatever space the caller passes in; standardise
    beforehand if the coordinates are on different scales.

    ``weighting='uniform'`` gives every neighbour the same weight.
    ``weighting='distance'`` weights neighbour :math:`i` by
    :math:`\exp(-d_i/\tau)` with :math:`\tau` the median distance to the
    :math:`k`-th neighbour, so a query whose neighbourhood is tight is not
    diluted by its furthest members. Which one to use is a choice to be made on
    the fit split, not a default to be trusted.
    """
    if len(pool) < k:
        raise ValueError(f"pool has {len(pool)} rows but k={k}")
    pool_sq = (pool ** 2).sum(1)
    out = np.empty((len(query), values.shape[1]), dtype=np.float64)
    for i in range(0, len(query), chunk):
        q = query[i:i + chunk]
        d2 = (q ** 2).sum(1)[:, None] + pool_sq[None, :] - 2.0 * (q @ pool.T)
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        if weighting == "uniform":
            out[i:i + chunk] = values[idx].mean(1)
        elif weighting == "distance":
            d = np.sqrt(np.maximum(np.take_along_axis(d2, idx, 1), 0.0))
            tau = np.median(d.max(1)) + 1e-12
            w = np.exp(-d / tau)
            w /= w.sum(1, keepdims=True)
            out[i:i + chunk] = (values[idx] * w[:, :, None]).sum(1)
        else:
            raise ValueError(f"unknown weighting {weighting!r}")
    return out


def standardise(reference: np.ndarray):
    """Return a function that z-scores features using ``reference``'s statistics."""
    mu = reference.mean(0)
    sd = reference.std(0) + 1e-9
    return lambda x: ((x - mu) / sd).astype(np.float64)


def retrieval_space(reference: np.ndarray, kind: str = "standardise"):
    """Return a map into the space retrieval measures distances in.

    ``standardise``  z-score each coordinate using ``reference``'s statistics.
    ``cosine``       z-score, then project onto the unit sphere, so distance
                     depends on direction alone and not on feature magnitude.
    ``whiten``       z-score, then rotate onto ``reference``'s principal axes
                     and equalise their scales, keeping 95% of the variance.
                     Useful when a few coordinates dominate the Euclidean
                     distance for reasons unrelated to the label.

    All three are fitted on ``reference`` only, which must be the retrieval
    pool: fitting them on deployment data would leak it into the target.
    """
    mu = reference.mean(0)
    sd = reference.std(0) + 1e-9
    z = lambda x: (np.asarray(x, dtype=np.float64) - mu) / sd
    if kind == "standardise":
        return z
    if kind == "cosine":
        def _cos(x):
            v = z(x)
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        return _cos
    if kind == "whiten":
        ref = z(reference)
        centre = ref.mean(0)
        _, sv, vt = np.linalg.svd(ref - centre, full_matrices=False)
        keep = int(np.searchsorted(np.cumsum(sv ** 2) / np.sum(sv ** 2), 0.95) + 1)
        scale = np.maximum(sv[:keep] / np.sqrt(len(ref)), 1e-8)
        basis = vt[:keep].T
        return lambda x: (z(x) - centre) @ basis / scale
    raise ValueError(f"unknown retrieval space {kind!r}")


def temper(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Raise or flatten a host's confidence before it is blended.

    A host calibrated for one deployment is rarely calibrated for a degraded
    one, and the blend weight cannot fix a scale error on its own. Temperature
    is chosen on the fit split like every other choice here.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, None)
    logits = np.log(p) / float(temperature)
    logits -= logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


def hard_label_values(pool_labels: np.ndarray, n_out: int, simplex: bool) -> np.ndarray:
    """``t(y)`` for every pool element -- the values ``hard`` retrieval averages."""
    if simplex:
        return np.eye(n_out)[np.asarray(pool_labels, dtype=int)]
    return np.asarray(pool_labels, dtype=np.float64)


def cross_mask_values(richer_outputs_on_pool: np.ndarray) -> np.ndarray:
    """``m_T`` for every pool element -- the values ``cross_mask`` averages.

    No labels appear here.  Note the pool should ideally be data the host was
    not trained on: on a memorised pool ``m_T`` approaches the labels, and the
    label-free claim becomes hollow even though it stays literally true.
    """
    return np.asarray(richer_outputs_on_pool, dtype=np.float64)


def richer_is_richer(
    richer_probs: np.ndarray,
    poorer_probs: np.ndarray,
    labels: np.ndarray,
    simplex: bool = True,
) -> dict:
    """Pre-flight check for the cross-mask precondition. Necessary, not sufficient.

    Compares the richer and poorer hosts *on the deployment distribution*.
    Proposition 4 needs the richer host to be conditionally correct, and this
    tests only that it is more accurate -- a strictly weaker property. Read the
    result accordingly:

    * ``precondition_met=False`` is decisive. The richer host cannot be
      conditionally correct if it is not even more accurate, and cross-mask
      lost essentially all of the label-based gain in every run where this
      fired (5 of 5 in our measurements).
    * ``precondition_met=True`` clears the necessary condition and nothing
      more. Cross-mask still lost most of the label-based gain in 13 of the 18
      runs where the check passed, because clearing an inequality is far short
      of being right almost everywhere.

    What tracked the outcome in those runs was the richer host's *absolute*
    accuracy rather than its margin over the poorer one: the runs where
    cross-mask held its gain had a richer host well above the runs where it did
    not, while the margin between the two hosts overlapped across both groups.
    That is the shape conditional correctness implies, so treat a richer host
    that is only marginally better as unproven rather than cleared. No threshold
    is offered here; the separation was measured over too few dataset-condition
    cells to fix one.

    Must be evaluated on deployment-distribution data. Measuring it on a
    different population can green-light cross-mask wrongly -- we observed
    exactly that on a cross-subject benchmark.
    """
    acc = (lambda p: float((p.argmax(1) == labels).mean())) if simplex else (
        lambda p: float(((p > 0.5).astype(int) == np.asarray(labels, int)).mean()))
    a_rich, a_poor = acc(richer_probs), acc(poorer_probs)
    return {
        "richer_accuracy": a_rich,
        "poorer_accuracy": a_poor,
        "margin": a_rich - a_poor,
        "precondition_met": bool(a_rich > a_poor),
    }
