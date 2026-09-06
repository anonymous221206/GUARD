"""Certify: a learned action score, thresholded by conformal risk control.

The score orders interventions; the threshold decides how many to make. They are
fitted on different splits on purpose: the score on ``D_fit``, alongside the blend
weight, and the threshold on ``D_conf``, which nothing before it has touched. The
guarantee comes from conformal risk control (Angelopoulos et al., ICLR 2024) and
holds for whatever the score learned, so a poor score costs gain and not safety.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def action_features(base: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Five deployment-computable summaries of the (base, target) pair.

    None of them reads a label, so the same code runs at calibration and at
    deployment: the base model's confidence and entropy, the target's
    confidence, whether the two already agree, and how far apart they are.
    """
    b = np.clip(np.asarray(base, dtype=np.float64), EPS, None)
    t = np.asarray(target, dtype=np.float64)
    return np.column_stack([
        b.max(1),
        -(b * np.log(b)).sum(1),
        t.max(1),
        (np.abs(b - t).mean(1) < 0.1).astype(float),
        np.abs(b - t).sum(1),
    ])


def fit_action_score(base_fit, target_fit, corrected_fit, labels_fit, loss):
    """Fit the scorer on D_fit, against whether the correction lowered the loss."""
    from sklearn.linear_model import LogisticRegression

    z = (loss(corrected_fit, labels_fit) < loss(base_fit, labels_fit)).astype(int)
    x = action_features(base_fit, target_fit)
    if len(set(z.tolist())) < 2:            # nothing to learn from
        return None
    return LogisticRegression(max_iter=2000).fit(x, z)


def score(model, base, target) -> np.ndarray:
    """Higher means the correction is more worth applying."""
    if model is None:
        return np.zeros(len(base))
    return model.predict_proba(action_features(base, target))[:, 1]


def crc_threshold(score_conf: np.ndarray, harmful_conf: np.ndarray,
                  alpha: float) -> float | None:
    """The most permissive threshold conformal risk control admits.

    The deployed rule is ``r > lambda``, which makes the loss non-increasing and
    right-continuous in ``lambda`` as the theorem requires. With a loss bounded by
    one, its calibration condition ``n/(n+1) * Rhat + 1/(n+1) <= alpha`` is
    ``Rhat <= alpha - (1-alpha)/n``. The risk is evaluated on the set the rule
    actually admits, so tied scores are charged for.
    """
    n = len(score_conf)
    level = alpha - (1.0 - alpha) / n
    if level <= 0:
        return None
    s = np.asarray(score_conf, dtype=np.float64)
    h = np.asarray(harmful_conf, dtype=float)
    # candidates just below each observed score, so that ``> c`` admits it
    cand = np.unique(s)
    cand = np.concatenate([np.nextafter(cand, -np.inf), [np.nextafter(s.max(), np.inf)]])
    risk = np.array([float(h[s > c].sum()) / n for c in cand])
    ok = np.nonzero(risk <= level)[0]
    if len(ok) == 0:
        return None
    return float(cand[ok[0]])          # smallest admissible threshold = most permissive


def _tighten(scores, gain, lam):
    """The most conservative admissible threshold that is still within one SE of the best.

    ``gain`` is the per-sample loss improvement, positive when the correction
    helps. Thresholds below ``lam`` are inadmissible; among the rest we take the
    total improvement, keep the best, and then step back to the most conservative
    threshold whose improvement is within one standard error of it.
    """
    cand = np.unique(scores[scores > lam])
    cand = np.concatenate([[lam], cand]) if len(cand) else np.array([lam])
    tot, se = [], []
    for c in cand:
        g = gain[scores > c]
        tot.append(float(g.sum()))
        se.append(float(g.std(ddof=1) * np.sqrt(len(g))) if len(g) > 1 else 0.0)
    tot, se = np.array(tot), np.array(se)
    b = int(np.argmax(tot))
    if tot[b] <= 0:                      # nothing worth applying
        return float(np.nextafter(scores.max(), np.inf))
    ok = np.nonzero(tot >= tot[b] - se[b])[0]
    return float(cand[ok[-1]])           # cand is ascending: last = most conservative


def certify_action(model, base_conf, target_conf, corrected_conf, labels_conf,
                   base_test, target_test, loss, alpha, delta,
                   fit=None) -> dict:
    """Calibrate on D_conf, tighten on D_fit, then decide on the deployment split.

    ``fit`` is ``(base, target, corrected, labels)`` on the fit split. Without it
    the calibrated threshold is used as is.
    """
    harmful = (loss(corrected_conf, labels_conf) - loss(base_conf, labels_conf)) > delta
    s_conf = score(model, base_conf, target_conf)
    lam = crc_threshold(s_conf, harmful, alpha)
    lam_crc = lam
    if lam is not None and model is not None and fit is not None:
        b_f, t_f, c_f, y_f = fit
        lam = _tighten(score(model, b_f, t_f), loss(b_f, y_f) - loss(c_f, y_f), lam)
    s_test = score(model, base_test, target_test)
    apply = np.zeros(len(s_test), bool) if lam is None else (s_test > lam)
    return {"apply": apply, "lambda": lam, "lambda_crc": lam_crc,
            "score_test": s_test, "calibration_risk": float(harmful.mean())}


def harm_accounting(
    delta_loss: np.ndarray, apply: np.ndarray, delta: float
) -> dict:
    """Joint and conditional harm, reported side by side on purpose."""
    return {
        "joint_harm": float((delta_loss > delta).mean()),
        "cond_harm": float((delta_loss[apply] > delta).mean()) if apply.any() else 0.0,
        "apply_rate": float(apply.mean()),
    }
