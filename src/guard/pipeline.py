"""The whole method in one function: Measure -> Recalibrate -> Certify.

Every experiment in ``experiments/`` calls :func:`run` and does nothing else
numerically.  If you want to know what GUARD does, this file is the answer;
the experiment files only load a frozen host's outputs and choose a split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from typing import Sequence

import numpy as np

from . import certify as _certify
from . import losses as _losses
from . import measure as _measure
from . import targets as _targets
from .splits import Split

BETA_GRID = np.linspace(0.0, 1.0, 41)


@dataclass
class HostOutputs:
    """Everything the frozen host contributes, for one deployment condition."""

    probs: np.ndarray           #: m(z) for every sample, shape (n, d)
    features: np.ndarray        #: representation used for retrieval, shape (n, f)
    labels: np.ndarray          #: ground truth, only read on fit/conf/test
    #: outputs of a *richer* mask on the same samples; needed for cross-mask only
    richer_probs: np.ndarray | None = None
    #: labels before binarisation, if the benchmark scores on a subset of them.
    #: CMU-MOSEI is the case we hit: its published accuracy excludes neutral
    #: samples, so a run that scores over all of them is not comparable with
    #: the numbers the dataset is published under.
    raw_labels: np.ndarray | None = None


@dataclass
class Result:
    condition: str
    target: str
    beta: float
    base_metric: float
    blanket_metric_delta: float
    gate_metric_delta: float
    base_loss: float
    gate_loss_gain: float
    blanket_loss_gain: float
    #: harm the *ungated* correction would incur -- the comparison the gate exists for
    blanket_joint_harm: float
    target_accuracy: float
    ph: float
    ph_se: float
    apply_rate: float
    joint_harm: float
    cond_harm: float
    q_hat: float
    n_pool: int
    n_fit: int
    n_conf: int
    n_test: int
    alpha: float
    delta: float
    k: int
    exchangeable: bool
    notes: list = field(default_factory=list)

    def as_row(self) -> dict:
        return asdict(self)


def _select_beta(m_fit, t_fit, y_fit, loss, objective):
    """Choose the blend weight on the fit split.

    ``objective='loss'`` follows the theory (beta minimises the canonical loss).
    ``objective='metric'`` optimises the deployment metric instead, which is a
    legitimate choice when the metric is thresholded and diverges from the loss.
    """
    if objective == "loss":
        return float(min(BETA_GRID,
                         key=lambda b: loss((1 - b) * m_fit + b * t_fit, y_fit).mean()))
    if objective == "metric":
        return float(max(BETA_GRID,
                         key=lambda b: _losses.f1_macro((1 - b) * m_fit + b * t_fit, y_fit)))
    raise ValueError(f"unknown beta objective {objective!r}")


def run(
    host: HostOutputs,
    split: Split,
    *,
    condition: str = "unnamed",
    loss_name: str = "cross_entropy",
    target: str = "hard",
    k: int = 50,
    alpha: float = 0.2,
    delta: float = 0.05,
    beta_objective: str = "loss",
    metric: str = "accuracy",
    space: str = "standardise",
    weighting: str = "uniform",
    temperature: float = 1.0,
) -> Result:
    """Run GUARD end to end on one frozen host under one deployment condition.

    ``space``, ``weighting`` and ``temperature`` shape the retrieval target and
    the host's confidence. Their defaults reproduce the plain construction; the
    values that earn their keep are chosen on ``split.fit`` by
    :func:`select_on_fit`, never on the test split.
    """
    loss = _losses.get(loss_name)
    n_out = host.probs.shape[1]
    notes: list[str] = []

    warning = split.warn_if_not_exchangeable()
    if warning:
        notes.append("EXCHANGEABILITY: " + warning)

    z = _targets.retrieval_space(host.features[split.pool], space)
    f_pool, f_fit = z(host.features[split.pool]), z(host.features[split.fit])
    f_conf, f_test = z(host.features[split.conf]), z(host.features[split.test])

    probs = (host.probs if temperature == 1.0
             else _targets.temper(host.probs, temperature))
    if target == "hard":
        values = _targets.hard_label_values(host.labels[split.pool], n_out, loss.simplex)
    elif target == "cross_mask":
        if host.richer_probs is None:
            raise ValueError("cross_mask target needs richer_probs")
        richer = (host.richer_probs if temperature == 1.0
                  else _targets.temper(host.richer_probs, temperature))
        values = _targets.cross_mask_values(richer[split.pool])
    else:
        raise ValueError(f"unknown target {target!r}")

    k_eff = min(k, len(split.pool) - 1)
    if k_eff < k:
        notes.append(f"k reduced from {k} to {k_eff}: pool has {len(split.pool)} rows")
    t_fit = _targets.knn_average(f_fit, f_pool, values, k_eff, weighting=weighting)
    t_conf = _targets.knn_average(f_conf, f_pool, values, k_eff, weighting=weighting)
    t_test = _targets.knn_average(f_test, f_pool, values, k_eff, weighting=weighting)

    m_fit, m_conf, m_test = (probs[split.fit], probs[split.conf], probs[split.test])
    y_fit, y_conf, y_test = (host.labels[split.fit], host.labels[split.conf],
                             host.labels[split.test])

    beta = _select_beta(m_fit, t_fit, y_fit, loss, beta_objective)
    corrected_conf = (1 - beta) * m_conf + beta * t_conf
    corrected_test = (1 - beta) * m_test + beta * t_test

    gate = _certify.certify(corrected_conf, y_conf, corrected_test, m_test,
                            loss, alpha, delta)
    apply = gate["apply"]

    base_loss = loss(m_test, y_test)
    corrected_loss = loss(corrected_test, y_test)
    gated_loss = np.where(apply, corrected_loss, base_loss)
    delta_loss = gated_loss - base_loss
    blanket_delta = corrected_loss - base_loss
    gated_probs = np.where(apply[:, None], corrected_test, m_test)

    if metric == "accuracy":
        score = lambda p: _losses.accuracy(p, y_test, loss)
    elif metric == "f1_macro":
        score = lambda p: _losses.f1_macro(p, y_test)
    elif metric == "accuracy_nonzero":
        if host.raw_labels is None:
            raise ValueError("accuracy_nonzero needs raw_labels on HostOutputs")
        raw_test = np.asarray(host.raw_labels).reshape(-1)[split.test]
        score = lambda p: _losses.accuracy_nonzero(p, y_test, loss, raw_test)
    else:
        raise ValueError(f"unknown metric {metric!r}")

    sub = np.random.default_rng(0).permutation(len(y_test))[:6000]
    nn = _measure.nearest_neighbour_index(f_test[sub])
    t_self = loss.target(y_test[sub], n_out)
    ph = _measure.potential_headroom(m_test[sub], t_self, loss.target(y_test[sub][nn], n_out),
                                     scale=1.0 if loss.simplex else n_out)
    if ph["negative_and_significant"]:
        notes.append("PH significantly negative: neighbours carry opposing labels, "
                     "retrieval-based correction is not applicable here")

    base = score(m_test)
    res = Result(
        condition=condition, target=target, beta=beta,
        base_metric=base,
        blanket_metric_delta=score(corrected_test) - base,
        gate_metric_delta=score(gated_probs) - base,
        base_loss=float(base_loss.mean()),
        gate_loss_gain=float((base_loss - gated_loss).mean()),
        blanket_loss_gain=float((base_loss - corrected_loss).mean()),
        blanket_joint_harm=float((blanket_delta > delta).mean()),
        target_accuracy=_losses.accuracy(t_test, y_test, loss),
        ph=ph["ph"], ph_se=ph["ph_se"],
        q_hat=gate["q_hat"],
        **_certify.harm_accounting(delta_loss, apply, delta),
        n_pool=len(split.pool), n_fit=len(split.fit),
        n_conf=len(split.conf), n_test=len(split.test),
        alpha=alpha, delta=delta, k=k_eff,
        exchangeable=split.exchangeable, notes=notes,
    )
    # Attached, not declared: ``as_row`` runs ``asdict`` over declared fields,
    # so arrays here would break CSV writing and deep-copy on every call.
    # Ranking metrics such as AUROC need the scores themselves, which no scalar
    # in ``Result`` can reconstruct.
    res.test_arrays = {
        "base_probs": m_test,
        "blanket_probs": corrected_test,
        "gated_probs": gated_probs,
        "applied": apply,
        "labels": y_test,
        "index": np.asarray(split.test),
    }
    return res


def select_on_fit(
    host: HostOutputs,
    split: Split,
    *,
    loss_name: str = "cross_entropy",
    k_grid: Sequence[int] = (5, 10, 20, 35, 50),
    target_grid: Sequence[str] = ("hard",),
    space_grid: Sequence[str] = ("standardise",),
    weighting_grid: Sequence[str] = ("uniform",),
    temperature_grid: Sequence[float] = (1.0,),
    metric: str = "accuracy",
) -> dict:
    """Choose the retrieval settings on ``split.fit`` alone.

    Every combination is scored by the accuracy of the blended prediction on the
    fit split, with the blend weight refitted for each. The test split is never
    read. A grid of one value each reproduces the fixed construction, so this is
    an opt-in: pass wider grids only where the paper reports them.

    Selection on a fit split can itself overfit when the grid is large and the
    split is small; that shows up as a chosen setting that does not transfer,
    not as an invalid certificate, because the conformal gate is calibrated
    afterwards on ``split.conf``.
    """
    loss = _losses.get(loss_name)
    n_out = host.probs.shape[1]
    y_fit = host.labels[split.fit]
    # Selection must be scored the way the benchmark is scored. Choosing on
    # plain accuracy and reporting on a subset of the labels picks a different
    # setting, and the gap shows up as a method that does not reproduce.
    if metric == "accuracy":
        fit_score = lambda p: _losses.accuracy(p, y_fit, loss)
    elif metric == "f1_macro":
        fit_score = lambda p: _losses.f1_macro(p, y_fit)
    elif metric == "accuracy_nonzero":
        if host.raw_labels is None:
            raise ValueError("accuracy_nonzero needs raw_labels on HostOutputs")
        raw_fit = np.asarray(host.raw_labels).reshape(-1)[split.fit]
        fit_score = lambda p: _losses.accuracy_nonzero(p, y_fit, loss, raw_fit)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    best = None
    for space in space_grid:
        z = _targets.retrieval_space(host.features[split.pool], space)
        f_pool, f_fit = z(host.features[split.pool]), z(host.features[split.fit])
        for temperature in temperature_grid:
            probs = (host.probs if temperature == 1.0
                     else _targets.temper(host.probs, temperature))
            m_fit = probs[split.fit]
            for target in target_grid:
                if target == "hard":
                    values = _targets.hard_label_values(
                        host.labels[split.pool], n_out, loss.simplex)
                elif target == "cross_mask":
                    if host.richer_probs is None:
                        raise ValueError("cross_mask target needs richer_probs")
                    richer = (host.richer_probs if temperature == 1.0
                              else _targets.temper(host.richer_probs, temperature))
                    values = _targets.cross_mask_values(richer[split.pool])
                else:
                    raise ValueError(f"unknown target {target!r}")
                for weighting in weighting_grid:
                    for k in k_grid:
                        k_eff = min(k, len(split.pool) - 1)
                        t_fit = _targets.knn_average(
                            f_fit, f_pool, values, k_eff, weighting=weighting)
                        beta = _select_beta(m_fit, t_fit, y_fit, loss, "loss")
                        blended = (1 - beta) * m_fit + beta * t_fit
                        score = fit_score(blended)
                        if best is None or score > best[0]:
                            best = (score, dict(k=k, target=target, space=space,
                                                weighting=weighting,
                                                temperature=temperature))
    chosen = best[1]
    chosen["fit_score"] = float(best[0])
    return chosen
