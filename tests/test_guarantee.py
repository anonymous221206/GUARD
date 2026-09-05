"""The guarantee, checked on synthetic data -- no dataset download required.

Run with ``pytest -q``.  These tests are the fastest way for a reader to
convince themselves the certificate does what the paper claims.
"""
import numpy as np
import pytest

from guard import HostOutputs, random_split, run
from guard.certify import conformal_quantile


def make_problem(n=6000, d=4, gap=0.6, seed=0):
    """A host whose outputs are miscalibrated by a known amount."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8))
    logits = x[:, :d] * 1.5
    p_true = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(d, p=row) for row in p_true])
    skew = np.exp(logits * (1 - gap))
    probs = skew / skew.sum(1, keepdims=True)
    return HostOutputs(probs=probs, features=x, labels=y)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_joint_harm_respects_budget(alpha):
    """P(harm and applied) <= alpha, the paper's actual claim."""
    host = make_problem()
    for seed in range(5):
        split = random_split(np.arange(len(host.labels)), seed=seed)
        r = run(host, split, alpha=alpha, delta=0.05)
        assert r.joint_harm <= alpha + 0.02, (alpha, seed, r.joint_harm)


def test_conditional_harm_may_exceed_alpha():
    """Conditional harm is NOT bounded by alpha -- guard against over-claiming."""
    host = make_problem(gap=0.9)
    split = random_split(np.arange(len(host.labels)), seed=0)
    r = run(host, split, alpha=0.2, delta=0.05)
    assert r.cond_harm >= r.joint_harm


def test_conformal_quantile_is_finite_sample_correct():
    scores = np.arange(100, dtype=float)
    q = conformal_quantile(scores, alpha=0.1)
    assert scores[scores <= q].size >= 90


def test_conformal_quantile_is_infinite_when_sample_too_small():
    """Too few points to support the level means no finite threshold certifies.

    Returning the largest observed score instead would shrink the plausible set
    and make the gate easier to pass, i.e. err towards applying.
    """
    scores = np.linspace(0.1, 0.9, 3)
    assert conformal_quantile(scores, 0.2) == float("inf")     # ceil(4*0.8)=4 > 3
    assert np.isfinite(conformal_quantile(np.linspace(0.1, 0.9, 4), 0.2))
    assert conformal_quantile(np.linspace(0.1, 0.9, 18), 0.05) == float("inf")
    assert np.isfinite(conformal_quantile(np.linspace(0.1, 0.9, 19), 0.05))
    assert conformal_quantile(np.array([]), 0.2) == float("inf")


def test_infinite_quantile_makes_every_label_plausible():
    """An infinite threshold does not abstain by itself.

    It makes the plausible set the whole label space, so the gate applies only
    where *every* label would stay within delta -- a stricter test, not a
    refusal.
    """
    from guard.certify import _plausible_simplex
    corrected = np.array([[0.7, 0.2, 0.1], [0.34, 0.33, 0.33]])
    assert _plausible_simplex(corrected, float("inf")).all()


def test_split_roles_must_be_disjoint():
    from guard.splits import Split
    with pytest.raises(ValueError, match="overlap"):
        Split(np.arange(10), np.arange(5, 15), np.arange(20, 30), np.arange(30, 40))


def test_non_exchangeable_split_is_flagged():
    from guard.splits import Split
    s = Split(np.arange(10), np.arange(10, 20), np.arange(20, 30), np.arange(30, 40),
              origin={"conf": "subject 3", "test": "subject 4"})
    assert not s.exchangeable
    assert "does not apply" in s.warn_if_not_exchangeable()


def test_no_gain_when_target_is_useless():
    """When the retrieval target is noise, GUARD must decline, not damage."""
    host = make_problem()
    host.features = np.random.default_rng(1).normal(size=host.features.shape)
    split = random_split(np.arange(len(host.labels)), seed=0)
    r = run(host, split, alpha=0.2, delta=0.05)
    assert r.gate_metric_delta > -0.02
    assert r.joint_harm <= 0.22
