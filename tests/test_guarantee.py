"""The guarantee, checked on synthetic data -- no dataset download required.

Run with ``pytest -q``.  These tests are the fastest way for a reader to
convince themselves the certificate does what the paper claims.
"""
import numpy as np
import pytest

from guard import HostOutputs, random_split, run


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
    """E[harm and applied] <= alpha, averaged over calibration draws.

    Conformal risk control bounds the expectation over the calibration draw and
    the deployment point, not the realised harm of any single split, so a per-seed
    assertion would be testing something the procedure never promised. The mean
    over many draws is what the guarantee constrains; the tolerance is Monte Carlo
    slack for the number of draws used here.
    """
    host = make_problem()
    harms = []
    for seed in range(40):
        split = random_split(np.arange(len(host.labels)), seed=seed)
        harms.append(run(host, split, alpha=alpha, delta=0.05).joint_harm)
    mean = float(np.mean(harms))
    assert mean <= alpha + 0.02, (alpha, mean, max(harms))


def test_conditional_harm_may_exceed_alpha():
    """Conditional harm is NOT bounded by alpha -- guard against over-claiming."""
    host = make_problem(gap=0.9)
    split = random_split(np.arange(len(host.labels)), seed=0)
    r = run(host, split, alpha=0.2, delta=0.05)
    assert r.cond_harm >= r.joint_harm





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


def test_crc_level_is_the_published_correction():
    """The calibration level must match conformal risk control, not plain alpha."""
    from guard.action import crc_threshold
    n, alpha = 200, 0.2
    score = np.linspace(0, 1, n)
    harmful = np.zeros(n, bool)
    # with no harm anywhere the most permissive threshold admits every point;
    # the rule is strict, so it sits just below the smallest score
    lam = crc_threshold(score, harmful, alpha)
    assert (score > lam).all()
    # with harm everywhere the threshold may still admit a few, but no more than
    # the level allows: floor(n * (alpha - (1-alpha)/n)) of them
    lam = crc_threshold(score, np.ones(n, bool), alpha)
    admitted = int((score > lam).sum())
    assert admitted <= int(n * (alpha - (1 - alpha) / n)) + 1


def test_useless_score_costs_gain_not_safety():
    """A score that ranks at random must not break the bound."""
    from guard.action import crc_threshold
    rng = np.random.default_rng(0)
    n, alpha = 400, 0.2
    score = rng.random(n)
    harmful = rng.random(n) < 0.5
    lam = crc_threshold(score, harmful, alpha)
    if lam is not None:
        applied = score >= lam
        assert float((applied & harmful).mean()) <= alpha
