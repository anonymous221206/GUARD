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
        applied = score > lam
        assert float((applied & harmful).mean()) <= alpha


def test_ties_are_priced_on_the_set_that_is_deployed():
    """A block of equal scores is admitted or refused whole, and paid for whole.

    Pricing the risk by walking a sorted order charges only the ties it reaches;
    the policy then deploys ``score > lambda``, which admits every one of them.
    On a score with heavy ties that gap is a real under-count, so the threshold
    is chosen against the set the policy will actually apply to.
    """
    from guard.action import crc_threshold
    n, alpha = 100, 0.2
    # forty points share one score, and every one of them is harmful
    score = np.concatenate([np.zeros(60), np.full(40, 0.7)])
    harmful = np.concatenate([np.zeros(60, bool), np.ones(40, bool)])
    lam = crc_threshold(score, harmful, alpha)
    applied = np.zeros(n, bool) if lam is None else (score > lam)
    # admitting the tied block would cost 0.40, twice the budget
    assert float((applied & harmful).mean()) <= alpha - (1 - alpha) / n
    assert not applied[60:].any()


def test_tightening_never_reopens_the_gate():
    """The fit split must not be able to lower a threshold the calibration set set.

    When conformal risk control admits nothing, lambda sits above every
    calibration score. If the fit scores happen to top out below that, returning
    their maximum would admit queries the certificate refused.
    """
    from guard.action import _tighten
    scores = np.linspace(0.0, 0.4, 50)          # fit scores, all low
    gain = -np.ones(50)                         # correction never helps
    lam = 0.9                                   # calibration admitted nothing
    assert _tighten(scores, gain, lam) >= lam


def test_temperature_keeps_multilabel_outputs_independent():
    """A multi-label row is not a distribution and must not be softmaxed."""
    from guard.targets import temper
    p = np.array([[0.9, 0.8, 0.7], [0.2, 0.3, 0.1]])
    out = temper(p, 2.0, simplex=False)
    want = 1.0 / (1.0 + np.exp(-np.log(p / (1 - p)) / 2.0))
    assert np.allclose(out, want)                        # each label on its own logit
    assert ((out > 0.5) == (p > 0.5)).all()              # no decision moves
    assert not np.allclose(out.sum(1), 1.0)              # the row is not a distribution
    soft = temper(p, 2.0, simplex=True)
    assert np.allclose(soft.sum(1), 1.0)                 # the simplex path still normalises
    assert not np.allclose(soft, out)


def test_retrieval_target_does_not_depend_on_the_batch():
    """p_corr has to be a function of the query and the pool, nothing else.

    The distance kernel once took its width from the queries in the batch, so a
    point got one target during calibration and another at deployment, and the
    conformal argument no longer applied to it.
    """
    from guard.targets import knn_average
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(300, 6))
    values = np.eye(3)[rng.integers(0, 3, 300)]
    q = rng.normal(size=(40, 6))
    alone = knn_average(q[:1], pool, values, 10, weighting="distance")
    with_others = knn_average(q, pool, values, 10, weighting="distance")[:1]
    assert np.allclose(alone, with_others), (alone, with_others)

