"""Swap the action score, keep everything else: which learner, and which features.

Proposition 1 holds for any score fixed before ``D_conf``, so nothing here can
move the harm bound; what it can move is gated gain and how many queries the
gate admits. Two questions are asked on one run. First, is a logistic model on
five features enough, or does a richer learner rank interventions better?
Second, which of the five features the gate actually needs, by dropping them one
at a time.

The corrector is selected once, by the shared core's own rule, and every variant
then sees the same frozen model, the same blend, the same splits and the same
budget. Only the map from features to a score changes. The ``logistic`` entry is
the deployed policy recomputed here, so it reproduces Table 3's GUARD column and
acts as this module's control.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from guard import action as _A
from guard import losses as _L, targets as _T
from guard.pipeline import _select_beta
from gates_core import _score, KS, TS, SPACES, WTS, ALPHA, DELTA
from scipy.special import expit

# the five columns of guard.action.action_features, in order
FEATURES = ('baseconf', 'baseent', 'targetconf', 'agree', 'l1')
ALL = tuple(range(len(FEATURES)))

# Deliberately shallow: each learner keeps its library default apart from one
# size knob, because the point is whether the family matters, not whether a
# tuned member of it does. No search, so nothing here can overfit D_fit through
# a hyperparameter the deployed policy would not have.
FAMILIES = {
    'logistic': lambda: LogisticRegression(max_iter=2000),
    'tree3':    lambda: DecisionTreeClassifier(max_depth=3, random_state=0),
    'forest':   lambda: RandomForestClassifier(n_estimators=200, random_state=0),
    'knn25':    lambda: KNeighborsClassifier(n_neighbors=25),
    'mlp16':    lambda: MLPClassifier(hidden_layer_sizes=(16,), max_iter=800,
                                      random_state=0),
}

# name -> (family, feature columns)
VARIANTS = {k: (k, ALL) for k in FAMILIES}
# dropping one feature says whether it is redundant given the other four; keeping
# one says whether it carries the signal by itself. The second question is the
# one that decides how many features the deployed score needs, and leave-one-out
# cannot answer it: four mutually redundant features each look free to drop.
VARIANTS.update({f'drop:{FEATURES[i]}': ('logistic', tuple(c for c in ALL if c != i))
                 for i in ALL})
VARIANTS.update({f'keep:{FEATURES[i]}': ('logistic', (i,)) for i in ALL})
# The deployed score is fitted against the sign of the improvement. Under a harm
# budget the oracle policy is instead a threshold on the ratio of expected
# benefit to the probability of exceeding the tolerance, so this variant fits
# those two quantities separately and ranks by their ratio. It changes what the
# score is asked to predict, not the calibration that follows it.
VARIANTS['ratio'] = ('ratio', ALL)
DROP = tuple(f'drop:{f}' for f in FEATURES)
KEEP = tuple(f'keep:{f}' for f in FEATURES)
NAMES = tuple(FAMILIES) + ('ratio',) + DROP + KEEP


class _Sub:
    """Read a feature subset at scoring time, so ``guard.action`` stays untouched.

    ``guard.action.score`` always builds all five columns and calls
    ``predict_proba`` on them; this wrapper slices to the columns its estimator
    was fitted on, which keeps the calibration and deployment paths identical.
    """

    def __init__(self, model, cols):
        self.model, self.cols = model, list(cols)

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X)[:, self.cols])


class _Ratio:
    """Rank by estimated benefit over estimated probability of excessive harm."""

    def __init__(self, u, h, cols):
        self.u, self.h, self.cols = u, h, list(cols)

    def predict_proba(self, X):
        x = np.asarray(X)[:, self.cols]
        u = self.u.predict(x)
        h = (self.h.predict_proba(x)[:, 1] if self.h is not None
             else np.full(len(x), 1e-3))
        s = expit(u / np.clip(h, 1e-3, None))
        return np.column_stack([1.0 - s, s])


def _fit(name, base_fit, target_fit, corrected_fit, labels_fit, loss):
    """The variant's score, fitted on D_fit against the same label the core uses."""
    family, cols = VARIANTS[name]
    if family == 'ratio':
        d = loss(corrected_fit, labels_fit) - loss(base_fit, labels_fit)
        x = _A.action_features(base_fit, target_fit, loss.simplex)[:, list(cols)]
        hz = (d > DELTA).astype(int)
        h = (LogisticRegression(max_iter=2000).fit(x, hz)
             if len(set(hz.tolist())) > 1 else None)
        return _Ratio(Ridge().fit(x, -d), h, cols)
    z = (loss(corrected_fit, labels_fit) < loss(base_fit, labels_fit)).astype(int)
    if len(set(z.tolist())) < 2:                 # nothing to learn from
        return None
    x = _A.action_features(base_fit, target_fit, loss.simplex)[:, list(cols)]
    return _Sub(FAMILIES[family]().fit(x, z), cols)


def _auroc(score_test, helped):
    if len(set(helped.tolist())) < 2:
        return float('nan')
    return float(roc_auc_score(helped, score_test))


def rows(probs, feats, labels, split, loss_name='cross_entropy', keep=None,
         targets=('hard',), richer=None, variants=NAMES):
    """{variant: dict(gain, harm, ...)} on one condition, one corrector for all."""
    loss = _L.get(loss_name)
    pool, fit, conf, test = split
    acc = lambda P, idx: _score(P, labels[idx], None if keep is None else keep[idx])
    n_out = probs.shape[1]
    SP = {}
    for sp in SPACES:
        z = _T.retrieval_space(feats[pool], sp)
        SP[sp] = {n: z(feats[i]) for n, i in
                  (('pool', pool), ('fit', fit), ('conf', conf), ('test', test))}

    def values(tg, T):
        if tg == 'hard':
            return _T.hard_label_values(labels[pool], n_out, loss.simplex)
        r = richer if T == 1.0 else _T.temper(richer, T, loss.simplex)
        return _T.cross_mask_values(r[pool])

    # identical nesting to gates_core.gate_row, so an exact tie breaks the same
    # way and the corrector chosen here is the one Table 3 reports
    best = None
    for T in TS:
        pr = probs if T == 1.0 else _T.temper(probs, T, loss.simplex)
        for tg in targets:
            if tg != 'hard' and richer is None:
                continue
            vals = values(tg, T)
            for sp in SPACES:
                for wt in WTS:
                    for k in KS:
                        ke = min(k, len(pool) - 1)
                        tf = _T.knn_average(SP[sp]['fit'], SP[sp]['pool'], vals, ke,
                                            weighting=wt)
                        b = _select_beta(pr[fit], tf, labels[fit], loss, 'loss')
                        s = acc((1 - b) * pr[fit] + b * tf, fit)
                        if best is None or s > best[0]:
                            best = (s, T, tg, sp, wt, k, b)
    _, T, tg, sp, wt, k, b = best
    pr = probs if T == 1.0 else _T.temper(probs, T, loss.simplex)
    vals = values(tg, T)
    ke = min(k, len(pool) - 1)
    tt = {n: _T.knn_average(SP[sp][n], SP[sp]['pool'], vals, ke, weighting=wt)
          for n in ('fit', 'conf', 'test')}
    # the temperature belongs to the correction; a declined query keeps the
    # frozen model's own output, exactly as in the shared core
    mc, mt = probs[conf], probs[test]
    cc = (1 - b) * pr[conf] + b * tt['conf']
    ct = (1 - b) * pr[test] + b * tt['test']
    cf = (1 - b) * pr[fit] + b * tt['fit']
    dl = loss(ct, labels[test]) - loss(mt, labels[test])
    helped = (dl < 0).astype(int)
    base = acc(mt, test)
    meta = dict(target=tg, space=sp, weighting=wt, k=k, temperature=T, beta=b)

    out = {}
    for name in variants:
        sc = _fit(name, probs[fit], tt['fit'], cf, labels[fit], loss)
        g = _A.certify_action(sc, mc, tt['conf'], cc, labels[conf], mt, tt['test'],
                              loss, ALPHA, DELTA,
                              fit=(probs[fit], tt['fit'], cf, labels[fit]))
        ap = g['apply']
        out[name] = dict(
            gain=acc(np.where(ap[:, None], ct, mt), test) - base,
            harm=float((ap & (dl > DELTA)).mean()),
            blanket_gain=acc(ct, test) - base,
            blanket_harm=float((dl > DELTA).mean()),
            apply=float(ap.mean()), base=base,
            auroc=_auroc(g['score_test'], helped),
            meta=meta)
    return out
