"""Swap the corrector, keep everything else: kNN retrieval against a trained probe.

A reviewer who grants us labelled deployment-matched splits will ask why the
corrector is a nearest-neighbour average rather than a model trained on the same
pool. This module answers that empirically. The frozen host, the pool, the
splits, the blend weight, the action score and the conformal threshold are the
shared core's code, unchanged; only the map from a query representation to a
target changes, and each corrector picks its own hyperparameters on D_fit by the
same rule.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from guard import action as _A
from guard import losses as _L, targets as _T
from guard.pipeline import _select_beta
from gates_core import _score, KS, TS, SPACES, WTS, ALPHA, DELTA

CS = (0.01, 0.1, 1.0, 10.0)          # linear probe regularisation
HID = ((128,), (256,))               # MLP probe widths
EPS = 1e-12


def _norm(P, simplex):
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    if simplex:
        P = np.clip(P, EPS, None)
        return P / P.sum(1, keepdims=True)
    return np.clip(P, EPS, 1.0 - EPS)


def _fit_hard(kind, hp, Fp, Yp, n_out, simplex):
    """Probe trained on the pool's labels: the probe analogue of hard retrieval."""
    if simplex:
        m = (LogisticRegression(C=hp, max_iter=2000) if kind == 'linear' else
             MLPClassifier(hidden_layer_sizes=hp, max_iter=300, early_stopping=True,
                           random_state=0))
        m.fit(Fp, np.asarray(Yp, dtype=int))
        cls = np.asarray(m.classes_, dtype=int)

        def f(Fq):
            P = np.zeros((len(Fq), n_out))
            P[:, cls] = m.predict_proba(Fq)
            return _norm(P, True)
        return f
    # multi-label: one probe per column; a column with one value keeps its prior
    Y = np.asarray(Yp, dtype=float)
    cols = []
    for j in range(Y.shape[1]):
        yj = (Y[:, j] > 0.5).astype(int)
        if len(np.unique(yj)) < 2:
            cols.append(float(yj.mean()))
            continue
        m = (LogisticRegression(C=hp, max_iter=2000) if kind == 'linear' else
             MLPClassifier(hidden_layer_sizes=hp, max_iter=300, early_stopping=True,
                           random_state=0))
        m.fit(Fp, yj)
        cols.append(m)

    def f(Fq):
        P = np.zeros((len(Fq), Y.shape[1]))
        for j, m in enumerate(cols):
            P[:, j] = m if np.isscalar(m) else m.predict_proba(Fq)[:, 1]
        return _norm(P, False)
    return f


def _fit_cross(kind, hp, Fp, Vp, simplex):
    """Probe regressed onto the richer mask's outputs: the label-free target."""
    a = 1.0 / max(hp, 1e-6) if kind == 'linear' else 1e-4
    m = (Ridge(alpha=a) if kind == 'linear' else
         MLPRegressor(hidden_layer_sizes=hp, max_iter=300, early_stopping=True,
                      random_state=0))
    m.fit(Fp, np.asarray(Vp, dtype=np.float64))
    return lambda Fq: _norm(m.predict(Fq), simplex)


def rows(probs, feats, labels, split, loss_name='cross_entropy', keep=None,
         targets=('hard',), richer=None, correctors=('knn', 'linear')):
    """{corrector: dict(gain, harm, ...)} on one condition.

    The 'knn' entry is the shared core's own policy, recomputed here on the same
    splits, so it reproduces Table 3's GUARD column and acts as a control.
    """
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

    def target(kind, tg, T, sp, wt, hp):
        vals = values(tg, T)
        if kind == 'knn':
            ke = min(hp, len(pool) - 1)
            return {n: _T.knn_average(SP[sp][n], SP[sp]['pool'], vals, ke, weighting=wt)
                    for n in ('fit', 'conf', 'test')}
        f = (_fit_hard(kind, hp, SP[sp]['pool'], labels[pool], n_out, loss.simplex)
             if tg == 'hard' else
             _fit_cross(kind, hp, SP[sp]['pool'], vals, loss.simplex))
        return {n: f(SP[sp][n]) for n in ('fit', 'conf', 'test')}

    out = {}
    for kind in correctors:
        best = None
        for T in TS:
            pr = probs if T == 1.0 else _T.temper(probs, T, loss.simplex)
            for tg in targets:
                if tg != 'hard' and richer is None:
                    continue
                # same nesting as the shared core, so an exact tie breaks the
                # same way and the knn control reproduces its selection
                grid = ([(wt, k) for wt in WTS for k in KS] if kind == 'knn'
                        else [(None, hp) for hp in (CS if kind == 'linear' else HID)])
                for sp in SPACES:
                    for wt, hp in grid:
                        tt = target(kind, tg, T, sp, wt, hp)
                        b = _select_beta(pr[fit], tt['fit'], labels[fit], loss, 'loss')
                        s = acc((1 - b) * pr[fit] + b * tt['fit'], fit)
                        if best is None or s > best[0]:
                            best = (s, T, tg, sp, wt, hp, b)
        _, T, tg, sp, wt, hp, b = best
        pr = probs if T == 1.0 else _T.temper(probs, T, loss.simplex)
        tt = target(kind, tg, T, sp, wt, hp)
        # the temperature belongs to the correction; a declined query keeps the
        # host's own output, exactly as in the shared core
        mc, mt = probs[conf], probs[test]
        cc = (1 - b) * pr[conf] + b * tt['conf']
        ct = (1 - b) * pr[test] + b * tt['test']
        cf = (1 - b) * pr[fit] + b * tt['fit']
        sc = _A.fit_action_score(probs[fit], tt['fit'], cf, labels[fit], loss)
        g = _A.certify_action(sc, mc, tt['conf'], cc, labels[conf], mt, tt['test'],
                              loss, ALPHA, DELTA,
                              fit=(probs[fit], tt['fit'], cf, labels[fit]))
        ap = g['apply']
        dl = loss(ct, labels[test]) - loss(mt, labels[test])
        base = acc(mt, test)
        out[kind] = dict(
            gain=acc(np.where(ap[:, None], ct, mt), test) - base,
            harm=float((ap & (dl > DELTA)).mean()),
            blanket_gain=acc(ct, test) - base,
            blanket_harm=float((dl > DELTA).mean()),
            apply=float(ap.mean()), base=base,
            meta=dict(target=tg, space=sp, weighting=wt, hp=hp, temperature=T, beta=b))
    return out
