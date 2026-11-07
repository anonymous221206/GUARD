"""Total deployment-label budget, with a fixed test split and fixed role allocation.

Regime A: the pool is carved from deployment data, so hard labels on it cost budget.
  hard   pool:fit:conf = 1:1:1 of B
  cross  pool stays full and unlabeled; fit:conf = 1:1 of B
Regime B: the pool carries existing training labels, so B covers fit and conf only.
Allocations were fixed before any run. Each arm selects its settings on D_fit as GUARD does.
"""
import sys, time, json, argparse
from multiprocessing import Pool
from rebuttal_common import *
from rebuttal_common import _A, _L, _T
from probe_core import _fit_hard, _fit_cross, CS
from gates_core import _score, KS, TS, SPACES, WTS
from guard.pipeline import _select_beta
def run_arm(P, F, Y, Rh, split, keep, targets, kind, ln):
    loss = _L.get(ln); ml = not loss.simplex; pool, fit, conf, test = split; n_out = P.shape[1]
    acc_i = lambda Q, idx: _score(Q, Y[idx], None if keep is None else keep[idx])
    SP = {}
    for sp in SPACES:
        z = _T.retrieval_space(F[pool], sp)
        SP[sp] = {n: z(F[i]) for n, i in (('pool', pool), ('fit', fit), ('conf', conf), ('test', test))}
    def vals(tg, T):
        if tg == 'hard': return _T.hard_label_values(Y[pool], n_out, loss.simplex)
        return _T.cross_mask_values((Rh if T == 1.0 else _T.temper(Rh, T, loss.simplex))[pool])
    cache = {}
    def tgt(tg, T, sp, wt, hp):
        if kind == 'knn':
            v = vals(tg, T); ke = min(hp, len(pool) - 1)
            return {n: _T.knn_average(SP[sp][n], SP[sp]['pool'], v, ke, weighting=wt) for n in ('fit', 'conf', 'test')}
        key = (tg, T if tg == 'cross' else 1.0, sp, hp)
        if key not in cache:
            f = _fit_hard('linear', hp, SP[sp]['pool'], Y[pool], n_out, loss.simplex) if tg == 'hard' else _fit_cross('linear', hp, SP[sp]['pool'], vals(tg, T), loss.simplex)
            cache[key] = {n: f(SP[sp][n]) for n in ('fit', 'conf', 'test')}
        return cache[key]
    best = None
    for T in TS:
        pr = P if T == 1.0 else _T.temper(P, T, loss.simplex)
        for tg in targets:
            grid = [(wt, k) for wt in WTS for k in KS] if kind == 'knn' else [(None, c) for c in CS]
            for sp in SPACES:
                for wt, hp in grid:
                    try: tt = tgt(tg, T, sp, wt, hp)
                    except ValueError: continue
                    b = _select_beta(pr[fit], tt['fit'], Y[fit], loss, 'loss')
                    s = acc_i((1 - b) * pr[fit] + b * tt['fit'], fit)
                    if best is None or s > best[0]: best = (s, T, tg, sp, wt, hp, b)
    if best is None: return None
    _, T, tg, sp, wt, hp, b = best
    pr = P if T == 1.0 else _T.temper(P, T, loss.simplex); tt = tgt(tg, T, sp, wt, hp)
    mf, mc, mt = P[fit], P[conf], P[test]; yf, yc, yt = Y[fit], Y[conf], Y[test]
    cf, cc, ct = (1 - b) * pr[fit] + b * tt['fit'], (1 - b) * pr[conf] + b * tt['conf'], (1 - b) * pr[test] + b * tt['test']
    dl = loss(ct, yt) - loss(mt, yt); kt = None if keep is None else keep[test]; acc = acc_fn(yt, kt, ml); base = acc(mt)
    harm_c = (loss(cc, yc) - loss(mc, yc)) > DELTA; gain_f = loss(mf, yf) - loss(cf, yf)
    sc = _A.fit_action_score(mf, tt['fit'], cf, yf, loss); none = np.zeros(len(test), bool); o = {}
    if sc is None: ap = dict(crc=none, ltt=none, crc_t=none, ltt_t=none)
    else:
        sf, sv, st = _A.score(sc, mf, tt['fit'], loss.simplex), _A.score(sc, mc, tt['conf'], loss.simplex), _A.score(sc, mt, tt['test'], loss.simplex)
        lc = _A.crc_threshold(sv, harm_c, ALPHA); ll = ltt_threshold(sv, harm_c, ALPHA, ZETA, grid_scores=sf)
        ap = dict(crc=none if lc is None else st > lc, ltt=none if ll is None else st >= ll,
                  crc_t=none if lc is None else st > _A._tighten(sf, gain_f, lc), ltt_t=none if ll is None else st > _A._tighten(sf, gain_f, ll))
    for k, a in ap.items():
        o[k] = dict(gain=acc(np.where(a[:, None], ct, mt)) - base, harm=float((a & (dl > DELTA)).mean()), apply=float(a.mean()))
    o['blanket'] = dict(gain=acc(ct) - base, harm=float((dl > DELTA).mean()), apply=1.0)
    o['meta'] = dict(T=T, target=tg, space=sp, weighting=wt, hp=hp, beta=b, n_pool=len(pool), n_fit=len(fit), n_conf=len(conf), n_test=len(test))
    return o
LOAD = dict(mosei=mosei, opportunity=opportunity, drugban_deploy=drugban_deploy, ptbxl=ptbxl, ave=ave, iemocap=iemocap)
A_LEVELS = (60, 120, 240, 480, 960, 1800, 'full'); B_LEVELS = (40, 80, 160, 320, 640, 'full')
J = []
for bench, ctxs in (('mosei', [dict(cond=c, seed=s) for c in ('a', 'v', 'av') for s in range(5)]),
                    ('opportunity', [dict(cfg=c, s=s) for c in ('low_cost_accels_only', 'no_imu_family', 'no_shoes', 'severe_three_sensors') for s in (0, 1, 2)]),
                    ('drugban_deploy', [dict(dataset=d, seed=s) for d in ('biosnap', 'bindingdb') for s in range(3)])):
    for c in ctxs:
        for B in A_LEVELS:
            for kind in ('knn', 'linear'):
                for tg in ('hard', 'cross'):
                    J.append(dict(bench=bench, ctx=c, B=B, kind=kind, target=tg, regime='A'))
for bench, ctxs in (('ptbxl', [dict(c=c, fk=fk, seed=s) for c, fk in (('a', 'a'), ('t', 't'), ('v', 'v'), ('at', 'a'), ('av', 'a'), ('tv', 't')) for s in range(3)]),
                    ('ave', [dict(cond=c, seed=s) for c in ('audio_only', 'visual_only') for s in range(3)]),
                    ('iemocap', [dict(f=f, m=m, cut=0) for f in range(5) for m in MS[:-1]])):
    for c in ctxs:
        for B in B_LEVELS:
            J.append(dict(bench=bench, ctx=c, B=B, kind='knn', target='auto', regime='B'))
def job(j):
    try:
        P, F, Y, Rh, split, keep, tgs, ln = LOAD[j['bench']](**j['ctx'])
        pool, fit, conf, test = split; B = j['B']
        # one fixed shuffle per role, so smaller budgets are nested in larger ones and a
        # small sample is not a single contiguous block of one activity or video
        sh = lambda a: np.random.default_rng(12345).permutation(a)
        pool_s, fit, conf = sh(pool), sh(fit), sh(conf)
        if j['regime'] == 'B':
            n = None if B == 'full' else B // 2
            if n is not None and n > min(len(fit), len(conf)): return dict(j, skip=True)
            sp = (pool, fit[:n], conf[:n], test); targets = tgs
        elif j['target'] == 'hard':
            n = None if B == 'full' else B // 3
            if n is not None and n > min(len(pool), len(fit), len(conf)): return dict(j, skip=True)
            sp = (pool_s[:n], fit[:n], conf[:n], test); targets = ('hard',)
        else:
            n = None if B == 'full' else B // 2
            if n is not None and n > min(len(fit), len(conf)): return dict(j, skip=True)
            sp = (pool, fit[:n], conf[:n], test); targets = ('cross',)
        return dict(j, r=run_arm(P, F, Y, Rh, sp, keep, targets, j['kind'], ln))
    except Exception:
        import traceback
        return dict(j, error=traceback.format_exc()[-1500:])
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--procs', type=int, default=8); ap.add_argument('--only'); a = ap.parse_args()
    jobs = [j for j in J if a.only is None or j['bench'] == a.only]; t0 = time.time()
    print('jobs', len(jobs), flush=True)
    with Pool(a.procs) as p, open(OUT / f"budget{'_' + a.only if a.only else ''}.jsonl", 'w') as fh:
        for i, r in enumerate(p.imap_unordered(job, jobs, chunksize=1)):
            fh.write(json.dumps(jsonable(r)) + '\n'); fh.flush()
            if i % 25 == 0: print(i + 1, len(jobs), round(time.time() - t0), flush=True)
    print('DONE', round(time.time() - t0), flush=True)
