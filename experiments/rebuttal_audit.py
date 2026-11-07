"""Independent input audit, run after the IEMOCAP errors.

For every benchmark this checks what the two errors broke, rather than comparing a
rerun with saved outputs:
  outputs   finite, inside [0, 1], rows of a simplex sum to one
  roles     pool, fit, conf and test share no row
  labels    retrieval from the pool beats the same retrieval with shuffled pool labels
  richer    richer-view outputs reach their accuracy on the rows they are paired with,
            and fall to chance when those rows are shuffled
The IEMOCAP control re-creates the old pairing, so the richer check is shown to detect it.
"""
import json
from multiprocessing import Pool
from rebuttal_common import *
from rebuttal_common import _L, _T
J = [('mosei', dict(cond=c, seed=s)) for c in ('a', 'v', 'av') for s in (0, 1)]
J += [('iemocap', dict(f=f, m=m, cut=0)) for f in range(5) for m in MS]
J += [('opportunity', dict(cfg=c, s=s)) for c in ('low_cost_accels_only', 'no_imu_family', 'no_shoes', 'severe_three_sensors') for s in (0, 1, 2)]
J += [('ave', dict(cond=c, seed=0)) for c in ('audio_only', 'visual_only')]
J += [('ninapro', dict(sub=sub, c=c)) for sub in nina_subjects() for c in ('12', '4')]
J += [('ptbxl', dict(c=c, fk=fk, seed=0)) for c, fk in (('a', 'a'), ('t', 't'), ('v', 'v'), ('at', 'a'), ('av', 'a'), ('tv', 't'))]
_D = [('drugban_biosnap_random_s42', c) for c in ('prot25', 'prot50', 'scaffold')] + [('drugban_bindingdb_random_s42', c) for c in ('prot25', 'prot50', 'scaffold')] + [('drugban_human_random_s42', c) for c in ('prot25', 'prot50')] + [('drugban_biosnap_cluster_s42', 'prot50')]
J += [('drugban', dict(d=d, cond=c, seed=0)) for d, c in _D]
LOAD = dict(mosei=mosei, iemocap=iemocap, opportunity=opportunity, ave=ave, ninapro=ninapro, ptbxl=ptbxl, drugban=drugban)
def acc(p, y, ml): return float(((p > .5) == (y > .5)).mean()) if ml else float((p.argmax(1) == y).mean())
def job(a):
    b, ctx = a
    P, F, Y, Rh, split, keep, tg, ln = LOAD[b](**ctx); loss = _L.get(ln); ml = not loss.simplex
    pool, fit, conf, test = split; rng = np.random.default_rng(0)
    q = np.asarray(P, dtype=np.float64); o = dict(bench=b, ctx=ctx)
    o['finite'] = bool(np.isfinite(q).all()); o['range_ok'] = bool(q.min() >= -1e-6 and q.max() <= 1 + 1e-6)
    o['simplex_err'] = None if ml else float(np.abs(q.sum(1) - 1).max())
    roles = [set(map(int, r)) for r in split]
    o['roles_disjoint'] = all(not (roles[i] & roles[j]) for i in range(4) for j in range(i + 1, 4))
    o['frozen_acc'] = acc(q[test], Y[test], ml)
    z = _T.retrieval_space(F[pool], 'standardise'); fp, ft = z(F[pool]), z(F[test])
    n_out = P.shape[1]; v = _T.hard_label_values(Y[pool], n_out, loss.simplex)
    o['knn_acc'] = acc(_T.knn_average(ft, fp, v, 20), Y[test], ml)
    o['knn_acc_shuffled'] = acc(_T.knn_average(ft, fp, v[rng.permutation(len(v))], 20), Y[test], ml)
    if Rh is not None and 'cross' in tg:
        o['richer_acc_pool'] = acc(Rh[pool], Y[pool], ml); o['richer_acc_pool_shuffled'] = acc(Rh[pool][rng.permutation(len(pool))], Y[pool], ml)
        o['richer_acc_test'] = acc(Rh[test], Y[test], ml)
    if b == 'iemocap':
        ref = np.load(f"{A}/iemocap_momke/folds/fold{ctx['f']}_atv.npz", allow_pickle=True)
        o['richer_acc_pool_old_pairing'] = acc(softmax(ref['pool_probs']), Y[pool], ml) if ctx['m'] != 'atv' else None
    return jsonable(o)
if __name__ == '__main__':
    with Pool(8) as p: rows = p.map(job, J, chunksize=1)
    (OUT / 'audit.json').write_text(json.dumps(rows))
    S = {}
    for b in LOAD:
        c = [r for r in rows if r['bench'] == b]; mn = lambda k: min((r[k] for r in c if r.get(k) is not None), default=None); mx = lambda k: max((r[k] for r in c if r.get(k) is not None), default=None)
        S[b] = dict(cells=len(c), finite=all(r['finite'] for r in c), range_ok=all(r['range_ok'] for r in c), max_simplex_err=mx('simplex_err'), roles_disjoint=all(r['roles_disjoint'] for r in c),
                    min_knn=mn('knn_acc'), max_knn_shuffled=mx('knn_acc_shuffled'), min_margin=min(r['knn_acc'] - r['knn_acc_shuffled'] for r in c),
                    min_richer_pool=mn('richer_acc_pool'), max_richer_shuffled=mx('richer_acc_pool_shuffled'), max_old_pairing=mx('richer_acc_pool_old_pairing'))
    (OUT / 'audit_summary.json').write_text(json.dumps(jsonable(S), indent=1)); print(json.dumps(jsonable(S), indent=0))
