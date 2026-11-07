"""Changing deployment conditions.

mix: every sample draws its own availability pattern. One corrector; one pooled threshold
     (GUARD) against one threshold per pattern (the mask-conditional corollary). Per-pattern
     harm is the number that matters: it is what a shift in the pattern mix re-weights.
s1:  corrector and score fitted at the deployed condition, calibration data from another.
s2:  corrector, score and calibration all from another condition, deployed unchanged.
Identity pairs are the reference.
"""
import sys, time, json, glob, argparse
from multiprocessing import Pool
from rebuttal_common import *
from rebuttal_common import _A, _L, _T
import gates_ltt, iemocap_eta as E
def mix_mosei(seed):
    probs, FE, y, keep = _mosei_raw_public(); A_, V_ = FE['a'], FE['v']; C = ('a', 'v', 'av'); PR = {c: probs(c) for c in C}
    n = len(y); rng = np.random.default_rng(seed); ci = rng.integers(0, 3, n); P = np.zeros((n, 2)); F = np.zeros((n, A_.shape[1] + V_.shape[1]))
    for j, c in enumerate(C):
        m = ci == j; P[m] = PR[c][m]
        if 'a' in c: F[m, :A_.shape[1]] = A_[m]
        if 'v' in c: F[m, A_.shape[1]:] = V_[m]
    return P, F, y, probs('tav'), tuple(np.array_split(rng.permutation(n), 4)), keep, ('hard', 'cross'), 'cross_entropy', ci, C
def mix_nina(sub, si):
    C = ('12', '8', '6', '4'); p = np.load(f'{A}/{sub}/preds.npz', allow_pickle=True); e = np.load(f'{A}/{sub}/masked_embeddings.npz', allow_pickle=True)
    ntr = len(p['train_y']); m = len(p['sess1_y']); rng = np.random.default_rng(si); ci = rng.integers(0, 4, m); cp = rng.integers(0, 4, ntr)
    Ptr = np.choose(cp[:, None], [p[f'train_{c}'] for c in C])
    Ftr = np.choose(cp[:, None], [e[f'train_{c}'] for c in C])
    Pd = np.choose(ci[:, None], [p[f'sess1_{c}'] for c in C]); Fd = np.choose(ci[:, None], [e[f'sess1_{c}'] for c in C])
    P = np.concatenate([Ptr, Pd]).astype(np.float64); P = np.clip(P, 1e-12, None); P /= P.sum(1, keepdims=True); F = np.concatenate([Ftr, Fd]).astype(np.float64)
    Y = np.concatenate([p['train_y'], p['sess1_y']]).astype(int); perm = rng.permutation(m) + ntr; k = m // 3
    return P, F, Y, None, (np.arange(ntr), perm[:k], perm[k:2*k], perm[2*k:]), None, ('hard',), 'cross_entropy', np.concatenate([cp, ci]), C
def mix_iem(fold, draw, eta=0.5):
    ref, Q, nt, npool = E._aligned(fold); y = np.concatenate([ref['pool_labels'], ref['labels']]); rich = np.concatenate([Q['atv'][2], Q['atv'][0]]); uv = np.unique(ref['vid'])
    rng = np.random.default_rng(1000 * draw + fold); mt, mp = E.draw_masks(rng, nt, eta), E.draw_masks(rng, npool, eta)
    pr = np.zeros_like(Q['atv'][0]); ft = np.zeros_like(Q['atv'][1]); pp = np.zeros_like(Q['atv'][2]); pf = np.zeros_like(Q['atv'][3])
    for m in E.MS:
        s = mt == m
        if s.any(): pr[s], ft[s] = Q[m][0][s], Q[m][1][s]
        s = mp == m
        if s.any(): pp[s], pf[s] = Q[m][2][s], Q[m][3][s]
    perm = np.random.default_rng(draw).permutation(len(uv)); parts = [np.where(np.isin(ref['vid'], g))[0] + npool for g in np.array_split(uv[perm], 3)]
    idx = {m: i for i, m in enumerate(E.MS)}; grp = np.array([idx[m] for m in mp] + [idx[m] for m in mt])
    return np.concatenate([pp, pr]), np.concatenate([pf, ft]), y, rich, (np.arange(npool), parts[0], parts[1], parts[2]), None, ('hard', 'cross'), 'cross_entropy', grp, tuple(E.MS)
from rebuttal_common import _mosei_raw as _mosei_raw_public
def job_mix(j):
    try:
        P, F, Y, Rh, split, keep, tg, ln, grp, C = dict(mosei=mix_mosei, nina=mix_nina, iem=mix_iem)[j['bench']](**j['ctx'])
        loss = _L.get(ln); pool, fit, conf, test = split
        r = gates_ltt.gate_row(P, F, Y, split, loss_name=ln, keep=keep, targets=tg, richer=Rh)
        tt, cc, _ = rebuild(r['_meta'], P, F, Y, Rh, split, loss)
        mf, mc, mt = P[fit], P[conf], P[test]; yf, yc, yt = Y[fit], Y[conf], Y[test]; dl = loss(cc['test'], yt) - loss(mt, yt)
        gf, gc, gt = grp[fit], grp[conf], grp[test]; kt = None if keep is None else keep[test]; acc = acc_fn(yt, kt, False); base = acc(mt)
        harm_c = (loss(cc['conf'], yc) - loss(mc, yc)) > DELTA; gain_f = loss(mf, yf) - loss(cc['fit'], yf)
        sc = _A.fit_action_score(mf, tt['fit'], cc['fit'], yf, loss); sf, sv, st = [_A.score(sc, b, t, True) for b, t in ((mf, tt['fit']), (mc, tt['conf']), (mt, tt['test']))]
        none = np.zeros(len(test), bool); lc = _A.crc_threshold(sv, harm_c, ALPHA)
        ap = dict(blanket=np.ones(len(test), bool), pooled_crc=none if lc is None else st > lc, pooled_crc_t=none if lc is None else st > _A._tighten(sf, gain_f, lc))
        m_crc, m_crc_t = none.copy(), none.copy()
        for g in np.unique(gt):
            s = gc == g; f = gf == g; t = gt == g
            if s.sum() < 2: continue
            l = _A.crc_threshold(sv[s], harm_c[s], ALPHA)
            if l is None: continue
            m_crc |= t & (st > l)
            m_crc_t |= t & (st > (_A._tighten(sf[f], gain_f[f], l) if f.sum() else l))
        ap.update(mask_crc=m_crc, mask_crc_t=m_crc_t); out = {}
        for k, a in ap.items():
            per = {C[int(g)]: dict(harm=float((a & (dl > DELTA))[gt == g].mean()), apply=float(a[gt == g].mean()), n=int((gt == g).sum())) for g in np.unique(gt)}
            out[k] = dict(gain=acc(np.where(a[:, None], cc['test'], mt)) - base, harm=float((a & (dl > DELTA)).mean()), apply=float(a.mean()), per=per)
        return dict(j, kind='mix', out=out, meta=r['_meta'])
    except Exception:
        import traceback
        return dict(j, kind='mix', error=traceback.format_exc()[-1500:])
def mosei_c(c, seed):
    probs, FE, y, keep = _mosei_raw_public(); F = np.zeros((len(y), FE['a'].shape[1] + FE['v'].shape[1]))
    if 'a' in c: F[:, :FE['a'].shape[1]] = FE['a']
    if 'v' in c: F[:, FE['a'].shape[1]:] = FE['v']
    return probs(c), F, y, probs('tav'), tuple(np.array_split(np.random.default_rng(seed).permutation(len(y)), 4)), keep, ('hard', 'cross'), 'cross_entropy'
def job_pair(j):
    try:
        L = dict(mosei=mosei_c, iem=iemocap, nina=ninapro)[j['bench']]
        Pm, Fm, Ym, Rh, sp_m, keep, tg, ln = L(**j['mild']); Ps, Fs, Ys, Rs, sp_s, _, _, _ = L(**j['sev'])
        loss = _L.get(ln); pool, fit, conf, test = sp_s
        assert all(np.array_equal(a, b) for a, b in zip(sp_m[1:], sp_s[1:])); yt = Ys[test]
        assert np.array_equal(Ym[conf], Ys[conf]) and np.array_equal(Ym[test], Ys[test])
        kt = None if keep is None else keep[test]; acc = acc_fn(yt, kt, False); mt = Ps[test]; base = acc(mt)
        if j['scen'] == 's2':
            r = gates_ltt.gate_row(Pm, Fm, Ym, sp_m, loss_name=ln, keep=keep, targets=tg, richer=Rh); m = r['_meta']
            T, tgt, sp, wt, k, b = [m[x] for x in ('temperature', 'target', 'space', 'weighting', 'k', 'beta')]
            tt, cc, _ = rebuild(m, Pm, Fm, Ym, Rh, sp_m, loss)
            z = _T.retrieval_space(Fm[sp_m[0]], sp); vals = (_T.hard_label_values(Ym[sp_m[0]], Pm.shape[1], True) if tgt == 'hard' else _T.cross_mask_values((Rh if T == 1.0 else _T.temper(Rh, T, True))[sp_m[0]]))
            t_dep = _T.knn_average(z(Fs[test]), z(Fm[sp_m[0]]), vals, min(k, len(sp_m[0]) - 1), weighting=wt)
            pr_dep = Ps if T == 1.0 else _T.temper(Ps, T, True); c_dep = (1 - b) * pr_dep[test] + b * t_dep
            mf, mc = Pm[fit], Pm[conf]; tf, tc, cf, cv = tt['fit'], tt['conf'], cc['fit'], cc['conf']
        else:
            r = gates_ltt.gate_row(Ps, Fs, Ys, sp_s, loss_name=ln, keep=keep, targets=tg, richer=Rs); m = r['_meta']
            T, tgt, sp, wt, k, b = [m[x] for x in ('temperature', 'target', 'space', 'weighting', 'k', 'beta')]
            tt, cc, _ = rebuild(m, Ps, Fs, Ys, Rs, sp_s, loss)
            z = _T.retrieval_space(Fs[pool], sp); vals = (_T.hard_label_values(Ys[pool], Ps.shape[1], True) if tgt == 'hard' else _T.cross_mask_values((Rs if T == 1.0 else _T.temper(Rs, T, True))[pool]))
            tc = _T.knn_average(z(Fm[conf]), z(Fs[pool]), vals, min(k, len(pool) - 1), weighting=wt)
            pr_m = Pm if T == 1.0 else _T.temper(Pm, T, True); cv = (1 - b) * pr_m[conf] + b * tc
            mf, mc = Ps[fit], Pm[conf]; tf, cf = tt['fit'], cc['fit']; t_dep, c_dep = tt['test'], cc['test']
        yf, yc = Ys[fit], Ys[conf]; dl = loss(c_dep, yt) - loss(mt, yt)
        harm_c = (loss(cv, yc) - loss(mc, yc)) > DELTA; gain_f = loss(mf, yf) - loss(cf, yf); sc = _A.fit_action_score(mf, tf, cf, yf, loss)
        none = np.zeros(len(test), bool)
        if sc is None: ap = dict(crc=none, crc_t=none)
        else:
            sf, sv, st = _A.score(sc, mf, tf, True), _A.score(sc, mc, tc, True), _A.score(sc, mt, t_dep, True); lc = _A.crc_threshold(sv, harm_c, ALPHA)
            ap = dict(crc=none if lc is None else st > lc, crc_t=none if lc is None else st > _A._tighten(sf, gain_f, lc))
        ap['blanket'] = np.ones(len(test), bool)
        return dict(j, kind='pair', out={k: dict(gain=acc(np.where(a[:, None], c_dep, mt)) - base, harm=float((a & (dl > DELTA)).mean()), apply=float(a.mean())) for k, a in ap.items()}, meta=m, base=base)
    except Exception:
        import traceback
        return dict(j, kind='pair', error=traceback.format_exc()[-1500:])
def run(job_):
    return job_mix(job_) if 'scen' not in job_ else job_pair(job_)
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--procs', type=int, default=6); a = ap.parse_args()
    subs = nina_subjects('seed0'); J = []
    J += [dict(bench='mosei', ctx=dict(seed=s)) for s in range(10)]
    J += [dict(bench='nina', ctx=dict(sub=s, si=i)) for i, s in enumerate(subs)]
    J += [dict(bench='iem', ctx=dict(fold=f, draw=d, eta=e)) for f in range(5) for d in range(5) for e in (0.3, 0.5)]
    for sc in ('s1', 's2'):
        for mi, se in (('av', 'a'), ('av', 'v'), ('a', 'av'), ('v', 'av'), ('a', 'a'), ('v', 'v'), ('av', 'av')):
            J += [dict(bench='mosei', scen=sc, pair=[mi, se], mild=dict(c=mi, seed=s), sev=dict(c=se, seed=s)) for s in range(5)]
        for mi, se in (('at', 'a'), ('at', 't'), ('tv', 't'), ('tv', 'v'), ('av', 'a'), ('av', 'v'), ('a', 'at'), ('t', 'at'), ('t', 'tv'), ('v', 'tv'), ('a', 'av'), ('v', 'av'), ('a', 'a'), ('t', 't'), ('v', 'v'), ('at', 'at'), ('av', 'av'), ('tv', 'tv')):
            J += [dict(bench='iem', scen=sc, pair=[mi, se], mild=dict(f=f, m=mi, cut=0), sev=dict(f=f, m=se, cut=0)) for f in range(5)]
        for mi, se in (('12', '4'), ('12', '6'), ('8', '4'), ('4', '12'), ('12', '12'), ('6', '6'), ('4', '4')):
            J += [dict(bench='nina', scen=sc, pair=[mi, se], mild=dict(sub=s, c=mi), sev=dict(sub=s, c=se)) for s in subs]
    t0 = time.time(); print('jobs', len(J), flush=True)
    with Pool(a.procs) as p, open(OUT / 'shift.jsonl', 'w') as fh:
        for i, r in enumerate(p.imap_unordered(run, J, chunksize=1)):
            fh.write(json.dumps(jsonable(r)) + '\n'); fh.flush()
            if i % 25 == 0: print(i + 1, len(J), round(time.time() - t0), flush=True)
    print('DONE', round(time.time() - t0), flush=True)
