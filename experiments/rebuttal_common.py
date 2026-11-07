"""Shared loaders and policy metrics for the rebuttal experiments.

Loaders mirror the gate drivers exactly, so the deployed cell of each comparison
reproduces the saved GUARD cell and serves as the control.
"""
import os, sys, glob, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent / 'src')]
ROOT = _HERE.parent
A = str(Path(os.environ.get('GUARD_ARTIFACTS', ROOT / 'artifacts')))
OUT = ROOT / 'results/rebuttal'; OUT.mkdir(exist_ok=True)
from guard import action as _A, losses as _L, targets as _T
from guard.losses import softmax
from ltt_core import ltt_threshold
from opp_split import deploy_split
from iemocap_io import align_pool
ALPHA, DELTA, ZETA = 0.2, 0.05, 0.05
MS = ['a', 't', 'v', 'at', 'av', 'tv', 'atv']

def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        o = float(o); return None if o != o else (o if abs(o) == float('inf') else round(o, 12))
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o

# ---- loaders: each returns probs, feats, labels, richer, split, keep, targets, loss_name
def _mosei_raw():
    D = f'{A}/mosei_cmad/dumps'
    p = np.load(f'{D}/student_preds.npz', allow_pickle=True); r = np.load(f'{D}/raw_features.npz', allow_pickle=True)
    raw = r['test_y'].reshape(-1); y = (raw > 0).astype(int); keep = raw != 0
    FE = {'v': r['test_vis'].astype(np.float64), 'a': r['test_ac'].astype(np.float64)}
    def probs(m):
        s = p[f'test_{m}'].astype(np.float64); e = 1 / (1 + np.exp(-s)); return np.stack([1 - e, e], 1)
    return probs, FE, y, keep
def mosei(cond, seed):
    probs, FE, y, keep = _mosei_raw()
    F = np.concatenate([FE[m] for m in cond], 1)
    split = tuple(np.array_split(np.random.default_rng(seed).permutation(len(y)), 4))
    return probs(cond), F, y, probs('tav'), split, keep, ('hard', 'cross'), 'cross_entropy'
def _vid_key(vid):
    cnt, out = {}, []
    for v in vid:
        cnt[v] = cnt.get(v, 0) + 1; out.append((v, cnt[v] - 1))
    return out
def iemocap(f, m, cut):
    D = f'{A}/iemocap_momke/folds'
    ref = np.load(f'{D}/fold{f}_atv.npz', allow_pickle=True)
    kr = {k: i for i, k in enumerate(_vid_key(ref['vid']))}; n = len(ref['labels'])
    rich = np.zeros((n, ref['probs'].shape[1])); rich[[kr[k] for k in _vid_key(ref['vid'])]] = softmax(ref['probs'])
    d = np.load(f'{D}/fold{f}_{m}.npz', allow_pickle=True)
    idx = np.array([kr[k] for k in _vid_key(d['vid'])])
    pr = np.zeros((n, d['probs'].shape[1])); pr[idx] = softmax(d['probs'])
    ft = np.zeros((n, d['feats'].shape[1])); ft[idx] = d['feats']
    P = np.concatenate([softmax(d['pool_probs']), pr]); F = np.concatenate([d['pool_feats'].astype(np.float64), ft])
    Y = np.concatenate([d['pool_labels'], ref['labels']]); Rh = np.concatenate([softmax(ref['pool_probs'])[align_pool(ref, d)], rich]); npool = len(d['pool_labels'])
    uv = np.unique(ref['vid']); perm = np.random.default_rng(cut).permutation(len(uv))
    parts = [np.where(np.isin(ref['vid'], g))[0] + npool for g in np.array_split(uv[perm], 3)]
    return P, F, Y, Rh, (np.arange(npool), parts[0], parts[1], parts[2]), None, ('hard', 'cross'), 'cross_entropy'
def opportunity(cfg, s):
    D = f'{A}/opportunity_dcl_v2'; y = np.load(f'{D}/deploy_y.npy'); F = np.load(f'{D}/retfeat_{cfg}_deploy.npy').astype(np.float64)
    P = np.load(f'{D}/probs_condition_specialist_{cfg}_deploy_s{s}.npy').astype(np.float64); rich = np.load(f'{D}/richer_deploy_s{s}.npy').astype(np.float64)
    return P, F, y, rich, deploy_split(len(y), s), None, ('hard', 'cross'), 'cross_entropy'
def ave(cond, seed):
    d = np.load(f'{A}/ave_av_att/dumps/AV_att_{cond}.npz', allow_pickle=True); r = np.load(f'{A}/ave_av_att/dumps/AV_att_{cond}_retrieval_paper.npz', allow_pickle=True)
    P = d['probs'].reshape(-1, d['probs'].shape[-1]).astype(np.float64); Y = d['labels'].reshape(-1, d['labels'].shape[-1]).argmax(1)
    F = r['deploy_features'].astype(np.float64); pP = r['pool_probs'].astype(np.float64); pF = r['pool_features'].astype(np.float64); pY = r['pool_labels']
    n = len(pY); perm = np.random.default_rng(seed).permutation(len(Y)) + n; k = len(perm) // 3
    return np.concatenate([pP, P]), np.concatenate([pF, F]), np.concatenate([pY, Y]), None, (np.arange(n), perm[:k], perm[k:2*k], perm[2*k:]), None, ('hard',), 'cross_entropy'
def nina_subjects(seed='seed*'):
    """The first ten subjects of each seed, relative to the artifact folder."""
    return [os.path.relpath(s, A) for d in sorted(glob.glob(f'{A}/ninapro_cnn/{seed}')) for s in sorted(glob.glob(f'{d}/subject*'))[:10]]

def ninapro(sub, c):
    p = np.load(f'{A}/{sub}/preds.npz', allow_pickle=True); e = np.load(f'{A}/{sub}/masked_embeddings.npz', allow_pickle=True)
    P = np.concatenate([p[f'train_{c}'], p[f'sess1_{c}']]).astype(np.float64); P = np.clip(P, 1e-12, None); P /= P.sum(1, keepdims=True)
    F = np.concatenate([e[f'train_{c}'], e[f'sess1_{c}']]).astype(np.float64); Y = np.concatenate([p['train_y'], p['sess1_y']]).astype(int)
    n = len(p['train_y']); m = len(p['sess1_y']); perm = np.random.default_rng(0).permutation(m) + n; k = m // 3
    return P, F, Y, None, (np.arange(n), perm[:k], perm[k:2*k], perm[2*k:]), None, ('hard',), 'cross_entropy'
def ptbxl(c, fk, seed):
    D = f'{A}/ptbxl_resnet1d_wang'; p = np.load(D + '/preds.npz', allow_pickle=True); r = np.load(D + '/raw_features.npz', allow_pickle=True)
    P = np.concatenate([p[f'train_{c}'], p[f'sess1_{c}']]).astype(np.float64); F = np.concatenate([r[f'train_{fk}'], r[f'sess1_{fk}']]).astype(np.float64)
    Y = np.concatenate([r['train_y'], r['sess1_y']]).astype(np.float64); n = len(r['train_y']); m = len(r['sess1_y'])
    perm = np.random.default_rng(seed).permutation(m) + n; k = m // 3
    return P, F, Y, None, (np.arange(n), perm[:k], perm[k:2*k], perm[2*k:]), None, ('hard',), 'bernoulli'
def drugban(d, cond, seed):
    z = np.load(f'{A}/drugban_processed/{d}/{cond}.npz', allow_pickle=True)
    P = np.concatenate([z['pool_probs'], z['calib_probs'], z['test_probs']]).astype(np.float64); F = np.concatenate([z['pool_feats'], z['calib_feats'], z['test_feats']]).astype(np.float64)
    Y = np.concatenate([z['pool_labels'], z['calib_labels'], z['test_labels']]); npool, ncal = len(z['pool_labels']), len(z['calib_labels'])
    zr = np.load(f'{A}/drugban_processed/{d}/full.npz', allow_pickle=True); R = np.concatenate([zr['pool_probs'], zr['calib_probs'], zr['test_probs']]).astype(np.float64)
    perm = np.random.default_rng(seed).permutation(ncal) + npool
    return P, F, Y, R, (np.arange(npool), perm[:ncal//2], perm[ncal//2:], np.arange(len(z['test_labels'])) + npool + ncal), None, ('hard', 'cross'), 'cross_entropy'
def drugban_deploy(dataset, seed):
    from exp_drugban import build
    h, s = build(Path(A) / f'drugban_processed/drugban_{dataset}_cluster_s42', 'prot50', 'deployment', seed)
    return h.probs, np.asarray(h.features, dtype=np.float64), h.labels, h.richer_probs, (s.pool, s.fit, s.conf, s.test), None, ('hard', 'cross'), 'cross_entropy'

# ---- policy metrics
def acc_fn(yt, keep_t, multilabel):
    def acc(P):
        ok = ((P > 0.5) == (yt > 0.5)).mean(1) if multilabel else (P.argmax(1) == yt)
        return float(ok[keep_t].mean() if keep_t is not None else ok.mean())
    return acc
def policy_row(ap, ct, mt, yt, dl, base_acc, acc, keep_t, multilabel):
    k = np.ones(len(yt), bool) if keep_t is None else keep_t
    H = dl > DELTA
    o = dict(gain=acc(np.where(ap[:, None], ct, mt)) - base_acc, harm=float((ap & H).mean()), apply=float(ap.mean()),
             eff_apply=float((ap & (np.abs(ct - mt).sum(1) > 1e-9)).mean()), dce=float((dl * ap).mean()))
    if ap.any():
        o.update(mean_d=float(dl[ap].mean()), help=float((dl[ap] < 0).mean()))
    if multilabel:
        pb, pc, y = mt > .5, ct > .5, yt > .5; a = ap[:, None]
        o.update(flip_m=float((a & (pb == y) & (pc != y)).mean()), flip_p=float((a & (pb != y) & (pc == y)).mean()))
        return o
    pb, pc = mt.argmax(1), ct.argmax(1); s = ap & k; n = max(int(k.sum()), 1)
    o.update(flip_m=float((s & (pb == yt) & (pc != yt)).sum() / n), flip_p=float((s & (pb != yt) & (pc == yt)).sum() / n))
    if s.any():
        m = max(int(s.sum()), 1); h = s & H
        o.update(n_app=int(s.sum()), harm_k=float(h.sum() / m), err_corr=float((pc[s] != yt[s]).mean()), err_base=float((pb[s] != yt[s]).mean()),
                 amb_corr=float((1 - ct[s].max(1)).mean()), h_right=float((h & (pc == yt)).sum() / m), h_flip=float((h & (pb == yt) & (pc != yt)).sum() / m),
                 h_wrong=float((h & (pb != yt) & (pc != yt)).sum() / m))
    return o
def rebuild(meta, P, F, Y, Rh, split, loss, roles=('fit', 'conf', 'test')):
    pool, fit, conf, test = split; idx = dict(fit=fit, conf=conf, test=test)
    T, tg, sp, wt, k, b = [meta[x] for x in ('temperature', 'target', 'space', 'weighting', 'k', 'beta')]
    z = _T.retrieval_space(F[pool], sp); Fp = z(F[pool])
    pr = P if T == 1.0 else _T.temper(P, T, loss.simplex)
    vals = (_T.hard_label_values(Y[pool], P.shape[1], loss.simplex) if tg == 'hard' else _T.cross_mask_values((Rh if T == 1.0 else _T.temper(Rh, T, loss.simplex))[pool]))
    ke = min(k, len(pool) - 1); tt, cc = {}, {}
    for n in roles:
        tt[n] = _T.knn_average(z(F[idx[n]]), Fp, vals, ke, weighting=wt); cc[n] = (1 - b) * pr[idx[n]] + b * tt[n]
    return tt, cc, pr
def factorial(r, P, F, Y, Rh, split, loss_name, keep):
    """Same candidate, five scores, CRC or LTT, with and without the fit-split tightening."""
    loss = _L.get(loss_name); multilabel = not loss.simplex; pool, fit, conf, test = split
    tt, cc, pr = rebuild(r['_meta'], P, F, Y, Rh, split, loss)
    mf, mc, mt = P[fit], P[conf], P[test]; yf, yc, yt = Y[fit], Y[conf], Y[test]
    dl = loss(cc['test'], yt) - loss(mt, yt)
    assert np.array_equal(dl > DELTA, r['_harmful']), 'reconstruction mismatch'
    kt = None if keep is None else keep[test]; acc = acc_fn(yt, kt, multilabel); base = acc(mt)
    harm_c = (loss(cc['conf'], yc) - loss(mc, yc)) > DELTA; gain_f = loss(mf, yf) - loss(cc['fit'], yf)
    sc = _A.fit_action_score(mf, tt['fit'], cc['fit'], yf, loss)
    conf_score = (lambda b: -np.abs(b - .5).mean(1)) if multilabel else (lambda b: -b.max(1))
    def scores(b, t, n):
        return dict(learned=_A.score(sc, b, t, loss.simplex), confidence=conf_score(b), distance=np.abs(b - t).sum(1),
                    near=-np.abs(b - t).sum(1), random=np.random.default_rng(0).random(n))
    sf, sv, st = scores(mf, tt['fit'], len(fit)), scores(mc, tt['conf'], len(conf)), scores(mt, tt['test'], len(test))
    none = np.zeros(len(test), bool); out = {}
    H = dl > DELTA; changed = np.abs(cc['test'] - mt).sum(1) > 1e-9
    for name in sf:
        if name == 'learned' and sc is None:
            z = policy_row(none, cc['test'], mt, yt, dl, base, acc, kt, multilabel)
            out[name] = dict(crc=z, ltt=z, crc_t=z, ltt_t=z, lam=[None] * 4, same_tight=True); continue
        lc = _A.crc_threshold(sv[name], harm_c, ALPHA); ll = ltt_threshold(sv[name], harm_c, ALPHA, ZETA, grid_scores=sf[name])
        lct = None if lc is None else _A._tighten(sf[name], gain_f, lc); llt = None if ll is None else _A._tighten(sf[name], gain_f, ll)
        a_c = none if lc is None else st[name] > lc; a_l = none if ll is None else st[name] >= ll
        a_ct = none if lct is None else st[name] > lct; a_lt = none if llt is None else st[name] > llt
        row = lambda a: policy_row(a, cc['test'], mt, yt, dl, base, acc, kt, multilabel)
        out[name] = dict(crc=row(a_c), ltt=row(a_l), crc_t=row(a_ct), ltt_t=row(a_lt), lam=[lc, ll, lct, llt], same_tight=bool(np.array_equal(a_ct, a_lt)),
                         auc_help=_L.auroc(st[name][changed], (dl[changed] < 0).astype(int)) if changed.sum() > 1 else None,
                         auc_harm=_L.auroc(-st[name][changed], H[changed].astype(int)) if changed.sum() > 1 else None)
    ptemp = pr[test]
    diag = dict(nll_base=float(loss(mt, yt).mean()), nll_corr=float(loss(cc['test'], yt).mean()), nll_temp=float(loss(ptemp, yt).mean()),
                conf_base=float(mt.max(1).mean()), z_rate_fit=float((gain_f > 0).mean()), noop=bool(np.abs(cc['test'] - mt).sum(1).max() < 1e-9),
                model_none=sc is None, n_test=len(test), n_fit=len(fit), n_conf=len(conf), n_pool=len(pool))
    return dict(fact=out, blanket=policy_row(np.ones(len(test), bool), cc['test'], mt, yt, dl, base, acc, kt, multilabel), diag=diag)
