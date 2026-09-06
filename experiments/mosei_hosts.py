#!/usr/bin/env python3
"""GUARD on the three frozen CMU-MOSEI hosts, across every modality pattern.

This is the driver behind the main CMU-MOSEI table. It was the one piece of the
paper that lived outside the repository, which is why two of its three rows could
not be reproduced. The hosts disagree on two conventions: CMAD evaluates 4643
clips and names the language-audio mask "ta", the other two evaluate 4659 and
name it "at", so masks are mapped rather than assumed.
"""
import os, sys, json
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))
sys.path.insert(0, str(_ROOT / 'src'))
from guard import HostOutputs, run
from guard.pipeline import select_on_fit
from guard.splits import Split

MASKS = ['t', 'a', 'v', 'ta', 'tv', 'av', 'tav']
HOSTS = {
    # CMAD ships no text features, so masks that observe only text leave the
    # retrieval with nothing to search on and the frozen output stands.
    'CMAD':  dict(dir='mosei_cmad/dumps', preds='student_preds.npz', alias={},
                  feat={'a': 'ac', 'v': 'vis'}),
    'TMDC':  dict(dir='mosei_tmdc/dumps', preds='preds.npz', alias={'ta': 'at', 'tav': 'atv'},
                  feat={'a': 'a', 't': 't', 'v': 'v'}),
    'MoMKE': dict(dir='mosei_momke/dumps', preds='preds.npz', alias={'ta': 'at', 'tav': 'atv'},
                  feat={'a': 'a', 't': 't', 'v': 'v'}),
}
ALPHA, DELTA, SEEDS = 0.2, 0.05, 5

def acc_f1(p, y, keep):
    """Binary accuracy and weighted F1 over non-neutral clips, the CMU-MOSEI convention."""
    from sklearn.metrics import f1_score
    pred = p.argmax(1)[keep]; gold = y[keep]
    return 100 * float((pred == gold).mean()), 100 * f1_score(gold, pred, average='weighted')

out = {}
for name, cfg in HOSTS.items():
    D = ARTIFACTS / cfg['dir']
    P = np.load(D / cfg['preds'], allow_pickle=True)
    R = np.load(D / 'raw_features.npz', allow_pickle=True)
    raw = R['test_y'].reshape(-1); y = (raw > 0).astype(int); keep = raw != 0
    feats = {m: R[f'test_{k}'].astype(np.float64)
             for m, k in cfg['feat'].items() if f'test_{k}' in R.files}
    def probs(mask):
        k = cfg['alias'].get(mask, mask)
        s = P[f'test_{k}'].astype(np.float64)
        if s.ndim == 1 or s.shape[1] == 1:
            e = 1 / (1 + np.exp(-s.reshape(-1))); return np.stack([1 - e, e], 1)
        e = np.exp(s - s.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
    rich = probs('tav')
    for mask in MASKS:
        cols = [feats[c] for c in mask if c in feats]
        if not cols:                      # nothing to retrieve on: keep the host
            a, f = acc_f1(probs(mask), y, keep)
            out[f'{name}|{mask}'] = (a, f, a, f)
            print(f'{name:6} {mask:4} {a:5.1f}/{f:5.1f}  (khong co dac trung)', flush=True)
            continue
        F = np.concatenate(cols, 1)
        accs, f1s, bases, basef, phs, phses = [], [], [], [], [], []
        for seed in range(SEEDS):
            perm = np.random.default_rng(seed).permutation(len(y))
            pool, fit, conf, test = np.array_split(perm, 4)
            host = HostOutputs(probs=probs(mask), features=F, labels=y,
                               richer_probs=rich, raw_labels=raw)
            sp = Split(pool=pool, fit=fit, conf=conf, test=test)
            # the retrieval settings are chosen on the fit split, never on test
            cfg_r = select_on_fit(host, sp, k_grid=(5, 10, 20, 35, 50),
                                  target_grid=('hard', 'cross_mask'),
                                  space_grid=('standardise', 'cosine'),
                                  weighting_grid=('uniform', 'distance'))
            r = run(host, sp, alpha=ALPHA, delta=DELTA,
                    **{k: cfg_r[k] for k in ('k', 'target', 'space', 'weighting')
                       if k in cfg_r})
            gp = r.test_arrays['gated_probs']          # the policy's own output
            a, f = acc_f1(gp, y[test], keep[test])
            # the frozen host on the same rows, so the two are comparable
            ba, bf = acc_f1(r.test_arrays['base_probs'], y[test], keep[test])
            accs.append(a); f1s.append(f); bases.append(ba); basef.append(bf)
            phs.append(r.ph); phses.append(r.ph_se)
        out[f'{name}|{mask}'] = (float(np.mean(accs)), float(np.mean(f1s)),
                                  float(np.mean(bases)), float(np.mean(basef)))
        d=np.array(accs)-np.array(bases); se=d.std(ddof=1)/np.sqrt(len(d)) if len(d)>1 else 0.0
        print(f'{name:6} {mask:4} frozen {np.mean(bases):5.1f} -> GUARD {np.mean(accs):5.1f}'
      f'  delta {d.mean():+5.2f} +- {se:.2f}  PH {np.mean(phs):.4f}', flush=True)
(_ROOT / 'results/gates').mkdir(parents=True, exist_ok=True)
json.dump(out, open(_ROOT / 'results/gates/mosei_hosts.json', 'w'), indent=1)
print('DA GHI')
