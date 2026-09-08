"""How large is the blend weight, mask by mask, on the CMU-MOSEI hosts?

Section 5.1 reads the gain as tracking how much the frozen model left on the
table. The blend weight is the direct test of that: if the claim holds it should
collapse where the host still sees language and grow where it does not. Prints
the per-mask weights and the two group means the paper quotes.
"""
import numpy as np, sys, os, collections
from pathlib import Path
_R = Path('.')
sys.path.insert(0, 'src'); sys.path.insert(0, 'experiments')
from guard import HostOutputs, run
from guard.pipeline import select_on_fit
from guard.splits import Split
A = Path(os.environ.get('GUARD_ARTIFACTS', 'artifacts'))
HOSTS = {'CMAD': dict(dir='mosei_cmad/dumps', preds='student_preds.npz', alias={},
                      feat={'a':'ac','v':'vis'}),
         'TMDC': dict(dir='mosei_tmdc/dumps', preds='preds.npz', alias={'ta':'at','tav':'atv'}, feat={'a':'a','t':'t','v':'v'}),
         'MoMKE': dict(dir='mosei_momke/dumps', preds='preds.npz',
                       alias={'ta':'at','tav':'atv'}, feat={'a':'a','t':'t','v':'v'})}
MASKS = ['t','a','v','ta','tv','av','tav']
out = collections.defaultdict(list)
for name, cfg in HOSTS.items():
    D = A / cfg['dir']
    P = np.load(D/cfg['preds'], allow_pickle=True); R = np.load(D/'raw_features.npz', allow_pickle=True)
    raw = R['test_y'].reshape(-1); y = (raw > 0).astype(int)
    feats = {m: R[f'test_{k}'].astype(np.float64) for m,k in cfg['feat'].items() if f'test_{k}' in R.files}
    def probs(mask):
        k = cfg['alias'].get(mask, mask); s = P[f'test_{k}'].astype(np.float64)
        if s.ndim == 1 or s.shape[1] == 1:
            e = 1/(1+np.exp(-s.reshape(-1))); return np.stack([1-e, e], 1)
        e = np.exp(s - s.max(1, keepdims=True)); return e/e.sum(1, keepdims=True)
    rich = probs('tav')
    for mask in MASKS:
        cols = [feats[c] for c in mask if c in feats]
        if not cols: continue
        F = np.concatenate(cols, 1)
        for seed in range(3):
            perm = np.random.default_rng(seed).permutation(len(y))
            pool, fit, conf, test = np.array_split(perm, 4)
            host = HostOutputs(probs=probs(mask), features=F, labels=y,
                               richer_probs=rich, raw_labels=raw)
            sp = Split(pool=pool, fit=fit, conf=conf, test=test)
            c = select_on_fit(host, sp, k_grid=(3,5,8,12,20,35,50),
                              target_grid=('hard','cross_mask'),
                              space_grid=('standardise','cosine'),
                              weighting_grid=('uniform','distance'),
                              temperature_grid=(1.0,2.0), metric='accuracy_nonzero')
            r = run(host, sp, metric='accuracy_nonzero',
                    **{k: c[k] for k in ('k','target','space','weighting','temperature')})
            out[f'{name}|{mask}'].append(r.beta)
        b = out[f'{name}|{mask}']
        print('%-12s beta = %s   (co t: %s)' % (f'{name}|{mask}', [round(x,3) for x in b], 't' in mask), flush=True)
keep = [v for k,v in out.items() if 't' in k.split('|')[1] for v in v]
lose = [v for k,v in out.items() if 't' not in k.split('|')[1] for v in v]
print('\nmask GIU ngon ngu: %d cell, beta=0 o %d, trung binh %.3f' % (len(keep), sum(1 for x in keep if x==0), np.mean(keep)))
print('mask MAT ngon ngu: %d cell, beta=0 o %d, trung binh %.3f' % (len(lose), sum(1 for x in lose if x==0), np.mean(lose)))
