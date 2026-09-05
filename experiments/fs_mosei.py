"""Calibration-resampling spread on the frozen CMAD host, MOSEI."""
import numpy as np, sys, json, collections
from finite_sample import curve
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))

D=f'{ARTIFACTS}/mosei_cmad/dumps'
p=np.load(f'{D}/student_preds.npz',allow_pickle=True); r=np.load(f'{D}/raw_features.npz',allow_pickle=True)
raw=r['test_y'].reshape(-1); y=(raw>0).astype(int); keep=raw!=0
FE={'v':r['test_vis'].astype(np.float64),'a':r['test_ac'].astype(np.float64)}
def probs(m):
    s=p[f'test_{m}'].astype(np.float64); e=1/(1+np.exp(-s)); return np.stack([1-e,e],1)
agg=collections.defaultdict(list); NC=None
for cond in ('a','v','av'):
    F=np.concatenate([FE[m] for m in cond],1)
    for seed in range(5):
        perm=np.random.default_rng(seed).permutation(len(y))
        sp=np.array_split(perm,4)
        res,nc=curve(probs(cond),F,y,sp,keep=keep,targets=('hard','cross'),richer=probs('tav'))
        NC=nc
        for n,(m,s) in res.items(): agg[n].append(s)
print('conf day du =',NC)
print(f"{'n':>6}{'sd harm':>10}")
out={}
for n in sorted(agg):
    v=float(np.mean(agg[n])); out[n]=v; print(f'{n:6d}{v:10.4f}')
json.dump(out,open(str(_ROOT / 'results/gates/fs_mosei.json'),'w'))
