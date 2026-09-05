"""Mixed availability on NinaPro: each window draws its own electrode count.

Same question as mixed_deploy.py on a second benchmark and a longer ladder, so the
comparison does not rest on one dataset. Per subject: pool is the training session,
the deployment session is split into fit, calibration and test thirds, and each
deployment window is assigned an electrode count at random.
"""
import numpy as np, sys, json, glob, collections
from gates_ltt import gate_row
from gates_core import ALPHA
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))

CONDS=('12','8','6','4')
RULES=('blanket','confidence','GUARD','LTT-confidence','LTT-agreement','LTT-learned','LTT-mask-confidence','LTT-mask-agreement','LTT-mask-learned','GUARD-mask')
res=collections.defaultdict(list); percond=collections.defaultdict(lambda: collections.defaultdict(list))
subs=sorted(glob.glob('artifacts/ninapro_cnn/seed*/subject*'))[:10]
for si,sub in enumerate(subs):
    p=np.load(f'{sub}/preds.npz',allow_pickle=True); e=np.load(f'{sub}/masked_embeddings.npz',allow_pickle=True)
    ntr=len(p['train_y']); m=len(p['sess1_y'])
    rng=np.random.default_rng(si); ci=rng.integers(0,len(CONDS),m)
    P0=np.stack([p[f'train_{c}'] for c in CONDS]); F0=np.stack([e[f'train_{c}'] for c in CONDS])
    # pool keeps a fixed pattern per row too, drawn the same way
    cp=rng.integers(0,len(CONDS),ntr)
    Ptr=np.stack([P0[cp[i],i] for i in range(ntr)]); Ftr=np.stack([F0[cp[i],i] for i in range(ntr)])
    Pd=np.stack([p[f'sess1_{c}'][i] for i,c in enumerate([CONDS[j] for j in ci])])
    Fd=np.stack([e[f'sess1_{c}'][i] for i,c in enumerate([CONDS[j] for j in ci])])
    P=np.concatenate([Ptr,Pd]).astype(np.float64); P=np.clip(P,1e-12,None); P/=P.sum(1,keepdims=True)
    F=np.concatenate([Ftr,Fd]).astype(np.float64)
    Y=np.concatenate([p['train_y'],p['sess1_y']]).astype(int)
    perm=rng.permutation(m)+ntr; k=m//3
    sp=(np.arange(ntr),perm[:k],perm[k:2*k],perm[2*k:])
    grp=np.concatenate([cp,ci])
    row=gate_row(P,F,Y,sp,groups=grp)
    for r_ in RULES:
        if r_ in row: res[r_].append(row[r_])
    ct_=ci[perm[2*k:]-ntr]; hm=row['_harmful']
    for r_ in RULES:
        if r_ not in row['_apply']: continue
        ap=row['_apply'][r_]
        for j,c in enumerate(CONDS):
            mm=ct_==j
            if mm.sum(): percond[r_][c].append(float((ap[mm]&hm[mm]).mean()))
print(f"{'rule':17}{'gain':>9}{'harm':>8}")
pooled={}
for k_ in [r for r in RULES if r in res]:
    g=np.nanmean([x[0] for x in res[k_]]); h=np.nanmean([x[1] for x in res[k_]])
    pooled[k_]=(float(g),float(h)); print(f'{k_:17}{g:+9.4f}{h:8.3f}')
print()
print('harm theo tung so dien cuc trong cung mot hon hop (budget %.2f)'%ALPHA)
print(f"{'rule':17}"+''.join(f'{c:>9}' for c in CONDS))
pc={}
for k_ in percond:
    v=[float(np.mean(percond[k_][c])) for c in CONDS]; pc[k_]=v
    print(f'{k_:17}'+''.join(f'{x:9.3f}' for x in v))
json.dump({'pooled':pooled,'per_condition':pc,'raw':{k_:{c:percond[k_][c] for c in CONDS} for k_ in percond}},
          open('mixed_nina.json','w'))
