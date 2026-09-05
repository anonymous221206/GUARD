"""Mixed availability: one calibration set, many conditions at deployment.

Every table so far calibrates and tests inside a single availability pattern, which is
the friendliest possible setting for a rule that tunes one scalar. A deployed model does
not get that: the pattern varies sample by sample and the operator calibrates once. Here
each sample draws its own pattern, features are zero-filled where a modality is absent,
and both rules see the same mixture. Harm is reported pooled and per pattern.
"""
import numpy as np, sys, json, collections
from gates_ltt import gate_row
from gates_core import ALPHA, DELTA
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))

D=f'{ARTIFACTS}/mosei_cmad/dumps'
p=np.load(f'{D}/student_preds.npz',allow_pickle=True); r=np.load(f'{D}/raw_features.npz',allow_pickle=True)
raw=r['test_y'].reshape(-1); y=(raw>0).astype(int); keep=raw!=0
A=r['test_ac'].astype(np.float64); V=r['test_vis'].astype(np.float64)
def probs(m):
    s=p[f'test_{m}'].astype(np.float64); e=1/(1+np.exp(-s)); return np.stack([1-e,e],1)
CONDS=('a','v','av'); PR={c:probs(c) for c in CONDS}; RICH=probs('tav')
n=len(y); res=collections.defaultdict(list); percond=collections.defaultdict(lambda: collections.defaultdict(list))
for seed in range(10):
    rng=np.random.default_rng(seed)
    ci=rng.integers(0,3,n)                      # each sample gets its own pattern
    P=np.zeros((n,2)); F=np.zeros((n,A.shape[1]+V.shape[1]))
    for j,c in enumerate(CONDS):
        m=ci==j; P[m]=PR[c][m]
        if 'a' in c: F[m,:A.shape[1]]=A[m]
        if 'v' in c: F[m,A.shape[1]:]=V[m]
    perm=rng.permutation(n); sp=np.array_split(perm,4)
    row=gate_row(P,F,y,sp,keep=keep,targets=('hard','cross'),richer=RICH,groups=ci)
    RULES=('blanket','confidence','GUARD','LTT-confidence','LTT-agreement','LTT-learned','LTT-mask-confidence','LTT-mask-agreement','LTT-mask-learned','GUARD-mask')
    for k in RULES:
        if k in row: res[k].append(row[k])
    ct_=ci[sp[3]]; hm=row['_harmful']
    for k in RULES:
        if k not in row['_apply']: continue
        ap=row['_apply'][k]
        for j,c in enumerate(CONDS):
            m=ct_==j
            percond[k][c].append(float((ap[m]&hm[m]).mean()))
print(f"{'rule':17}{'gain':>9}{'harm':>8}")
out={}
for k in [r for r in RULES if r in res]:
    g=np.nanmean([x[0] for x in res[k]]); h=np.nanmean([x[1] for x in res[k]])
    out[k]=(float(g),float(h)); print(f'{k:17}{g:+9.4f}{h:8.3f}')
print()
print('harm theo tung pattern trong cung mot hon hop (budget %.2f)'%ALPHA)
print(f"{'rule':17}" + ''.join(f'{c:>9}' for c in CONDS))
pc={}
for k in percond:
    v=[float(np.mean(percond[k][c])) for c in CONDS]; pc[k]=v
    print(f'{k:17}' + ''.join(f'{x:9.3f}' for x in v))
json.dump({'pooled':out,'per_condition':pc},open('mixed_deploy.json','w'))
