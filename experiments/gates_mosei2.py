"""Gate comparison on the frozen CMAD host, MOSEI, through the shared core."""
import numpy as np, sys, json, collections
from gates_core import gate_row
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))
os.makedirs(_ROOT / 'results/gates', exist_ok=True)

D=f'{ARTIFACTS}/mosei_cmad/dumps'
p=np.load(f'{D}/student_preds.npz',allow_pickle=True); r=np.load(f'{D}/raw_features.npz',allow_pickle=True)
raw=r['test_y'].reshape(-1); y=(raw>0).astype(int); keep=raw!=0
FE={'v':r['test_vis'].astype(np.float64),'a':r['test_ac'].astype(np.float64)}
def probs(m):
    s=p[f'test_{m}'].astype(np.float64); e=1/(1+np.exp(-s)); return np.stack([1-e,e],1)
RULES=['blanket','random','confidence','agreement','learned','GUARD']
acc=collections.defaultdict(list)
for cond in ('a','v','av'):
    F=np.concatenate([FE[m] for m in cond],1)
    for seed in range(10):
        perm=np.random.default_rng(seed).permutation(len(y))
        pool,fit,conf,test=np.array_split(perm,4)
        row=gate_row(probs(cond),F,y,(pool,fit,conf,test),keep=keep,
                     targets=('hard','cross'),richer=probs('tav'))
        for k in RULES: acc[k].append(row[k])
        acc['_a'].append(row['_meta']['apply']); acc['_b'].append(row['_meta']['base'])
out={}
print(f"{'rule':11}{'gain':>9}{'harm':>8}")
for k in RULES:
    g=np.nanmean([x[0] for x in acc[k]]); h=np.nanmean([x[1] for x in acc[k]])
    out[k]=(float(g),float(h)); print(f'{k:11}{g:+9.4f}{h:8.3f}')
print(f"base {np.mean(acc['_b']):.4f}  apply {np.mean(acc['_a']):.3f}")
json.dump(out,open(str(_ROOT / 'results/gates/gates_mosei2.json'),'w'))
