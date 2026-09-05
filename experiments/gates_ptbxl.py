"""Gate comparison on the frozen PTB-XL host. Multi-label, so the loss is Bernoulli
and accuracy is per diagnostic superclass."""
import numpy as np, sys, json, collections
from gates_core import gate_row
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))

A=f'{ARTIFACTS}/ptbxl_resnet1d_wang'
RULES=['blanket','random','confidence','agreement','learned','GUARD']
p=np.load(A+'/preds.npz',allow_pickle=True); r=np.load(A+'/raw_features.npz',allow_pickle=True)
acc=collections.defaultdict(list)
for c,fk in (('a','a'),('t','t'),('v','v'),('at','a'),('av','a'),('tv','t')):
    P=np.concatenate([p[f'train_{c}'],p[f'sess1_{c}']]).astype(np.float64)
    F=np.concatenate([r[f'train_{fk}'],r[f'sess1_{fk}']]).astype(np.float64)
    Y=np.concatenate([r['train_y'],r['sess1_y']]).astype(np.float64)
    n=len(r['train_y']); m=len(r['sess1_y'])
    for seed in range(3):
        perm=np.random.default_rng(seed).permutation(m)+n; k=m//3
        row=gate_row(P,F,Y,(np.arange(n),perm[:k],perm[k:2*k],perm[2*k:]),loss_name='bernoulli')
        for r_ in RULES: acc[r_].append(row[r_])
        acc['_a'].append(row['_meta']['apply']); acc['_b'].append(row['_meta']['base'])
    print('done',c,flush=True)
out={}
print(f"{'rule':11}{'gain':>9}{'harm':>8}")
for k in RULES:
    g=np.nanmean([x[0] for x in acc[k]]); h=np.nanmean([x[1] for x in acc[k]])
    out[k]=(float(g),float(h)); print(f'{k:11}{g:+9.4f}{h:8.3f}')
print(f"base {np.mean(acc['_b']):.4f}  apply {np.mean(acc['_a']):.3f}  cells {len(acc['GUARD'])}")
json.dump(out,open(str(_ROOT / 'results/gates/gates_ptbxl.json'),'w'))
