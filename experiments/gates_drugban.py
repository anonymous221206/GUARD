"""Gate comparison on the frozen DrugBAN host, over its nine dataset-split-condition cells."""
import numpy as np, sys, json, collections, glob, os
from gates_core import gate_row
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))
os.makedirs(_ROOT / 'results/gates', exist_ok=True)

ROOT=str(_ROOT / 'data/processed')
RULES=['blanket','random','confidence','agreement','learned','GUARD']
CELLS=[('drugban_biosnap_random_s42',c) for c in ('prot25','prot50','scaffold')] + \
      [('drugban_bindingdb_random_s42',c) for c in ('prot25','prot50','scaffold')] + \
      [('drugban_human_random_s42',c) for c in ('prot25','prot50')] + \
      [('drugban_biosnap_cluster_s42','prot50')]
acc=collections.defaultdict(list)
for d,cond in CELLS:
    p=f'{ROOT}/{d}/{cond}.npz'
    if not os.path.exists(p): print('skip',p); continue
    z=np.load(p,allow_pickle=True)
    P=np.concatenate([z['pool_probs'],z['calib_probs'],z['test_probs']]).astype(np.float64)
    F=np.concatenate([z['pool_feats'],z['calib_feats'],z['test_feats']]).astype(np.float64)
    Y=np.concatenate([z['pool_labels'],z['calib_labels'],z['test_labels']])
    npool,ncal=len(z['pool_labels']),len(z['calib_labels'])
    # the richer condition on the same rows supplies the label-free target
    rp=f'{ROOT}/{d}/full.npz'
    R=None
    if os.path.exists(rp):
        zr=np.load(rp,allow_pickle=True)
        R=np.concatenate([zr['pool_probs'],zr['calib_probs'],zr['test_probs']]).astype(np.float64)
    for seed in range(3):
        perm=np.random.default_rng(seed).permutation(ncal)+npool
        split=(np.arange(npool), perm[:ncal//2], perm[ncal//2:],
               np.arange(len(z['test_labels']))+npool+ncal)
        r=gate_row(P,F,Y,split,targets=('hard','cross') if R is not None else ('hard',),richer=R)
        for k_ in RULES: acc[k_].append(r[k_])
        acc['_apply'].append(r['_meta']['apply']); acc['_base'].append(r['_meta']['base'])
    print('done',d,cond,flush=True)
print(f"{'rule':11}{'gain':>9}{'harm':>8}")
res={}
for k_ in RULES:
    g=np.nanmean([x[0] for x in acc[k_]]); h=np.nanmean([x[1] for x in acc[k_]])
    res[k_]=(float(g),float(h)); print(f'{k_:11}{g:+9.4f}{h:8.3f}')
print(f"base {np.mean(acc['_base']):.4f}  apply {np.mean(acc['_apply']):.3f}  cells {len(acc['GUARD'])}")
json.dump(res,open(str(_ROOT / 'results/gates/gates_drugban.json'),'w'))
