"""Does the released code reproduce the paper's CMAD row? Audio-only, ten splits."""
from pathlib import Path
import sys, numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'src'))
from guard import HostOutputs, run
from guard.pipeline import select_on_fit
from guard.splits import Split
D=str(ROOT / 'artifacts/mosei_cmad/dumps')
p=np.load(f'{D}/student_preds.npz',allow_pickle=True); r=np.load(f'{D}/raw_features.npz',allow_pickle=True)
raw=r['test_y'].reshape(-1); y=(raw>0).astype(int)
def probs(m):
    s=p[f'test_{m}'].astype(np.float64); e=1/(1+np.exp(-s)); return np.stack([1-e,e],1)
for cond, feats in (('a',['test_ac']), ('v',['test_vis']), ('av',['test_vis','test_ac'])):
    F=np.concatenate([r[k].astype(np.float64) for k in feats],1)
    host=HostOutputs(probs=probs(cond), features=F, labels=y,
                     richer_probs=probs('tav'), raw_labels=raw)
    b,g=[],[]
    for seed in range(10):
        perm=np.random.default_rng(seed).permutation(len(y))
        pool,fit,conf,test=np.array_split(perm,4)
        sp=Split(pool,fit,conf,test,origin={k:'deployment' for k in ('pool','fit','conf','test')})
        c=select_on_fit(host,sp,k_grid=(3,5,8,12,20,35,50),
                        target_grid=('hard','cross_mask'),
                        space_grid=('cosine',), weighting_grid=('distance',),
                        temperature_grid=(1.0,2.0),metric='accuracy_nonzero')
        res=run(host,sp,condition=cond,metric='accuracy_nonzero',
                k=c['k'],target=c['target'],space=c['space'],
                weighting=c['weighting'],temperature=c['temperature'])
        b.append(res.base_metric); g.append(res.base_metric+res.gate_metric_delta)
    print(f"{cond:3}  released code {100*np.mean(b):.1f} -> {100*np.mean(g):.1f}")
print("paper    a  63.1 -> 68.6 | v  63.7 -> 68.3 | av  64.5 -> 70.3")
