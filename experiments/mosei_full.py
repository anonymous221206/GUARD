"""Base metric, gated metric, joint harm and apply rate on CMU-MOSEI.

Same path as repro_check.py, so the numbers reconcile with Table 1 by construction.
"""
import sys, json, numpy as np
from guard import HostOutputs, run
from guard.pipeline import select_on_fit
from guard.splits import Split
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))

D=f'{ARTIFACTS}/mosei_cmad/dumps'
p=np.load(f'{D}/student_preds.npz',allow_pickle=True); r=np.load(f'{D}/raw_features.npz',allow_pickle=True)
raw=r['test_y'].reshape(-1); y=(raw>0).astype(int)
def probs(m):
    s=p[f'test_{m}'].astype(np.float64); e=1/(1+np.exp(-s)); return np.stack([1-e,e],1)
out={}
for cond, feats in (('a',['test_ac']), ('v',['test_vis']), ('av',['test_vis','test_ac']), ('tav',['test_vis','test_ac'])):
    F=np.concatenate([r[k].astype(np.float64) for k in feats],1)
    host=HostOutputs(probs=probs(cond),features=F,labels=y,richer_probs=probs('tav'),raw_labels=raw)
    b,g,h,ap,bh=[],[],[],[],[]
    for seed in range(10):
        perm=np.random.default_rng(seed).permutation(len(y))
        pool,fit,conf,test=np.array_split(perm,4)
        sp=Split(pool,fit,conf,test,origin={k:'deployment' for k in ('pool','fit','conf','test')})
        c=select_on_fit(host,sp,k_grid=(3,5,8,12,20,35,50),target_grid=('hard','cross_mask'),
                        space_grid=('cosine',),weighting_grid=('distance',),
                        temperature_grid=(1.0,2.0),metric='accuracy_nonzero')
        res=run(host,sp,condition=cond,metric='accuracy_nonzero',
                k=c['k'],target=c['target'],space=c['space'],
                weighting=c['weighting'],temperature=c['temperature'])
        b.append(res.base_metric); g.append(res.base_metric+res.gate_metric_delta)
        h.append(res.joint_harm); ap.append(res.apply_rate)
        bh.append(getattr(res,'blanket_joint_harm',float('nan')))
    out[cond]=dict(base=float(np.mean(b)),guard=float(np.mean(g)),harm=float(np.mean(h)),
                   apply=float(np.mean(ap)),blanket_harm=float(np.nanmean(bh)))
    o=out[cond]
    print("%-4s base %.4f  GUARD %.4f  harm %.4f  apply %.3f  blanket_harm %.4f"
          %(cond,o['base'],o['guard'],o['harm'],o['apply'],o['blanket_harm']),flush=True)
json.dump(out,open(str(_ROOT / 'results/gates/mosei_full.json'),'w'),indent=1)
print("DA GHI")
