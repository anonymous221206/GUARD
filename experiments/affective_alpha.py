"""Alpha sweep for CMU-MOSEI on the authoritative repro_check path."""
import sys, csv, numpy as np
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
ALPHAS=[0.05,0.10,0.20,0.30,0.50]
def probs(m):
    s=p[f'test_{m}'].astype(np.float64); e=1/(1+np.exp(-s)); return np.stack([1-e,e],1)
rows=[]
for cond, feats in (('a',['test_ac']), ('v',['test_vis']), ('av',['test_vis','test_ac'])):
    F=np.concatenate([r[k].astype(np.float64) for k in feats],1)
    host=HostOutputs(probs=probs(cond),features=F,labels=y,richer_probs=probs('tav'),raw_labels=raw)
    for seed in range(10):
        perm=np.random.default_rng(seed).permutation(len(y))
        pool,fit,conf,test=np.array_split(perm,4)
        sp=Split(pool,fit,conf,test,origin={k:'deployment' for k in ('pool','fit','conf','test')})
        c=select_on_fit(host,sp,k_grid=(3,5,8,12,20,35,50),target_grid=('hard','cross_mask'),
                        space_grid=('cosine',),weighting_grid=('distance',),
                        temperature_grid=(1.0,2.0),metric='accuracy_nonzero')
        for al in ALPHAS:
            res=run(host,sp,condition=cond,metric='accuracy_nonzero',alpha=al,delta=0.05,
                    k=c['k'],target=c['target'],space=c['space'],
                    weighting=c['weighting'],temperature=c['temperature'])
            rows.append(dict(family='affective',dataset='cmu_mosei',condition=cond,
                             target=c['target'],seed=seed,exchangeable=True,alpha=al,
                             apply_rate=res.apply_rate,joint_harm=res.joint_harm,
                             acc_gain=res.gate_metric_delta,
                             blanket_joint_harm=getattr(res,'blanket_joint_harm',float('nan')),
                             blanket_acc_gain=getattr(res,'blanket_metric_delta',float('nan'))))
    print('cond',cond,'xong',flush=True)
with open(str(_ROOT / 'results/alpha/alpha_affective.csv'),'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('DA GHI',len(rows))
