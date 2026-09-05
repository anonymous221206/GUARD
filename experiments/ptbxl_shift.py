"""Dangerous-direction shift: calibrate at a mild lead set, deploy at a severe one."""
from pathlib import Path
import numpy as np, sys, csv
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'src'))
from guard import losses as _L, targets as _T, certify as _C
from guard.pipeline import _select_beta
B=str(ROOT / 'experiments')
BE=_L.get('bernoulli'); ham=lambda P,Y: float((((P>0.5)==(Y>0.5)).mean(1)).mean())
ALPHAS=[0.05,0.10,0.20,0.30,0.50]; DELTA=0.05
KS=(5,10,20,35,50); SPACES=('standardise','cosine'); WTS=('uniform','distance')
r=np.load(f'{B}/artifacts/ptbxl_resnet1d_wang/raw_features.npz',allow_pickle=True)
F=np.concatenate([r['train_a'],r['sess1_a']]).astype(np.float64)
def load(k):
    d=np.load(f'{B}/artifacts/ptbxl_dropladder/drop{k}.npz',allow_pickle=True)
    P=np.concatenate([d['train_probs'],d['sess1_probs']]).astype(np.float64)
    Y=np.concatenate([d['train_y'],d['sess1_y']]).astype(np.float64)
    return P,Y,len(d['train_y']),len(d['sess1_y'])
rows=[]
for mild,severe in [(12,4),(12,2),(10,3),(9,2)]:
    Pm,Y,n,m=load(mild); Ps,_,_,_=load(severe)
    for seed in range(3):
        perm=np.random.default_rng(seed).permutation(m)+n; q=m//3
        pool,fit,conf,test=np.arange(n),perm[:q],perm[q:2*q],perm[2*q:]
        SP={sp:{k2:_T.retrieval_space(F[pool],sp)(F[i]) for k2,i in
                (('pool',pool),('fit',fit),('conf',conf),('test',test))} for sp in SPACES}
        vals=_T.hard_label_values(Y[pool],Ps.shape[1],BE.simplex); best=None
        for sp in SPACES:
            for wt in WTS:
                for k3 in KS:
                    tf=_T.knn_average(SP[sp]['fit'],SP[sp]['pool'],vals,k3,weighting=wt)
                    b=_select_beta(Ps[fit],tf,Y[fit],BE,'loss')
                    sc=ham((1-b)*Ps[fit]+b*tf,Y[fit])
                    if best is None or sc>best[0]: best=(sc,sp,wt,k3,b)
        _,sp,wt,k3,b=best; f=SP[sp]
        tc=_T.knn_average(f['conf'],f['pool'],vals,k3,weighting=wt)
        tt=_T.knn_average(f['test'],f['pool'],vals,k3,weighting=wt)
        # calibration sees the MILD condition, deployment is SEVERE
        cc=(1-b)*Pm[conf]+b*tc
        ct=(1-b)*Ps[test]+b*tt
        blo=BE(Ps[test],Y[test]); cl=BE(ct,Y[test]); hurt=(cl-blo)>DELTA
        base=ham(Ps[test],Y[test])
        for al in ALPHAS:
            g=_C.certify(cc,Y[conf],ct,Ps[test],BE,al,DELTA); ap=g['apply']
            gp=np.where(ap[:,None],ct,Ps[test])
            rows.append(dict(family='ptbxl',dataset='ptbxl_severity_shift',
                             condition=f'calib{mild}_deploy{severe}',target='hard',seed=seed,
                             exchangeable=False,alpha=al,apply_rate=float(ap.mean()),
                             joint_harm=float((ap&hurt).mean()),acc_gain=ham(gp,Y[test])-base,
                             blanket_joint_harm=float(hurt.mean()),blanket_acc_gain=ham(ct,Y[test])-base))
    print('calib',mild,'-> deploy',severe,'xong',flush=True)
with open(str(ROOT / 'results/alpha/alpha_ptbxl_shift.csv'),'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('DA GHI',len(rows))
