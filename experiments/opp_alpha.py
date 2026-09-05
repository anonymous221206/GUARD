import numpy as np, sys, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import os
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', ARTIFACTS))
sys.path.insert(0,str(ROOT / 'experiments'))
sys.path.insert(0,str(ROOT / 'src'))
from gates_core import gate_row
from guard import losses as _L, targets as _T, certify as _C
from guard.pipeline import _select_beta
from sklearn.metrics import f1_score
D=str(ARTIFACTS / 'opportunity_dcl_v2')
CFG=['low_cost_accels_only','no_imu_family','no_shoes','severe_three_sensors']
loss=_L.get('cross_entropy'); DELTA=0.05; ALPHAS=[0.05,0.10,0.20,0.30,0.50]
y=np.load(f'{D}/deploy_y.npy')
rows=[]
for cfg in CFG:
    F=np.load(f'{D}/retfeat_{cfg}_deploy.npy').astype(np.float64)
    for s in (0,1,2):
        P=np.load(f'{D}/probs_condition_specialist_{cfg}_deploy_s{s}.npy').astype(np.float64)
        rich=np.load(f'{D}/richer_deploy_s{s}.npy').astype(np.float64)
        rng=np.random.default_rng(s); perm=rng.permutation(len(y)); q=len(y)//4
        split=(perm[:q],perm[q:2*q],perm[2*q:3*q],perm[3*q:])
        r=gate_row(P,F,y,split,targets=('hard','cross'),richer=rich)
        pool,fit,conf,test=split
        z=_T.retrieval_space(F[pool],r['_meta']['space'])
        fp,ff,fc,ft=z(F[pool]),z(F[fit]),z(F[conf]),z(F[test])
        vals=(_T.hard_label_values(y[pool],P.shape[1],loss.simplex)
              if r['_meta']['target']=='hard' else _T.cross_mask_values(rich[pool]))
        k_=r['_meta']['k']; wt=r['_meta']['weighting']
        tf=_T.knn_average(ff,fp,vals,k_,weighting=wt)
        b=_select_beta(P[fit],tf,y[fit],loss,'loss')
        tc=_T.knn_average(fc,fp,vals,k_,weighting=wt); tt=_T.knn_average(ft,fp,vals,k_,weighting=wt)
        cc=(1-b)*P[conf]+b*tc; ct=(1-b)*P[test]+b*tt
        blo=loss(P[test],y[test]); cl=loss(ct,y[test]); hurt=(cl-blo)>DELTA
        wf=lambda Q: f1_score(y[test],Q.argmax(1),average='weighted')
        base=wf(P[test]); bl=wf(ct)
        for al in ALPHAS:
            g=_C.certify(cc,y[conf],ct,P[test],loss,al,DELTA); ap=g['apply']
            gp=np.where(ap[:,None],ct,P[test])
            rows.append(dict(family='opportunity',dataset='opportunity',condition=cfg,
                             target=r['_meta']['target'],seed=s,exchangeable=True,alpha=al,
                             apply_rate=float(ap.mean()),joint_harm=float((ap&hurt).mean()),
                             acc_gain=wf(gp)-base,blanket_joint_harm=float(hurt.mean()),
                             blanket_acc_gain=bl-base))
    print('cfg',cfg,'xong',flush=True)
with open(str(ROOT / 'results/alpha/alpha_opportunity.csv'),'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('DA GHI',len(rows))
