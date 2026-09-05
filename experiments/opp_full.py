"""OPPORTUNITY on the DeepConvLSTM hosts the paper actually corrects.

Reads the dumps left by dcl_hosts2.py rather than retraining: a second training run
does not reproduce the first closely enough to mix outputs, as that script records.
"""
import numpy as np, sys, json, collections
from gates_core import gate_row
from guard import losses as _L, targets as _T, certify as _C
from guard.pipeline import _select_beta
from sklearn.metrics import f1_score
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))

D=f'{ARTIFACTS}/opportunity_dcl_v2'
CFG=['low_cost_accels_only','no_imu_family','no_shoes','severe_three_sensors']
RULES=['blanket','random','confidence','agreement','learned','GUARD']
loss=_L.get('cross_entropy'); ALPHA,DELTA=0.2,0.05
y=np.load(f'{D}/deploy_y.npy')
print(f"{'config':24}{'seed':>5}{'base acc':>10}{'wtd F1':>9}")
gates=collections.defaultdict(list); sev={}
for cfg in CFG:
    F=np.load(f'{D}/retfeat_{cfg}_deploy.npy').astype(np.float64)
    bs,gs,bl_,ap_,hm_=[],[],[],[],[]; ab,ag,bh_=[],[],[]
    for s in (0,1,2):
        P=np.load(f'{D}/probs_condition_specialist_{cfg}_deploy_s{s}.npy').astype(np.float64)
        rich=np.load(f'{D}/richer_deploy_s{s}.npy').astype(np.float64)
        acc=float((P.argmax(1)==y).mean()); wf=f1_score(y,P.argmax(1),average='weighted')
        print(f'{cfg:24}{s:5d}{acc:10.4f}{wf:9.4f}',flush=True)
        rng=np.random.default_rng(s); perm=rng.permutation(len(y)); q=len(y)//4
        split=(perm[:q],perm[q:2*q],perm[2*q:3*q],perm[3*q:])
        r=gate_row(P,F,y,split,targets=('hard','cross'),richer=rich)
        for k in RULES: gates[k].append(r[k])
        gates['_a'].append(r['_meta']['apply'])
        # the severity panel needs frozen / blanket / GUARD on the same split
        pool,fit,conf,test=split
        z=_T.retrieval_space(F[pool],r['_meta']['space'])
        fp,ff,fc,ft=z(F[pool]),z(F[fit]),z(F[conf]),z(F[test])
        vals=(_T.hard_label_values(y[pool],P.shape[1],loss.simplex) if r['_meta']['target']=='hard'
              else _T.cross_mask_values(rich[pool]))
        k_=r['_meta']['k']; wt=r['_meta']['weighting']
        tf=_T.knn_average(ff,fp,vals,k_,weighting=wt)
        b=_select_beta(P[fit],tf,y[fit],loss,'loss')
        tc=_T.knn_average(fc,fp,vals,k_,weighting=wt); tt=_T.knn_average(ft,fp,vals,k_,weighting=wt)
        cc=(1-b)*P[conf]+b*tc; ct=(1-b)*P[test]+b*tt
        g=_C.certify(cc,y[conf],ct,P[test],loss,ALPHA,DELTA); ap=g['apply']
        blo=loss(P[test],y[test]); cl=loss(ct,y[test])
        gp=np.where(ap[:,None],ct,P[test])
        wf_=lambda Q: f1_score(y[test],Q.argmax(1),average="weighted"); ac_=lambda Q: float((Q.argmax(1)==y[test]).mean())
        bs.append(wf_(P[test])); bl_.append(wf_(ct)); gs.append(wf_(gp))
        ab.append(ac_(P[test])); ag.append(ac_(gp)); bh_.append(float((((cl-blo)>DELTA)).mean()))
        ap_.append(float(ap.mean())); hm_.append(float((ap&((cl-blo)>DELTA)).mean()))
    sev[cfg]=dict(frozen=float(np.mean(bs)),blanket=float(np.mean(bl_)),guard=float(np.mean(gs)),
                  apply=float(np.mean(ap_)),harm=float(np.mean(hm_)),acc_frozen=float(np.mean(ab)),acc_guard=float(np.mean(ag)),bharm=float(np.mean(bh_)))
    print(f"  -> wtdF1 frozen {sev[cfg]['frozen']:.4f} blanket {sev[cfg]['blanket']:.4f} GUARD {sev[cfg]['guard']:.4f}",flush=True)
out={}
print(f"\n{'rule':11}{'gain':>9}{'harm':>8}")
for k in RULES:
    g=np.nanmean([x[0] for x in gates[k]]); h=np.nanmean([x[1] for x in gates[k]])
    out[k]=(float(g),float(h)); print(f'{k:11}{g:+9.4f}{h:8.3f}')
json.dump({'gates':out,'severity':sev},open(str(_ROOT / 'results/gates/opportunity_dcl_full.json'),'w'))
