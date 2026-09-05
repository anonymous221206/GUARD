"""Non-exchangeable branch: conformal calibration on subject 3, deployment on another."""
from pathlib import Path
import numpy as np, sys, csv
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'experiments'))
sys.path.insert(0,str(ROOT / 'src'))
from gates_core import gate_row
from guard import losses as _L, targets as _T, certify as _C
from guard.pipeline import _select_beta
from sklearn.metrics import f1_score
D=str(ROOT / 'artifacts/opportunity_dcl_v2')
CFG=['low_cost_accels_only','no_imu_family','no_shoes','severe_three_sensors']
loss=_L.get('cross_entropy'); DELTA=0.05; ALPHAS=[0.05,0.10,0.20,0.30,0.50]
yd=np.load(f'{D}/deploy_y.npy'); yc=np.load(f'{D}/calib_y.npy')
rows=[]
for cfg in CFG:
    Fd=np.load(f'{D}/retfeat_{cfg}_deploy.npy').astype(np.float64)
    Fc=np.load(f'{D}/retfeat_{cfg}_calib.npy').astype(np.float64)
    for s in (0,1,2):
        Pd=np.load(f'{D}/probs_condition_specialist_{cfg}_deploy_s{s}.npy').astype(np.float64)
        Pc=np.load(f'{D}/probs_condition_specialist_{cfg}_calib_s{s}.npy').astype(np.float64)
        Rd=np.load(f'{D}/richer_deploy_s{s}.npy').astype(np.float64)
        rng=np.random.default_rng(s); perm=rng.permutation(len(yd)); q=len(yd)//4
        r=gate_row(Pd,Fd,yd,(perm[:q],perm[q:2*q],perm[2*q:3*q],perm[3*q:]),
                   targets=('hard','cross'),richer=Rd)
        pool,fit,test=perm[:q],perm[q:2*q],perm[3*q:]
        z=_T.retrieval_space(Fd[pool],r['_meta']['space'])
        fp,ff,ft=z(Fd[pool]),z(Fd[fit]),z(Fd[test]); fc=z(Fc)      # conf = subject 3
        vals=(_T.hard_label_values(yd[pool],Pd.shape[1],loss.simplex)
              if r['_meta']['target']=='hard' else _T.cross_mask_values(Rd[pool]))
        k_=r['_meta']['k']; wt=r['_meta']['weighting']
        tf=_T.knn_average(ff,fp,vals,k_,weighting=wt)
        b=_select_beta(Pd[fit],tf,yd[fit],loss,'loss')
        tc=_T.knn_average(fc,fp,vals,k_,weighting=wt); tt=_T.knn_average(ft,fp,vals,k_,weighting=wt)
        cc=(1-b)*Pc+b*tc; ct=(1-b)*Pd[test]+b*tt
        blo=loss(Pd[test],yd[test]); cl=loss(ct,yd[test]); hurt=(cl-blo)>DELTA
        wf=lambda Q: f1_score(yd[test],Q.argmax(1),average='weighted')
        base=wf(Pd[test]); bl=wf(ct)
        for al in ALPHAS:
            g=_C.certify(cc,yc,ct,Pd[test],loss,al,DELTA); ap=g['apply']
            gp=np.where(ap[:,None],ct,Pd[test])
            rows.append(dict(family='opportunity',dataset='opportunity_cross_subject',condition=cfg,
                             target=r['_meta']['target'],seed=s,exchangeable=False,alpha=al,
                             apply_rate=float(ap.mean()),joint_harm=float((ap&hurt).mean()),
                             acc_gain=wf(gp)-base,blanket_joint_harm=float(hurt.mean()),
                             blanket_acc_gain=bl-base))
    print('cfg',cfg,'xong',flush=True)
with open(str(ROOT / 'results/alpha/alpha_opp_nx.csv'),'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('DA GHI',len(rows))
