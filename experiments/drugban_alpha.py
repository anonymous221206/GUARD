import numpy as np, sys, csv, os, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'experiments'))
sys.path.insert(0,str(ROOT / 'src'))
from gates_core import gate_row
from guard import losses as _L, targets as _T, certify as _C
from guard.pipeline import _select_beta
DATA_ROOT=ROOT / 'data/processed'
CELLS=[('drugban_biosnap_random_s42',c) for c in ('prot25','prot50','scaffold')] +       [('drugban_bindingdb_random_s42',c) for c in ('prot25','prot50','scaffold')] +       [('drugban_human_random_s42',c) for c in ('prot25','prot50')] +       [('drugban_biosnap_cluster_s42','prot50')]
loss=_L.get('cross_entropy'); DELTA=0.05; ALPHAS=[0.05,0.10,0.20,0.30,0.50]
acc=lambda Q,Y: float((Q.argmax(1)==Y).mean())
rows=[]
for d,cond in CELLS:
    p=str(DATA_ROOT / d / f'{cond}.npz')
    if not os.path.exists(p): print('skip',p); continue
    z=np.load(p,allow_pickle=True)
    P=np.concatenate([z['pool_probs'],z['calib_probs'],z['test_probs']]).astype(np.float64)
    F=np.concatenate([z['pool_feats'],z['calib_feats'],z['test_feats']]).astype(np.float64)
    Y=np.concatenate([z['pool_labels'],z['calib_labels'],z['test_labels']])
    npool,ncal=len(z['pool_labels']),len(z['calib_labels'])
    rp=str(DATA_ROOT / d / 'full.npz'); R=None
    if os.path.exists(rp):
        zr=np.load(rp,allow_pickle=True)
        R=np.concatenate([zr['pool_probs'],zr['calib_probs'],zr['test_probs']]).astype(np.float64)
    exch = 'cluster' not in d
    for seed in range(3):
        perm=np.random.default_rng(seed).permutation(ncal)+npool
        split=(np.arange(npool),perm[:ncal//2],perm[ncal//2:],np.arange(len(z['test_labels']))+npool+ncal)
        r=gate_row(P,F,Y,split,targets=('hard','cross') if R is not None else ('hard',),richer=R)
        pool,fit,conf,test=split
        zz=_T.retrieval_space(F[pool],r['_meta']['space'])
        fp,ff,fc,ft=zz(F[pool]),zz(F[fit]),zz(F[conf]),zz(F[test])
        vals=(_T.hard_label_values(Y[pool],P.shape[1],loss.simplex)
              if r['_meta']['target']=='hard' else _T.cross_mask_values(R[pool]))
        k_=r['_meta']['k']; wt=r['_meta']['weighting']
        tf=_T.knn_average(ff,fp,vals,k_,weighting=wt)
        b=_select_beta(P[fit],tf,Y[fit],loss,'loss')
        tc=_T.knn_average(fc,fp,vals,k_,weighting=wt); tt=_T.knn_average(ft,fp,vals,k_,weighting=wt)
        cc=(1-b)*P[conf]+b*tc; ct=(1-b)*P[test]+b*tt
        blo=loss(P[test],Y[test]); cl=loss(ct,Y[test]); hurt=(cl-blo)>DELTA
        base=acc(P[test],Y[test]); bl=acc(ct,Y[test])
        for al in ALPHAS:
            g=_C.certify(cc,Y[conf],ct,P[test],loss,al,DELTA); ap=g['apply']
            gp=np.where(ap[:,None],ct,P[test])
            rows.append(dict(family='drugban',dataset=d,condition=cond,target=r['_meta']['target'],
                             seed=seed,exchangeable=exch,alpha=al,apply_rate=float(ap.mean()),
                             joint_harm=float((ap&hurt).mean()),acc_gain=acc(gp,Y[test])-base,
                             blanket_joint_harm=float(hurt.mean()),blanket_acc_gain=bl-base))
    print('done',d,cond,flush=True)
with open(str(ROOT / 'results/alpha/alpha_drugban.csv'),'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('DA GHI',len(rows))
