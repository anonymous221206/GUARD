import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'src'))
from guard import losses as _L, targets as _T, certify as _C
from guard.pipeline import _select_beta
B=str(ROOT)
ALPHA,DELTA=0.2,0.05; KS=(5,10,20,35,50); SPACES=('standardise','cosine'); WTS=('uniform','distance')
BE=_L.get('bernoulli')
from sklearn.metrics import roc_auc_score
ham=lambda P,Y: float(roc_auc_score(Y,P,average='macro'))
def run(P,F,Y,split,loss,score):
    pool,fit,conf,test=split
    SP={sp:{q:_T.retrieval_space(F[pool],sp)(F[i]) for q,i in (('pool',pool),('fit',fit),('conf',conf),('test',test))} for sp in SPACES}
    vals=_T.hard_label_values(Y[pool],P.shape[1],loss.simplex); best=None
    for sp in SPACES:
        for wt in WTS:
            for k in KS:
                tf=_T.knn_average(SP[sp]['fit'],SP[sp]['pool'],vals,k,weighting=wt)
                b=_select_beta(P[fit],tf,Y[fit],loss,'loss')
                sc=score((1-b)*P[fit]+b*tf,Y[fit])
                if best is None or sc>best[0]: best=(sc,sp,wt,k,b)
    _,sp,wt,k,b=best; f=SP[sp]
    tc=_T.knn_average(f['conf'],f['pool'],vals,k,weighting=wt); tt=_T.knn_average(f['test'],f['pool'],vals,k,weighting=wt)
    cc=(1-b)*P[conf]+b*tc; ct=(1-b)*P[test]+b*tt
    g=_C.certify(cc,Y[conf],ct,P[test],loss,ALPHA,DELTA); ap=g['apply']
    bl=loss(P[test],Y[test]); cl=loss(ct,Y[test]); gp=np.where(ap[:,None],ct,P[test])
    return (score(P[test],Y[test]),score(ct,Y[test]),score(gp,Y[test]),float(ap.mean()),
            float((ap&((cl-bl)>DELTA)).mean()),float((((cl-bl)>DELTA)).mean()))
r=np.load(f'{B}/artifacts/ptbxl_resnet1d_wang/raw_features.npz',allow_pickle=True)
F=np.concatenate([r['train_a'],r['sess1_a']]).astype(np.float64)   # limb leads: derivable from I,II
res=[]
for kk in [12,11,10,9,8,7,6,5,4,3,2]:
    d=np.load(f'{B}/artifacts/ptbxl_dropladder/drop{kk}.npz',allow_pickle=True)
    P=np.concatenate([d['train_probs'],d['sess1_probs']]).astype(np.float64)
    Y=np.concatenate([d['train_y'],d['sess1_y']]).astype(np.float64)
    n,m=len(d['train_y']),len(d['sess1_y']); acc=[]
    for seed in range(3):
        perm=np.random.default_rng(seed).permutation(m)+n; q=m//3
        acc.append(run(P,F,Y,(np.arange(n),perm[:q],perm[q:2*q],perm[2*q:]),BE,ham))
    a=np.mean(acc,0)
    res.append(dict(level=f'drop{kk}',leads=kk,frozen=a[0],blanket=a[1],guard=a[2],apply=a[3],harm=a[4],bharm=a[5]))
    print('ptbxl',kk,np.round(a[:3],4),flush=True)
json.dump(res,open(f'{B}/artifacts/ptbxl_dropladder/severity_dense_auc.json','w'),indent=1)
print('DA GHI')
