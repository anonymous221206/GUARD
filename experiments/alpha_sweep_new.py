"""Alpha sweep for the benchmarks missing from alpha_frontier_all.csv.
from pathlib import Path
The kNN and blend are computed once per cell; only Certify depends on alpha."""
import sys, json, csv, numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'src'))
sys.path.insert(0,str(ROOT / 'scripts'))
from guard import losses as _L, targets as _T, certify as _C
from guard.pipeline import _select_beta
from sklearn.metrics import roc_auc_score
B=str(ROOT / 'experiments')
ALPHAS=[0.05,0.10,0.20,0.30,0.50]; DELTA=0.05
KS=(5,10,20,35,50); SPACES=('standardise','cosine'); WTS=('uniform','distance')
def cell(P,F,Y,split,loss,score,tgt):
    pool,fit,conf,test=split
    SP={sp:{q:_T.retrieval_space(F[pool],sp)(F[i]) for q,i in (('pool',pool),('fit',fit),('conf',conf),('test',test))} for sp in SPACES}
    vals=_T.hard_label_values(Y[pool],P.shape[1],loss.simplex) if tgt=='hard' else None
    best=None
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
    bl=loss(P[test],Y[test]); cl=loss(ct,Y[test]); hurt=(cl-bl)>DELTA
    base=score(P[test],Y[test]); out=[]
    for al in ALPHAS:
        g=_C.certify(cc,Y[conf],ct,P[test],loss,al,DELTA); ap=g['apply']
        gp=np.where(ap[:,None],ct,P[test])
        out.append(dict(alpha=al,apply_rate=float(ap.mean()),joint_harm=float((ap&hurt).mean()),
                        acc_gain=score(gp,Y[test])-base,blanket_joint_harm=float(hurt.mean()),
                        blanket_acc_gain=score(ct,Y[test])-base))
    return out
rows=[]
BE=_L.get('bernoulli'); CE=_L.get('cross_entropy')
mac=lambda P,Y: float(roc_auc_score(Y,P,average='macro'))
amax=lambda P,Y: float((P.argmax(1)==Y).mean())
# PTB-XL
r=np.load(f'{B}/artifacts/ptbxl_resnet1d_wang/raw_features.npz',allow_pickle=True)
F=np.concatenate([r['train_a'],r['sess1_a']]).astype(np.float64)
for kk in [11,10,9,8,7,6,5,4,3,2]:
    d=np.load(f'{B}/artifacts/ptbxl_dropladder/drop{kk}.npz',allow_pickle=True)
    P=np.concatenate([d['train_probs'],d['sess1_probs']]).astype(np.float64)
    Y=np.concatenate([d['train_y'],d['sess1_y']]).astype(np.float64)
    n,m=len(d['train_y']),len(d['sess1_y'])
    for seed in range(3):
        perm=np.random.default_rng(seed).permutation(m)+n; q=m//3
        for o in cell(P,F,Y,(np.arange(n),perm[:q],perm[q:2*q],perm[2*q:]),BE,mac,'hard'):
            rows.append(dict(family='ptbxl',dataset='ptbxl',condition=f'drop{kk}',target='hard',seed=seed,exchangeable=True,**o))
    print('ptbxl',kk,flush=True)
with open(str(ROOT / 'results/alpha/alpha_new.csv'),'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('DA GHI',len(rows))
