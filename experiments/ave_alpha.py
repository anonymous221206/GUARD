import numpy as np, sys, csv
from guard import losses as _L, targets as _T, certify as _C
from guard.pipeline import _select_beta
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))

loss=_L.get('cross_entropy'); DELTA=0.05; ALPHAS=[0.05,0.10,0.20,0.30,0.50]
KS=(5,10,20,35,50); SPACES=('standardise','cosine'); WTS=('uniform','distance')
def sweep(P,F,Y,split,rich=None):
    pool,fit,conf,test=split
    SP={sp:{q:_T.retrieval_space(F[pool],sp)(F[i]) for q,i in
            (('pool',pool),('fit',fit),('conf',conf),('test',test))} for sp in SPACES}
    V={'hard':_T.hard_label_values(Y[pool],P.shape[1],loss.simplex)}; tg_list=['hard']
    if rich is not None: V['cross']=_T.cross_mask_values(rich[pool]); tg_list.append('cross')
    best=None
    for tg in tg_list:
        for sp in SPACES:
            for wt in WTS:
                for k in KS:
                    tf=_T.knn_average(SP[sp]['fit'],SP[sp]['pool'],V[tg],k,weighting=wt)
                    b=_select_beta(P[fit],tf,Y[fit],loss,'loss')
                    sc=float((((1-b)*P[fit]+b*tf).argmax(1)==Y[fit]).mean())
                    if best is None or sc>best[0]: best=(sc,tg,sp,wt,k,b)
    _,tg,sp,wt,k,b=best; f=SP[sp]
    tc=_T.knn_average(f['conf'],f['pool'],V[tg],k,weighting=wt)
    tt=_T.knn_average(f['test'],f['pool'],V[tg],k,weighting=wt)
    cc=(1-b)*P[conf]+b*tc; ct=(1-b)*P[test]+b*tt
    bl=loss(P[test],Y[test]); cl=loss(ct,Y[test]); hurt=(cl-bl)>DELTA
    a=lambda Q: float((Q.argmax(1)==Y[test]).mean()); base=a(P[test]); out=[]
    for al in ALPHAS:
        g=_C.certify(cc,Y[conf],ct,P[test],loss,al,DELTA); ap=g['apply']
        gp=np.where(ap[:,None],ct,P[test])
        out.append(dict(alpha=al,target=tg,apply_rate=float(ap.mean()),joint_harm=float((ap&hurt).mean()),
                        acc_gain=a(gp)-base,blanket_joint_harm=float(hurt.mean()),blanket_acc_gain=a(ct)-base))
    return out
A=str(ARTIFACTS / 'ave_av_att/dumps')
def ld(f):
    d=np.load(f'{A}/{f}.npz',allow_pickle=True)
    return (d['probs'].reshape(-1,d['probs'].shape[-1]).astype(np.float64),
            d['labels'].reshape(-1,d['labels'].shape[-1]).argmax(1))
Pf,Y=ld('AV_att_full'); Pa,_=ld('AV_att_audio_only'); Pv,_=ld('AV_att_visual_only')
r=np.load(f'{A}/AV_att_audio_only_retrieval_paper.npz',allow_pickle=True)
F=r['deploy_features'].astype(np.float64)
pF,pY,pP=r['pool_features'].astype(np.float64),r['pool_labels'],r['pool_probs'].astype(np.float64)
n=len(pY); rows=[]
for eta in [0.2,0.4,0.6,0.8,1.0]:
    wa,wv,wf=eta*(1-eta),(1-eta)*eta,(1-eta)**2; tot=max(wa+wv+wf,1e-9)
    for seed in range(3):
        rng=np.random.default_rng(seed)
        sel=rng.choice(3,size=len(Y),p=[wf/tot,wa/tot,wv/tot]) if tot>1e-6 else rng.choice([1,2],size=len(Y))
        P=np.where(sel[:,None]==0,Pf,np.where(sel[:,None]==1,Pa,Pv))
        PP=np.concatenate([pP,P]); FF=np.concatenate([pF,F]); YY=np.concatenate([pY,Y]); RICH=np.concatenate([pP,Pf])
        perm=rng.permutation(len(Y))+n; k=len(perm)//3
        for o in sweep(PP,FF,YY,(np.arange(n),perm[:k],perm[k:2*k],perm[2*k:]),RICH):
            rows.append(dict(family='ave',dataset='ave',condition=f'eta{eta}',seed=seed,exchangeable=True,**o))
    print('ave',eta,'xong',flush=True)
with open(str(_ROOT / 'results/alpha/alpha_ave.csv'),'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('DA GHI',len(rows))
