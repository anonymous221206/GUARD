"""Blanket correction for the two panels that lack it: IEMOCAP and AVE."""
from pathlib import Path
import numpy as np, sys, json, collections
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'src'))
from guard import losses as _L, targets as _T, certify as _C
from guard.pipeline import _select_beta
B=str(ROOT)
loss=_L.get('cross_entropy'); ALPHA,DELTA=0.2,0.05
KS=(5,10,20,35,50); SPACES=('standardise','cosine'); WTS=('uniform','distance')
def three(P,F,Y,split,rich=None):
    pool,fit,conf,test=split
    SP={sp:{q:_T.retrieval_space(F[pool],sp)(F[i]) for q,i in
            (('pool',pool),('fit',fit),('conf',conf),('test',test))} for sp in SPACES}
    V={'hard':_T.hard_label_values(Y[pool],P.shape[1],loss.simplex)}
    tg_list=['hard']
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
    g=_C.certify(cc,Y[conf],ct,P[test],loss,ALPHA,DELTA); ap=g['apply']
    gp=np.where(ap[:,None],ct,P[test])
    a=lambda Q: float((Q.argmax(1)==Y[test]).mean())
    return a(P[test]), a(ct), a(gp)
out={}
# ---- AVE
A=f'{B}/artifacts/ave_av_att/dumps'
def ld(f):
    d=np.load(f'{A}/{f}.npz',allow_pickle=True)
    return (d['probs'].reshape(-1,d['probs'].shape[-1]).astype(np.float64),
            d['labels'].reshape(-1,d['labels'].shape[-1]).argmax(1))
Pf,Y=ld('AV_att_full'); Pa,_=ld('AV_att_audio_only'); Pv,_=ld('AV_att_visual_only')
r=np.load(f'{A}/AV_att_audio_only_retrieval_paper.npz',allow_pickle=True)
F=r['deploy_features'].astype(np.float64); pF,pY,pP=r['pool_features'].astype(np.float64),r['pool_labels'],r['pool_probs'].astype(np.float64)
n=len(pY); res=[]
for eta in [0.0,0.2,0.4,0.6,0.8,1.0]:
    wa,wv,wf=eta*(1-eta),(1-eta)*eta,(1-eta)**2; tot=max(wa+wv+wf,1e-9)
    acc=[]
    for seed in range(3):
        rng=np.random.default_rng(seed)
        sel=rng.choice(3,size=len(Y),p=[wf/tot,wa/tot,wv/tot]) if tot>1e-6 else rng.choice([1,2],size=len(Y))
        P=np.where(sel[:,None]==0,Pf,np.where(sel[:,None]==1,Pa,Pv))
        PP=np.concatenate([pP,P]); FF=np.concatenate([pF,F]); YY=np.concatenate([pY,Y])
        RICH=np.concatenate([pP,Pf])
        perm=rng.permutation(len(Y))+n; k=len(perm)//3
        acc.append(three(PP,FF,YY,(np.arange(n),perm[:k],perm[k:2*k],perm[2*k:]),RICH))
    m=np.mean(acc,0); res.append(dict(eta=eta,frozen=m[0],blanket=m[1],guard=m[2]))
    print('ave',eta,np.round(m,4),flush=True)
out['ave']=res
# ---- IEMOCAP: per pattern, then reweighted to the missing-rate protocol
D=f'{B}/artifacts/iemocap_momke/folds'; MS=['a','t','v','at','av','tv','atv']
def key(d):
    c={}; o=[]
    for v in d['vid']:
        c[v]=c.get(v,0)+1; o.append((v,c[v]-1))
    return o
per={}
for m_ in MS:
    fr,bl,gu=[],[],[]
    for fo in range(5):
        ref=np.load(f'{D}/fold{fo}_atv.npz',allow_pickle=True); kr={k:i for i,k in enumerate(key(ref))}
        d=np.load(f'{D}/fold{fo}_{m_}.npz',allow_pickle=True)
        idx=np.array([kr[k] for k in key(d)]); N=len(ref['labels'])
        pr=np.zeros((N,d['probs'].shape[1])); pr[idx]=d['probs']
        ft=np.zeros((N,d['feats'].shape[1])); ft[idx]=d['feats']
        P=np.concatenate([d['pool_probs'].astype(np.float64),pr])
        F2=np.concatenate([d['pool_feats'].astype(np.float64),ft])
        Y2=np.concatenate([d['pool_labels'],ref['labels']]); npool=len(d['pool_labels'])
        uv=np.unique(ref['vid']); perm=np.random.default_rng(0).permutation(len(uv))
        parts=[np.where(np.isin(ref['vid'],g))[0]+npool for g in np.array_split(uv[perm],3)]
        a,b,c=three(P,F2,Y2,(np.arange(npool),parts[0],parts[1],parts[2]))
        fr.append(a); bl.append(b); gu.append(c)
    per[m_]=(np.mean(fr),np.mean(bl),np.mean(gu))
    print('iemocap',m_,np.round(per[m_],4),flush=True)
def w(m_,eta): return eta**(3-len(m_))*(1-eta)**len(m_)
res=[]
for eta in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]:
    Z=sum(w(m_,eta) for m_ in MS)
    v=[sum(w(m_,eta)/Z*per[m_][i] for m_ in MS) for i in range(3)]
    res.append(dict(eta=eta,frozen=v[0],blanket=v[1],guard=v[2]))
out['iemocap']=res
json.dump(out,open(str(ROOT / 'results/blanket_two.json'),'w')); print('saved')
