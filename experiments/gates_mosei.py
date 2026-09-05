"""Same frozen model, same correction, different gate.

Isolates what the certificate buys: a learned gate (the principle behind
trained whether-to-repair methods) and confidence gates are all held to the
same apply rate as GUARD, so only the decision rule differs.
"""
import numpy as np, sys, json, collections
from guard import losses as _L, targets as _T, certify as _C
from sklearn.linear_model import LogisticRegression
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))
os.makedirs(_ROOT / 'results/gates', exist_ok=True)

D=f'{ARTIFACTS}/mosei_cmad/dumps'
p=np.load(f'{D}/student_preds.npz',allow_pickle=True)
r=np.load(f'{D}/raw_features.npz',allow_pickle=True)
raw=r['test_y'].reshape(-1); y=(raw>0).astype(int); keep=raw!=0
VIS=r['test_vis'].astype(np.float64); AC=r['test_ac'].astype(np.float64)
loss=_L.get('cross_entropy'); GRID=np.linspace(0,1,41); ALPHA,DELTA=0.2,0.05
AVAIL={'a':[AC],'v':[VIS],'av':[VIS,AC]}
KS=[3,5,8,12,20,35,50]; TS=[1.0,2.0]; TGTS=['hard','cross']
def probs(m,T):
    s=p['test_'+m].astype(np.float64)/T; e=1/(1+np.exp(-s)); return np.stack([1-e,e],1)
def acc_nz(P,idx): m=keep[idx]; return float((P.argmax(1)==y[idx])[m].mean())
def embed(F,pool):
    X=F[pool]; mu=X.mean(0); sd=X.std(0); sd[sd==0]=1
    Z=(F-mu)/sd; return Z/np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
def neigh(Q,P,kmax):
    d=((Q**2).sum(1)[:,None]+(P**2).sum(1)[None,:]-2*Q@P.T)
    o=np.argsort(d,axis=1)[:,:kmax]; return o,np.take_along_axis(d,o,1)
def blend(V,o,dist,k):
    dd=np.sqrt(np.maximum(dist[:,:k],0)); tau=np.median(dd[:,-1:])+1e-12
    w=np.exp(-dd/tau); w/=w.sum(1,keepdims=True)
    return (V[o[:,:k]]*w[:,:,None]).sum(1)
def feats_for_gate(m,t,o,dist,k):
    ent=-(m*np.log(np.clip(m,1e-12,None))).sum(1)
    return np.column_stack([m.max(1), ent, np.abs(m[:,1]-m[:,0]), t.max(1),
        (m.argmax(1)==t.argmax(1)).astype(float), np.abs(m-t).sum(1),
        np.sqrt(np.maximum(dist[:,:k],0)).mean(1)])
def at_rate(score,rate):
    n=len(score); nk=int(round(rate*n))
    out=np.zeros(n,bool)
    if nk>0: out[np.argsort(-score)[:nk]]=True
    return out

rows=[]
for cond in ['a','v','av']:
    F=np.concatenate(AVAIL[cond],1)
    for seed in range(10):
        perm=np.random.default_rng(seed).permutation(len(y))
        pool,fit,conf,test=np.array_split(perm,4)
        Z=embed(F,pool); P=Z[pool]
        NB={n:neigh(Z[i],P,max(KS)) for n,i in (('fit',fit),('conf',conf),('test',test))}
        best=None
        for T in TS:
            V={'hard':_T.hard_label_values(y[pool],2,loss.simplex),
               'cross':_T.cross_mask_values(probs('tav',T)[pool])}
            mf=probs(cond,T)[fit]
            for tg in TGTS:
                for k in KS:
                    tf=blend(V[tg],*NB['fit'],k)
                    b=float(min(GRID,key=lambda b: loss((1-b)*mf+b*tf,y[fit]).mean()))
                    sc=acc_nz((1-b)*mf+b*tf,fit)
                    if best is None or sc>best[0]: best=(sc,T,tg,k,b)
        _,T,tg,k,b=best
        V={'hard':_T.hard_label_values(y[pool],2,loss.simplex),
           'cross':_T.cross_mask_values(probs('tav',T)[pool])}[tg]
        tf,tc,tt=(blend(V,*NB[n],k) for n in ('fit','conf','test'))
        mf,mc,mt=(probs(cond,T)[i] for i in (fit,conf,test))
        cf,cc,ct=((1-b)*x+b*t for x,t in ((mf,tf),(mc,tc),(mt,tt)))
        bl=loss(mt,y[test]); cl=loss(ct,y[test]); dl=cl-bl
        base=acc_nz(mt,test)
        def score_row(name,ap):
            gp=np.where(ap[:,None],ct,mt)
            return dict(cond=cond,seed=seed,gate=name,gain=acc_nz(gp,test)-base,
                        apply=float(ap.mean()),harm=float((ap&(dl>DELTA)).mean()))
        g=_C.certify(cc,y[conf],ct,mt,loss,ALPHA,DELTA); apG=g['apply']
        R=float(apG.mean())
        rows.append(score_row('GUARD',apG))
        rows.append(score_row('blanket',np.ones(len(test),bool)))
        # learned gate: trained on D_fit to predict "correction helps"
        Xf=feats_for_gate(mf,tf,*NB['fit'],k); zf=(loss(cf,y[fit])<loss(mf,y[fit])).astype(int)
        Xt=feats_for_gate(mt,tt,*NB['test'],k)
        if len(set(zf))>1:
            lr=LogisticRegression(max_iter=2000).fit(Xf,zf)
            rows.append(score_row('learned',at_rate(lr.predict_proba(Xt)[:,1],R)))
        rows.append(score_row('maxprob',at_rate(-mt.max(1),R)))
        rows.append(score_row('margin',at_rate(-np.abs(mt[:,1]-mt[:,0]),R)))
        rows.append(score_row('entropy',at_rate((-(mt*np.log(np.clip(mt,1e-12,None))).sum(1)),R)))
        rows.append(score_row('agree',at_rate((mt.argmax(1)==tt.argmax(1)).astype(float),R)))
        rng=np.random.default_rng(1000+seed)
        rows.append(score_row('random',at_rate(rng.random(len(test)),R)))
json.dump(rows,open(str(_ROOT / 'results/gates/gates_mosei.json'),'w'))
print(f"{'gate':9} {'gain':>8} {'apply':>7} {'harm':>7}   (alpha=0.2)")
for gname in ['blanket','random','maxprob','margin','entropy','agree','learned','GUARD']:
    v=[x for x in rows if x['gate']==gname]
    if not v: continue
    print(f"{gname:9} {np.mean([x['gain'] for x in v]):+8.4f} "
          f"{np.mean([x['apply'] for x in v]):7.3f} {np.mean([x['harm'] for x in v]):7.3f}")
print()
print('per condition (gain / harm)')
print(f"{'gate':9} " + ' '.join(f'{c:>16}' for c in ['a','v','av']))
for gname in ['blanket','learned','GUARD']:
    cells=[]
    for c in ['a','v','av']:
        v=[x for x in rows if x['gate']==gname and x['cond']==c]
        cells.append(f"{np.mean([x['gain'] for x in v]):+.4f}/{np.mean([x['harm'] for x in v]):.3f}")
    print(f"{gname:9} " + ' '.join(f'{c:>16}' for c in cells))
