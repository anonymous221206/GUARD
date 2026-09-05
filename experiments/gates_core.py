"""Same frozen model, same corrector, same intervention rate: only the rule varies.

Every benchmark is reduced to (probs, features, labels, split) and handed to one
routine, so a difference between benchmarks cannot come from a difference in how
the comparison was run.
"""
import numpy as np, sys
sys.path.insert(0,'/vol/dl-nguyenb5-solar/users/hoangpc/guard-release/src')
from guard import losses as _L, targets as _T, certify as _C
from guard.pipeline import _select_beta
from sklearn.linear_model import LogisticRegression

GRID=np.linspace(0,1,41); ALPHA,DELTA=0.2,0.05
KS=(3,5,8,12,20,35,50); TS=(1.0,2.0); SPACES=('standardise','cosine'); WTS=('uniform','distance')

def _score(P,y,keep=None):
    # multi-label targets arrive as (n, L); score them per label, not by argmax
    ok = ((P > 0.5) == (y > 0.5)).mean(1) if np.ndim(y) == 2 else (P.argmax(1) == y)
    return float(ok[keep].mean() if keep is not None else ok.mean())

def _at_rate(score,rate):
    n=len(score); nk=int(round(rate*n)); out=np.zeros(n,bool)
    if nk>0: out[np.argsort(-score)[:nk]]=True
    return out

def gate_row(probs, feats, labels, split, loss_name='cross_entropy',
             keep=None, targets=('hard',), richer=None, seeds=1):
    """Return {rule: (mean gain, mean joint harm)} at GUARD's own apply rate."""
    loss=_L.get(loss_name)
    pool,fit,conf,test=split
    acc=lambda P,idx: _score(P,labels[idx], None if keep is None else keep[idx])
    n_out=probs.shape[1]
    SP={}
    for sp in SPACES:
        z=_T.retrieval_space(feats[pool],sp)
        SP[sp]={n:z(feats[i]) for n,i in (('pool',pool),('fit',fit),('conf',conf),('test',test))}
    best=None
    for T in TS:
        pr = probs if T==1.0 else _T.temper(probs,T)
        for tg in targets:
            vals=(_T.hard_label_values(labels[pool],n_out,loss.simplex) if tg=='hard'
                  else _T.cross_mask_values((richer if T==1.0 else _T.temper(richer,T))[pool]))
            for sp in SPACES:
                for wt in WTS:
                    for k in KS:
                        ke=min(k,len(pool)-1)
                        tf=_T.knn_average(SP[sp]['fit'],SP[sp]['pool'],vals,ke,weighting=wt)
                        b=_select_beta(pr[fit],tf,labels[fit],loss,'loss')
                        s=acc((1-b)*pr[fit]+b*tf,fit)
                        if best is None or s>best[0]: best=(s,T,tg,sp,wt,k,b)
    _,T,tg,sp,wt,k,b=best
    f=SP[sp]
    pr = probs if T==1.0 else _T.temper(probs,T)
    vals=(_T.hard_label_values(labels[pool],n_out,loss.simplex) if tg=='hard'
          else _T.cross_mask_values((richer if T==1.0 else _T.temper(richer,T))[pool]))
    ke=min(k,len(pool)-1)
    tt={n:_T.knn_average(f[n],f['pool'],vals,ke,weighting=wt) for n in ('fit','conf','test')}
    mc,mt=pr[conf],pr[test]
    cc=(1-b)*mc+b*tt['conf']; ct=(1-b)*mt+b*tt['test']
    g=_C.certify(cc,labels[conf],ct,mt,loss,ALPHA,DELTA); apG=g['apply']
    bl=loss(mt,labels[test]); cl=loss(ct,labels[test]); dl=cl-bl
    base=acc(mt,test); R=float(apG.mean())
    def row(ap): return (acc(np.where(ap[:,None],ct,mt),test)-base, float((ap&(dl>DELTA)).mean()))
    out={'GUARD':row(apG), 'blanket':row(np.ones(len(test),bool))}
    ent=-(mt*np.log(np.clip(mt,1e-12,None))).sum(1)
    out['confidence']=row(_at_rate(-np.abs(mt-0.5).mean(1) if mt.shape[1]>2 and np.ndim(labels)==2 else -mt.max(1),R))
    out['agreement']=row(_at_rate((np.abs(mt-tt['test']).mean(1)<0.1).astype(float),R))
    out['random']=row(_at_rate(np.random.default_rng(0).random(len(test)),R))
    Xf=np.column_stack([pr[fit].max(1), -(pr[fit]*np.log(np.clip(pr[fit],1e-12,None))).sum(1),
                        tt['fit'].max(1), (np.abs(pr[fit]-tt['fit']).mean(1)<0.1).astype(float),
                        np.abs(pr[fit]-tt['fit']).sum(1)])
    Xt=np.column_stack([mt.max(1), ent, tt['test'].max(1),
                        (np.abs(mt-tt['test']).mean(1)<0.1).astype(float),
                        np.abs(mt-tt['test']).sum(1)])
    zf=(loss((1-b)*pr[fit]+b*tt['fit'],labels[fit])<loss(pr[fit],labels[fit])).astype(int)
    out['learned']=(row(_at_rate(LogisticRegression(max_iter=2000).fit(Xf,zf).predict_proba(Xt)[:,1],R))
                    if len(set(zf))>1 else (np.nan,np.nan))
    out['_meta']=dict(base=base, apply=R, k=k, target=tg, space=sp, weighting=wt, temperature=T, beta=b)
    return out
