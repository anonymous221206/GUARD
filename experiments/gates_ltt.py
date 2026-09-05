"""gate_row with two Learn-then-Test policies added.

Everything up to the corrector is the shared core's code, unchanged, so GUARD and the
LTT policies see the same frozen model, the same retrieval target and the same blend.
The LTT policies then calibrate one threshold on D_conf instead of screening per sample.
"""
import numpy as np, sys
from guard import losses as _L, targets as _T, certify as _C
from guard.pipeline import _select_beta
from gates_core import _score, _at_rate, KS, TS, SPACES, WTS, ALPHA, DELTA
from ltt_core import ltt_threshold
from sklearn.linear_model import LogisticRegression
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))


def _scores(m,t,multilabel):
    """High score means apply. Least-confident first, most-agreeing first."""
    conf = -np.abs(m-0.5).mean(1) if multilabel else -m.max(1)
    agree = -np.abs(m-t).mean(1)
    return {'LTT-confidence':conf,'LTT-agreement':agree}

def gate_row(probs, feats, labels, split, loss_name='cross_entropy',
             keep=None, targets=('hard',), richer=None, seeds=1, groups=None):
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
    AP={}
    def row(ap, tag=None):
        if tag is not None: AP[tag]=ap
        return (acc(np.where(ap[:,None],ct,mt),test)-base, float((ap&(dl>DELTA)).mean()))
    out={'GUARD':row(apG,'GUARD'), 'blanket':row(np.ones(len(test),bool),'blanket')}
    ent=-(mt*np.log(np.clip(mt,1e-12,None))).sum(1)
    multilabel = mt.shape[1]>2 and np.ndim(labels)==2
    out['confidence']=row(_at_rate(-np.abs(mt-0.5).mean(1) if multilabel else -mt.max(1),R))
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
    # --- Learn-then-Test: thresholds certified on D_conf ---
    harm_c = (loss(cc,labels[conf])-loss(mc,labels[conf])) > DELTA
    sc_c=_scores(mc,tt['conf'],multilabel); sc_t=_scores(mt,tt['test'],multilabel)
    # the learned gate's own score, so LTT is not handicapped by a weak family
    if len(set(zf))>1:
        lr=LogisticRegression(max_iter=2000).fit(Xf,zf)
        Xc=np.column_stack([mc.max(1), -(mc*np.log(np.clip(mc,1e-12,None))).sum(1),
                            tt['conf'].max(1), (np.abs(mc-tt['conf']).mean(1)<0.1).astype(float),
                            np.abs(mc-tt['conf']).sum(1)])
        sc_c['LTT-learned']=lr.predict_proba(Xc)[:,1]; sc_t['LTT-learned']=lr.predict_proba(Xt)[:,1]
    for name in list(sc_c):
        lam=ltt_threshold(sc_c[name],harm_c,ALPHA,0.05)
        ap = np.zeros(len(test),bool) if lam is None else (sc_t[name]>=lam)
        r=row(ap,name); out[name]=r
        out[name+'-rate']=float(ap.mean())
        if groups is None: continue
        # fair comparator: the mask is known at deployment, so calibrate one threshold per mask
        gc,gt=np.asarray(groups)[conf],np.asarray(groups)[test]
        apm=np.zeros(len(test),bool)
        for gv in np.unique(gt):
            sel=gc==gv
            if sel.sum()<10: continue
            lm=ltt_threshold(sc_c[name][sel],harm_c[sel],ALPHA,0.05)
            if lm is not None: apm |= (gt==gv)&(sc_t[name]>=lm)
        mname=name.replace('LTT-','LTT-mask-')
        out[mname]=row(apm,mname); out[mname+'-rate']=float(apm.mean())
    # symmetric comparator: GUARD calibrated per mask too (the group-conditional corollary)
    if groups is not None:
        gc,gt=np.asarray(groups)[conf],np.asarray(groups)[test]
        apm=np.zeros(len(test),bool)
        for gv in np.unique(gt):
            sc,st=gc==gv,gt==gv
            if sc.sum()<10 or st.sum()==0: continue
            gg=_C.certify(cc[sc],labels[conf][sc],ct[st],mt[st],loss,ALPHA,DELTA)
            apm[st]=gg['apply']
        out['GUARD-mask']=row(apm,'GUARD-mask'); out['GUARD-mask-rate']=float(apm.mean())
    out['_apply']=AP; out['_harmful']=(dl>DELTA)
    out['_meta']=dict(base=base, apply=R, k=k, target=tg, space=sp, weighting=wt, temperature=T, beta=b)
    return out
