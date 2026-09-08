"""gate_row with two Learn-then-Test policies added.

Everything up to the corrector is the shared core's code, unchanged, so GUARD and the
LTT policies see the same frozen model, the same retrieval target and the same blend.
The LTT policies then calibrate one threshold on D_conf instead of screening per sample.
"""
import numpy as np, sys
from guard import action as _A
from guard import losses as _L, targets as _T
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


ETAS = (0.05, 0.1, 0.2, 0.3, 0.4)

def _auroc(score, pos):
    """Rank quality of `score` for predicting `pos`; ties get half credit."""
    import numpy as _np
    pos = _np.asarray(pos, bool)
    if pos.all() or not pos.any():
        return 0.5
    r = _np.argsort(_np.argsort(score, kind='mergesort'), kind='mergesort').astype(float) + 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


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
        pr = probs if T==1.0 else _T.temper(probs,T,loss.simplex)
        for tg in targets:
            vals=(_T.hard_label_values(labels[pool],n_out,loss.simplex) if tg=='hard'
                  else _T.cross_mask_values((richer if T==1.0 else _T.temper(richer,T,loss.simplex))[pool]))
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
    pr = probs if T==1.0 else _T.temper(probs,T,loss.simplex)
    vals=(_T.hard_label_values(labels[pool],n_out,loss.simplex) if tg=='hard'
          else _T.cross_mask_values((richer if T==1.0 else _T.temper(richer,T,loss.simplex))[pool]))
    ke=min(k,len(pool)-1)
    tt={n:_T.knn_average(f[n],f['pool'],vals,ke,weighting=wt) for n in ('fit','conf','test')}
    # temperature is part of the correction, so it goes into the blend only; the
    # baseline and the output a declined query gets are the host's own
    mc,mt=probs[conf],probs[test]
    cc=(1-b)*pr[conf]+b*tt['conf']; ct=(1-b)*pr[test]+b*tt['test']
    cf=(1-b)*pr[fit]+b*tt['fit']
    _sc=_A.fit_action_score(probs[fit],tt['fit'],cf,labels[fit],loss)
    g=_A.certify_action(_sc,mc,tt['conf'],cc,labels[conf],mt,tt['test'],loss,ALPHA,DELTA,fit=(probs[fit], tt['fit'], cf, labels[fit])); apG=g['apply']
    bl=loss(mt,labels[test]); cl=loss(ct,labels[test]); dl=cl-bl
    base=acc(mt,test); R=float(apG.mean())
    AP={}
    def row(ap, tag=None):
        if tag is not None: AP[tag]=ap
        return (acc(np.where(ap[:,None],ct,mt),test)-base, float((ap&(dl>DELTA)).mean()))
    out={'GUARD':row(apG,'GUARD'), 'blanket':row(np.ones(len(test),bool),'blanket')}
    # what conformal risk control alone would apply, before the fit-split
    # tightening: the price of that step is the difference between the two
    _lc = g['lambda_crc']
    out['GUARD-untightened'] = row(
        np.zeros(len(test), bool) if _lc is None else g['score_test'] > _lc,
        'GUARD-untightened')
    ent=-(mt*np.log(np.clip(mt,1e-12,None))).sum(1)
    multilabel = mt.shape[1]>2 and np.ndim(labels)==2
    out['confidence']=row(_at_rate(-np.abs(mt-0.5).mean(1) if multilabel else -mt.max(1),R))
    out['agreement']=row(_at_rate((np.abs(mt-tt['test']).mean(1)<0.1).astype(float),R))
    out['random']=row(_at_rate(np.random.default_rng(0).random(len(test)),R))
    # the raw host, so this is the scorer certify_action fits, not a second one
    pf=probs[fit]
    Xf=np.column_stack([pf.max(1), -(pf*np.log(np.clip(pf,1e-12,None))).sum(1),
                        tt['fit'].max(1), (np.abs(pf-tt['fit']).mean(1)<0.1).astype(float),
                        np.abs(pf-tt['fit']).sum(1)])
    Xt=np.column_stack([mt.max(1), ent, tt['test'].max(1),
                        (np.abs(mt-tt['test']).mean(1)<0.1).astype(float),
                        np.abs(mt-tt['test']).sum(1)])
    zf=(loss(cf,labels[fit])<loss(pf,labels[fit])).astype(int)
    out['learned']=(row(_at_rate(LogisticRegression(max_iter=2000).fit(Xf,zf).predict_proba(Xt)[:,1],R))
                    if len(set(zf))>1 else (np.nan,np.nan))
    # --- Learn-then-Test: thresholds certified on D_conf ---
    harm_c = (loss(cc,labels[conf])-loss(mc,labels[conf])) > DELTA
    sc_c=_scores(mc,tt['conf'],multilabel); sc_t=_scores(mt,tt['test'],multilabel)
    # the candidate thresholds have to be fixed before D_conf is read, so they
    # come from the fit split, which every stage before calibration already saw
    sc_f=_scores(probs[fit],tt['fit'],multilabel)
    # the learned gate's own score, so LTT is not handicapped by a weak family
    if len(set(zf))>1:
        lr=LogisticRegression(max_iter=2000).fit(Xf,zf)
        Xc=np.column_stack([mc.max(1), -(mc*np.log(np.clip(mc,1e-12,None))).sum(1),
                            tt['conf'].max(1), (np.abs(mc-tt['conf']).mean(1)<0.1).astype(float),
                            np.abs(mc-tt['conf']).sum(1)])
        sc_c['LTT-learned']=lr.predict_proba(Xc)[:,1]; sc_t['LTT-learned']=lr.predict_proba(Xt)[:,1]
        sc_f['LTT-learned']=lr.predict_proba(Xf)[:,1]
    # confidence + CRC: the same calibration, without the learned score
    # same CRC routine GUARD is held to, so the two differ only in the score
    _lmc = _A.crc_threshold(sc_c['LTT-confidence'], harm_c, ALPHA)
    _apc = np.zeros(len(test), bool) if _lmc is None else (sc_t['LTT-confidence'] > _lmc)
    out['CRC-confidence'] = row(_apc, 'CRC-confidence')
    out['CRC-confidence-rate'] = float(_apc.mean())

    for name in list(sc_c):
        lam=ltt_threshold(sc_c[name],harm_c,ALPHA,0.05,grid_scores=sc_f.get(name))
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
            lm=ltt_threshold(sc_c[name][sel],harm_c[sel],ALPHA,0.05,grid_scores=sc_f.get(name))
            if lm is not None: apm |= (gt==gv)&(sc_t[name]>=lm)
        mname=name.replace('LTT-','LTT-mask-')
        out[mname]=row(apm,mname); out[mname+'-rate']=float(apm.mean())
        # GUARD's own risk-utility frontier: same corrector, alpha swept
    fr=[]
    for a_ in (0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6):
        gg=_A.certify_action(_sc,mc,tt['conf'],cc,labels[conf],mt,tt['test'],loss,a_,DELTA,fit=(probs[fit], tt['fit'], cf, labels[fit]))
        ap_=gg['apply']
        fr.append((a_, acc(np.where(ap_[:,None],ct,mt),test)-base,
                   float((ap_&(dl>DELTA)).mean()), float(ap_.mean())))
    out['GUARD-frontier']=fr
    out['_apply']=AP; out['_harmful']=(dl>DELTA)
    out['_meta']=dict(base=base, apply=R, k=k, target=tg, space=sp, weighting=wt, temperature=T, beta=b)
    return out
