"""How much does the realized harm move when the calibration set is redrawn?

Theorem 1 is marginal over the calibration draw. This measures the spread that
statement leaves open: hold the frozen model, the corrector and the test third
fixed, redraw D_conf at a range of sizes, and record the standard deviation of
the harm actually realized on test.
"""
import numpy as np, sys, json
from guard import losses as _L, targets as _T, certify as _C
from guard.pipeline import _select_beta
from gates_core import _score, KS, TS, SPACES, WTS, ALPHA, DELTA
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))


NS=(25,50,75,100,150,200,300,400,600)
B=200

def curve(probs, feats, labels, split, keep=None, targets=('hard',), richer=None):
    loss=_L.get('cross_entropy'); pool,fit,conf,test=split
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
    f=SP[sp]; pr = probs if T==1.0 else _T.temper(probs,T)
    vals=(_T.hard_label_values(labels[pool],n_out,loss.simplex) if tg=='hard'
          else _T.cross_mask_values((richer if T==1.0 else _T.temper(richer,T))[pool]))
    ke=min(k,len(pool)-1)
    tt={n:_T.knn_average(f[n],f['pool'],vals,ke,weighting=wt) for n in ('conf','test')}
    mc,mt=pr[conf],pr[test]
    cc=(1-b)*mc+b*tt['conf']; ct=(1-b)*mt+b*tt['test']
    bl=loss(mt,labels[test]); dl=loss(ct,labels[test])-bl
    harmful=dl>DELTA
    rng=np.random.default_rng(0); res={}
    for n in NS:
        if n>len(conf): continue
        hs=[]
        for _ in range(B):
            sub=rng.choice(len(conf),n,replace=False)
            g=_C.certify(cc[sub],labels[conf][sub],ct,mt,loss,ALPHA,DELTA)
            hs.append(float((g['apply']&harmful).mean()))
        res[n]=(float(np.mean(hs)),float(np.std(hs)))
    return res, len(conf)
