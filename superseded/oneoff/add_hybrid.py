#!/usr/bin/env python3
"""Add the GUARD-score + LTT hybrid to the rule comparison.

Direct GUARD ties one parameter to two jobs: alpha sets the conformal coverage of
the plausible set and, through it, how aggressive the rule is. The hybrid splits
them. eta shapes the plausible set and so the score; alpha stays the risk budget
and is spent by LTT. The score is frozen before D_conf is touched: its quantile
comes from D_fit, so D_conf is used only to calibrate the threshold.
"""
from pathlib import Path

p = Path('experiments/gates_ltt.py')
s = p.read_text()

helper = '''
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

'''
anchor = "def gate_row("
assert s.count(anchor) == 1
s = s.replace(anchor, helper + anchor)

old = """    out['_apply']=AP; out['_harmful']=(dl>DELTA)"""
new = """    # --- GUARD score + LTT: same score, but LTT picks the operating point ---
    cf = (1-b)*pr[fit]+b*tt['fit']; mf = pr[fit]
    half = len(fit)//2
    best_eta, best_auc = None, -1.0
    harm_fit_b = (loss(cf[half:],labels[fit][half:])-loss(mf[half:],labels[fit][half:])) > DELTA
    for eta in ETAS:
        # quantile from one half of the fit set, ranking measured on the other
        gg=_C.certify(cf[:half],labels[fit][:half],cf[half:],mf[half:],loss,eta,DELTA)
        a=_auroc(-gg['worst_case_extra_loss'], ~harm_fit_b)
        if a>best_auc: best_auc, best_eta = a, eta
    gq=_C.certify(cf,labels[fit],cc,mc,loss,best_eta,DELTA)     # score on conf
    gt=_C.certify(cf,labels[fit],ct,mt,loss,best_eta,DELTA)     # score on test
    sg_c, sg_t = -gq['worst_case_extra_loss'], -gt['worst_case_extra_loss']
    lam=ltt_threshold(sg_c,harm_c,ALPHA,0.05)
    ap = np.zeros(len(test),bool) if lam is None else (sg_t>=lam)
    out['GUARD-LTT']=row(ap,'GUARD-LTT'); out['GUARD-LTT-rate']=float(ap.mean())
    if groups is not None:
        gc,gt_=np.asarray(groups)[conf],np.asarray(groups)[test]
        apm=np.zeros(len(test),bool)
        for gv in np.unique(gt_):
            sel=gc==gv
            if sel.sum()<10: continue
            lm=ltt_threshold(sg_c[sel],harm_c[sel],ALPHA,0.05)
            if lm is not None: apm |= (gt_==gv)&(sg_t>=lm)
        out['GUARD-mask-LTT']=row(apm,'GUARD-mask-LTT'); out['GUARD-mask-LTT-rate']=float(apm.mean())
    out['_eta']=best_eta; out['_lambda']=(None if lam is None else -lam)
    out['_apply']=AP; out['_harmful']=(dl>DELTA)"""
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
print('da them GUARD-LTT va GUARD-mask-LTT')
