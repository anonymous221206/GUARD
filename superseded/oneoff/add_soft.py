#!/usr/bin/env python3
"""A soft counterpart to the worst-case score.

The max over the plausible set is what makes the direct certificate work: one bad
label is enough to refuse. As a *ranking* signal that is wasteful, because it
throws away how likely that label is. Under LTT the score carries no guarantee of
its own, so it can be softened. This adds the plausible-mass-weighted mean of the
positive part, which keeps the same action-specific construction.
"""
from pathlib import Path

p = Path('experiments/gates_ltt.py')
s = p.read_text()

old = """    gq=_C.certify(cf,labels[fit],cc,mc,loss,best_eta,DELTA)     # score on conf
    gt=_C.certify(cf,labels[fit],ct,mt,loss,best_eta,DELTA)     # score on test
    sg_c, sg_t = -gq['worst_case_extra_loss'], -gt['worst_case_extra_loss']"""
new = """    gq=_C.certify(cf,labels[fit],cc,mc,loss,best_eta,DELTA)     # score on conf
    gt=_C.certify(cf,labels[fit],ct,mt,loss,best_eta,DELTA)     # score on test
    sg_c, sg_t = -gq['worst_case_extra_loss'], -gt['worst_case_extra_loss']

    def _soft(corrected, base, q_hat):
        \"\"\"Plausible-mass-weighted mean of the positive hypothetical loss change.\"\"\"
        if not loss.simplex:
            return None
        keep = (1-corrected) <= q_hat
        keep[~keep.any(1)] = True
        d = (-np.log(np.clip(np.asarray(corrected,dtype=np.float64),1e-12,None))
             + np.log(np.clip(np.asarray(base,dtype=np.float64),1e-12,None)))
        w = np.where(keep, np.asarray(corrected,dtype=np.float64), 0.0)
        w = w / np.clip(w.sum(1,keepdims=True), 1e-12, None)
        return (w*np.clip(d,0,None)).sum(1)

    so_c = _soft(cc,mc,gq['q_hat']); so_t = _soft(ct,mt,gt['q_hat'])
    if so_c is not None:
        lam_s=ltt_threshold(-so_c,harm_c,ALPHA,0.05)
        aps = np.zeros(len(test),bool) if lam_s is None else (-so_t>=lam_s)
        out['GUARD-soft-LTT']=row(aps,'GUARD-soft-LTT'); out['GUARD-soft-LTT-rate']=float(aps.mean())"""
assert s.count(old) == 1
p.write_text(s.replace(old, new))
print('da them GUARD-soft-LTT')
