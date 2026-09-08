#!/usr/bin/env python3
"""Who spends the budget better, at matched realised harm rather than matched alpha.

At the same nominal alpha the two rules spend very different amounts of it: GUARD
realises far less harm than the LTT rules. Comparing gains there says as much
about how much budget each chose to use as about how well it chose. Sweeping
GUARD's alpha traces its risk-utility frontier, so the LTT points can be read
against the frontier instead of against a single point.
"""
from pathlib import Path

p = Path('experiments/gates_ltt.py')
s = p.read_text()

anchor = "    out['_apply']=AP; out['_harmful']=(dl>DELTA)"
new = """    # GUARD's own risk-utility frontier: same corrector, alpha swept
    fr=[]
    for a_ in (0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6):
        gg=_C.certify(cc,labels[conf],ct,mt,loss,a_,DELTA)
        ap_=gg['apply']
        fr.append((a_, acc(np.where(ap_[:,None],ct,mt),test)-base,
                   float((ap_&(dl>DELTA)).mean()), float(ap_.mean())))
    out['GUARD-frontier']=fr
""" + anchor
assert s.count(anchor) == 1
p.write_text(s.replace(anchor, new))
print('da them GUARD-frontier')
