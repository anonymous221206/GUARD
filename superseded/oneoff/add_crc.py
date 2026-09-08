#!/usr/bin/env python3
"""Conformal risk control on the same monotone families.

Certify bounds P(apply and harm) by the miscoverage of the plausible set, which
is why its realised harm sits far under the budget: the two events overlap much
less than the bound allows. Conformal risk control calibrates the threshold
against the harm itself, keeps a marginal guarantee of the same shape as
Theorem 2, and needs no multiple testing. The family only has to be monotone in
the threshold, which 1{s(X) <= lambda} is.
"""
from pathlib import Path

p = Path('experiments/gates_ltt.py')
s = p.read_text()

helper = '''
def crc_threshold(score_c, harm_c, alpha):
    """Largest apply set whose calibration risk stays under the CRC level.

    Angelopoulos et al. (2023): with a loss bounded by B=1 and n calibration
    points, picking the most permissive lambda whose empirical risk is at most
    alpha - (1 - alpha)/n gives E[risk] <= alpha over calibration and test.
    """
    import numpy as _np
    n = len(score_c)
    level = alpha - (1.0 - alpha) / n
    if level <= 0:
        return None
    order = _np.argsort(-score_c, kind='mergesort')       # admit high score first
    risk = _np.cumsum(harm_c[order]) / n
    ok = _np.nonzero(risk <= level)[0]
    if len(ok) == 0:
        return None
    return float(score_c[order[ok[-1]]])

'''
anchor = "def gate_row("
assert s.count(anchor) == 1
s = s.replace(anchor, helper + anchor)

old = "    # GUARD's own risk-utility frontier: same corrector, alpha swept"
new = """    # --- conformal risk control on the same families ---
    for nm, sc, stt in (('CRC-guard', sg_c, sg_t),
                        ('CRC-confidence', sc_c['LTT-confidence'], sc_t['LTT-confidence']),
                        ('CRC-learned', sc_c.get('LTT-learned'), sc_t.get('LTT-learned'))):
        if sc is None: continue
        lm = crc_threshold(sc, harm_c, ALPHA)
        a_ = np.zeros(len(test), bool) if lm is None else (stt >= lm)
        out[nm] = row(a_, nm); out[nm+'-rate'] = float(a_.mean())

""" + old
assert s.count(old) == 1
p.write_text(s.replace(old, new))
print('da them CRC-guard, CRC-confidence, CRC-learned')
