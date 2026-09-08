from pathlib import Path

p = Path('experiments/gates_ltt.py')
s = p.read_text()

old = """    so_c = _soft(cc,mc,gq['q_hat']); so_t = _soft(ct,mt,gt['q_hat'])"""
new = """    def _agg(corrected, base):
        \"\"\"Aggregate the hypothetical loss changes under the corrected posterior.

        Returns the signed expected change and the plug-in probability that the
        realised change exceeds delta, which is the per-sample contribution to
        exactly the risk LTT controls.
        \"\"\"
        if not loss.simplex:
            return None, None
        q = np.clip(np.asarray(corrected,dtype=np.float64),1e-12,None)
        q = q/q.sum(1,keepdims=True)
        d = (-np.log(q) + np.log(np.clip(np.asarray(base,dtype=np.float64),1e-12,None)))
        return (q*d).sum(1), (q*(d>DELTA)).sum(1)

    ex_c, ph_c = _agg(cc,mc); ex_t, ph_t = _agg(ct,mt)
    for nm, sc, stt in (('GUARD-exp-LTT', ex_c, ex_t), ('GUARD-phat-LTT', ph_c, ph_t)):
        if sc is None: continue
        lm = ltt_threshold(-sc, harm_c, ALPHA, 0.05)
        a_ = np.zeros(len(test),bool) if lm is None else (-stt >= lm)
        out[nm] = row(a_, nm); out[nm+'-rate'] = float(a_.mean())
        if groups is None: continue
        gc2, gt2 = np.asarray(groups)[conf], np.asarray(groups)[test]
        am = np.zeros(len(test),bool)
        for gv in np.unique(gt2):
            sel = gc2 == gv
            if sel.sum() < 10: continue
            l2 = ltt_threshold(-sc[sel], harm_c[sel], ALPHA, 0.05)
            if l2 is not None: am |= (gt2 == gv) & (-stt >= l2)
        mn = nm.replace('GUARD-', 'GUARD-mask-')
        out[mn] = row(am, mn); out[mn+'-rate'] = float(am.mean())

    so_c = _soft(cc,mc,gq['q_hat']); so_t = _soft(ct,mt,gt['q_hat'])"""
assert s.count(old) == 1
p.write_text(s.replace(old, new))
print('da them GUARD-exp-LTT va GUARD-phat-LTT (kem ban per-mask)')
