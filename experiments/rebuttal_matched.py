"""The l1 distance at GUARD's own intervention rate, on the saved corrector of every cell.

Table 13 ranks by confidence, agreement and a random score at GUARD's rate. This adds the
strongest simple rule on the same terms. No setting is selected again: each cell reuses
the configuration stored in factorial_<bench>.jsonl, and confidence and random are
recomputed as controls against the saved columns. The same comparison is repeated at the
rate of the calibrated l1 rule, so the check does not rest on GUARD's own operating point.
"""
import argparse, json
from multiprocessing import Pool
from rebuttal_common import *
from rebuttal_common import _A, _L
from gates_core import _at_rate
LOAD = dict(mosei=mosei, iemocap=iemocap, opportunity=opportunity, ave=ave, ninapro=ninapro, ptbxl=ptbxl, drugban=drugban)
def job(a):
    b, r = a
    P, F, Y, Rh, split, keep, tg, ln = LOAD[b](**r['ctx']); loss = _L.get(ln); ml = not loss.simplex; fit, test = split[1], split[3]
    tt, cc, _ = rebuild(r['meta'], P, F, Y, Rh, split, loss, roles=('fit', 'test'))
    mt, yt = P[test], Y[test]; dl = loss(cc['test'], yt) - loss(mt, yt); kt = None if keep is None else keep[test]
    acc = acc_fn(yt, kt, ml); base = acc(mt); R = r['rate']
    row = lambda ap: policy_row(ap, cc['test'], mt, yt, dl, base, acc, kt, ml)
    conf = -np.abs(mt - .5).mean(1) if ml else -mt.max(1); l1 = np.abs(mt - tt['test']).sum(1)
    sc = _A.fit_action_score(P[fit], tt['fit'], cc['fit'], Y[fit], loss); Rl = r['x']['fact']['distance']['crc_t']['apply']
    learned = None if sc is None else _A.score(sc, mt, tt['test'], loss.simplex)
    out = {k: row(_at_rate(s, R)) for k, s in (('l1', l1), ('confidence', conf), ('random', np.random.default_rng(0).random(len(test))))}
    at_l1 = dict(l1=row(_at_rate(l1, Rl)), learned=row(np.zeros(len(test), bool) if learned is None else _at_rate(learned, Rl)))
    return dict(driver=b, ctx=r['ctx'], rate=R, rate_l1=Rl, **{k: [v['gain'], v['harm']] for k, v in out.items()}, at_l1={k: [v['gain'], v['harm']] for k, v in at_l1.items()})
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--procs', type=int, default=8); a = ap.parse_args()
    jobs = [(b, json.loads(l)) for b in LOAD for l in open(OUT / f'factorial_{b}.jsonl')]
    with Pool(a.procs) as p: rows = p.map(job, jobs, chunksize=1)
    with open(OUT / 'matched_l1.jsonl', 'w') as fh:
        for r in rows: fh.write(json.dumps(jsonable(r)) + '\n')
    print('DONE', len(rows), flush=True)
