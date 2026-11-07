"""Score x calibration x tightening on one fixed candidate, on every benchmark.

Usage: rebuttal_factorial.py BENCH [BENCH ...] [--procs N]
One JSON line per cell in results/rebuttal/factorial_<bench>.jsonl. The flat keys
match results/gates/cells_*.json, so the saved GUARD cell is the control.
"""
import time, json, argparse
from multiprocessing import Pool
from rebuttal_common import *
from rebuttal_common import _A, _L, _T
import gates_ltt
J = {}
J['mosei'] = [dict(cond=c, seed=s) for c in ('a', 'v', 'av') for s in range(10)]
J['iemocap'] = [dict(f=f, m=m, cut=c) for f in range(5) for m in MS for c in range(5)]
J['opportunity'] = [dict(cfg=c, s=s) for c in ('low_cost_accels_only', 'no_imu_family', 'no_shoes', 'severe_three_sensors') for s in (0, 1, 2)]
J['ave'] = [dict(cond=c, seed=s) for c in ('audio_only', 'visual_only') for s in range(3)]
J['ninapro'] = [dict(sub=sub, c=c) for sub in nina_subjects() for c in ('12', '8', '6', '4')]
J['ptbxl'] = [dict(c=c, fk=fk, seed=s) for c, fk in (('a', 'a'), ('t', 't'), ('v', 'v'), ('at', 'a'), ('av', 'a'), ('tv', 't')) for s in range(3)]
_D = [('drugban_biosnap_random_s42', c) for c in ('prot25', 'prot50', 'scaffold')] + [('drugban_bindingdb_random_s42', c) for c in ('prot25', 'prot50', 'scaffold')] + [('drugban_human_random_s42', c) for c in ('prot25', 'prot50')] + [('drugban_biosnap_cluster_s42', 'prot50')]
J['drugban'] = [dict(d=d, cond=c, seed=s) for d, c in _D for s in range(3)]
LOAD = dict(mosei=mosei, iemocap=iemocap, opportunity=opportunity, ave=ave, ninapro=ninapro, ptbxl=ptbxl, drugban=drugban)
def job(a):
    b, ctx = a
    try:
        P, F, Y, Rh, split, keep, tg, ln = LOAD[b](**ctx)
        r = gates_ltt.gate_row(P, F, Y, split, loss_name=ln, keep=keep, targets=tg, richer=Rh)
        o = dict(driver=b, ctx=ctx, rate=r['_meta']['apply'], base=r['_meta']['base'], meta=r['_meta'],
                 **{k: v for k, v in r.items() if not k.startswith('_')}, x=factorial(r, P, F, Y, Rh, split, ln, keep))
        return b, json.dumps(jsonable(o))
    except Exception:
        import traceback
        return b, json.dumps(dict(driver=b, ctx=ctx, error=traceback.format_exc()[-1500:]))
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('bench', nargs='+'); ap.add_argument('--procs', type=int, default=6); a = ap.parse_args()
    jobs = [(b, c) for b in a.bench for c in J[b]]; t0 = time.time(); fh = {b: open(OUT / f'factorial_{b}.jsonl', 'w') for b in a.bench}
    with Pool(a.procs) as p:
        for i, (b, s) in enumerate(p.imap_unordered(job, jobs, chunksize=1)):
            fh[b].write(s + '\n'); fh[b].flush()
            if i % 10 == 0: print(i + 1, len(jobs), round(time.time() - t0), flush=True)
    print('DONE', len(jobs), round(time.time() - t0), flush=True)
