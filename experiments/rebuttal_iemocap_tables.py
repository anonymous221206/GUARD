"""The retrieval-probe and action-score tables on IEMOCAP, in parallel. Same rows() as the originals."""
import sys, time, json
from multiprocessing import Pool
from rebuttal_common import *
from rebuttal_common import _A, _L, _T
MODE = sys.argv[1]; PROCS = int(sys.argv[2])
if MODE == 'action':
    from action_core import rows, NAMES as KINDS
    kw = dict(variants=KINDS); FIELDS = ('gain', 'harm', 'blanket_gain', 'blanket_harm', 'apply', 'base', 'auroc')
else:
    from probe_core import rows
    KINDS = ('knn', 'linear'); kw = dict(correctors=KINDS); FIELDS = ('gain', 'harm', 'blanket_gain', 'blanket_harm', 'apply', 'base')
def job(a):
    f, m, cut = a
    P, F, Y, Rh, split, keep, tg, ln = iemocap(f, m, cut)
    return dict(ctx=dict(f=f, m=m, cut=cut), r=rows(P, F, Y, split, targets=tg, richer=Rh, **kw))
if __name__ == '__main__':
    jobs = [(f, m, c) for f in range(5) for m in MS for c in range(5)]; t0 = time.time(); cells = []
    with Pool(PROCS) as p:
        for i, r in enumerate(p.imap_unordered(job, jobs, chunksize=1)):
            cells.append(r)
            if i % 10 == 0: print(i + 1, len(jobs), round(time.time() - t0), flush=True)
    summary = {k: {f: float(np.nanmean([c['r'][k][f] for c in cells])) for f in FIELDS} for k in KINDS}
    summary['cells'] = len(cells)
    json.dump(jsonable(dict(summary=summary, cells=cells)), open(OUT / f'{MODE}_iemocap.json', 'w'))
    print(json.dumps(summary, indent=1)); print('DONE', round(time.time() - t0), flush=True)
