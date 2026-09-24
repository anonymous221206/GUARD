"""OPPORTUNITY DeepConvLSTM under splits that keep temporal neighbours apart.

Deployment windows are stored in temporal order and consecutive windows overlap,
so a split by row lets the retrieval pool hold the window right next to a test
query. This driver reruns the paper's OPPORTUNITY path (opp_dcl.py) under:

  row          the paper's split: one random permutation of all windows
  dec2         the same, on every second window, so no two kept windows overlap
  block<B>     contiguous blocks of B windows assigned at random to the four roles,
               with the last `gap` windows of each block dropped; exchangeable at
               the block level, and no role holds a window adjacent to another's
  quarters     four contiguous quarters, as in the earlier audit; this breaks
               exchangeability between D_conf and D_test and is a lower bound

    python experiments/opp_dcl_blocks.py row dec2 block50 block200 quarters
"""
import collections, json, os, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'src')); sys.path.insert(0, str(_ROOT / 'experiments'))
from gates_core import gate_row                          # noqa: E402
from guard import losses as _L, targets as _T            # noqa: E402
from guard import action as _A                           # noqa: E402

ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))
D = str(ARTIFACTS / 'opportunity_dcl_v2')
CFG = ['low_cost_accels_only', 'no_imu_family', 'no_shoes', 'severe_three_sensors']
loss = _L.get('cross_entropy'); ALPHA, DELTA = 0.2, 0.05
GAP = 2
y = np.load(f'{D}/deploy_y.npy'); n = len(y)


def make_split(mode, s):
    rng = np.random.default_rng(s)
    if mode in ('row', 'dec2'):
        u = np.arange(n)[::2] if mode == 'dec2' else np.arange(n)
        perm = u[rng.permutation(len(u))]; q = len(u) // 4
        return perm[:q], perm[q:2*q], perm[2*q:3*q], perm[3*q:]
    if mode == 'quarters':
        q = n // 4; order = rng.permutation(4)
        parts = [np.arange(i*q, (i+1)*q - GAP) for i in range(4)]
        return tuple(parts[o] for o in order)
    B = int(mode[5:])
    blocks = [np.arange(i, min(i+B, n) - GAP) for i in range(0, n, B)]
    blocks = [b for b in blocks if len(b)]
    roles = np.arange(len(blocks)) % 4
    roles = roles[rng.permutation(len(blocks))]
    return tuple(np.concatenate([blocks[j] for j in np.where(roles == r)[0]]) for r in range(4))


out = {}
for mode in sys.argv[1:]:
    res = {}
    for cfg in CFG:
        F = np.load(f'{D}/retfeat_{cfg}_deploy.npy').astype(np.float64)
        bs, gs, bl_, ap_, hm_, bh_ = [], [], [], [], [], []
        for s in (0, 1, 2):
            P = np.load(f'{D}/probs_condition_specialist_{cfg}_deploy_s{s}.npy').astype(np.float64)
            rich = np.load(f'{D}/richer_deploy_s{s}.npy').astype(np.float64)
            split = make_split(mode, s)
            r = gate_row(P, F, y, split, targets=('hard', 'cross'), richer=rich)
            pool, fit, conf, test = split
            z = _T.retrieval_space(F[pool], r['_meta']['space'])
            fp, ff, fc, ft = z(F[pool]), z(F[fit]), z(F[conf]), z(F[test])
            vals = (_T.hard_label_values(y[pool], P.shape[1], loss.simplex) if r['_meta']['target'] == 'hard'
                    else _T.cross_mask_values(rich[pool]))
            k_ = r['_meta']['k']; wt = r['_meta']['weighting']
            tf = _T.knn_average(ff, fp, vals, k_, weighting=wt)
            T_ = r['_meta']['temperature']; Pt = P if T_ == 1.0 else _T.temper(P, T_, loss.simplex); b = r['_meta']['beta']
            tc = _T.knn_average(fc, fp, vals, k_, weighting=wt); tt = _T.knn_average(ft, fp, vals, k_, weighting=wt)
            cc = (1-b)*Pt[conf] + b*tc; ct = (1-b)*Pt[test] + b*tt
            g = _A.certify_action(_A.fit_action_score(P[fit], tf, (1-b)*Pt[fit] + b*tf, y[fit], loss),
                                  P[conf], tc, cc, y[conf], P[test], tt, loss, ALPHA, DELTA,
                                  fit=(P[fit], tf, (1-b)*Pt[fit] + b*tf, y[fit])); ap = g['apply']
            blo = loss(P[test], y[test]); cl = loss(ct, y[test])
            gp = np.where(ap[:, None], ct, P[test])
            wf = lambda Q: f1_score(y[test], Q.argmax(1), average='weighted')
            bs.append(wf(P[test])); bl_.append(wf(ct)); gs.append(wf(gp))
            ap_.append(float(ap.mean())); hm_.append(float((ap & ((cl-blo) > DELTA)).mean()))
            bh_.append(float(((cl-blo) > DELTA).mean()))
        res[cfg] = dict(frozen=float(np.mean(bs)), blanket=float(np.mean(bl_)), guard=float(np.mean(gs)),
                        apply=float(np.mean(ap_)), harm=float(np.mean(hm_)), blanket_harm=float(np.mean(bh_)),
                        target=r['_meta']['target'])
        c = res[cfg]
        print(f"{mode:9} {cfg:22} F1 {c['frozen']:.3f} -> {c['guard']:.3f} (blanket {c['blanket']:.3f})"
              f"  harm {c['harm']:.3f}  apply {c['apply']:.2f}  bl_harm {c['blanket_harm']:.3f}", flush=True)
    m = lambda k: float(np.mean([res[c][k] for c in CFG]))
    print(f"{mode:9} MEAN F1 {m('frozen'):.3f} -> {m('guard'):.3f}  gain {m('guard')-m('frozen'):+.3f}  "
          f"harm {min(res[c]['harm'] for c in CFG):.3f}-{max(res[c]['harm'] for c in CFG):.3f}", flush=True)
    out[mode] = res
p = _ROOT / 'results/gates/opportunity_dcl_blocks.json'
old = json.loads(p.read_text()) if p.exists() else {}
old.update(out); p.write_text(json.dumps(old, indent=1))
