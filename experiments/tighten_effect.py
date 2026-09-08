"""What the fit-split tightening changes, measured against the untightened gate.

`certify_action` returns both thresholds: the one conformal risk control selects
and the tightened one GUARD deploys. This runs the AVE cells of the rule
comparison and scores the same test split under each, so the two differ only in
the threshold. gate_row calls the gate once per alpha on its frontier curve as
well, so the record keeps alpha and the deployed call is checked against the
apply set gate_row itself reports.
"""
import numpy as np, sys, os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'experiments'))
sys.path.insert(0, str(_ROOT / 'src'))
from guard import action as _A

REC = []
_orig = _A.certify_action


def patched(model, base_conf, target_conf, corrected_conf, labels_conf,
            base_test, target_test, loss, alpha, delta, fit):
    out = _orig(model, base_conf, target_conf, corrected_conf, labels_conf,
                base_test, target_test, loss, alpha, delta, fit=fit)
    s, lam_crc = out["score_test"], out["lambda_crc"]
    REC.append(dict(alpha=float(alpha), guard=out["apply"].copy(),
                    crc=(np.zeros(len(s), bool) if lam_crc is None else s > lam_crc),
                    tightened=bool(out["lambda"] is not None and lam_crc is not None
                                   and out["lambda"] > lam_crc)))
    return out


_A.certify_action = patched
import gates_core, gates_ltt                                   # noqa: E402
gates_core._A.certify_action = patched
gates_ltt._A.certify_action = patched
from gates_ltt import gate_row                                 # noqa: E402
from gates_core import ALPHA                                   # noqa: E402

A = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))
rows = []
for cond in ('audio_only', 'visual_only'):
    d = np.load(A / f'ave_av_att/dumps/AV_att_{cond}.npz', allow_pickle=True)
    r = np.load(A / f'ave_av_att/dumps/AV_att_{cond}_retrieval_paper.npz', allow_pickle=True)
    P = d['probs'].reshape(-1, d['probs'].shape[-1]).astype(np.float64)
    Y = d['labels'].reshape(-1, d['labels'].shape[-1]).argmax(1)
    F = r['deploy_features'].astype(np.float64)
    pP = r['pool_probs'].astype(np.float64)
    pF = r['pool_features'].astype(np.float64)
    pY = r['pool_labels']
    n = len(pY)
    PP, FF, YY = np.concatenate([pP, P]), np.concatenate([pF, F]), np.concatenate([pY, Y])
    for seed in range(3):
        perm = np.random.default_rng(seed).permutation(len(Y)) + n
        k = len(perm) // 3
        REC.clear()
        row = gate_row(PP, FF, YY, (np.arange(n), perm[:k], perm[k:2 * k], perm[2 * k:]))
        h = row['_harmful']
        g = next(x for x in REC if abs(x['alpha'] - ALPHA) < 1e-9)
        assert np.array_equal(g['guard'], row['_apply']['GUARD']), 'khong phai lan goi trien khai'
        rows.append(dict(cell=f'AVE/{cond}',
                         crc_harm=float((g['crc'] & h).mean()), crc_apply=float(g['crc'].mean()),
                         g_harm=float((g['guard'] & h).mean()), g_apply=float(g['guard'].mean()),
                         tightened=g['tightened']))
        x = rows[-1]
        print('%-20s CRC harm %.3f apply %.3f | GUARD harm %.3f apply %.3f | siet=%s'
              % (x['cell'], x['crc_harm'], x['crc_apply'], x['g_harm'], x['g_apply'],
                 x['tightened']), flush=True)

m = lambda k: float(np.mean([x[k] for x in rows]))
print('\ntrung binh over %d cell: CRC harm %.3f apply %.3f -> GUARD harm %.3f apply %.3f'
      % (len(rows), m('crc_harm'), m('crc_apply'), m('g_harm'), m('g_apply')))
print('so cell bi siet: %d/%d' % (sum(x['tightened'] for x in rows), len(rows)))
