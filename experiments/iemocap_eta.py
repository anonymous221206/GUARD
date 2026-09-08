"""IEMOCAP under a deployment missing rate, the axis MoMKE and SIEVE report on.

The per-mask dumps give the frozen model's output under each of the seven patterns.
A missing rate eta is a distribution over those patterns: every modality drops
independently with probability eta, and a sample that loses all three keeps one at
random, matching the protocol the two published baselines are scored under. Each
sample is then read from the dump for the pattern it drew, so no model is retrained
and the frozen row here is a reproduction of MoMKE, not a rescoring of it.

Writes one row per (eta, fold, draw); Figure 3 and Table 8 both read this file.
"""
import numpy as np, sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gates_ltt import gate_row

_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))
D = f'{ARTIFACTS}/iemocap_momke/folds'
MS = ['a', 't', 'v', 'at', 'av', 'tv', 'atv']
MODS = 'atv'
DRAWS = 5
ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
OUT = _ROOT / 'results/iemocap_eta.json'


def _order(vid):
    """Row key that survives the reordering between mask dumps."""
    cnt, out = {}, []
    for v in vid:
        cnt[v] = cnt.get(v, 0) + 1
        out.append((v, cnt[v] - 1))
    return out


def _aligned(fold):
    """Every mask's test and pool rows, put in the atv dump's row order."""
    ref = np.load(f'{D}/fold{fold}_atv.npz', allow_pickle=True)
    kt = {k: i for i, k in enumerate(_order(ref['vid']))}
    kp = {k: i for i, k in enumerate(_order(ref['pool_vid']))}
    nt, np_ = len(ref['labels']), len(ref['pool_labels'])
    P = {}
    for m in MS:
        d = np.load(f'{D}/fold{fold}_{m}.npz', allow_pickle=True)
        it = np.array([kt[k] for k in _order(d['vid'])])
        ip = np.array([kp[k] for k in _order(d['pool_vid'])])
        pr = np.zeros((nt, d['probs'].shape[1])); pr[it] = d['probs']
        ft = np.zeros((nt, d['feats'].shape[1])); ft[it] = d['feats']
        pp = np.zeros((np_, d['pool_probs'].shape[1])); pp[ip] = d['pool_probs']
        pf = np.zeros((np_, d['pool_feats'].shape[1])); pf[ip] = d['pool_feats']
        P[m] = (pr.astype(np.float64), ft.astype(np.float64),
                pp.astype(np.float64), pf.astype(np.float64))
    return ref, P, nt, np_


def draw_masks(rng, n, eta):
    """Pattern per sample: each modality gone w.p. eta, never all three."""
    keep = rng.random((n, 3)) >= eta
    dead = ~keep.any(1)
    if dead.any():
        keep[dead, rng.integers(0, 3, dead.sum())] = True
    return np.array([''.join(MODS[j] for j in range(3) if k[j]) for k in keep])


def main():
    rows = []
    for fold in range(5):
        ref, P, nt, npool = _aligned(fold)
        y = np.concatenate([ref['pool_labels'], ref['labels']])
        rich = np.concatenate([P['atv'][2], P['atv'][0]])
        uv = np.unique(ref['vid'])
        for eta in ETAS:
            for draw in range(DRAWS):
                rng = np.random.default_rng(1000 * draw + fold)
                mt, mp = draw_masks(rng, nt, eta), draw_masks(rng, npool, eta)
                pr = np.zeros_like(P['atv'][0]); ft = np.zeros_like(P['atv'][1])
                pp = np.zeros_like(P['atv'][2]); pf = np.zeros_like(P['atv'][3])
                for m in MS:
                    s = mt == m
                    if s.any(): pr[s], ft[s] = P[m][0][s], P[m][1][s]
                    s = mp == m
                    if s.any(): pp[s], pf[s] = P[m][2][s], P[m][3][s]
                probs = np.concatenate([pp, pr]); feats = np.concatenate([pf, ft])
                perm = np.random.default_rng(draw).permutation(len(uv))
                parts = [np.where(np.isin(ref['vid'], g))[0] + npool
                         for g in np.array_split(uv[perm], 3)]
                r = gate_row(probs, feats, y, (np.arange(npool), parts[0], parts[1], parts[2]),
                             targets=('hard', 'cross'), richer=rich)
                rows.append(dict(eta=eta, fold=fold, draw=draw,
                                 base=r['_meta']['base'], rate=r['_meta']['apply'],
                                 **{k: list(v) for k, v in r.items()
                                    if k in ('GUARD', 'blanket', 'confidence', 'learned')}))
                print(f"eta={eta:.1f} fold{fold} draw{draw} base={r['_meta']['base']:.4f} "
                      f"guard={r['GUARD'][0]:+.4f} harm={r['GUARD'][1]:.3f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT, 'w'))
    print('wrote', OUT, len(rows), 'rows')
    for eta in ETAS:
        sel = [r for r in rows if r['eta'] == eta]
        b = np.mean([r['base'] for r in sel]); g = np.mean([r['GUARD'][0] for r in sel])
        print(f'eta {eta:.1f}  frozen {b:.3f}  +GUARD {b+g:.3f}  harm '
              f'{np.mean([r["GUARD"][1] for r in sel]):.3f}')


if __name__ == '__main__':
    main()
