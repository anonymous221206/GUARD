"""AVE under a deployment missing rate.

The three sidecars give the frozen AV-att model's outputs and retrieval features
with both streams, with audio alone, and with vision alone. A missing rate eta is
a distribution over those three: each stream is absent independently with
probability eta, and a segment that would lose both keeps one at random. Every
segment is then read from the sidecar for the pattern it drew, so nothing is
retrained and eta=0 is the intact model.

Writes one row per (eta, seed); the AVE panel of Figure 4 reads this file.
"""
import numpy as np, sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gates_ltt import gate_row

_ROOT = Path(__file__).resolve().parents[1]
A = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))
D = A / 'ave_av_att/dumps'
CONDS = ('full', 'audio_only', 'visual_only')
ETAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = 3
OUT = _ROOT / 'results/ave_eta.json'


def load():
    P, F, PP, PF = {}, {}, {}, {}
    y = pool_y = None
    for c in CONDS:
        d = np.load(D / f'AV_att_{c}.npz', allow_pickle=True)
        r = np.load(D / f'AV_att_{c}_retrieval_paper.npz', allow_pickle=True)
        P[c] = d['probs'].reshape(-1, d['probs'].shape[-1]).astype(np.float64)
        F[c] = r['deploy_features'].astype(np.float64)
        PP[c] = r['pool_probs'].astype(np.float64)
        PF[c] = r['pool_features'].astype(np.float64)
        lab = d['labels'].reshape(-1, d['labels'].shape[-1]).argmax(1)
        pl = r['pool_labels']
        if y is None:
            y, pool_y = lab, pl
        # the three sidecars must describe the same rows, or a mixture is meaningless
        assert np.array_equal(y, lab) and np.array_equal(pool_y, pl), c
    return P, F, PP, PF, y, pool_y


def draw(rng, n, eta):
    gone_a, gone_v = rng.random(n) < eta, rng.random(n) < eta
    both = gone_a & gone_v
    if both.any():                       # never lose everything
        keep_a = rng.random(both.sum()) < 0.5
        idx = np.flatnonzero(both)
        gone_a[idx[keep_a]] = False
        gone_v[idx[~keep_a]] = False
    out = np.full(n, 'full', dtype=object)
    out[gone_v & ~gone_a] = 'audio_only'
    out[gone_a & ~gone_v] = 'visual_only'
    return out


def main():
    P, F, PP, PF, y, pool_y = load()
    n, npool = len(y), len(pool_y)
    rich = np.concatenate([PP['full'], P['full']])
    Y = np.concatenate([pool_y, y])
    rows = []
    for eta in ETAS:
        for seed in range(SEEDS):
            rng = np.random.default_rng(1000 * seed + int(round(100 * eta)))
            mt, mp = draw(rng, n, eta), draw(rng, npool, eta)
            pr = np.zeros_like(P['full']); ft = np.zeros_like(F['full'])
            pp = np.zeros_like(PP['full']); pf = np.zeros_like(PF['full'])
            for c in CONDS:
                s = mt == c
                if s.any(): pr[s], ft[s] = P[c][s], F[c][s]
                s = mp == c
                if s.any(): pp[s], pf[s] = PP[c][s], PF[c][s]
            perm = np.random.default_rng(seed).permutation(n) + npool
            k = n // 3
            r = gate_row(np.concatenate([pp, pr]), np.concatenate([pf, ft]), Y,
                         (np.arange(npool), perm[:k], perm[k:2 * k], perm[2 * k:]),
                         targets=('hard', 'cross'), richer=rich)
            rows.append(dict(eta=eta, seed=seed, base=r['_meta']['base'],
                             rate=r['_meta']['apply'],
                             **{k_: list(v) for k_, v in r.items()
                                if k_ in ('GUARD', 'blanket', 'confidence', 'learned')}))
            print(f"eta={eta:.1f} seed{seed} base={r['_meta']['base']:.4f} "
                  f"guard={r['GUARD'][0]:+.4f} harm={r['GUARD'][1]:.3f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT, 'w'))
    print('wrote', OUT, len(rows), 'rows')
    for eta in ETAS:
        s = [r for r in rows if r['eta'] == eta]
        b = np.mean([r['base'] for r in s]); g = np.mean([r['GUARD'][0] for r in s])
        print(f'eta {eta:.1f}  frozen {b:.3f}  +GUARD {b+g:.3f}  '
              f'harm {np.mean([r["GUARD"][1] for r in s]):.3f}')


if __name__ == '__main__':
    main()
