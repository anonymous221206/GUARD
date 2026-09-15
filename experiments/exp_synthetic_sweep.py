#!/usr/bin/env python3
"""A dense sweep of the same synthetic family, with raw error held fixed.

exp_synthetic_matched.py tunes three configurations to one accuracy. Three
points ask the reader to accept that they are matched; a sweep shows it. Here
tau runs over a grid and g is bisected at each step so that base accuracy stays
at TARGET_ACC throughout. Small tau needs a large g and puts the error in the
Bayes gap; at the largest tau, g reaches zero and the model is Bayes-optimal
with the same accuracy. The three configurations of the table are points on
this line.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from guard import losses as _L                            # noqa: E402
from guard import targets as _T                           # noqa: E402
from guard.pipeline import _select_beta                   # noqa: E402

D, C = 10, 4
N_SRC, N_FIT, N_TEST = 4000, 600, 4000
TARGET_ACC, K = 0.65, 50
CE = _L.get("cross_entropy")


def softmax(u):
    u = u - u.max(1, keepdims=True)
    e = np.exp(u)
    return e / e.sum(1, keepdims=True)


def gen(rng, n, W, V, tau, g):
    X = rng.standard_normal((n, D))
    z = X @ W / tau
    pi = softmax(z)
    y = np.array([rng.choice(C, p=pi[i]) for i in range(n)])
    return X, y, pi, softmax(z + g * (X @ V))


def base_acc(rng, W, V, tau, g, n=6000):
    _, y, _, m = gen(rng, n, W, V, tau, g)
    return (m.argmax(1) == y).mean()


def tune(lo, hi, target, f, iters=18):
    """Bisect a knob whose accuracy falls monotonically in it."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > target else (lo, mid)
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--points", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("results/synthetic_sweep"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(1000 + seed)
        W = rng.standard_normal((D, C))
        V = rng.standard_normal((D, C)) / np.sqrt(D)
        probe = lambda t, g: base_acc(np.random.default_rng(7), W, V, t, g)
        # the ambiguity end: g=0 already gives the target accuracy
        tau_amb = tune(0.2, 30.0, TARGET_ACC, lambda t: probe(t, 0.0))
        for tau in np.linspace(0.6, tau_amb, a.points):
            # hold accuracy fixed by moving the bias knob against the noise knob
            g = tune(0.0, 8.0, TARGET_ACC, lambda gg: probe(tau, gg))
            Xs, ys, _, _ = gen(rng, N_SRC, W, V, tau, g)
            Xf, yf, _, mf = gen(rng, N_FIT, W, V, tau, g)
            Xt, yt, pit, mt = gen(rng, N_TEST, W, V, tau, g)
            acc0 = float((mt.argmax(1) == yt).mean())
            vals = _T.hard_label_values(ys, C, True)
            pi_f = _T.knn_average(Xf, Xs, vals, K, weighting="uniform")
            pi_t = _T.knn_average(Xt, Xs, vals, K, weighting="uniform")
            b = _select_beta(mf, pi_f, yf, CE, "loss")
            pc = (1 - b) * mt + b * pi_t
            rows.append(dict(seed=seed, tau=float(tau), g=float(g), base_acc=acc0,
                             ph_true=float(((mt - pit) ** 2).sum(1).mean()),
                             amb_true=float((pit * (1 - pit)).sum(1).mean()),
                             beta=float(b),
                             loss_gain=float((CE(mt, yt) - CE(pc, yt)).mean()),
                             d_acc=float((pc.argmax(1) == yt).mean() - acc0)))
        print(f"seed {seed} done ({a.points} points)", flush=True)

    with open(a.out / "sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f"\n{'PH_true':>9}{'base acc':>10}{'beta':>7}{'d acc':>9}{'loss gain':>11}   (mean +- se over seeds)")
    # rows are seed-major, so every a.points-th row is the same grid point
    for i in range(a.points):
        v = [r for j, r in enumerate(rows) if j % a.points == i]
        m = lambda k: np.mean([x[k] for x in v])
        se = lambda k: np.std([x[k] for x in v], ddof=1) / np.sqrt(len(v))
        print(f"{m('ph_true'):9.4f}{m('base_acc'):10.3f}{m('beta'):7.2f}"
              f"{m('d_acc'):+9.4f}{m('loss_gain'):+11.4f}   +-{se('d_acc'):.4f}")
    print(f"\nbase accuracy across the sweep: "
          f"{np.mean([r['base_acc'] for r in rows]):.4f} "
          f"+- {np.std([r['base_acc'] for r in rows]):.4f}")


if __name__ == "__main__":
    main()
