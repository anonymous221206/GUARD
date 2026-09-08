#!/usr/bin/env python3
"""Three synthetic hosts tuned to the same accuracy, differing in why they err.

The point of the reduction is that raw error does not say how much is
correctable. To test it, the error has to be held fixed while its source moves,
which a single knob cannot do: here ``tau`` sets how ambiguous the label is
given X, and ``g`` sets how far the host sits from the Bayes prediction. Each
cell bisects one knob until the host reaches the same accuracy as the others, so
the three differ in the Bayes gap and in nothing else that the metric can see.

    bias_dominant   sharp tau, large g   -- error is mostly miscalibration
    mixed           both intermediate
    ambig_dominant  g = 0, flat tau      -- host is Bayes-optimal, error is noise

This is the driver behind the synthetic table; ``exp_synthetic.py`` is a
smaller, faster check that shares no numbers with it, because the skew it uses
preserves the arg-max and so cannot move accuracy at all.
"""
from __future__ import annotations
import argparse, csv, json, sys
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
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=Path("results/synthetic_matched"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in a.seeds:
        rng = np.random.default_rng(1000 + seed)
        W = rng.standard_normal((D, C))
        V = rng.standard_normal((D, C)) / np.sqrt(D)
        probe = lambda t, g: base_acc(np.random.default_rng(7), W, V, t, g)
        tau_amb = tune(0.2, 30.0, TARGET_ACC, lambda t: probe(t, 0.0))
        tau_sharp = 0.6
        g_bias = tune(0.0, 8.0, TARGET_ACC, lambda g: probe(tau_sharp, g))
        tau_mid = tune(0.2, 30.0, 0.5 * (TARGET_ACC + 0.92), lambda t: probe(t, 0.0))
        g_mid = tune(0.0, 8.0, TARGET_ACC, lambda g: probe(tau_mid, g))
        for cname, (tau, g) in (("bias_dominant", (tau_sharp, g_bias)),
                                ("mixed", (tau_mid, g_mid)),
                                ("ambig_dominant", (tau_amb, 0.0))):
            Xs, ys, _, _ = gen(rng, N_SRC, W, V, tau, g)
            Xf, yf, _, mf = gen(rng, N_FIT, W, V, tau, g)
            Xt, yt, pit, mt = gen(rng, N_TEST, W, V, tau, g)
            acc0 = float((mt.argmax(1) == yt).mean())
            vals = _T.hard_label_values(ys, C, True)
            pi_f = _T.knn_average(Xf, Xs, vals, K, weighting="uniform")
            pi_t = _T.knn_average(Xt, Xs, vals, K, weighting="uniform")
            b = _select_beta(mf, pi_f, yf, CE, "loss")
            pc = (1 - b) * mt + b * pi_t
            rows.append(dict(condition=cname, seed=seed, tau=round(float(tau), 3),
                             g=round(float(g), 3), base_acc=acc0,
                             ph_true=float(((mt - pit) ** 2).sum(1).mean()),
                             amb_true=float((pit * (1 - pit)).sum(1).mean()),
                             beta=float(b),
                             loss_gain=float((CE(mt, yt) - CE(pc, yt)).mean()),
                             d_acc=float((pc.argmax(1) == yt).mean() - acc0)))
            r = rows[-1]
            print(f"s{seed} {cname:15} acc={r['base_acc']:.3f} PH={r['ph_true']:.4f} "
                  f"amb={r['amb_true']:.3f} beta={r['beta']:.2f} "
                  f"gain={r['loss_gain']:+.4f} d_acc={r['d_acc']:+.4f}", flush=True)
    with open(a.out / "guard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\n%-16s %8s %8s %8s %10s %8s" % ("cell", "base acc", "PH_true", "beta", "loss gain", "d acc"))
    for cname in ("bias_dominant", "mixed", "ambig_dominant"):
        v = [r for r in rows if r["condition"] == cname]
        m = lambda k: float(np.mean([x[k] for x in v]))
        print("%-16s %8.3f %8.4f %8.2f %+10.4f %+8.4f"
              % (cname, m("base_acc"), m("ph_true"), m("beta"), m("loss_gain"), m("d_acc")))


if __name__ == "__main__":
    main()
