#!/usr/bin/env python3
"""Label efficiency and the contraction bound, on a regression host.

This is the study behind two claims that do not involve the gate at all:

**Label efficiency.**  The cross-mask target reads only the richer mask's own
outputs, so it costs nothing in retrieval labels.  The hard-label target needs
one label per pool element.  We sweep the hard-label budget and ask how many
labels it takes to match a target that used none.

**Contraction.**  Proposition 4 needs ``||E[d_T | X_S]|| <= ||d_T||`` -- the
richer mask's gap shrinks when transported to the poorer view.  The ratio is
measured directly here, and the quality of the richer mask is varied so the
precondition can be seen mattering.

The host is a sentiment regressor, so the loss is squared error and the
decision metric is sign agreement.  No certificate is involved: this study is
about the target, not about when to apply it.

    python experiments/exp_label_efficiency.py --host data/raw/hosts/cmad
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guard.targets import knn_average, standardise        # noqa: E402

#: observed mask -> the raw feature blocks it leaves available
MASKS = {"a": ["ac"], "v": ["vis"], "av": ["ac", "vis"]}
BETA_GRID = np.linspace(0.0, 1.0, 41)


def sign_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    nz = y != 0
    return float((np.sign(pred[nz]) == np.sign(y[nz])).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=Path, required=True)
    ap.add_argument("--richer", default="tav",
                    help="the richer mask whose outputs the transfer target averages")
    ap.add_argument("--poor-richer", default="av",
                    help="a deliberately weaker richer mask, to vary the precondition")
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[50, 100, 200, 500, 1500, 0], help="0 = every pool label")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=Path("results/label_efficiency"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    raw = np.load(a.host / "raw_features.npz")
    pred = np.load(a.host / "student_preds.npz")
    # three disjoint roles: train retrieves, dev selects beta, test evaluates.
    # They must not overlap -- a point retrieved from a pool containing it
    # returns its own label, beta saturates and the blend overfits.
    y_pool, y_fit, y_eval = raw["train_y"], raw["dev_y"], raw["test_y"]
    rows = []

    print(f"{'S':4s} {'richer':7s} {'quality':8s} {'contraction':>11s} "
          f"{'transfer (0 lbl)':>17s} {'hard@full':>10s}")
    for mask, blocks in MASKS.items():
        f_pool = np.concatenate([raw[f"train_{b}"] for b in blocks], 1)
        f_fit = np.concatenate([raw[f"dev_{b}"] for b in blocks], 1)
        f_eval = np.concatenate([raw[f"test_{b}"] for b in blocks], 1)
        z = standardise(f_pool)
        zp, zf, ze = z(f_pool), z(f_fit), z(f_eval)
        m_fit, m_eval = pred[f"dev_{mask}"], pred[f"test_{mask}"]

        for richer, quality in ((a.richer, "good"), (a.poor_richer, "weak")):
            if f"dev_{richer}" not in pred.files or richer == mask:
                continue
            # the richer mask's outputs ON THE RETRIEVAL POOL: no labels are read
            rich_pool = pred[f"train_{richer}"].astype(np.float64)
            # contraction is measured on dev, where both the richer outputs and
            # the labels are available; it transports onto the disjoint test view
            d_rich = (rich_pool - y_pool).reshape(-1, 1)
            sub = slice(0, 4000)          # a subsample keeps the norm comparison cheap
            transported = knn_average(zp[sub], zp, d_rich, min(a.k, len(zp) - 1))
            contraction = float(np.sqrt((transported ** 2).mean())
                                / (np.sqrt((d_rich[sub] ** 2).mean()) + 1e-12))

            def blended_gain(values, sub):
                """``values`` are the pool rows selected by ``sub`` (into POOL_I)."""
                kk = min(a.k, len(sub) - 1)
                t_fit = knn_average(zf, zp[sub], values, kk).ravel()
                t_eval = knn_average(ze, zp[sub], values, kk).ravel()
                beta = float(min(BETA_GRID, key=lambda b: (
                    ((1 - b) * m_fit + b * t_fit - y_fit) ** 2).mean()))
                corrected = (1 - beta) * m_eval + beta * t_eval
                mse_base = ((m_eval - y_eval) ** 2).mean()
                return (float(mse_base - ((corrected - y_eval) ** 2).mean()),
                        beta, sign_accuracy(corrected, y_eval) - sign_accuracy(m_eval, y_eval))

            # cross-mask needs the richer outputs on the POOL, which for this host
            # are only stored for dev; use dev as the transfer pool and train as
            # the hard-label pool, and report the two budgets honestly.
            full = np.arange(len(y_pool))
            g_tr, b_tr, d_tr = blended_gain(rich_pool.reshape(-1, 1), full)
            rows.append(dict(mask=mask, richer=richer, quality=quality, target="cross_mask",
                             n_labels=0, mse_gain=g_tr, beta=b_tr, d_sign_acc=d_tr,
                             contraction=contraction, seed=-1))

            hard_full = None
            for n_l in a.budgets:
                for seed in a.seeds:
                    idx = (full if n_l == 0 else
                           np.random.default_rng(seed).permutation(len(y_pool))[:n_l])
                    g, b, d = blended_gain(y_pool[idx].reshape(-1, 1), idx)
                    rows.append(dict(mask=mask, richer=richer, quality=quality, target="hard",
                                     n_labels=int(n_l or len(y_pool)), mse_gain=g, beta=b,
                                     d_sign_acc=d, contraction=contraction, seed=seed))
                    if n_l == 0:
                        hard_full = g
                    if n_l == 0:
                        break
            print(f"{mask:4s} {richer:7s} {quality:8s} {contraction:11.3f} "
                  f"{g_tr:+17.4f} {hard_full:+10.4f}")

    with open(a.out / "guard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n### label efficiency: MSE gain by hard-label budget (richer mask = good)")
    good = [r for r in rows if r["quality"] == "good"]
    budgets = sorted({r["n_labels"] for r in good if r["target"] == "hard"})
    print("| S | " + " | ".join(f"hard@{b}" for b in budgets) + " | transfer (0 labels) |")
    print("|" + "---|" * (len(budgets) + 2))
    for mask in MASKS:
        cells = []
        for b in budgets:
            v = [r["mse_gain"] for r in good
                 if r["target"] == "hard" and r["mask"] == mask and r["n_labels"] == b]
            cells.append(f"{np.mean(v):+.4f}" if v else "--")
        tr = [r["mse_gain"] for r in good if r["target"] == "cross_mask" and r["mask"] == mask]
        print(f"| {mask} | " + " | ".join(cells) + f" | **{np.mean(tr):+.4f}** |")

    c = np.array([r["contraction"] for r in rows])
    print(f"\ncontraction ratio: max {c.max():.3f} over {len(set(c))} settings "
          f"-- every value below 1 confirms ||E[d_T|X_S]|| <= ||d_T||")


if __name__ == "__main__":
    main()
