#!/usr/bin/env python3
"""Two group-level checks on the certificate.

``class_harm``
    The guarantee is marginal over the test draw, so it says nothing per class.
    This measures harm within each class under the ordinary marginal gate, with
    the group sizes printed beside it -- a rate over a dozen samples is not a
    rate, and the reader needs to see which is which.

``calib_size``
    How the spread of realised harm across calibration draws behaves as the
    calibration set shrinks, against the binomial scale sqrt(alpha(1-alpha)/n).
    That expression is evaluated at p = alpha, so it bounds the spread rather
    than predicting it: where realised harm sits well below the budget, the
    observed spread is several times smaller and much flatter in n.

    python experiments/exp_groupwise.py --study class_harm
    python experiments/exp_groupwise.py --study calib_size

Both run on the affective hosts, which are consumed exactly as their authors
released them, and both reuse the pipeline's own primitives.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from guard import losses as L                             # noqa: E402
from guard.certify import certify                         # noqa: E402
from guard.pipeline import BETA_GRID                      # noqa: E402
from guard.splits import Split                            # noqa: E402
from guard.targets import hard_label_values, knn_average, standardise  # noqa: E402
from hosts.dumps import MASKS, build                      # noqa: E402

LOSS = L.CROSS_ENTROPY


def prepare(host_dir: Path, mask: str, seed: int, k: int = 50) -> dict:
    """Corrected and host outputs on one mask, with the usual thirds split."""
    probs, feats, labels, richer, n_pool, n_dep = build(host_dir, mask, "train")
    perm = np.random.default_rng(seed).permutation(n_dep) + n_pool
    thirds = np.array_split(perm, 3)
    split = Split(pool=np.arange(n_pool), fit=thirds[0], conf=thirds[1], test=thirds[2],
                  origin={"pool": "training session",
                          **{r: "deployment session" for r in ("fit", "conf", "test")}})
    z = standardise(feats[split.pool])
    f_pool = z(feats[split.pool])
    values = hard_label_values(labels[split.pool], probs.shape[1], True)
    k_eff = min(k, len(split.pool) - 1)
    est = lambda q: knn_average(z(q), f_pool, values, k_eff)
    t_fit, t_conf, t_test = (est(feats[split.fit]), est(feats[split.conf]),
                             est(feats[split.test]))
    beta = float(min(BETA_GRID, key=lambda b: LOSS(
        (1 - b) * probs[split.fit] + b * t_fit, labels[split.fit]).mean()))
    return dict(beta=beta,
                pc_conf=(1 - beta) * probs[split.conf] + beta * t_conf,
                pc_test=(1 - beta) * probs[split.test] + beta * t_test,
                m_test=probs[split.test],
                y_conf=labels[split.conf], y_test=labels[split.test])


def study_class_harm(hosts, seeds, alpha, delta, out: Path):
    fh = open(out / "class_harm.csv", "w", newline="")
    cols = ["host", "mask", "seed", "cls", "n_test_cls", "n_applied_cls",
            "joint_harm_cls", "cond_harm_cls", "joint_harm_overall"]
    wr = csv.DictWriter(fh, fieldnames=cols)
    wr.writeheader()
    by_class = defaultdict(list)
    for hdir in hosts:
        for mask in MASKS:
            for seed in seeds:
                d = prepare(hdir, mask, seed)
                apply = certify(d["pc_conf"], d["y_conf"], d["pc_test"], d["m_test"],
                                LOSS, alpha, delta)["apply"]
                base = LOSS(d["m_test"], d["y_test"])
                dl = np.where(apply, LOSS(d["pc_test"], d["y_test"]), base) - base
                harm = dl > delta
                for c in np.unique(d["y_test"]):
                    sel = d["y_test"] == c
                    r = dict(host=hdir.name, mask=mask, seed=seed, cls=int(c),
                             n_test_cls=int(sel.sum()),
                             n_applied_cls=int((sel & apply).sum()),
                             joint_harm_cls=round(float(harm[sel].mean()), 6),
                             cond_harm_cls=(round(float(harm[sel & apply].mean()), 6)
                                            if (sel & apply).any() else 0.0),
                             joint_harm_overall=round(float(harm.mean()), 6))
                    wr.writerow(r)
                    by_class[int(c)].append(r)
            print(f"  {hdir.name}/{mask}", flush=True)
    fh.close()

    print(f"\n{'class':>6s} {'mean n':>8s} {'joint harm':>11s} {'worst cell':>11s} "
          f"{'cells over alpha':>17s}")
    for c in sorted(by_class):
        v = by_class[c]
        jh = np.array([r["joint_harm_cls"] for r in v])
        print(f"{c:6d} {np.mean([r['n_test_cls'] for r in v]):8.0f} {jh.mean():11.4f} "
              f"{jh.max():11.4f} {int((jh > alpha).sum()):9d}/{len(v)}")
    allc = [r for v in by_class.values() for r in v]
    over = sum(1 for r in allc if r["joint_harm_cls"] > alpha)
    print(f"\n{over}/{len(allc)} (host, mask, seed, class) cells exceed alpha={alpha}; "
          f"smallest class group {min(r['n_test_cls'] for r in allc)} samples")


def study_calib_size(hosts, seeds, alpha, delta, out: Path,
                     sizes=(25, 50, 100, 200, 0), draws=40):
    fh = open(out / "calib_size.csv", "w", newline="")
    wr = csv.DictWriter(fh, fieldnames=["host", "mask", "n_conf", "draw",
                                        "joint_harm", "apply_rate"])
    wr.writeheader()
    by_n = defaultdict(list)
    for hdir in hosts:
        for mask in MASKS:
            d = prepare(hdir, mask, seeds[0])
            base = LOSS(d["m_test"], d["y_test"])
            n_full = len(d["y_conf"])
            for size in sizes:
                n = n_full if size == 0 else size
                if n > n_full:
                    continue
                for draw in range(draws):
                    rng = np.random.default_rng(9000 + 13 * draw + n)
                    idx = (np.arange(n_full) if size == 0
                           else rng.choice(n_full, n, replace=False))
                    apply = certify(d["pc_conf"][idx], d["y_conf"][idx], d["pc_test"],
                                    d["m_test"], LOSS, alpha, delta)["apply"]
                    dl = np.where(apply, LOSS(d["pc_test"], d["y_test"]), base) - base
                    r = dict(host=hdir.name, mask=mask, n_conf=n, draw=draw,
                             joint_harm=round(float((dl > delta).mean()), 6),
                             apply_rate=round(float(apply.mean()), 6))
                    wr.writerow(r)
                    by_n[n].append(r["joint_harm"])
                    if size == 0:
                        break                   # the full set is deterministic
            print(f"  {hdir.name}/{mask}", flush=True)
    fh.close()

    print(f"\n{'n_conf':>7s} {'mean harm':>10s} {'observed sd':>12s} "
          f"{'sqrt(a(1-a)/n)':>15s} {'ratio':>7s}")
    for n in sorted(by_n):
        v = np.array(by_n[n])
        pred = np.sqrt(alpha * (1 - alpha) / n)
        obs = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        print(f"{n:7d} {v.mean():10.4f} {obs:12.4f} {pred:15.4f} {obs / pred:7.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=["class_harm", "calib_size"])
    ap.add_argument("--hosts", type=Path, default=ROOT / "data/raw/hosts")
    ap.add_argument("--out", type=Path, default=ROOT / "results/groupwise")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    hosts = [h for h in sorted(a.hosts.glob("*/")) if (h / "preds.npz").exists()]
    if not hosts:
        raise SystemExit(f"no host with preds.npz under {a.hosts}")
    print(f"hosts: {[h.name for h in hosts]}")
    if a.study == "class_harm":
        study_class_harm(hosts, a.seeds, a.alpha, a.delta, a.out)
    else:
        study_calib_size(hosts, a.seeds, a.alpha, a.delta, a.out)


if __name__ == "__main__":
    main()
