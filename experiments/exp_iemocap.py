#!/usr/bin/env python3
"""IEMOCAP through the five-fold leave-one-session-out dumps.

The standard ``raw_features.npz`` / ``preds.npz`` pair carries a single session
and cannot rebuild the reported five-fold numbers; the artefact therefore ships
``folds/fold{0..4}_{mask}.npz``, each holding its own retrieval pool and query
side.  This entry point consumes those.

Two protocol details are load-bearing.  The split is drawn over **dialogues**,
not utterances, because utterances from one dialogue are far from independent;
and the fold index seeds the generator, so fold ``k`` always draws the same
dialogue partition.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from guard import HostOutputs, run                       # noqa: E402
from guard.pipeline import select_beta                   # noqa: E402
from guard.splits import Split                           # noqa: E402
from guard import targets as _targets                    # noqa: E402

MASKS = ("v", "a", "av", "tv", "t", "at", "atv")
N_SPLIT = 5          # independent dialogue cuts per fold; see the module docstring


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True,
                    help="artifacts/iemocap_momke (or iemocap_tmdc)")
    ap.add_argument("--masks", nargs="+", default=["a"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--splits", type=int, default=N_SPLIT,
                    help="independent dialogue cuts per fold, averaged.  A session "
                         "holds 28-32 dialogues, so one cut leaves ten per part and "
                         "the printed number inherits the luck of that cut.")
    ap.add_argument("--beta-scope", default="pooled", choices=["pooled", "per-fold"],
                    help="'pooled' selects beta on every fold's fit third together, "
                         "roughly five times the data; it is what the reported rows "
                         "use, because per-fold fit is too small for a stable choice.")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--beta-objective", default="crossfit",
                    choices=["loss", "metric", "crossfit"],
                    help="the reported IEMOCAP rows use rule D ('crossfit').")
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()

    d_dir = a.artifact / "folds" if (a.artifact / "folds").is_dir() else a.artifact
    rows = []
    print("%-4s %8s %8s %9s %7s %7s" %
          ("mask", "frozen", "GUARD", "change", "apply", "harm"))
    for mask in a.masks:
        # One pass over the folds: predictions, retrieval target, labels, dialogue id.
        data = []
        for fold in a.folds:
            d = np.load(d_dir / ("fold%d_%s.npz" % (fold, mask)), allow_pickle=True)
            Pp = d["pool_probs"].astype(np.float64)
            Fp = d["pool_feats"].astype(np.float64)
            Yp = d["pool_labels"]
            Pq = d["probs"].astype(np.float64)
            Fq = d["feats"].astype(np.float64)
            Yq = d["labels"]
            z = _targets.standardise(Fp)
            values = _targets.hard_label_values(Yp, Pq.shape[1], True)
            tgt = _targets.knn_average(z(Fq), z(Fp), values, a.k)
            data.append((Pq, Fq, tgt, Yq, d["vid"], Pp, Fp, Yp))

        base_l, guard_l, harm_l, apply_l = [], [], [], []
        for cut in range(a.splits):
            cuts = []
            for i, (_, _, _, _, vid, _, _, _) in enumerate(data):
                uv = np.unique(vid)
                rng = np.random.default_rng(1000 * cut + a.folds[i])
                perm = uv[rng.permutation(len(uv))]
                t3 = len(uv) // 3
                sel = lambda vs: np.flatnonzero(np.isin(vid, vs))
                cuts.append((sel(perm[:t3]), sel(perm[t3:2 * t3]), sel(perm[2 * t3:])))

            beta = None
            if a.beta_scope == "pooled":
                Pf = np.concatenate([data[i][0][cuts[i][0]] for i in range(len(data))])
                Tf = np.concatenate([data[i][2][cuts[i][0]] for i in range(len(data))])
                Yf = np.concatenate([data[i][3][cuts[i][0]] for i in range(len(data))])
                beta = select_beta(Pf, Tf, Yf, objective="crossfit",
                                   alpha=a.alpha, delta=a.delta)

            for i, (Pq, Fq, tgt, Yq, vid, Pp, Fp, Yp) in enumerate(data):
                fit, conf, test = cuts[i]
                n_pool = len(Yp)
                probs = np.concatenate([Pp, Pq])
                feats = np.concatenate([Fp, Fq])
                labels = np.concatenate([Yp, Yq])
                split = Split(pool=np.arange(n_pool), fit=fit + n_pool,
                              conf=conf + n_pool, test=test + n_pool)
                host = HostOutputs(probs=probs, features=feats, labels=labels)
                r = run(host, split, condition=mask, target="hard", alpha=a.alpha,
                        delta=a.delta, k=a.k, beta=beta,
                        beta_objective=a.beta_objective)
                rows.append({**r.as_row(), "fold": a.folds[i], "cut": cut})
                base_l.append(r.base_metric)
                guard_l.append(r.base_metric + r.gate_metric_delta)
                harm_l.append(r.joint_harm)
                apply_l.append(r.apply_rate)

        m = lambda v: sum(v) / len(v)
        print("%-4s %8.4f %8.4f %+9.4f %7.3f %7.4f   <- %d cap (fold, cut)" %
              (mask, m(base_l), m(guard_l), m(guard_l) - m(base_l),
               m(apply_l), m(harm_l), len(base_l)))

    tag = "iemocap_%s" % a.artifact.name
    if a.beta_objective != "crossfit":
        tag += "_" + a.beta_objective
    out_dir = a.out / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "guard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s/guard.csv" % out_dir)


if __name__ == "__main__":
    main()
