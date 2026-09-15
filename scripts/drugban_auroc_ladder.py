#!/usr/bin/env python3
"""Re-score the DrugBAN protein ladder under AUROC, the metric the benchmark is published with.

The saved ``guard_results.json`` reports accuracy, because ``gate_row`` scores
with the accuracy helper that every benchmark shares. DrugBAN is published under
AUROC, so the severity figure was labelling an accuracy curve with DrugBAN's
name. This script repeats the same corrector selection, on the same dumps and
the same splits, and reports AUROC for the frozen model, for blanket correction
and for the certified policy.

Only the reported metric changes. The policy is selected exactly as before, on
the canonical loss, and the harm numbers are defined through that loss, so they
are copied from the existing file rather than recomputed.
"""
import importlib.util, json, sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B / "experiments"))
sys.path.insert(0, str(B / "src"))

from guard import action as _A, losses as _L, targets as _T   # noqa: E402
from guard.pipeline import _select_beta                        # noqa: E402
from gates_core import _score, KS, TS, SPACES, WTS, ALPHA, DELTA  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "protladder", B / "scripts/drugban_protladder_v2.py")
PL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PL)


def one(P, F, Y, split, richer):
    """AUROC of frozen, blanket and certified, under gate_row's own selection."""
    loss = _L.get("cross_entropy")
    pool, fit, conf, test = split
    acc = lambda Q, idx: _score(Q, Y[idx])
    n_out = P.shape[1]
    SP = {}
    for sp in SPACES:
        z = _T.retrieval_space(F[pool], sp)
        SP[sp] = {n: z(F[i]) for n, i in
                  (("pool", pool), ("fit", fit), ("conf", conf), ("test", test))}

    def values(tg, T):
        if tg == "hard":
            return _T.hard_label_values(Y[pool], n_out, loss.simplex)
        r = richer if T == 1.0 else _T.temper(richer, T, loss.simplex)
        return _T.cross_mask_values(r[pool])

    best = None
    for T in TS:
        pr = P if T == 1.0 else _T.temper(P, T, loss.simplex)
        for tg in ("hard", "cross"):
            vals = values(tg, T)
            for sp in SPACES:
                for wt in WTS:
                    for k in KS:
                        ke = min(k, len(pool) - 1)
                        tf = _T.knn_average(SP[sp]["fit"], SP[sp]["pool"], vals, ke,
                                            weighting=wt)
                        b = _select_beta(pr[fit], tf, Y[fit], loss, "loss")
                        s = acc((1 - b) * pr[fit] + b * tf, fit)
                        if best is None or s > best[0]:
                            best = (s, T, tg, sp, wt, k, b)
    _, T, tg, sp, wt, k, b = best
    pr = P if T == 1.0 else _T.temper(P, T, loss.simplex)
    vals = values(tg, T)
    ke = min(k, len(pool) - 1)
    tt = {n: _T.knn_average(SP[sp][n], SP[sp]["pool"], vals, ke, weighting=wt)
          for n in ("fit", "conf", "test")}
    mc, mt = P[conf], P[test]
    cc = (1 - b) * pr[conf] + b * tt["conf"]
    ct = (1 - b) * pr[test] + b * tt["test"]
    cf = (1 - b) * pr[fit] + b * tt["fit"]
    sc = _A.fit_action_score(P[fit], tt["fit"], cf, Y[fit], loss)
    g = _A.certify_action(sc, mc, tt["conf"], cc, Y[conf], mt, tt["test"],
                          loss, ALPHA, DELTA, fit=(P[fit], tt["fit"], cf, Y[fit]))
    ap = g["apply"]
    auroc = lambda Q: float(roc_auc_score(Y[test], Q[:, 1]))
    return dict(frozen=auroc(mt), blanket=auroc(ct),
                guard=auroc(np.where(ap[:, None], ct, mt)))


def main():
    # the figure reads artifacts first and falls back to results; two copies of
    # this file exist and they disagree on harm, so follow the same order here
    src = (B / "artifacts/drugban_protladder_v2/guard_results.json")
    if not src.exists():
        src = B / "results/drugban_protladder_v2/guard_results.json"
    old = json.load(open(src))
    out = {"metric": "AUROC", "alpha": old["alpha"], "delta": old["delta"],
           "fractions_percent": old["fractions_percent"], "per_fraction": {}}
    for pct in PL.PCTS:
        z = np.load(PL.OUT / PL.file_name(pct), allow_pickle=False)
        full = np.load(PL.OUT / "full.npz", allow_pickle=False)
        P = np.concatenate([z["pool_probs"], z["calib_probs"], z["test_probs"]]).astype(np.float64)
        F = np.concatenate([z["pool_feats"], z["calib_feats"], z["test_feats"]]).astype(np.float64)
        Y = np.concatenate([z["pool_labels"], z["calib_labels"], z["test_labels"]])
        R = np.concatenate([full["pool_probs"], full["calib_probs"], full["test_probs"]]).astype(np.float64)
        npool, ncal = len(z["pool_labels"]), len(z["calib_labels"])
        vals = []
        for seed in range(3):
            perm = np.random.default_rng(seed).permutation(ncal) + npool
            split = (np.arange(npool), perm[:ncal // 2], perm[ncal // 2:],
                     np.arange(len(z["test_labels"])) + npool + ncal)
            vals.append(one(P, F, Y, split, R))
        mean = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        # harm is a property of the loss, not of the reported metric
        src = old["per_fraction"][str(pct)]["mean"]
        mean["joint_harm"] = src["joint_harm"]
        mean["guard_without_certify_joint_harm"] = src["guard_without_certify_joint_harm"]
        out["per_fraction"][str(pct)] = {"mean": mean, "seeds": vals}
        print(f"{pct:3d}%  frozen {mean['frozen']:.4f}  blanket {mean['blanket']:.4f} "
              f"  GUARD {mean['guard']:.4f}", flush=True)
    p = src.parent / "guard_results_auroc.json"
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
