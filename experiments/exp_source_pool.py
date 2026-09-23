#!/usr/bin/env python3
"""What the retrieval pool D_source contributes, and what it cannot break.

Four studies, each varying one thing about the offline data while every other
role stays fixed:

``size``    subsample D_source; hard-label and cross-mask targets side by side.
            A hard-label pool of N rows costs N labels; a cross-mask pool costs none.
``noise``   flip a fraction of the D_source labels; the cross-mask target reads
            no labels and is the reference.
``origin``  cross-domain DrugBAN: the pool from the deployment domain, from the
            source domain at matched size, the whole source pool, or both.
``conf``    subsample D_conf; the certificate is calibrated there, so joint harm
            should stay inside the budget at every size and only the apply rate
            should pay.
``clean``   in-domain DrugBAN with the pool carved from deployment data, which the
            host never trained on (the released random pool is its training data).

    python experiments/exp_source_pool.py size --bench drugban --out results/source_pool
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guard import HostOutputs, run                                # noqa: E402
from guard.losses import accuracy as _accuracy, get as _get_loss  # noqa: E402
from guard.pipeline import select_on_fit                          # noqa: E402
from guard.splits import Split                                    # noqa: E402
from guard import targets as _targets                             # noqa: E402

_knn = _targets.knn_average


def _knn_stable(query, pool, values, k, chunk=1024, weighting="uniform"):
    """``knn_average`` with the distance kernel shifted by each row's nearest hit.

    With a pool of a few dozen rows the kernel width is small against the query
    distances and every weight underflows, giving 0/0. Shifting by the smallest
    distance leaves the normalised weights unchanged in exact arithmetic. Only
    rows that came back NaN are recomputed, so every other row is bit-identical
    to the library's, and the full-pool arms reproduce the paper's cells.
    """
    out = _knn(query, pool, values, k, chunk=chunk, weighting=weighting)
    bad = np.isnan(out).any(1)
    if bad.any():
        q = query[bad]
        pool_sq = (pool ** 2).sum(1)
        tau = _targets._pool_scale(pool, pool_sq, k)
        d2 = (q ** 2).sum(1)[:, None] + pool_sq[None, :] - 2.0 * (q @ pool.T)
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        d = np.sqrt(np.maximum(np.take_along_axis(d2, idx, 1), 0.0))
        w = np.exp(-(d - d.min(1, keepdims=True)) / tau)
        w /= w.sum(1, keepdims=True)
        out[bad] = (values[idx] * w[:, :, None]).sum(1)
    return out


_targets.knn_average = _knn_stable

ARTIFACTS = Path(os.environ.get("GUARD_ARTIFACTS", ROOT / "artifacts"))
ALPHA, DELTA = 0.2, 0.05
GRID = dict(k_grid=(3, 5, 8, 12, 20, 35, 50), space_grid=("standardise", "cosine"),
            weighting_grid=("uniform", "distance"), temperature_grid=(1.0, 2.0))
POLICIES = {"selected": ("hard", "cross_mask"), "hard": ("hard",),
            "cross_mask": ("cross_mask",)}
CE = _get_loss("cross_entropy")


# ---------------------------------------------------------------- benchmarks
def drugban_host(stem, cond):
    d = np.load(ROOT / "data/processed" / f"drugban_{stem}" / f"{cond}.npz")
    rich = np.load(ROOT / "data/processed" / f"drugban_{stem}" / "full.npz")
    parts = ("pool", "calib", "test")
    cat = lambda z, k: np.concatenate([z[f"{p}_{k}"] for p in parts])
    n = {p: len(d[f"{p}_labels"]) for p in parts}
    host = HostOutputs(probs=cat(d, "probs").astype(np.float64), features=cat(d, "feats"),
                       labels=cat(d, "labels"),
                       richer_probs=cat(rich, "probs").astype(np.float64))
    return host, n


def drugban_split(n, pool_from, seed=0):
    """The split exp_drugban.py uses, so the full-pool arm reproduces the paper."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n["calib"]) + n["pool"]
    test = np.arange(n["test"]) + n["pool"] + n["calib"]
    if pool_from == "source":
        h = n["calib"] // 2
        return dict(pool=np.arange(n["pool"]), fit=perm[:h], conf=perm[h:], test=test)
    t = n["calib"] // 3
    return dict(pool=perm[:t], fit=perm[t:2 * t], conf=perm[2 * t:3 * t], test=test,
                source=np.arange(n["pool"]))


MOSEI = {
    "CMAD": dict(dir="mosei_cmad/dumps", preds="student_preds.npz", alias={},
                 feat={"a": "ac", "v": "vis"}),
    "TMDC": dict(dir="mosei_tmdc/dumps", preds="preds.npz", alias={"tav": "atv"},
                 feat={"a": "a", "v": "v"}),
    "MoMKE": dict(dir="mosei_momke/dumps", preds="preds.npz", alias={"tav": "atv"},
                  feat={"a": "a", "v": "v"}),
}


def mosei_host(name, mask):
    cfg = MOSEI[name]
    D = ARTIFACTS / cfg["dir"]
    P = np.load(D / cfg["preds"], allow_pickle=True)
    R = np.load(D / "raw_features.npz", allow_pickle=True)
    raw = R["test_y"].reshape(-1)

    def probs(m):
        s = P[f"test_{cfg['alias'].get(m, m)}"].astype(np.float64)
        if s.ndim == 1 or s.shape[1] == 1:
            e = 1 / (1 + np.exp(-s.reshape(-1)))
            return np.stack([1 - e, e], 1)
        e = np.exp(s - s.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    F = np.concatenate([R[f"test_{cfg['feat'][c]}"].astype(np.float64) for c in mask], 1)
    return HostOutputs(probs=probs(mask), features=F, labels=(raw > 0).astype(int),
                       richer_probs=probs("tav"), raw_labels=raw)


def mosei_split(n, seed):
    pool, fit, conf, test = np.array_split(np.random.default_rng(seed).permutation(n), 4)
    return dict(pool=pool, fit=fit, conf=conf, test=test)


# ---------------------------------------------------------------- one cell
def cell(host, roles, policy, metric, labels=None, cfg=None):
    """Select on fit, run, and return the row. ``labels`` overrides pool labels.

    Selection reads only pool and fit, so a study that varies D_conf passes the
    configuration it already chose instead of repeating the search."""
    if labels is not None:
        host = HostOutputs(probs=host.probs, features=host.features, labels=labels,
                           richer_probs=host.richer_probs, raw_labels=host.raw_labels)
    sp = Split(roles["pool"], roles["fit"], roles["conf"], roles["test"],
               origin={k: "deployment" for k in ("pool", "fit", "conf", "test")})
    if cfg is None:
        cfg = select_on_fit(host, sp, target_grid=POLICIES[policy], metric=metric, **GRID)
        cfg = {k: cfg[k] for k in ("k", "target", "space", "weighting", "temperature")}
    r = run(host, sp, alpha=ALPHA, delta=DELTA, metric=metric, **cfg)
    ta = r.test_arrays
    return dict(policy=policy, chosen=cfg["target"], k=r.k, beta=r.beta,
                n_pool=len(sp.pool), n_fit=len(sp.fit), n_conf=len(sp.conf),
                n_test=len(sp.test), base_metric=r.base_metric,
                gate_metric_delta=r.gate_metric_delta,
                blanket_metric_delta=r.blanket_metric_delta,
                base_accuracy=_accuracy(ta["base_probs"], ta["labels"], CE),
                gate_accuracy=_accuracy(ta["gated_probs"], ta["labels"], CE),
                joint_harm=r.joint_harm, blanket_joint_harm=r.blanket_joint_harm,
                apply_rate=r.apply_rate, target_accuracy=r.target_accuracy, _cfg=cfg)


def settings(bench, only=None):
    """(name, seed, condition, host, roles, metric) for every setting."""
    if bench == "drugban":
        for ds in ("biosnap", "bindingdb", "human"):
            if only and ds != only:
                continue
            for s in ("s42", "s1", "s2"):
                for cond in ("prot25", "scaffold_prot50"):
                    host, n = drugban_host(f"{ds}_random_{s}", cond)
                    yield ds, s, cond, host, drugban_split(n, "source"), "auroc"
    else:
        for name in MOSEI:
            if only and name != only:
                continue
            for mask in ("a", "v", "av"):
                host = mosei_host(name, mask)
                for seed in range(5):
                    yield name, seed, mask, host, mosei_split(len(host.labels), seed), \
                        "accuracy_nonzero"


# ---------------------------------------------------------------- studies
def study_size(bench, emit, only=None):
    sizes = (100, 300, 1000, 3000) if bench == "drugban" else (50, 150, 400)
    reps = 3 if bench == "drugban" else 1
    for name, seed, cond, host, roles, metric in settings(bench, only):
        full = roles["pool"]
        arms = [(len(full), 0, full)]
        for N in sizes:
            if N >= len(full):
                continue
            for rep in range(reps):
                rng = np.random.default_rng(1000 * rep + N)
                arms.append((N, rep, np.sort(rng.choice(full, N, replace=False))))
        for N, rep, pool in arms:
            for policy in POLICIES:
                row = cell(host, {**roles, "pool": pool}, policy, metric)
                emit(dict(study="size", bench=bench, name=name, seed=seed, cond=cond,
                          rep=rep, pool_frac=N / len(full), **row))


def study_noise(bench, emit, only=None):
    for name, seed, cond, host, roles, metric in settings(bench, only):
        pool = roles["pool"]
        for rate in (0.1, 0.2, 0.3, 0.4):
            rng = np.random.default_rng(int(rate * 100))
            y = host.labels.copy()
            flip = pool[rng.random(len(pool)) < rate]
            y[flip] = 1 - y[flip]                 # every benchmark here is binary
            # only the pool is corrupted; fit, conf and test keep their labels
            for policy in ("selected", "hard"):
                row = cell(host, roles, policy, metric, labels=y)
                emit(dict(study="noise", bench=bench, name=name, seed=seed, cond=cond,
                          noise=rate, **row))


def study_origin(bench, emit, only=None):
    assert bench == "drugban"
    for ds, seeds in (("biosnap", ("s42", "s1", "s2")), ("bindingdb", ("s42",))):
        if only and ds != only:
            continue
        for s in seeds:
            for cond in ("prot25", "scaffold_prot50"):
                host, n = drugban_host(f"{ds}_cluster_{s}", cond)
                roles = drugban_split(n, "deployment")
                dep, src = roles["pool"], roles["source"]
                arms = [("deployment", 0, dep), ("source_all", 0, src),
                        ("source_all+deployment", 0, np.concatenate([src, dep]))]
                for rep in range(3):
                    rng = np.random.default_rng(rep)
                    arms.append(("source_matched", rep,
                                 np.sort(rng.choice(src, len(dep), replace=False))))
                for arm, rep, pool in arms:
                    for policy in POLICIES:
                        row = cell(host, {**roles, "pool": pool}, policy, "auroc")
                        emit(dict(study="origin", bench=bench, name=ds, seed=s,
                                  cond=cond, arm=arm, rep=rep, **row))


def study_conf(bench, emit, only=None):
    sizes = (50, 100, 200, 400)
    for name, seed, cond, host, roles, metric in settings(bench, only):
        if bench == "drugban" and seed != "s42":
            continue
        full = roles["conf"]
        first = cell(host, roles, "selected", metric)
        cfg = first["_cfg"]
        emit(dict(study="conf", bench=bench, name=name, seed=seed, cond=cond, rep=0, **first))
        for N in sizes:
            if N >= len(full):
                continue
            for rep in range(20 if bench == "drugban" else 8):
                rng = np.random.default_rng(1000 * rep + N)
                conf = np.sort(rng.choice(full, N, replace=False))
                row = cell(host, {**roles, "conf": conf}, "selected", metric, cfg=cfg)
                emit(dict(study="conf", bench=bench, name=name, seed=seed, cond=cond,
                          rep=rep, **row))


def study_clean(bench, emit, only=None):
    """In-domain DrugBAN with the pool carved from deployment data.

    The released random split retrieves from the host's training set, where the
    all-inputs model is near-perfect because it was fitted there; its outputs on
    that pool are close to the labels themselves. A pool the host never trained
    on removes that route, so hard-label and cross-mask are compared cleanly.
    """
    assert bench == "drugban"
    for ds in ("biosnap", "bindingdb", "human"):
        if only and ds != only:
            continue
        for s in ("s42", "s1", "s2"):
            for cond in ("prot25", "scaffold_prot50"):
                host, n = drugban_host(f"{ds}_random_{s}", cond)
                roles = drugban_split(n, "deployment")
                for policy in POLICIES:
                    row = cell(host, roles, policy, "auroc")
                    emit(dict(study="clean", bench=bench, name=ds, seed=s, cond=cond,
                              arm="deployment", **row))


STUDIES = {"clean": study_clean, "size": study_size, "noise": study_noise, "origin": study_origin,
           "conf": study_conf}
COLS = ["study", "bench", "name", "seed", "cond", "arm", "rep", "noise", "pool_frac",
        "policy", "chosen", "k", "beta", "n_pool", "n_fit", "n_conf", "n_test",
        "base_metric", "gate_metric_delta", "blanket_metric_delta", "base_accuracy",
        "gate_accuracy", "joint_harm", "blanket_joint_harm", "apply_rate",
        "target_accuracy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=list(STUDIES))
    ap.add_argument("--bench", default="drugban", choices=["drugban", "mosei"])
    ap.add_argument("--only", default=None, help="one dataset or host, to shard a study")
    ap.add_argument("--out", type=Path, default=ROOT / "results/source_pool")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"{a.study}_{a.bench}{'_' + a.only if a.only else ''}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="raise")
        w.writeheader()

        def emit(row):
            row.pop("_cfg", None)
            w.writerow(row)
            fh.flush()
            print(" ".join(f"{k}={row[k]}" for k in
                           ("name", "seed", "cond", "policy", "n_pool", "n_conf")),
                  f"gain={row['gate_metric_delta']:+.4f} harm={row['joint_harm']:.3f} "
                  f"apply={row['apply_rate']:.2f}", flush=True)
        STUDIES[a.study](a.bench, emit, a.only)
    print("DA GHI", path)


if __name__ == "__main__":
    main()
