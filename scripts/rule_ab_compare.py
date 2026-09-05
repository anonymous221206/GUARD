#!/usr/bin/env python3
"""Re-evaluate saved GUARD dumps without mutating their source artifacts."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("$WORKSPACE")
OUT = ROOT / "results" / "rule_ab_compare_20260828"
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from guard import HostOutputs, run
from guard.splits import Split
from experiments.exp_drugban import build as drugban_build


def result_row(pair, source, condition, seed, objective, r, seconds):
    return dict(pair=pair, source=source, condition=condition, seed=seed,
                beta_objective=objective, beta=r.beta,
                base_accuracy=r.base_metric,
                guard_accuracy=r.base_metric + r.gate_metric_delta,
                joint_harm=r.joint_harm, apply_rate=r.apply_rate,
                seconds=seconds, n_pool=r.n_pool, n_fit=r.n_fit,
                n_conf=r.n_conf, n_test=r.n_test)


def run_drugban(rows, timings):
    for dataset in ("biosnap", "bindingdb", "human"):
        for objective in ("loss", "crossfit"):
            start = time.perf_counter()
            for seed in (1, 2, 42):
                dumps = ROOT / "data" / "processed" / f"drugban_{dataset}_random_s{seed}"
                host, split = drugban_build(dumps, "prot25", "source", 0)
                once = time.perf_counter()
                r = run(host, split, condition="prot25", target="hard",
                        alpha=0.2, delta=0.05, k=50, beta_objective=objective)
                rows.append(result_row(f"DrugBAN/{dataset}", str(dumps), "prot25",
                                       seed, objective, r, time.perf_counter() - once))
            timings.append(dict(pair=f"DrugBAN/{dataset}", beta_objective=objective,
                                seconds=time.perf_counter() - start, n_runs=3))
            print(f"DrugBAN/{dataset} {objective} done", flush=True)


def opportunity_split(y_dep, seed):
    dep_u = np.arange(len(y_dep))[::2]
    perm = dep_u[np.random.default_rng(700 + seed).permutation(len(dep_u))]
    available, test = perm[:3 * (len(dep_u) // 4)], perm[3 * (len(dep_u) // 4):]
    t3 = len(available) // 3
    return Split(available[:t3], available[t3:2*t3], available[2*t3:3*t3], test,
                 origin={k: "subject 4" for k in ("pool", "fit", "conf", "test")})


def run_opportunity(rows, timings):
    dump = ROOT / "artifacts" / "opportunity_deepconvlstm" / "dumps"
    y_dep = np.load(dump / "deploy_y.npy")
    cfgs = json.loads(str(np.load(ROOT / "data" / "processed" / "opportunity.npz",
                                  allow_pickle=True)["configs"]))
    for objective in ("loss", "crossfit"):
        start = time.perf_counter()
        for seed in (0, 1, 2):
            split = opportunity_split(y_dep, seed)
            for condition in cfgs:
                probs = np.load(dump / f"probs_condition_specialist_{condition}_deploy_s{seed}.npy")
                feats = np.load(dump / f"retfeat_{condition}_deploy.npy")
                host = HostOutputs(probs=probs, features=feats, labels=y_dep)
                once = time.perf_counter()
                r = run(host, split, condition=condition, target="hard", alpha=0.2,
                        delta=0.05, k=50, beta_objective=objective)
                rows.append(result_row("OPPORTUNITY/DeepConvLSTM", str(dump), condition,
                                       seed, objective, r, time.perf_counter() - once))
        timings.append(dict(pair="OPPORTUNITY/DeepConvLSTM", beta_objective=objective,
                            seconds=time.perf_counter() - start, n_runs=3 * len(cfgs)))
        print(f"OPPORTUNITY/DeepConvLSTM {objective} done", flush=True)


def simplex(x):
    p = 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))
    return np.stack([1.0 - p, p], 1)


def run_cmad(rows):
    d = ROOT / "artifacts" / "mosei_cmad" / "dumps"
    raw = np.load(d / "raw_features.npz")
    pr = np.load(d / "student_preds.npz")
    parts = ("train", "dev", "test")
    labels = np.concatenate([(raw[f"{s}_y"] > 0).astype(int) for s in parts])
    n_train, n_dev = len(raw["train_y"]), len(raw["dev_y"])
    pool = np.arange(n_train)
    deploy = np.arange(n_train + n_dev, len(labels))
    block = {"a": ("ac",), "v": ("vis",), "av": ("ac", "vis"),
             "intact": ("ac", "vis")}
    for condition, mods in block.items():
        key = "tav" if condition == "intact" else condition
        features = np.concatenate([np.concatenate([raw[f"{s}_{m}"] for s in parts])
                                   for m in mods], 1).astype(np.float64)
        features = (features - features[pool].mean(0)) / (features[pool].std(0) + 1e-8)
        probs = simplex(np.concatenate([pr[f"{s}_{key}"] for s in parts]))
        host = HostOutputs(probs=probs, features=features, labels=labels)
        for seed in range(20):
            perm = np.random.default_rng(1000 + seed).permutation(deploy)
            t3 = len(perm) // 3
            split = Split(pool, perm[:t3], perm[t3:2*t3], perm[2*t3:],
                          origin={"pool": "training population", "fit": "deployment",
                                  "conf": "deployment", "test": "deployment"})
            once = time.perf_counter()
            r = run(host, split, condition=condition, target="hard", alpha=0.2,
                    delta=0.05, k=50, beta_objective="loss")
            rows.append(result_row("CMU-MOSEI/CMAD", str(d), condition, seed, "loss",
                                   r, time.perf_counter() - once))
        print(f"CMU-MOSEI/CMAD {condition} done", flush=True)


def run_iemocap(rows):
    d = ROOT / "artifacts" / "iemocap_momke" / "folds"
    for fold in range(5):
        z = np.load(d / f"fold{fold}_a.npz", allow_pickle=True)
        p_pool, f_pool, y_pool = (z["pool_probs"].astype(np.float64),
                                  z["pool_feats"].astype(np.float64), z["pool_labels"])
        p_query, f_query, y_query = (z["probs"].astype(np.float64),
                                     z["feats"].astype(np.float64), z["labels"])
        probs = np.concatenate([p_pool, p_query])
        feats = np.concatenate([f_pool, f_query])
        labels = np.concatenate([y_pool, y_query])
        vids = z["vid"]
        order = np.unique(vids)[np.random.default_rng(fold).permutation(len(np.unique(vids)))]
        t3 = len(order) // 3
        sel = lambda a: np.flatnonzero(np.isin(vids, a)) + len(y_pool)
        split = Split(np.arange(len(y_pool)), sel(order[:t3]), sel(order[t3:2*t3]),
                      sel(order[2*t3:]),
                      origin={"pool": "training sessions", "fit": "held-out session",
                              "conf": "held-out session", "test": "held-out session"})
        once = time.perf_counter()
        r = run(HostOutputs(probs=probs, features=feats, labels=labels), split,
                condition="a", target="hard", alpha=0.2, delta=0.05, k=50,
                beta_objective="crossfit")
        rows.append(result_row("IEMOCAP/MoMKE", str(d), "audio", fold, "crossfit",
                               r, time.perf_counter() - once))
        print(f"IEMOCAP/MoMKE fold {fold} done", flush=True)


def write_rows(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    ab, timings, artifacts = [], [], []
    run_drugban(ab, timings)
    run_opportunity(ab, timings)
    run_cmad(artifacts)
    run_iemocap(artifacts)
    write_rows(OUT / "ab_guard.csv", ab)
    write_rows(OUT / "artifact_guard.csv", artifacts)
    write_rows(OUT / "timings.csv", timings)
    (OUT / "manifest.json").write_text(json.dumps({"source": "saved dumps only", "alpha": 0.2,
        "delta": 0.05, "drugban_seeds": [1, 2, 42], "opportunity_seeds": [0, 1, 2],
        "cmad_seeds": list(range(20)), "iemocap_folds": list(range(5))}, indent=2))
    print(f"DONE {OUT}", flush=True)


if __name__ == "__main__":
    main()
