#!/usr/bin/env python3
"""Create the CPU-only BindingDB-s1 protein-retention ladder and GUARD panel.

This is intentionally a create-only runner.  It calls the frozen DrugBAN
prefix path verbatim for 100/50/25 and uses its same integer-prefix rule for
the other retained fractions.  GUARD evaluation follows ``gates_drugban.py``:
three calibration permutations, its fixed alpha/delta and its unretuned gate
selection grid.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


B = Path(__file__).resolve().parents[1]
HOST = B / "hosts/drugban.py"
REPO = B.parent / "CURA/external/DrugBAN"
CKPT = B / "checkpoints/drugban/bindingdb_s1.pth"
CFG = B / "checkpoints/drugban/bindingdb_s1.yaml"
STORED = B / "data/processed/drugban_bindingdb_random_s1"
OUT = B / "artifacts/drugban_protladder_v2"
PCTS = (100, 80, 70, 60, 50, 40, 30, 25, 20, 15, 10)
ALPHA, DELTA = 0.2, 0.05


def load_host():
    spec = importlib.util.spec_from_file_location("frozen_drugban_host", HOST)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def file_name(pct: int) -> str:
    return "full.npz" if pct == 100 else f"prot{pct}.npz"


def degraded(df: pd.DataFrame, pct: int, host):
    if pct == 100:
        return host.degrade(df, "full")[0]
    if pct == 50:
        return host.degrade(df, "prot50")[0]
    if pct == 25:
        return host.degrade(df, "prot25")[0]
    d = host.degrade(df, "full")[0]
    d["Protein"] = [p[:max(1, len(p) * pct // 100)] for p in d["Protein"]]
    return d


def summary(parts: dict) -> dict:
    y, p = parts["test_labels"], parts["test_probs"]
    return {
        "test_auroc": float(roc_auc_score(y, p[:, 1])),
        "test_accuracy": float((p.argmax(1) == y).mean()),
        "pool_rows": int(len(parts["pool_labels"])),
        "calib_rows": int(len(parts["calib_labels"])),
        "test_rows": int(len(y)),
    }


def compare_stored(parts: dict, pct: int) -> dict:
    old = np.load(STORED / file_name(pct), allow_pickle=False)
    old_parts = {key: old[key] for key in parts}
    old_s = summary(old_parts)
    new_s = summary(parts)
    return {
        "stored_gpu": old_s,
        "regenerated_cpu": new_s,
        "auroc_difference_cpu_minus_gpu": new_s["test_auroc"] - old_s["test_auroc"],
        "accuracy_difference_cpu_minus_gpu": new_s["test_accuracy"] - old_s["test_accuracy"],
        "bitwise_identical": bool(all(np.array_equal(parts[k], old_parts[k]) for k in parts)),
        "max_abs_difference": max(
            float(np.max(np.abs(parts[k] - old_parts[k])))
            for k in parts if np.issubdtype(parts[k].dtype, np.number)
        ),
    }


def run_guard() -> dict:
    sys.path.insert(0, str(B / "experiments"))
    from gates_core import gate_row

    rows = {}
    for pct in PCTS:
        z = np.load(OUT / file_name(pct), allow_pickle=False)
        full = np.load(OUT / "full.npz", allow_pickle=False)
        P = np.concatenate([z["pool_probs"], z["calib_probs"], z["test_probs"]]).astype(np.float64)
        F = np.concatenate([z["pool_feats"], z["calib_feats"], z["test_feats"]]).astype(np.float64)
        Y = np.concatenate([z["pool_labels"], z["calib_labels"], z["test_labels"]])
        R = np.concatenate([full["pool_probs"], full["calib_probs"], full["test_probs"]]).astype(np.float64)
        npool, ncal = len(z["pool_labels"]), len(z["calib_labels"])
        values = []
        for seed in range(3):
            perm = np.random.default_rng(seed).permutation(ncal) + npool
            split = (np.arange(npool), perm[:ncal // 2], perm[ncal // 2:],
                     np.arange(len(z["test_labels"])) + npool + ncal)
            r = gate_row(P, F, Y, split, targets=("hard", "cross"), richer=R)
            meta = r["_meta"]
            base = float(meta["base"])
            values.append({
                "seed": seed,
                "frozen_metric": base,
                "guard_without_certify_metric": base + float(r["blanket"][0]),
                "guard_without_certify_gain": float(r["blanket"][0]),
                "guard_without_certify_joint_harm": float(r["blanket"][1]),
                "guard_metric": base + float(r["GUARD"][0]),
                "guard_gain": float(r["GUARD"][0]),
                "joint_harm": float(r["GUARD"][1]),
                "apply_rate": float(meta["apply"]),
                "selected_target": meta["target"], "k": int(meta["k"]),
                "space": meta["space"], "weighting": meta["weighting"],
                "temperature": float(meta["temperature"]), "beta": float(meta["beta"]),
            })
        metric_keys = [k for k in values[0] if k not in {"seed", "selected_target", "space", "weighting"}]
        mean = {k: float(np.mean([v[k] for v in values])) for k in metric_keys}
        rows[str(pct)] = {"alpha": ALPHA, "delta": DELTA, "seeds": values,
                          "mean": mean,
                          "configuration": "gates_drugban.py: 3 seeds; unretuned grid; hard/cross target selection"}
        print(f"GUARD {pct:3d}% gain={mean['guard_gain']:+.8f} harm={mean['joint_harm']:.8f} apply={mean['apply_rate']:.8f}", flush=True)
    return {"cpu_generated": True, "alpha": ALPHA, "delta": DELTA,
            "fractions_percent": PCTS, "per_fraction": rows}


def main() -> None:
    targets = [OUT / file_name(p) for p in PCTS] + [OUT / "ladder.json", OUT / "guard_results.json"]
    exists = [str(p) for p in targets if p.exists()]
    if exists:
        raise FileExistsError("create-only runner refuses to overwrite: " + ", ".join(exists))
    OUT.mkdir(parents=True, exist_ok=True)
    host = load_host()
    model, _ = host._load_model(REPO, CFG, CKPT, "cpu")
    folder = REPO / "datasets/bindingdb/random"
    pool = pd.read_csv(folder / "train.csv")
    calib = pd.read_csv(folder / "val.csv")
    test = pd.read_csv(folder / "test.csv")
    pool = pool.iloc[np.random.default_rng(0).permutation(len(pool))[:12000]].reset_index(drop=True)
    ladder, overlap = {}, {}
    for pct in PCTS:
        parts = {}
        for name, df in (("pool", pool), ("calib", calib), ("test", test)):
            prob, feat, label = host._forward(model, degraded(df, pct, host), "cpu")
            parts[f"{name}_probs"], parts[f"{name}_feats"], parts[f"{name}_labels"] = prob, feat, label
        np.savez_compressed(OUT / file_name(pct), **parts)
        ladder[str(pct)] = summary(parts)
        if pct in (100, 50, 25):
            overlap[str(pct)] = compare_stored(parts, pct)
        print(f"CPU {pct:3d}% AUROC={ladder[str(pct)]['test_auroc']:.12f} accuracy={ladder[str(pct)]['test_accuracy']:.8f}", flush=True)
    (OUT / "ladder.json").write_text(json.dumps({
        "dataset_split": "bindingdb_s1 (BindingDB random split, seed 1)",
        "environment_note": "All eleven fractions were regenerated on CPU in one environment; no stored GPU dump was reused.",
        "checkpoint": str(CKPT), "fractions_percent": PCTS,
        "per_fraction": ladder, "stored_gpu_comparison": overlap,
    }, indent=2) + "\n")
    (OUT / "guard_results.json").write_text(json.dumps(run_guard(), indent=2) + "\n")


if __name__ == "__main__":
    main()
