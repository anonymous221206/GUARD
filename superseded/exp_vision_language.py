#!/usr/bin/env python3
"""Vision--language hosts: ViLT with missing-aware prompts (Lee et al., CVPR 2023).

Reads the per-condition dumps produced by ``hosts/vilt_prompts.py`` and runs
GUARD on them.  No GPU and no model code needed here -- the host is frozen and
we only touch its outputs, which is the whole point of the method.

    python experiments/exp_vision_language.py --dataset hateful_memes

Outputs ``results/<dataset>/guard.csv`` and prints a table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guard import HostOutputs, run                       # noqa: E402
from guard.splits import Split                           # noqa: E402

CONDITIONS = ("complete", "textmiss", "imgmiss")
TARGETS = ("hard", "cross_mask")


def load(dump_dir: Path, condition: str):
    z = np.load(dump_dir / f"extract_{condition}.npz")
    return {k: z[k] for k in z.files}


def build_split(n_pool: int, n_val: int, n_test: int, seed: int) -> Split:
    """Roles in the concatenated index space ``[pool | val | test]``.

    The retrieval pool is the training split; ``fit`` and ``conf`` are a random
    halving of the validation split, and ``test`` is the official test split.
    Validation and test are iid draws of one population for these datasets, so
    calibration and evaluation are exchangeable in the sense the gate needs.
    """
    perm = np.random.default_rng(seed).permutation(n_val) + n_pool
    half = n_val // 2
    return Split(
        pool=np.arange(n_pool),
        fit=perm[:half],
        conf=perm[half:],
        test=np.arange(n_test) + n_pool + n_val,
        origin={"pool": "train split", "fit": "val/test population",
                "conf": "val/test population", "test": "val/test population"},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["food101", "hateful_memes"])
    ap.add_argument("--dumps", type=Path, required=True,
                    help="directory holding extract_<condition>.npz")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out_dir = a.out / a.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    richer = load(a.dumps, "complete")            # the all-modalities mask
    for cond in CONDITIONS:
        d = load(a.dumps, cond)
        n_val, n_test, n_pool = len(d["val_labels"]), len(d["test_labels"]), len(d["src_labels"])
        split = build_split(n_pool, n_val, n_test, a.seed)

        # one index space the pipeline can address: [pool | val | test]
        probs = np.concatenate([d["src_probs"], d["val_probs"], d["test_probs"]]).astype(np.float64)
        feats = np.concatenate([d["src_feats"], d["val_feats"], d["test_feats"]])
        labels = np.concatenate([d["src_labels"], d["val_labels"], d["test_labels"]])
        rich = np.concatenate([richer["src_probs"], richer["val_probs"],
                               richer["test_probs"]]).astype(np.float64)
        host = HostOutputs(probs=probs, features=feats, labels=labels, richer_probs=rich)
        for target in TARGETS:
            r = run(host, split, condition=cond, target=target,
                    alpha=a.alpha, delta=a.delta, k=a.k)
            rows.append(r.as_row())
            print(f"{cond:10s} {target:10s} base={r.base_metric:.4f} "
                  f"target={r.target_accuracy:.4f} beta={r.beta:.2f} "
                  f"gate={r.gate_metric_delta:+.4f} harm={r.joint_harm:.3f} "
                  f"apply={r.apply_rate:.2f}")
            for note in r.notes:
                print(f"           note: {note}")

    with open(out_dir / "guard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "notes"])
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != "notes"})
    (out_dir / "manifest.json").write_text(json.dumps(
        {"dataset": a.dataset, "alpha": a.alpha, "delta": a.delta,
         "k": a.k, "seed": a.seed, "n_cells": len(rows)}, indent=2))
    viol = sum(1 for r in rows if r["joint_harm"] > a.alpha)
    print(f"\n{viol}/{len(rows)} cells exceed the harm budget")


if __name__ == "__main__":
    main()
