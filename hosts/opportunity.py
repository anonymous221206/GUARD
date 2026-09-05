#!/usr/bin/env python3
"""Turn the OPPORTUNITY release into the single npz the experiment expects.

Writes ``H`` (samples x features), ``y``, ``subject``, ``block_slices`` (one
slice per sensor block) and ``configs`` (the deployment sensor subsets).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="npz of per-block features produced by the OPPORTUNITY preprocessing")
    ap.add_argument("--configs", type=Path, required=True,
                    help="json listing the deployment sensor subsets")
    ap.add_argument("--out", type=Path, default=Path("data/processed/opportunity.npz"))
    ap.add_argument("--pca-dim", type=int, default=6,
                    help="components kept per sensor block (matches the host setup)")
    a = ap.parse_args()

    z = np.load(a.source, allow_pickle=False)
    n_blocks = sum(1 for k in z.files if k.startswith("feat_"))
    blocks, cols, start = [], [], 0
    train = np.isin(z["subject"], [1, 2])
    for i in range(n_blocks):
        f = z[f"feat_{i:02d}"].astype(np.float64)
        mu, sd = f[train].mean(0, keepdims=True), f[train].std(0, keepdims=True) + 1e-6
        fs = (f - mu) / sd
        # each block is compressed to a fixed number of components, fitted on the
        # training subjects only; this keeps the blocks comparable in width so a
        # sensor subset is a clean coordinate subset.
        centre = fs[train].mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(fs[train] - centre, full_matrices=False)
        cols.append((fs - centre) @ vt[: a.pca_dim].T)
        blocks.append(slice(start, start + a.pca_dim))
        start += a.pca_dim
    H = np.concatenate(cols, 1)

    classes = np.array([0, 1, 2, 4, 5])          # the locomotion labels in use
    y = np.searchsorted(classes, z["y_locomotion"])
    cfg = {c["name"]: c["observed_idx"]
           for c in json.loads(a.configs.read_text())["fixed_deployment_configs"]}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, H=H, y=y, subject=z["subject"],
                        block_slices=np.array(blocks, dtype=object),
                        configs=json.dumps(cfg))
    print(f"{a.out}: H={H.shape}, {n_blocks} blocks, {len(classes)} classes, "
          f"subjects {sorted(set(z['subject'].tolist()))}")


if __name__ == "__main__":
    main()
