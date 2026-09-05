#!/usr/bin/env python3
"""UCI OPPORTUNITY -> windowed per-sensor features.

Run this once after ``data/download_opportunity.sh``; it turns the released
``.dat`` files into the per-sensor feature blocks the experiment expects.

Modalities (K=19, each a physical body-worn sensor):
  * 12 tri-axial accelerometers (cols 2-37, groups of 3)      cost 1.0
  * 5 IMUs BACK/RUA/RLA/LUA/LLA (9 ch: acc+gyro+mag;
    quaternions dropped)                                       cost 3.0
  * 2 instrumented shoes L/R (16 ch)                           cost 2.0
Object/ambient sensors (cols 135-243) are excluded (heavy missingness, not
body-worn).  Labels: Locomotion (col 244), HL_Activity (245),
ML_Both_Arms (250).  Subjects S1-S4, runs ADL1-5 + Drill.

Windows: 30 samples (1 s @30 Hz), step 15; features = per-channel mean+std;
NaNs linearly interpolated per run (sensors drop out briefly), windows with
any remaining NaN in a modality are marked invalid for it (we keep only
windows valid for ALL 19 body-worn modalities — complete-data requirement
of the acquisition protocol).

Output: results/phase0f/opportunity_features.npz
  feat_00..feat_18 (n, 2*ch), subject (n,), run (n,), y_locomotion,
  y_hl, y_ml (n,), names, costs
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ACC_NAMES = ["RKN_up", "HIP", "LUA_up", "RUA_lo", "LH", "BACK_acc",
             "RKN_lo", "RWR", "RUA_up", "LUA_lo", "LWR", "RH"]
MODS = []
for i, nm in enumerate(ACC_NAMES):                       # 1-based cols
    MODS.append((f"acc_{nm}", list(range(2 + 3 * i, 5 + 3 * i)), 1.0))
for j, nm in enumerate(["BACK", "RUA", "RLA", "LUA", "LLA"]):
    st = 38 + 13 * j
    MODS.append((f"imu_{nm}", list(range(st, st + 9)), 3.0))
MODS.append(("shoe_L", list(range(103, 119)), 2.0))
MODS.append(("shoe_R", list(range(119, 135)), 2.0))
LABELS = {"locomotion": 244, "hl": 245, "ml": 250}
WIN, STEP = 30, 15


def window_file(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, na_values="NaN",
                     engine="c")
    arr = df.to_numpy(float)
    outs = {k: [] for k, _, _ in MODS}
    labs = {t: [] for t in LABELS}
    ok_rows = []
    # interpolate short sensor dropouts per column
    for _, cols, _ in MODS:
        idx = [c - 1 for c in cols]
        sub = pd.DataFrame(arr[:, idx]).interpolate(
            limit=90, limit_direction="both").to_numpy()
        arr[:, idx] = sub
    n = arr.shape[0]
    for s in range(0, n - WIN + 1, STEP):
        w = arr[s:s + WIN]
        feats, valid = {}, True
        for k, cols, _ in MODS:
            x = w[:, [c - 1 for c in cols]]
            if np.isnan(x).any():
                valid = False
                break
            feats[k] = np.concatenate([x.mean(0), x.std(0)])
        if not valid:
            continue
        ok_rows.append(True)
        for k in feats:
            outs[k].append(feats[k])
        for t, c in LABELS.items():
            v = w[:, c - 1]
            v = v[~np.isnan(v)]
            labs[t].append(int(pd.Series(v).mode().iloc[0]) if len(v)
                           else 0)
    return outs, labs, len(ok_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path,
                    default=Path("data/raw/OpportunityUCIDataset/dataset"),
                    help="directory holding S1-ADL1.dat ...")
    ap.add_argument("--out", type=Path,
                    default=Path("data/processed/opportunity_features.npz"))
    a = ap.parse_args()
    DATA, OUT = a.source, a.out
    if not DATA.is_dir():
        raise SystemExit(f"{DATA} not found -- run data/download_opportunity.sh first")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_f = {k: [] for k, _, _ in MODS}
    all_l = {t: [] for t in LABELS}
    subj, run = [], []
    for s in (1, 2, 3, 4):
        for r in ["ADL1", "ADL2", "ADL3", "ADL4", "ADL5", "Drill"]:
            p = DATA / f"S{s}-{r}.dat"
            outs, labs, nw = window_file(p)
            for k in all_f:
                all_f[k].extend(outs[k])
            for t in all_l:
                all_l[t].extend(labs[t])
            subj.extend([s] * nw)
            run.extend([r] * nw)
            print(f"S{s}-{r}: {nw} windows", flush=True)
    save = {f"feat_{i:02d}": np.asarray(all_f[k], np.float64)
            for i, (k, _, _) in enumerate(MODS)}
    save |= {f"y_{t}": np.asarray(all_l[t]) for t in LABELS}
    save["subject"] = np.asarray(subj)
    save["run"] = np.asarray(run)
    save["names"] = np.asarray([k for k, _, _ in MODS])
    save["costs"] = np.asarray([c for _, _, c in MODS])
    np.savez(OUT, **save)
    n = len(subj)
    print(f"{OUT}: {n} windows, K={len(MODS)} sensor blocks")
    for t in LABELS:
        vals, cnts = np.unique(save[f"y_{t}"], return_counts=True)
        print(t, dict(zip(vals.tolist(), cnts.tolist())))


if __name__ == "__main__":
    main()
