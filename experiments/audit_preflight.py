#!/usr/bin/env python3
"""Tinh lai phep kiem richer_is_richer cho MOI dieu kien bi che cua DrugBAN.

Khong train gi, chi doc dumps/*.npz. Muc dich: xac dinh phep kiem dang duoc do
tren split nao, va sinh mot bang nhat quan thay cho cac dong pre-flight cu
(von chay o dieu kien "full" noi richer va poorer la cung mot model).

Ghi ra CSV o duong dan truyen bang --out. KHONG ghi de gi khac.
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np

CONDITIONS = ("prot50", "prot25", "scaffold", "scaffold_prot50")   # bo "full": degenerate

def acc(p, y):
    """Giong richer_is_richer: simplex thi argmax, mot cot thi nguong 0.5."""
    p = np.asarray(p)
    if p.ndim == 2 and p.shape[1] > 1:
        return float((p.argmax(1) == np.asarray(y)).mean())
    return float(((p.ravel() > 0.5).astype(int) == np.asarray(y, int)).mean())

def build_split(d, seed, pool_from):
    n_pool, n_cal, n_test = (len(d["pool_labels"]), len(d["calib_labels"]),
                             len(d["test_labels"]))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_cal) + n_pool
    if pool_from == "source":
        return dict(fit=perm[: n_cal // 2], conf=perm[n_cal // 2:],
                    test=np.arange(n_test) + n_pool + n_cal)
    t = n_cal // 3
    return dict(fit=perm[t:2 * t], conf=perm[2 * t:3 * t],
                test=np.arange(n_test) + n_pool + n_cal)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="thu muc data/processed")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        sys.exit("LOI: %s da ton tai, khong ghi de" % a.out)

    rows = []
    for dumps in sorted(p for p in a.root.iterdir() if p.is_dir() and p.name.startswith("drugban_")):
        name = dumps.name                       # drugban_<ds>_<split>_s<seed>
        parts = name.split("_")
        ds, protocol, seed = parts[1], parts[2], int(parts[3][1:])
        for pool_from in ("source", "deployment"):
            rich = np.load(dumps / "full.npz")
            richer = np.concatenate([rich["pool_probs"], rich["calib_probs"], rich["test_probs"]])
            for cond in CONDITIONS:
                f = dumps / f"{cond}.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                probs = np.concatenate([d["pool_probs"], d["calib_probs"], d["test_probs"]])
                labels = np.concatenate([d["pool_labels"], d["calib_labels"], d["test_labels"]])
                sp = build_split(d, seed, pool_from)
                r = dict(dataset=ds, protocol=protocol, pool=pool_from, seed=seed, condition=cond)
                for s in ("fit", "conf", "test"):
                    i = sp[s]
                    ra, pa = acc(richer[i], labels[i]), acc(probs[i], labels[i])
                    r["richer_%s" % s] = round(ra, 6)
                    r["poorer_%s" % s] = round(pa, 6)
                    r["met_%s" % s] = ra > pa
                rows.append(r)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("da ghi %d hang -> %s" % (len(rows), a.out))
    for s in ("fit", "conf", "test"):
        n = sum(1 for r in rows if r["met_%s" % s])
        print("  split %-5s: %d/%d hang co richer > poorer" % (s, n, len(rows)))

if __name__ == "__main__":
    main()
