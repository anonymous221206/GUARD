#!/usr/bin/env python3
"""Every number quoted in the budget-sweep subsection, recomputed from the sweep.

The subsection makes counting claims about the whole sweep -- how many runs, how
many exceed the budget, where the gain peaks. Those are exactly the claims that
rot silently when a driver is rerun, so they are printed here rather than counted
by hand.
"""
import csv, collections, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "results/alpha"
FILES = ["alpha_affective.csv", "alpha_ave.csv", "alpha_drugban.csv", "alpha_new.csv",
         "alpha_ninapro.csv", "alpha_opportunity.csv", "alpha_opp_nx.csv",
         "alpha_ptbxl_shift.csv"]

rows = []
for f in FILES:
    p = A / f
    if not p.exists():
        print(f"thieu {p}", file=sys.stderr); continue
    for r in csv.DictReader(open(p)):
        r["_file"] = f
        rows.append(r)

ex = lambda r: str(r.get("exchangeable")) in ("True", "true", "1")
fl = lambda r, k: float(r[k])
keep = [r for r in rows if ex(r)]
print(f"tong so run: {len(rows)}  (exchangeable {len(keep)}, khong {len(rows)-len(keep)})")

for a in (0.05, 0.2):
    at = [r for r in keep if abs(fl(r, "alpha") - a) < 1e-9]
    over = [r for r in at if fl(r, "joint_harm") > a]
    bover = [r for r in at if fl(r, "blanket_joint_harm") > a]
    print(f"alpha={a}: {len(at)} run, GUARD vuot {len(over)}, blanket vuot {len(bover)}")

bl = [fl(r, "blanket_joint_harm") for r in keep]
print(f"blanket joint harm: trung binh {np.mean(bl):.3f}, "
      f"khoang {min(bl):.3f}-{max(bl):.3f} (khong doi theo alpha)")

# per-condition means at alpha=0.2, the claim that none of them exceeds
g = collections.defaultdict(list)
for r in keep:
    if abs(fl(r, "alpha") - 0.2) < 1e-9:
        g[(r["family"], r["dataset"], r["condition"])].append(fl(r, "joint_harm"))
worst = max(g.items(), key=lambda kv: np.mean(kv[1]))
print(f"trung binh theo dieu kien cao nhat tai alpha=0.2: {np.mean(worst[1]):.3f} {worst[0]}")

# where the gain peaks, per family
print("dinh cua gated gain theo alpha:")
byfam = collections.defaultdict(lambda: collections.defaultdict(list))
for r in keep:
    byfam[r["family"]][fl(r, "alpha")].append(fl(r, "acc_gain"))
for fam in sorted(byfam):
    m = {a: float(np.mean(v)) for a, v in byfam[fam].items()}
    top = max(m, key=m.get)
    print(f"  {fam:14} dinh o alpha={top:.2f} (gain {m[top]:+.4f})")

# the two stress tests: same benchmark, exchangeable against not
print("hai phep thu pha vo tinh trao doi duoc (tai alpha=0.2):")
for name, keepf, brokef in (("OPPORTUNITY, doi chu the", "alpha_opportunity.csv", "alpha_opp_nx.csv"),
                            ("PTB-XL, doi muc suy giam", "alpha_new.csv", "alpha_ptbxl_shift.csv")):
    for tag, f in (("giu duoc", keepf), ("pha vo", brokef)):
        sel = [r for r in rows if r["_file"] == f and abs(fl(r, "alpha") - 0.2) < 1e-9]
        if not sel:
            print(f"  {name:26} {tag:9} (khong co du lieu)"); continue
        print(f"  {name:26} {tag:9} apply {np.mean([fl(r,'apply_rate') for r in sel]):.3f}  "
              f"joint harm {np.mean([fl(r,'joint_harm') for r in sel]):.3f}  n={len(sel)}")
