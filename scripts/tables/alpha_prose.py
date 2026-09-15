"""Recompute every number the budget subsection states, from results/alpha/.

The prose of that subsection is the one place in the paper that quotes figures
no table carries, so they are regenerated here rather than trusted.
"""
import csv, glob, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rows = []
for f in sorted(glob.glob(str(ROOT / "results/alpha/alpha_*.csv"))):
    rows += [r for r in csv.DictReader(open(f))
             if r["exchangeable"].strip().lower() in ("true", "1", "yes")]

by = {}
for r in rows:
    by.setdefault(float(r["alpha"]), []).append(r)

m = lambda v, k: float(np.mean([float(x[k]) for x in v]))
print(f"{'alpha':>6}{'n':>6}{'harm':>9}{'gain':>10}{'apply':>8}{'over %':>8}")
for a in sorted(by):
    v = by[a]
    over = 100 * np.mean([float(x["joint_harm"]) > a for x in v])
    print(f"{a:6.2f}{len(v):6d}{m(v,'joint_harm'):9.3f}"
          f"{m(v,'acc_gain'):+10.4f}{m(v,'apply_rate'):8.3f}{over:8.1f}")

lo, hi = min(by), max(by)
bh = m(rows, "blanket_joint_harm")
bg = m(rows, "blanket_acc_gain")
print(f"\nruns per budget: {len(by[lo])}, total {len(rows)}")
print(f"tightening {hi} -> {lo}: harm {m(by[hi],'joint_harm'):.3f} -> {m(by[lo],'joint_harm'):.3f}, "
      f"gain {m(by[hi],'acc_gain'):+.3f} -> {m(by[lo],'acc_gain'):+.3f} "
      f"({100*(1-m(by[lo],'acc_gain')/m(by[hi],'acc_gain')):.0f}% of the gain given up)")
print(f"saturation: gain {m(by[0.2],'acc_gain'):+.4f} at 0.2 vs {m(by[hi],'acc_gain'):+.4f} at {hi}")
print(f"at 0.2 against blanket: gain {100*m(by[0.2],'acc_gain')/bg:.0f}% , "
      f"harm {100*m(by[0.2],'joint_harm')/bh:.0f}%  (blanket gain {bg:+.4f}, harm {bh:.3f})")
for a in (lo, 0.2):
    print(f"runs over budget at alpha={a}: "
          f"{100*np.mean([float(x['joint_harm'])>a for x in by[a]]):.1f}%")

# Retention is a ratio per benchmark, then averaged. Averaging the raw gains
# instead pools accuracy, weighted F1 and AUROC into one mean and lets the
# benchmark with the largest gain set the answer, which is a different number.
fam = {}
for r in by[0.2]:
    fam.setdefault(r["family"], []).append(r)
print("\nretention of blanket gain at alpha=0.2, per benchmark:")
ratios = []
for f in sorted(fam):
    v = fam[f]
    g, b = m(v, "acc_gain"), m(v, "blanket_acc_gain")
    ratios.append(g / b)
    print(f"  {f:14} {g:+.4f} / {b:+.4f} = {100*g/b:6.1f}%")
print(f"  mean of the per-benchmark ratios: {100*np.mean(ratios):.1f}%")
print(f"  ratio of the pooled gains:        {100*m(by[0.2],'acc_gain')/m(by[0.2],'blanket_acc_gain'):.1f}%"
      "   <- not what the paper reports")
