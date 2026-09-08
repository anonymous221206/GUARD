#!/usr/bin/env python3
"""LaTeX rows for the DrugBAN table and for its line in the summary table.

Reads the per-cell CSVs written by ``experiments/exp_drugban.py`` and prints the
rows verbatim, so a rerun cannot leave the table and the results disagreeing.
The retrain-residual column comes from a separate retraining experiment and is
carried through as given; everything else is computed here.

The in-domain ``random`` rows retrieve from the source population, as the released
split defines it. The cross-domain ``cluster`` rows retrieve from the deployment
domain: with a source pool the neighbours come from the wrong distribution and the
gate declines almost everything, which is reported in the caption rather than
hidden by a choice of pool.
"""
import csv, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
POLICY = "selected"

# (label, dataset, split, seeds, condition, retrain residual as printed, pool)
ROWS = [
    ("BioSNAP",   "random$^{\\dagger}$",  "biosnap_random",     ("s1", "s2", "s42"), "prot25",          "$+0.007$", "source"),
    (None,        "random$^{\\ddagger}$", "biosnap_random",     ("s1", "s2", "s42"), "scaffold_prot50", "$+0.034$", "source"),
    (None,        "cluster$^{\\dagger}$", "biosnap_cluster",    ("s1", "s2", "s42"), "prot25",          "---",      "deployment"),
    ("BindingDB", "random$^{\\dagger}$",  "bindingdb_random",   ("s1", "s2", "s42"), "prot25",          "$+0.004$", "source"),
    (None,        "random$^{\\ddagger}$", "bindingdb_random",   ("s1", "s2", "s42"), "scaffold_prot50", "$+0.002$", "source"),
    (None,        "cluster$^{\\dagger}$", "bindingdb_cluster",  ("s42",),            "prot25",          "---",      "deployment"),
    (None,        "cluster$^{\\ddagger}$","bindingdb_cluster",  ("s42",),            "scaffold_prot50", "---",      "deployment"),
    ("Human",     "random$^{\\dagger}$",  "human_random",       ("s1", "s2", "s42"), "prot25",          "$-0.001$", "source"),
    (None,        "random$^{\\ddagger}$", "human_random",       ("s1", "s2", "s42"), "scaffold_prot50", "$+0.021$", "source"),
]
TARGET_POOL = {"human_random": "source", "bindingdb_random": "source",
               "biosnap_random": "source", "biosnap_cluster": "deployment",
               "bindingdb_cluster": "deployment"}


def cells(stem, seeds, cond, policy=POLICY, pool="source"):
    out = []
    for s in seeds:
        f = RES / f"drugban_{stem}_{s}_pool-{pool}" / "guard.csv"
        if not f.exists():
            print(f"thieu {f}", file=sys.stderr)
            continue
        for r in csv.DictReader(open(f)):
            if r["condition"] == cond and r.get("policy") == policy:
                out.append(r)
    return out


def agg(rs, key):
    return float(np.mean([float(r[key]) for r in rs]))


def main():
    print("% sinh boi scripts/tables/drugban_rows.py")
    summary = []
    for label, split, stem, seeds, cond, resid, pool in ROWS:
        rs = cells(stem, seeds, cond, pool=pool)
        if not rs:
            print(f"% THIEU {stem} {cond}")
            continue
        au = agg(rs, "base_metric")
        au_g = au + agg(rs, "gate_metric_delta")
        ac = agg(rs, "base_accuracy")
        ac_g = agg(rs, "gate_accuracy")
        h = agg(rs, "joint_harm")
        ap = 100 * agg(rs, "apply_rate")
        summary.append((au, au_g, h, ap / 100))
        b = lambda v, w: f"$\\mathbf{{{v:.3f}}}$" if w else f"${v:.3f}$"
        head = f"{label:<9} " if label else " " * 10
        print(f"{head}& {split:<21} & {b(au, False)} & {b(au_g, au_g > au)} & "
              f"{b(ac, False)} & {b(ac_g, ac_g > ac)} & ${h:.3f}$ & ${ap:.0f}$ & {resid} \\\\")
    if summary:
        au = np.mean([r[0] for r in summary]); ag = np.mean([r[1] for r in summary])
        hs = [r[2] for r in summary]; ra = [r[3] for r in summary]
        print(f"\n% dong Table 2: {len(summary)} dieu kien")
        print(f"DrugBAN ($3$ datasets)     & AUROC         & ${len(summary)}$ & ${au:.3f}$ & "
              f"$\\mathbf{{{ag:.3f}}}$ & ${min(hs):.3f}$--${max(hs):.3f}$ & "
              f"${100*min(ra):.0f}$--${100*max(ra):.0f}$ \\\\")

    # Table 12 needs the two targets side by side, in accuracy
    print("\n% Table 12: gain accuracy theo tung dich")
    for name, stem, seeds in (("Human, random", "human_random", ("s1", "s2", "s42")),
                              ("BindingDB, random", "bindingdb_random", ("s1", "s2", "s42")),
                              ("BioSNAP, random", "biosnap_random", ("s1", "s2", "s42")),
                              ("BioSNAP, cluster", "biosnap_cluster", ("s1", "s2", "s42")),
                              ("BindingDB, cluster", "bindingdb_cluster", ("s42",))):
        got = {}
        for pol in ("hard", "cross_mask"):
            rs = [r for c in ("prot25", "scaffold_prot50")
                  for r in cells(stem, seeds, c, pol, TARGET_POOL[stem])]
            if rs:
                got[pol] = agg(rs, "gate_accuracy") - agg(rs, "base_accuracy")
        rich = [r for c in ("full",)
                for r in cells(stem, seeds, c, "hard", TARGET_POOL[stem])]
        ra = agg(rich, "base_accuracy") if rich else float("nan")
        if len(got) == 2:
            print(f"{name:<18} & ${ra:.3f}$ & ${got['hard']:+.3f}$ & ${got['cross_mask']:+.3f}$ \\\\")


if __name__ == "__main__":
    main()
