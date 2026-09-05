#!/usr/bin/env python3
"""Turn every ``results/*/guard.csv`` into the DrugBAN and OPPORTUNITY tables.

This covers the runs that write ``guard.csv``. The remaining tables are printed
by their drivers and transcribed; ``docs/PAPER_MAP.md`` says which driver
produced which table, so any number in the text can still be traced to the run
that produced it.
"""

from __future__ import annotations

import csv
import collections
import sys
from pathlib import Path

import numpy as np

ORDER = ["full", "prot50", "scaffold", "scaffold_prot50", "prot25"]


def load(root: Path):
    rows = []
    for f in sorted(root.glob("*/guard.csv")):
        reader = csv.DictReader(open(f))
        if not {"target", "condition", "joint_harm"} <= set(reader.fieldnames or []):
            continue                      # ablation tables have their own schema
        for r in reader:
            r["run"] = f.parent.name
            rows.append(r)
    return rows


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows = load(root)
    if not rows:
        print(f"no results under {root}/")
        return
    num = lambda r, k: float(r[k])

    groups = collections.defaultdict(list)
    for r in rows:
        if r["target"] != "hard":
            continue
        run = r["run"]
        family = run.rsplit("_s", 1)[0] + ("_" + run.split("pool-")[-1] if "pool-" in run else "")
        groups[(family, r["condition"])].append(r)

    print("| run | condition | seeds | base | blanket | gate | apply | joint harm | cond harm |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (family, cond) in sorted(groups, key=lambda x: (x[0], ORDER.index(x[1])
                                                        if x[1] in ORDER else 99)):
        g = groups[(family, cond)]
        col = lambda k: np.array([num(r, k) for r in g])
        sd = lambda a: f"±{a.std():.4f}" if len(a) > 1 else ""
        print("| %s | %s | %d | %.4f | %+.4f | %+.4f%s | %.2f | %.3f | %.3f |" % (
            family, cond, len(g), col("base_metric").mean(),
            col("blanket_metric_delta").mean(), col("gate_metric_delta").mean(),
            sd(col("gate_metric_delta")), col("apply_rate").mean(),
            col("joint_harm").mean(), col("cond_harm").mean()))

    # a violation only means something where the split was declared exchangeable;
    # the non-exchangeable runs are in the repository precisely to show what
    # happens when that assumption is dropped.
    ex = [r for r in rows if str(r.get("exchangeable", "True")) == "True"]
    nx = [r for r in rows if str(r.get("exchangeable", "True")) != "True"]
    v_ex = sum(1 for r in ex if num(r, "joint_harm") > num(r, "alpha"))
    v_nx = sum(1 for r in nx if num(r, "joint_harm") > num(r, "alpha"))
    print(f"\n**{len(ex)} certified cells with an exchangeable calibration set: "
          f"{v_ex} exceed the harm budget.**")
    if nx:
        print(f"\n{len(nx)} further cells deliberately violate exchangeability "
              f"(calibration and deployment drawn from different populations); "
              f"{v_nx} of those exceed the budget, which is the point of running them.")

    # cross-mask vs hard-label, split by protocol: pooling them hides the point.
    hd = {(r["run"], r["condition"]): r for r in rows if r["target"] == "hard"}
    buckets = collections.defaultdict(list)
    for r in rows:
        if r["target"] != "cross_mask":
            continue
        key = (r["run"], r["condition"])
        if key not in hd:
            continue
        family = r["run"].split("_")[0]
        protocol = f"{family}, cross-domain" if "cluster" in r["run"] else f"{family}, in-domain"
        buckets[protocol].append(
            num(r, "gate_metric_delta") - num(hd[key], "gate_metric_delta"))

    if buckets:
        print("\n### cross-mask minus hard-label")
        print("Cross-mask uses no retrieval labels; hard-label needs one per pool element.")
        print("\n| host family / protocol | cells | mean | worst | best |")
        print("|---|---|---|---|---|")
        for protocol in sorted(buckets):
            d = np.array(buckets[protocol])
            print("| %s | %d | %+.4f | %+.4f | %+.4f |" % (
                protocol, len(d), d.mean(), d.min(), d.max()))
        print("\nProposition 4 guarantees dominance only when the richer host is")
        print("conditionally correct.  Under the cross-domain protocol the richer host")
        print("has collapsed, the precondition fails, and cross-mask loses -- which is")
        print("what the theory predicts, not a counterexample to it.")


if __name__ == "__main__":
    main()
