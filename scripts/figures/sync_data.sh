#!/usr/bin/env bash
# Refresh the figure inputs from the sweep results, then redraw.
set -euo pipefail
cd "$(dirname "$0")"
r=../../results/alpha; d=ablation_data/ALPHA_NEW
cp -f "$r/alpha_affective.csv" "$d/affective.csv"
cp -f "$r/alpha_ave.csv" "$d/ave.csv"
cp -f "$r/alpha_drugban.csv" "$d/drugban.csv"
cp -f "$r/alpha_ninapro.csv" "$d/ninapro.csv"
cp -f "$r/alpha_opportunity.csv" "$d/opportunity.csv"
cp -f "$r/alpha_opp_nx.csv" "$d/opportunity_nx.csv"
cp -f "$r/alpha_new.csv" "$d/ptbxl.csv"
cp -f "$r/alpha_ptbxl_shift.csv" "$d/ptbxl_shift.csv"
for f in make_fig_alpha2.py make_fig_apply.py make_fig_screen.py; do
  "${PYTHON:-python3}" "$f"
done
