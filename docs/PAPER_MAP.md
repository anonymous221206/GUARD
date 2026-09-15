# Which script produced which number

Every table and figure in the paper is listed here with the driver that produced
it and the artefacts that driver reads. Nothing reported in the paper comes from
a script outside this repository. Figures 1 and 2 are drawn by hand in draw.io
and are the only exception; their source is in the paper directory, not here.

Before any of it runs:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,figures]"
bash data/download_artifacts.sh          # ~2.9 GB into ./artifacts
```

Drivers read `./artifacts`; set `GUARD_ARTIFACTS` to point elsewhere. Outputs
land under `results/`.

## Main text

| # | Content | Driver | Reads |
|---|---|---|---|
| Table 1 | CMU-MOSEI against published methods | `experiments/mosei_hosts.py`, then `scripts/tables/mosei_rows.py` | `mosei_cmad`, `mosei_tmdc`, `mosei_momke` |
| Table 2 | All benchmarks, summary | the per-benchmark drivers of Tables 9-11 | all dumps |
| Table 3 | Intervention rules compared | `experiments/ltt_one.py <driver>` | all dumps |
| Figure 1 | Positioning against prior repair | `figures_drawio/guard_figures_v2.drawio`, page 1 | hand-drawn |
| Figure 2 | GUARD at deployment | `figures_drawio/guard_figures_v2.drawio`, page 2 | hand-drawn |
| Figure 3 | Synthetic sweep, gain against headroom | `experiments/exp_synthetic_sweep.py`, then `scripts/figures/make_fig_synthetic.py` | `results/synthetic_sweep/sweep.csv` |
| Figure 4 | Severity ladders | six ladder drivers (below), then `scripts/figures/make_fig_severity.py` | six dumps |

`ltt_one.py` wraps one gate driver and records every (condition, seed) cell to
`results/gates/cells_<driver>.json`:

```bash
for d in gates_mosei2 gates_iemocap gates_rest gates_drugban opp_dcl; do
  python experiments/ltt_one.py $d
done
```

Table 3 is the mean over those cells and Table 14 their range. They read the same
files, so they cannot disagree.

The six panels of Figure 4 come from `experiments/ptbxl_sev_dense.py`,
`experiments/nina_sev_dense.py`, `experiments/opp_dcl.py`,
`scripts/drugban_protladder_guard.py`, `experiments/iemocap_eta.py` and
`experiments/ave_eta.py`.

DrugBAN is published under AUROC, but the shared `gate_row` scores every
benchmark with the accuracy helper, so the ladder file it writes holds accuracy.
`scripts/drugban_auroc_ladder.py` repeats the same corrector selection on the
same dumps and records AUROC instead, writing `guard_results_auroc.json` beside
it; `make_fig_severity.py` prefers that file and labels the axis accordingly.
Run it after `drugban_protladder_guard.py`.

Two copies of `drugban_protladder_v2/guard_results.json` exist, under
`artifacts/` and under `results/`, and they disagree on the harm columns. The
one under `artifacts/` is the one the drivers write and the figure reads; the
other is stale.

## Appendix

| # | Content | Driver | Reads |
|---|---|---|---|
| Table 5 | Benchmarks, hosts, degradations | descriptive; hosts listed in `docs/REPRODUCTION.md` | none |
| Table 6 | Sample counts per role | `experiments/counts_splits.py` | `ave_av_att`, `ninapro_cnn`, `iemocap_momke` |
| Table 7 | Synthetic study, matched raw error | `experiments/exp_synthetic_matched.py` | none, generated in-process |
| Table 8 | IEMOCAP per missing rate | `experiments/iemocap_eta.py` | `iemocap_momke/folds` |
| Table 9 | DrugBAN per condition | `experiments/exp_drugban.py`, then `scripts/tables/drugban_rows.py` | `drugban_processed` |
| Table 10 | OPPORTUNITY under sensor loss | `experiments/opp_dcl.py` | `opportunity_dcl_v2` |
| Table 11 | Per-condition results | `experiments/mosei_full.py`, `experiments/gates_iemocap.py`, `experiments/gates_ave.py`, `experiments/nina_sev_dense.py`, `experiments/ptbxl_sev_dense.py` | five dumps |
| Table 12 | Cross-mask target | `scripts/tables/drugban_rows.py`, `experiments/opp_targets.py` | `drugban_processed`, `opportunity_dcl_v2` |
| Table 13 | CMU-MOSEI harm and apply rate | `experiments/mosei_full.py` | `mosei_cmad` |
| Table 14 | Removing the certificate | same cells as Table 3 | all dumps |
| Figure 6 | Cross-mask accuracy screen | `scripts/figures/make_fig_screen.py` | `scripts/figures/ablation_data/` |
| Figure 7 | Intervention rate | `scripts/figures/make_fig_apply.py` | `scripts/figures/ablation_data/` |

`scripts/tables/alpha_stats.py` recomputes every counting claim in the
budget-sweep subsection (how many runs, how many over budget, where the gain
peaks) from the same CSVs Figure 5 reads.

## The sweep behind Figure 5

| Curve | Driver | Output |
|---|---|---|
| CMU-MOSEI | `experiments/affective_alpha.py` | `results/alpha/alpha_affective.csv` |
| AVE | `experiments/ave_alpha.py` | `results/alpha/alpha_ave.csv` |
| DrugBAN | `experiments/drugban_alpha.py` | `results/alpha/alpha_drugban.csv` |
| NinaPro | `experiments/nina_alpha.py` | `results/alpha/alpha_ninapro.csv` |
| OPPORTUNITY | `experiments/opp_alpha.py` | `results/alpha/alpha_opportunity.csv` |
| OPPORTUNITY, non-exchangeable | `experiments/opp_alpha_nx.py` | `results/alpha/alpha_opp_nx.csv` |
| PTB-XL | `experiments/alpha_sweep_new.py` | `results/alpha/alpha_new.csv` |
| PTB-XL, shifted calibration | `experiments/ptbxl_shift.py` | `results/alpha/alpha_ptbxl_shift.csv` |

## Offline preparation

Two artefacts are produced rather than downloaded, because they need a
checkpoint and the original dataset:

```bash
# the AVE retrieval sidecars, one per condition including the intact one
python hosts/ave.py export --host artifacts/ave_av_att --source <AVE_ECCV18> \
       --conditions full --variant paper --cuda-device 0
# the DrugBAN protein ladder: eleven frozen dumps, then the gate over them
python scripts/drugban_protladder_v2.py          # needs dgl, dgllife
python scripts/drugban_protladder_guard.py       # numpy only
```

The AVE exporter verifies itself: it refuses to write a sidecar whose posteriors
do not reproduce the stored dump. The stored dumps were made on a GPU, so an
exact reproduction needs one; on CPU the check fails at around `5e-3`.

## Numbers quoted in the running text

| Claim | Driver |
|---|---|
| the blend weight by mask in §5.1 ($0.70$ against $0.12$) | `experiments/beta_by_mask.py` |
| what the fit-split tightening changes (§4.2) | `experiments/tighten_effect.py` |
| the counting claims of the budget sweep (§5.4) | `scripts/tables/alpha_stats.py` |

`scripts/rerun_all_results.sh` runs every driver behind a reported number in one
pass; `scripts/tables/verify_paper.py` then checks the LaTeX against what they
wrote, and `scripts/tables/rebuild_tables.py` prints the replacements for any
cell that moved.

## Checked automatically

```bash
pytest -q                          # ten tests on the certified decision rule
python experiments/repro_check.py  # frozen-host numbers against the paper
python experiments/audit_preflight.py
```

`experiments/exp_synthetic.py` is a second, smaller synthetic check that shares
no number with Table 7. Its skew preserves the arg-max, so it can move the loss
but never the accuracy; it runs in seconds and doubles as an install test.

## Honest limits of this map

Three tables now have generators (`scripts/tables/`); the rest of the gate
drivers print their numbers to standard output and those were transcribed into
the LaTeX source. Re-running a driver reproduces a number, it does not rewrite
the paper.

Code for the previous gate and for the experiments that were dropped is not in
this tree. Nothing in it produced a number in the paper; it is reachable in the
history if anyone wants to see what was replaced.
