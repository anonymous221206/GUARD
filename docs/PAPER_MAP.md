# Which script produced which number

Every table and figure in the paper is listed here with the driver that produced
it and the artefacts that driver reads. Nothing reported in the paper comes from
a script outside this repository.

Before any of it runs:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
bash data/download_artifacts.sh          # ~2.5 GB into ./artifacts
```

Drivers read `./artifacts`; set `GUARD_ARTIFACTS` to point elsewhere. Outputs
land under `results/`.

## Main text

| # | Content | Driver | Artefacts |
|---|---|---|---|
| Table 1 | CMU-MOSEI, frozen vs certified | `experiments/mosei_full.py` | `mosei_cmad/dumps` |
| Table 2 | All benchmarks, summary | the per-benchmark drivers below | all dumps |
| Table 3 | Intervention rules compared | `experiments/gates_ranges.py` | all dumps |
| Figure 3 | IEMOCAP | `experiments/gates_iemocap.py` | `iemocap_momke/folds` |
| Figure 4 | Severity ladders | `experiments/nina_sev_dense.py`, `experiments/ptbxl_sev_dense.py` | `ninapro_cnn`, `ptbxl_dropladder` |
| Figure 5 | Budget sweep | the eight sweep drivers, then `scripts/figures/make_fig_alpha2.py` | see below |

`gates_ranges.py` runs the five per-benchmark gate drivers under one wrapper and
writes `results/gates/gates_cells.json`, one row per (benchmark, condition, seed).
Table 3 is the mean over those rows, Table 14 their range. They read the same
file, so they cannot disagree.

## Appendix

| # | Content | Driver | Artefacts |
|---|---|---|---|
| Table 5 | Benchmarks, hosts, degradations | descriptive; hosts listed in `docs/REPRODUCTION.md` | none |
| Table 6 | Sample counts per role | `experiments/counts_splits.py` | `ave_av_att`, `ninapro_cnn`, `iemocap_momke` |
| Table 7 | Synthetic study | `experiments/exp_synthetic.py` | none, generated in-process |
| Table 8 | IEMOCAP per pattern | `experiments/gates_iemocap.py` | `iemocap_momke/folds` |
| Table 9 | DrugBAN | `experiments/exp_drugban.py` | `drugban_processed` |
| Table 10 | Gate ablation | `experiments/exp_ablations.py` | `drugban_processed` |
| Table 11 | Per-condition results | `experiments/mosei_full.py`, `experiments/opp_full.py`, `experiments/nina_ladder_dense.py`, `experiments/ptbxl_sev_dense.py` | four dumps |
| Table 12 | Cross-mask target | `experiments/gates_rest.py`, `experiments/gates_drugban.py` | `ave_av_att`, `ninapro_cnn`, `drugban_processed` |
| Table 13 | CMU-MOSEI harm and apply rate | `experiments/mosei_full.py` | `mosei_cmad/dumps` |
| Table 14 | Removing the certificate | `experiments/gates_ranges.py` | same file as Table 3 |
| Figure 6 | Cross-mask accuracy screen | `scripts/figures/make_fig_screen.py` | `scripts/figures/ablation_data/` |
| Figure 7 | Intervention rate | `scripts/figures/make_fig_apply.py` | `scripts/figures/ablation_data/` |
| §D.1 | Calibration resampling spread | `experiments/fs_mosei.py` | `mosei_cmad/dumps` |

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

## Checked automatically

```bash
pytest -q                          # ten tests on the certified decision rule
python experiments/repro_check.py  # frozen-host numbers against the paper
```

## Honest limits of this map

The gate drivers print their tables to standard output and those numbers were
transcribed into the LaTeX source; they are not written by a table generator.
`scripts/build_tables.py` generates only the DrugBAN and OPPORTUNITY tables from
`results/*/guard.csv`. Re-running a driver reproduces a number, it does not
rewrite the paper.

`experiments/exp_modules.py` and `experiments/exp_ablations.py` still accept a
`vilt` option. The paper reports no vision-language experiment; those code paths
are inert without dumps that this repository does not ship. The host itself is in
`superseded/`, which produces nothing in the paper.

## Figures

`scripts/figures/` holds the three generators and, under `ablation_data/`, the
exact CSVs they read, so every figure redraws from a clean clone:

```bash
cd scripts/figures && python3 make_fig_alpha2.py   # Figure 5
python3 make_fig_apply.py                          # Figure 7
python3 make_fig_screen.py                         # Figure 6
```

Figures 1 and 2 are drawn by hand and are not generated.
