# Paper-to-code map

This map follows the final manuscript labels. Table numbers can shift when the
venue style moves floats, so the LaTeX labels are the stable identifiers.

Set up the CPU reproduction environment and frozen-output dumps with:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-reproduce.txt
bash data/download_artifacts.sh dumps
```

Drivers read `artifacts/` by default. Set `GUARD_ARTIFACTS` to use another
location. Compact outputs live under `results/`.

## Main text

| Label | Final content | Driver / source |
|---|---|---|
| `tab:mosei-main` | CMU-MOSEI hosts and GUARD | `experiments/mosei_hosts.py`; `scripts/tables/mosei_rows.py` |
| `tab:allbench` | cross-benchmark summary | per-benchmark drivers below; MOSEI is explicitly the three language-absent masks |
| `tab:gates` | intervention rules | `experiments/ltt_one.py` over the gate drivers |
| `tab:crossmask` | hard-label and cross-mask targets | `scripts/tables/drugban_rows.py`; `experiments/opp_targets.py` |
| `fig:overview` | method positioning | hand-drawn publication figure |
| `fig:guard` | deployment pipeline | hand-drawn publication figure |
| `fig:synthetic` | gain against correctable headroom | `experiments/exp_synthetic_sweep.py`; `scripts/figures/make_fig_synthetic.py` |
| `fig:severity` | degradation ladders | ladder drivers; `scripts/figures/make_fig_severity.py` |

## Appendix tables

| Label | Final content | Driver / evidence |
|---|---|---|
| `tab:notation` | notation | descriptive |
| `tab:conditional-harm` | IEMOCAP joint and conditional harm | `experiments/review_conditional.py`; `results/review_20260917/conditional_iemocap.json` |
| `tab:hme-paired` | HME/CMAD/GUARD on paired MOSEI partitions | `experiments/review_hme.py`; `scripts/tables/verify_hme_run.py`; `results/external_review_20260917/hme/` |
| `tab:corrector` | retrieval and probe ablation | `experiments/probe_vs_knn.py` |
| `tab:probe-extension` | matched MOSEI and DrugBAN probe comparison | `experiments/review_probe_extension.py`; `results/review_20260917/probe_*.json` |
| `tab:source-pool` | varying the retrieval pool, its labels, its origin and the calibration size | `experiments/exp_source_pool.py`; `scripts/tables/source_pool_rows.py`; `results/source_pool/*.csv` |
| `tab:delta` | harm tolerance and decision flips | `experiments/exp_delta.py`; `scripts/tables/delta_rows.py`; `results/delta/*.csv` |
| `tab:efficiency` | correction latency and retained state | `experiments/review_efficiency.py`; `results/review_20260917/efficiency_*.json` |
| `tab:actionscore` | action-score learners/features | `experiments/action_family.py` |
| `tab:alpha` | harm-budget sweep | alpha drivers; `scripts/tables/alpha_stats.py` |
| `tab:iemocap` | IEMOCAP missing-rate study | `experiments/iemocap_eta.py` |
| `tab:main` | DrugBAN conditions | `experiments/exp_drugban.py`; `scripts/tables/drugban_rows.py` |
| `tab:gateablation` | OPPORTUNITY sensor loss | `experiments/opp_dcl.py` |
| `tab:fourdomains` | per-condition benchmark results | `experiments/mosei_full.py`, `gates_iemocap.py`, `gates_ave.py`, `nina_sev_dense.py`, `ptbxl_sev_dense.py` |
| `tab:mosei-harm` | CMU-MOSEI harm/apply | `experiments/mosei_full.py` |
| `tab:blanket` | removing the certificate | saved cells produced by the gate drivers |
| `tab:setup` | benchmark setup | descriptive; host evidence in `docs/REPRODUCTION.md` |
| `tab:splits` | sample counts and grids | `experiments/counts_splits.py` |
| `tab:app-synthetic` | matched-error synthetic study | `experiments/exp_synthetic_matched.py` |

## Appendix figures

| Label | Content | Driver |
|---|---|---|
| `fig:iemocap` | IEMOCAP missing-rate curves | `scripts/figures/make_fig_iemocap.py` |
| `fig:alpha` | harm budget sweep | alpha drivers; `scripts/figures/make_fig_alpha2.py` |
| `fig:apply` | intervention rate | `scripts/figures/make_fig_apply.py` |
| `fig:screen` | cross-mask accuracy screen | `scripts/figures/make_fig_screen.py` |

Figure 8's 28 cells come only from binary DrugBAN and five-class OPPORTUNITY; the OPPORTUNITY rows are written by `experiments/opp_targets.py` to `results/gates/opp_targets_cells.json`.
The manuscript explicitly states their chance levels and that absolute accuracy
does not transfer directly to 29-way AVE or many-class NinaPro.

## Automated checks

```bash
pytest -q
python experiments/repro_check.py
python scripts/verify_submission.py /path/to/guard_iclr2027.tex
```

`verify_paper.py` checks 796 cells from the original result tables against the
committed result files. `verify_review_additions.py` checks 88 fields in the
conditional-harm, probe and efficiency tables. `verify_hme_table.py` checks the
54 displayed HME/CMAD/GUARD values against the paired-comparison JSON, whose 70
underlying HME metrics can be recomputed by `verify_hme_run.py` when the CMAD
dumps are installed.

Several older table drivers print rows rather than editing LaTeX. The verifier,
not automatic manuscript rewriting, is the final synchronization check.

## OPPORTUNITY split

Deployment windows are in temporal order and consecutive windows overlap. Every
OPPORTUNITY driver takes its roles from `experiments/opp_split.py`: contiguous
200-window blocks assigned at random to pool, fit, conf and test, with two windows
dropped at each block end. `experiments/opp_dcl_blocks.py` compares this split with
a split by row and with other block sizes (`results/gates/opportunity_dcl_blocks.json`).
The Figure 3 correlation interval is printed by `scripts/tables/synthetic_ci.py`.
