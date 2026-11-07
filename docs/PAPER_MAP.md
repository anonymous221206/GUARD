# Paper-to-code map

Rows use the LaTeX labels of the manuscript. Saved outputs are under `results/`.

## Main text

| Label | Final content | Driver / source |
|---|---|---|
| `tab:mosei-main` | CMU-MOSEI hosts and GUARD | `experiments/mosei_hosts.py`; `scripts/tables/mosei_rows.py` |
| `tab:allbench` | cross-benchmark summary | per-benchmark drivers below; MOSEI is explicitly the three language-absent masks |
| `tab:gates` | score, calibration and tightening on one corrector | `experiments/rebuttal_factorial.py`; `scripts/tables/rebuttal_rows.py`; `results/rebuttal/factorial_*.jsonl` |
| `tab:budget` | total deployment-label budget | `experiments/rebuttal_budget.py`; `scripts/tables/rebuttal_rows.py`; `results/rebuttal/budget.jsonl` |
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
| `tab:reliability` | what applied corrections do, all benchmarks | `experiments/rebuttal_factorial.py` |
| `tab:paired` | paired differences on one corrector | `experiments/rebuttal_factorial.py` |
| `tab:matched` | rules at one intervention rate | `experiments/rebuttal_factorial.py`; `experiments/rebuttal_matched.py`; `results/rebuttal/matched_l1.jsonl` |
| `tab:budget-full`, `tab:budget-pool` | label budget in full | `experiments/rebuttal_budget.py` |
| `tab:mixed`, `tab:reuse` | mixed availability; adapter reused at another condition | `experiments/rebuttal_shift.py`; `results/rebuttal/shift.jsonl` |
| Sec. 5.6, App. B text | the two calibrators across budget settings; worst-pattern standard errors | `experiments/rebuttal_crc_ltt.py`; `results/rebuttal/crc_ltt.json` |
| App. A text | input audit of all seven benchmarks | `experiments/rebuttal_audit.py`; `results/rebuttal/audit*.json` |
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
| `tab:setup` | benchmark setup | descriptive |
| `tab:splits` | sample counts and grids | `experiments/counts_splits.py` |
| `tab:app-synthetic` | matched-error synthetic study | `experiments/exp_synthetic_matched.py` |

## Appendix figures

| Label | Content | Driver |
|---|---|---|
| `fig:iemocap` | IEMOCAP missing-rate curves | `scripts/figures/make_fig_iemocap.py` |
| `fig:alpha` | harm budget sweep | alpha drivers; `scripts/figures/make_fig_alpha2.py` |
| `fig:apply` | intervention rate | `scripts/figures/make_fig_apply.py` |
| `fig:screen` | cross-mask accuracy screen | `scripts/figures/make_fig_screen.py` |

## Checks

```bash
pytest -q
python scripts/verify_submission.py /path/to/guard_iclr2027.tex
```
