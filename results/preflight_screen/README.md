# Cross-mask accuracy screen, per condition

`drugban_preflight_all_splits.csv` — `richer_is_richer` evaluated for **every** degraded
condition of DrugBAN: 4 conditions x 2 pool modes x 13 dumps (dataset x protocol x seed) =
104 rows. Each row gives `richer_acc`, `poorer_acc` and `precondition_met` on all three of
the `fit`, `conf` and `test` splits, so the split dependence is visible.

`crossmask_precondition.csv` — the screen table the paper reports, 35 rows over DrugBAN and
OPPORTUNITY, with the certified gain under each retrieval target.

Regenerate the first with
`python3 experiments/audit_preflight.py --root data/processed --out <file>.csv`
(reads `dumps/*.npz` only, trains nothing).

## Why this directory exists

Before 2026-08-30, `exp_drugban.py` ran the screen at the `full` condition, where `build()`
loads `full.npz` for both the richer and the poorer host. The two sides are then the **same
model**, so `richer_acc` equals `poorer_acc` and `precondition_met` is always `False`.
**51 of the 71** `pre-flight` lines in `logs/` are a consequence of that and measure nothing.
They are left in place because logs are a record, but **do not read them as evidence**. This
table replaces them.

`exp_drugban.py` now runs the screen at every **degraded** condition, and evaluates it on
`split.fit` rather than `split.test`: the screen is described as a pre-flight check, so it
must not consume test labels.

`exp_opportunity.py` had two problems. On the `--features` route the screen ran only for the
first configuration of the first seed. On `artifact_route()`, which is the path that produces
the OPPORTUNITY rows the paper prints, the screen did not run at all. Both are fixed.

## Agreement with the numbers in the paper

**DrugBAN, 23/23 rows.** The `fit` split matches `crossmask_precondition.csv` to better than
5e-5. The `conf` and `test` splits match 0/23. That is the evidence the paper's screen was
measured on `fit`, and the reason the fix chose `fit`.

Running the patched `exp_drugban.py` on `drugban_biosnap_random_s42 --pool source --seed 42`
prints `prot25: 0.8478 vs 0.5462` and `scaffold_prot50: 0.8478 vs 0.6854`, matching the two
corresponding rows exactly. The same command before the fix printed
`0.8331 vs 0.8331 -> NOT applicable`.

**OPPORTUNITY, 12/12 rows.** The screen must be evaluated on a **single split fixed for the
whole seed**, the `fit` split of the first configuration, not each configuration's own `fit`.
The split is redrawn per (configuration, design), so measuring per configuration makes the
richer host's accuracy vary with the configuration and adds split noise to a comparison that
is meant to be across configurations. Measured on one split, the richer host is a single
number per seed, as in the paper's table. Verified with
`exp_opportunity.py --artifact artifacts/opportunity_deepconvlstm --protocol deployment`.

**All 35 rows of `crossmask_precondition.csv` are now reproducible from the code here.**

## The two rows that do not pass

On the `fit` split, 102 of 104 rows have the richer host ahead of the poorer one. The two
that do not:

| dump | condition | richer | poorer |
|---|---|---|---|
| `bindingdb_cluster` deployment s42 | `scaffold` | 0.5264 | 0.5310 |
| `human_random` source s2 | `prot50` | 0.9133 | 0.9167 |

Both are conditions the paper does **not** report, which uses `prot25` and `scaffold_prot50`,
so neither contradicts the screen results in the paper.
