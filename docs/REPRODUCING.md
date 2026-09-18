# Reproducing the release

The release separates a fast method check, CPU reruns from frozen-host outputs,
and host training. This matters because GUARD's certificate can be checked
without retraining any multimodal model.

## 1. Fast check: no downloads or GPU

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python experiments/exp_synthetic.py
python experiments/exp_synthetic_sweep.py --out results/synthetic_sweep
```

The current suite has 13 tests. They exercise the finite-sample decision rule,
including the distinction between joint and conditional harm. The guarantee is
marginal over the calibration draw and a fresh exchangeable deployment point;
it does not require every realised test split to have `joint_harm <= alpha`.

## 2. Validate the committed paper results

For the exact numerical environment used in the final CPU review runs:

```bash
pip install -r requirements-reproduce.txt
python scripts/verify_submission.py /path/to/guard_iclr2027.tex
```

The unified verifier checks the original paper tables, the conditional-harm,
probe and efficiency additions, and the paired HME table. The LaTeX source is
passed explicitly because the anonymous submission source is not bundled in
this code repository.

The compact JSON/CSV/NPZ evidence required for these checks is committed under
`results/`. The HME prediction archive is 169 KiB; its 583 MiB checkpoint and
524 MiB prepared-data cache are intentionally excluded.

## 3. Re-run GUARD from saved host outputs, no GPU

```bash
pip install -r requirements-reproduce.txt
bash data/download_artifacts.sh dumps
python experiments/repro_check.py
bash scripts/rerun_all_results.sh
```

`repro_check.py` must print the current CMAD rows:

```text
a    released code 63.1 -> 67.8
v    released code 63.7 -> 68.2
av   released code 64.5 -> 69.6
```

`rerun_all_results.sh` is deliberately portable, sequential and fail-fast. Logs
are written under `logs/rerun/`. This tier reproduces the CPU-side correction and
calibration results from saved outputs. It does not retrain HME or the frozen
hosts. The latency table is hardware-dependent and is left untouched unless
`RUN_TIMINGS=1` is set.

The older convenience runner remains useful for adapter smoke tests:

```bash
bash scripts/run_all.sh synthetic
bash scripts/run_all.sh drugban
bash scripts/run_all.sh affective
bash scripts/run_all.sh opportunity
```

Only `all`, `drugban`, `affective`, `opportunity`, and `synthetic` are accepted.
Its affective and OPPORTUNITY modes use the documented legacy adapter layouts;
they are not a substitute for the final-paper driver above.

## 4. Re-run the HME contextual comparator

HME is a retrained comparator, not a frozen-host baseline. Its exact protocol,
inputs, selected checkpoint and saved predictions are documented in
`REVIEW_ADDITIONS.md`. Use a separate Python 3.9 environment and install the
CUDA build of PyTorch 1.12.1 before `requirements-hme.txt`.

```bash
export HME_REPO=/path/to/HME
export HME_BERT_DIR=/path/to/BERT_EN
export HME_MOSEI_PICKLE=/path/to/mosei.pkl
python experiments/review_hme.py prepare
python experiments/review_hme.py smoke
python experiments/review_hme.py train
python experiments/review_hme.py evaluate
python experiments/review_hme_audit.py \
  --hme-repo "$HME_REPO" --bert-dir "$HME_BERT_DIR" --dataset "$HME_MOSEI_PICKLE"
python scripts/tables/verify_hme_run.py
```

The final paper reports one HME training seed and five deployment partitions;
those partitions are not independent retraining runs.

## 5. Retrain frozen hosts

Host retraining uses each upstream implementation and has separate dependencies.
Examples:

```bash
bash data/download_drugban.sh
python hosts/drugban.py train --dataset biosnap --split random --seed 42

bash data/download_opportunity.sh
python hosts/opportunity_prepare.py
python hosts/opportunity_dcl_train.py --help
```

IEMOCAP, NinaPro DB5, CMU-MOSEI and AVE require the reader to obtain the data
under their original licences. `scripts/download/` documents the expected
layouts. `docs/REPRODUCTION.md` compares the frozen hosts with published
anchors and records the BioSNAP-cluster shortfall rather than hiding it.
