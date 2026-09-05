# What you can reproduce, and what it costs

Three tiers, from "works the moment you clone" to "retrain every host".  Be
honest with yourself about which one you need; the first tier already checks
the paper's central claim.

## Tier 1 — no downloads, no GPU, about a minute

```bash
pip install -e ".[dev]"
pytest -q                              # the guarantee, on synthetic data
python experiments/exp_synthetic.py    # the synthetic table
```

`pytest` checks `P(Delta_loss > delta AND applied) <= alpha` directly, at three
values of `alpha`, and separately checks that conditional harm is *not*
bounded — so the paper's central claim, and the limit of that claim, are both
verifiable without touching a dataset.

`exp_synthetic.py` reproduces the synthetic table: measured gap grows with the
injected calibration gap and the recovered loss tracks it, while a host whose
error is label noise shows neither.

## Tier 2 — saved host outputs, no GPU, about ten minutes

```bash
export GUARD_ARTIFACT_URL=<release archive>
bash data/download_artifacts.sh dumps      # ~1 GB
bash scripts/run_all.sh
```

The dumps are the frozen hosts' outputs and representations under each
deployment condition.  Everything downstream of a dump is pure numpy, so this
tier reproduces **the certification numbers covered by the bundled artifact archive** on a laptop: the
DrugBAN tables, the vision--language tables, OPPORTUNITY under both calibration
protocols, the label-budget sweep, the selector and plug-in ablations, the
label-efficiency and contraction study, and the three studies below.

```bash
bash scripts/run_all.sh frontier    # alpha and delta sweeps, the validity figure
bash scripts/run_all.sh modules     # Measure x Certify, nulls as the deciding cells
bash scripts/run_all.sh groupwise   # class-conditional harm; harm spread vs n_conf
```

Two things in there are worth knowing before you read the output.

* **The alpha frontier turns over above alpha ~ 0.4.**  At large alpha the
  conformal quantile shrinks until the plausible set would be empty, and
  `certify` never returns an empty set, so those points fall back to the worst
  case over *all* labels and are refused.  Apply rate and harm both fall as a
  result.  That is the convention showing through, not the guarantee failing.
* **The budget sweep runs with `--fixed-host`.**  The host keeps whichever epoch
  scores best on its validation split, and that split is carved out of the
  labelled budget -- so without the flag a larger budget silently buys a better
  frozen host, base accuracy climbs across the sweep, and the table can no
  longer say whether the gain belongs to GUARD or to the host.  With the flag
  the model-selection split is the full one at every budget, base accuracy is
  bit-identical across levels for every (configuration, seed), and only what
  GUARD is given varies.  The honest reading of that table is therefore "labels
  GUARD needs, given a host that is already frozen".

This is the tier we recommend.  The method is what the paper is about, and the
method never sees a GPU.

## Tier 3 — retrain the hosts, GPU, one to two days

```bash
bash data/download_drugban.sh
python hosts/drugban.py train --dataset biosnap --split random --seed 42

bash data/download_vilt.sh                 # then fetch the three corpora by hand
python hosts/vilt_prompts.py train --task hateful_memes --repo data/raw/missing_aware_prompts

bash data/download_opportunity.sh
python hosts/opportunity_prepare.py
python hosts/opportunity.py --source data/processed/opportunity_features.npz \
    --configs configs/opportunity.json
```

Training calls each host's own entry point with its own configuration; we pass
only the dataset, split and seed.  Expect small differences from our numbers:
host training is stochastic, and one benchmark is badly so -- Hateful Memes
moved between AUROC 0.526 and 0.642 across seeds under the released
configuration.  See `docs/REPRODUCTION.md`.

## Licensed source data

The release contains the preprocessing and experiment drivers for every paper
experiment.  Four source datasets must first be obtained by the reader because
their licences do not permit us to redistribute them: IEMOCAP from **USC SAIL**
(signed release agreement), NinaPro DB5 from the **NinaPro project at
HEIA-FR/HES-SO** (registered download), CMU-MOSEI from the **CMU MultiComp Lab**
through the Multimodal SDK, and AVE from the **AVE-ECCV18 authors** through their
access-controlled Drive links.  After that one licence step, the corresponding
script checks the documented layout and continues with preparation and the
released run; it does not merely stop.

The affective hosts (MoMKE, TMDC, CMAD, IMDer, LNLN) are consumed through their
published-output layout. `hosts/dumps.py` documents that layout and the release
drivers run the complete GUARD experiment on those outputs.


## Paper coverage matrix

The release can run the synthetic study, DrugBAN, affective hosts (from their published-output layout), vision--language, and the MLP variant of OPPORTUNITY when their data/checkpoints or dumps are supplied.

The paper's reported OPPORTUNITY row is DeepConvLSTM and is now weights-backed.  The release includes retrained `opportunity_dcl_v2` checkpoints and their coherent DCL_HOSTS2-format dumps; regenerate a checkpoint's dumps with `python hosts/opportunity_dcl_infer.py --checkpoint <checkpoint.pt> --data <opportunity_ours3.npz> --repo <DeepConvLSTM_py3-repo> --out <DCL_HOSTS2>`.  The released checkpoints come from a new retraining run, rather than the unrecoverable original frozen hosts, and the paper reports this retraining run's numbers.  Training is available as `hosts/opportunity_dcl_train.py`; it saves a full-host checkpoint and one `condition_specialist` checkpoint for every configuration and seed, including the preprocessing metadata inference needs.

AVE, NinaPro DB5, and PTB-XL are also archived-dumps-only in this release. Their drivers are respectively `exp_ave.py`, `exp_ninapro.py`, and `exp_ptbxl.py`; each takes `--dumps`. AVE is preserved as frozen audio/visual-attention outputs; NinaPro's archive contains retrained seeds rather than the original paper checkpoint; PTB-XL is preserved as frozen ResNet-1D and reduced-lead outputs. These drivers reproduce the method on those saved hosts and do not claim to retrain the paper runs.

Thus every paper benchmark is either covered by a runnable release driver or explicitly identified above as dump-backed because the original frozen host cannot honestly be regenerated here.

## Source-download coverage

| Benchmark | Data tier | Driver | Available with no manual step |
|---|---|---|---|
| PTB-XL | automatic | `experiments/exp_ptbxl.py` | raw PTB-XL data and the public PTB-XL predictor |
| OPPORTUNITY | automatic | `experiments/exp_opportunity_dcl.py` | raw UCI data and 15 DeepConvLSTM checkpoints |
| DrugBAN | automatic | `experiments/exp_drugban.py` | authors' source/data and the release checkpoints |
| NinaPro DB5 | registered download from NinaPro/HEIA-FR | `scripts/download/ninapro_db5.sh` | preprocesses and runs after DB5 is installed |
| CMU-MOSEI / CMAD | SDK access from CMU MultiComp Lab | `scripts/download/cmu_mosei.sh` / `experiments/repro_check.py` | runs after MOSEI is installed; CMAD check dumps are public |
| AVE / AV-att | access-controlled Drive from AVE-ECCV18 authors | `scripts/download/ave.sh` | runs after the documented AVE layout is installed |
| IEMOCAP / MoMKE / TMDC / GCNet | signed USC SAIL agreement | `scripts/download/iemocap.sh` | runs after IEMOCAP is installed |

Run public downloads with `bash scripts/download/all.sh`; use `--dry-run` to check endpoints only. The only manual actions are accepting the named data licences and placing the resulting files in the documented layouts.

## Two-level check

Level one needs no dataset: `pytest -q` (10 tests), then `python experiments/repro_check.py`. The latter must print the committed-dump paper row: `a 63.1 -> 68.6`, `v 63.7 -> 68.3`, and `av 64.5 -> 70.3`.

Level two is the full rerun from raw data, using the source-download scripts and the driver in the table.
