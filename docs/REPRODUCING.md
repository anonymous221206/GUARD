# What you can reproduce, and what it costs

Three tiers, from "works the moment you clone" to "retrain every model".  Be
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

## Tier 2 — saved model outputs, no GPU, about ten minutes

```bash
export GUARD_ARTIFACT_REPO=<org>/<repo>
bash data/download_artifacts.sh                # list the twelve pairs
bash data/download_artifacts.sh ninapro_cnn    # one pair, ~1.6 GB
bash data/download_artifacts.sh all            # everything, ~12 GB
bash scripts/run_all.sh
```

The dumps are the frozen models' outputs and representations under each
deployment condition.  Everything downstream of a dump is pure numpy, so this
tier reproduces **every certification number in the paper** on a laptop: the
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

bash data/download_opportunity.sh
python hosts/opportunity_prepare.py
python hosts/opportunity.py --source data/processed/opportunity_features.npz \
    --configs configs/opportunity.json
```

Training calls each model's own entry point with its own configuration; we pass
only the dataset, split and seed.  Expect small differences from our numbers:
training is stochastic and none of these scripts is made bit-deterministic.  The
NinaPro CNN is the one we can quantify, because we retrained it twice: the two
seeds sat 0.005 apart and both landed within 0.004 of the published rows.

## What is *not* scriptable here

The affective-computing models (MoMKE, TMDC, CMAD, IMDer) are used through
their own released code, which we do not vendor.  We ship their **outputs** in
the dump archive, since those are ours to publish; regenerating them means
following each project's own instructions and pointing `hosts/dumps.py` at the
result.  The expected file layout is documented at the top of that file.

Some corpora sit behind click-through licences and cannot be fetched by script;
`data/README.md` lists where to get each one.

Finally, note the beta-selection gap recorded at the top of `README.md`: the
paper's rule D is not yet implemented in `src/guard/`, so Tier 2 does not yet
reproduce the published numbers exactly.

## The beta-selection rule

`beta_objective` defaults to `crossfit` (rule D), which is what the paper
reports: a four-fold cross-fit inside the fit split that runs the full conformal
gate on held-out rows and keeps the beta with the best mean certified gain. Run
with defaults and the numbers match.

The alternative is `loss` (rule A), which minimises the fit loss over the grid in
a single pass. It is the objective the theory motivates, and it was the default
in an earlier version of this package. Rule D performs 164 certifications where
rule A performs one sweep, but end to end the difference is small: measured over
DrugBAN and OPPORTUNITY, 38.4 s against 36.9 s, because retrieval and I/O
dominate.

The two rules do not always agree. On NinaPro DB5, alpha = 0.20, two seeds:

| electrodes | paper | rule D (default) | rule A |
|---|---|---|---|
| 12 | 0.780 | 0.782 / 0.781 | 0.766 / 0.763 |
| 8  | 0.750 | 0.748 / 0.749 | 0.722 / 0.721 |
| 6  | 0.729 | 0.726 / 0.725 | 0.694 / 0.691 |
| 4  | 0.686 | 0.685 / 0.686 | 0.636 / 0.638 |

Rule D lands within 0.004 of the paper, inside the spread between the two seeds
(0.005). Rule A is lower at every rung, by up to 0.049. Elsewhere the gap is
smaller: on DrugBAN and OPPORTUNITY the two differ by at most 0.019 corrected
accuracy, and in both directions.

`scripts/verify_crossfit_port.py` checks this implementation of rule D against
the one the experiments were run with, over 72 (seed, rung, subject, alpha)
combinations: the selected beta agrees exactly.

## What is guaranteed, precisely

`P(Delta_loss > delta AND applied) <= alpha`, over the draw of a test point, when
the calibration and deployment sets are exchangeable.

Three things this does **not** say, each reported separately so the distinction
cannot be lost:

* It is **joint**, not conditional. `cond_harm`, the harm among the points where
  we did intervene, is routinely several times `alpha`. That is not a violation;
  it is what a joint budget means.
* It bounds the **loss**, not a thresholded downstream metric. A certified-safe
  loss change can still move macro-F1 the wrong way.
* It assumes **exchangeability**. `Split` records where each role came from and
  the pipeline warns when calibration and evaluation do not share an origin.

## Two artefacts that are not what their names suggest

* **IEMOCAP is five-fold.** The standard `raw_features.npz` / `preds.npz` pair for
  MoMKE holds a single session, so it cannot rebuild the five-fold
  leave-one-session-out numbers on its own. The five-fold dumps ship separately
  under `<pair>/folds/`, thirty-five files each, and that is what the paper's
  IEMOCAP rows are computed from.
* **NinaPro checkpoints are retrained.** The originals were not retained. The
  released checkpoints come from re-running the training script unchanged; they
  land within 0.004 of the published rows under rule D, inside the seed-to-seed
  spread. Training was not made bit-deterministic, so a further retrain will
  differ again by about that much.

## Models corrected

| model | published in | used for |
|---|---|---|
| DrugBAN | Nature Mach. Intell. 2023 | drug--target, official random and cluster splits |
| DeepConvLSTM | Sensors 2016 | wearable activity recognition, OPPORTUNITY |
| CMAD | 2025 | affective computing on CMU-MOSEI |
| AV-att | ECCV 2018 | audio--visual event localisation on AVE |
| MoMKE | 2024 | affective computing on IEMOCAP |
| `resnet1d_wang` | PTB-XL benchmark, 2021 | ECG diagnosis |
| a small sEMG CNN (ours) | -- | NinaPro DB5, the only architecture we wrote |

Only the AV-att checkpoint is used as published; the rest are retrained from the
authors' released code. Reproduction is checked against each published table
before any correction is attached, and the one split that falls short is reported
rather than tuned away.
