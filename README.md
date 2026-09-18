# GUARD

Risk-controlled post-hoc correction for a **frozen** model deployed with missing
modalities. GUARD leaves the host's weights fixed and acts on its outputs: it
estimates correctable error, constructs a candidate correction, and gates that
correction with a finite-sample joint-harm guarantee.

```python
import numpy as np
from guard import HostOutputs, random_split, run

split  = random_split(np.arange(n), seed=0)
result = run(HostOutputs(probs, features, labels), split,
             target="hard", alpha=0.2, delta=0.05)

result.gate_metric_delta   # what you gained
result.joint_harm          # realised test statistic; not itself guaranteed <= alpha
```

## What is guaranteed, precisely

Conditioned on all pre-calibration fitting, GUARD guarantees
`P(Delta_loss > delta AND applied) <= alpha` marginally over the calibration
draw and a fresh deployment point, when they are exchangeable. It is a
population statement, so a finite realised test split can exceed `alpha`.

Three things this does **not** say, each of which we report separately so the
distinction cannot be lost:

* It is **joint**, not conditional.  `cond_harm` -- harm among the points where
  we did intervene -- is routinely several times `alpha`.  That is not a
  violation; it is what a joint budget means.
* It bounds the **loss**, not a thresholded downstream metric.  A certified-safe
  loss change can still move macro-F1 the wrong way.
* It assumes **exchangeability**.  `Split` records where each role came from and
  the pipeline emits a warning when calibration and evaluation do not share an
  origin.  We broke this by accident once and the budget broke with it.

The `cross_mask` target is also available, but it requires
`HostOutputs(..., richer_probs=...)`. See `guard.targets.cross_mask_values` and
the paper's cross-mask precondition before using it.

## Install

```bash
pip install -e .              # the method: numpy and scikit-learn
pip install -e ".[hosts]"     # to re-run the frozen hosts
pip install -e ".[dev]"       # tests
pip install -e ".[reproduce]" # artifact download and figure dependencies
```

## Reproduce

**Straight after cloning, with no downloads and no GPU:**

```bash
pip install -e ".[dev]"
pytest -q                              # the guarantee itself, on synthetic data
python experiments/exp_synthetic_sweep.py --out results/synthetic_sweep
```

That is enough to check the central claim: the tests verify
`P(Delta_loss > delta AND applied) <= alpha` at three values of `alpha`, and
separately verify that conditional harm is *not* bounded, so the claim and its
limit are both confirmed before any dataset is involved.

**Re-run the CPU-side paper drivers from saved frozen-host outputs:**

```bash
pip install -e ".[dev,reproduce]"
bash data/download_artifacts.sh dumps    # ~2.5 GB, from a public dataset repo
bash scripts/rerun_all_results.sh
```

`scripts/run_all.sh` is a smaller convenience runner for `drugban`,
`affective`, `opportunity`, or `synthetic`; it is not the complete paper suite.
The direct HME comparator retrains a model and therefore has a separate GPU
protocol in `docs/REVIEW_ADDITIONS.md`.

**From scratch, with a GPU**, retrain the hosts through their own entry points:

```bash
bash data/download_drugban.sh
python hosts/drugban.py train --dataset biosnap --split random --seed 42
python hosts/drugban.py dump  --dataset biosnap --split random --seed 42
python experiments/exp_drugban.py --dumps data/processed/drugban_biosnap_random_s42
```

`docs/REPRODUCING.md` sets out the tiers, what each costs, and which pieces
require licensed source data.

The release includes the compact result files used by the paper and automated
checks for all machine-checkable cells. Some older drivers print rows rather
than rewriting LaTeX; `docs/PAPER_MAP.md` states this explicitly.

## Layout

```
src/guard/        the method.  ~600 lines, no dataset knowledge
  losses.py       canonical losses and their links
  measure.py      neighbour-pair estimate of the calibration-gap energy
  targets.py      hard-label and cross-mask retrieval targets
  action.py       learned action score, conformal risk control, tightening
  splits.py       data roles, with exchangeability made explicit
  pipeline.py     Measure -> Recalibrate -> Certify, in one function
hosts/            adapters that turn a published model into HostOutputs
experiments/      paper drivers, ablations, and review-requested studies
scripts/          rerun, table-verification, download, and figure helpers
tests/            the guarantee checked on synthetic data
data/             download scripts only; no data is committed
docs/             upstream patches, protocol notes
```

The split between `src/guard/` and `experiments/` is the point: if you want to
know what the method does, read `pipeline.py`; the experiment files only load a
frozen host's outputs and choose a split.

## Hosts

| host | published in | what we use it for |
|---|---|---|
| DrugBAN | Nature Mach. Intell. 2023 | drug--target, official random and cluster splits |
| CMAD, TMDC, MoMKE | 2024--2026 | CMU-MOSEI and IEMOCAP |
| AV-att | ECCV 2018 | audio--visual event recognition |
| DeepConvLSTM | Sensors 2016 | wearable activity recognition |
| sEMG CNN | NinaPro DB5 | hand-movement recognition |
| resnet1d_wang | PTB-XL | multi-label ECG classification |

Host reproduction is checked before any correction is attached; see
`docs/REPRODUCTION.md` for our numbers against the published tables, including
the one setting where we fall short of the published figure.

## Citing

```bibtex
@inproceedings{guard,
  title     = {Risk-Controlled Correction for Frozen Multimodal Models under Missing Modalities},
  booktitle = {ICLR},
  year      = {2027}
}
```

Released under the MIT licence.  The datasets and host checkpoints keep their
own licences; see `data/README.md`.
