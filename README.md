# GUARD

Certified post-hoc correction for a **frozen** model deployed with missing
modalities.  GUARD never retrains, never adds a branch, and never touches the
host's weights: it reads the host's outputs, estimates how much of the error is
a calibration gap, applies a retrieval-based correction, and certifies that the
correction does not increase the loss by more than `delta` on more than an
`alpha` fraction of deployments.

```python
import numpy as np
from guard import HostOutputs, random_split, run

split  = random_split(np.arange(n), seed=0)
result = run(HostOutputs(probs, features, labels), split,
             target="cross_mask", alpha=0.2, delta=0.05)

result.gate_metric_delta   # what you gained
result.joint_harm          # <= alpha, the guarantee
```

## What is guaranteed, precisely

`P(Delta_loss > delta AND applied) <= alpha`, over the draw of a test point,
when the calibration and deployment sets are exchangeable.

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

## Install

```bash
pip install -e .              # the method: numpy only
pip install -e ".[hosts]"     # to re-run the frozen hosts
pip install -e ".[dev]"       # tests
```

## Reproduce

**Straight after cloning, with no downloads and no GPU:**

```bash
pip install -e ".[dev]"
pytest -q                              # the guarantee itself, on synthetic data
python experiments/exp_synthetic.py    # the synthetic table
```

That is enough to check the central claim: the tests verify
`P(Delta_loss > delta AND applied) <= alpha` at three values of `alpha`, and
separately verify that conditional harm is *not* bounded, so the claim and its
limit are both confirmed before any dataset is involved.

**Every certification number in the paper, still without a GPU** — download the
saved host outputs and run everything on numpy:

```bash
export GUARD_ARTIFACT_URL=<release archive>
bash data/download_artifacts.sh dumps    # ~1 GB
bash scripts/run_all.sh
```

**From scratch, with a GPU**, retrain the hosts through their own entry points:

```bash
bash data/download_drugban.sh
python hosts/drugban.py train --dataset biosnap --split random --seed 42
python hosts/drugban.py dump  --dataset biosnap --split random --seed 42
python experiments/exp_drugban.py --dumps data/processed/drugban_biosnap_random_s42
```

`docs/REPRODUCING.md` sets out the three tiers, what each costs, and which
pieces cannot be scripted because they sit behind a licence.

Every experiment writes `results/<name>/guard.csv` plus a `manifest.json`
recording the configuration and seeds, and `scripts/build_tables.py` turns those
into the paper's tables, so no number in the text is typed by hand.

## Layout

```
src/guard/        the method.  ~600 lines, numpy only, no dataset knowledge
  losses.py       canonical losses and their links
  measure.py      neighbour-pair estimate of the calibration-gap energy
  targets.py      hard-label and cross-mask retrieval targets
  certify.py      split-conformal LAC gate and harm accounting
  splits.py       data roles, with exchangeability made explicit
  pipeline.py     Measure -> Recalibrate -> Certify, in one function
hosts/            thin adapters that turn a published model into HostOutputs
experiments/      one file per table in the paper; they call run() and nothing else
scripts/          run_all.sh, build_tables.py
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
| ViLT + missing-aware prompts | CVPR 2023 | vision--language |
| MoMKE, TMDC | 2024, 2026 | affective computing |
| CMAD, IMDer, LNLN | 2023--2025 | dominance and label-efficiency studies |

Host reproduction is checked before any correction is attached; see
`docs/REPRODUCTION.md` for our numbers against the published tables, including
the one setting where we fall short of the published figure.

## Citing

```bibtex
@inproceedings{guard,
  title     = {Gated Utility-Aware Recalibration for Safe Missing-Modality Deployment},
  booktitle = {ICLR},
  year      = {2027}
}
```

Released under the MIT licence.  The datasets and host checkpoints keep their
own licences; see `data/README.md`.
