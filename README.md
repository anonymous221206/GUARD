# GUARD

Certified post-hoc correction for a **frozen** model deployed with missing
modalities.  GUARD never retrains, never adds a branch, and never touches the
model's weights: it reads the model's outputs, applies a retrieval-based
correction, and certifies that the correction does not increase the loss by more
than `delta` on more than an `alpha` fraction of deployments.

```python
from guard import HostOutputs, random_split, run

split  = random_split(np.arange(n), seed=0)
result = run(HostOutputs(probs, features, labels), split,
             target="cross_mask", alpha=0.2, delta=0.05)

result.gate_metric_delta   # what you gained
result.joint_harm          # <= alpha, the guarantee
```

## Install

```bash
pip install -e .              # the method: numpy only
pip install -e ".[hosts]"     # to re-run the frozen models
pip install -e ".[dev]"       # tests
```

## Run

**1. Straight after cloning. No downloads, no GPU, about a minute.**

```bash
pip install -e ".[dev]"
pytest -q                              # the guarantee, on synthetic data
python experiments/exp_synthetic.py    # the synthetic table
```

The tests check `P(Delta_loss > delta AND applied) <= alpha` at three values of
`alpha`, and separately check that conditional harm is *not* bounded, so the
claim and its limit are both confirmed before any dataset is involved.

**2. Every certification number in the paper. No GPU, about ten minutes plus the
download.**  Artefacts are one directory per (dataset, model) pair, nine in all,
one per row block of the paper's tables, so a single pair checks a single block.

```bash
export GUARD_ARTIFACT_REPO=<org>/<repo>
bash data/download_artifacts.sh                  # list the nine pairs
bash data/download_artifacts.sh ninapro_cnn      # one pair, ~1.6 GB
bash data/download_artifacts.sh all              # everything, ~10 GB
bash scripts/run_all.sh
```

**3. From scratch, with a GPU.**  Retrain each model through its own entry point.

```bash
bash data/download_drugban.sh
python hosts/drugban.py train --dataset biosnap --split random --seed 42
python hosts/drugban.py dump  --dataset biosnap --split random --seed 42
python experiments/exp_drugban.py --dumps data/processed/drugban_biosnap_random_s42
```

## Where the output goes

Every experiment writes `results/<name>/guard.csv` and a `manifest.json`
recording the configuration and seeds.  `scripts/build_tables.py` turns those
into the paper's tables, so no number in the text is typed by hand.

`beta_objective` defaults to `crossfit`, which is the rule the paper reports.
Run with defaults and the numbers match.

## Layout

```
src/guard/        the method.  ~600 lines, numpy only, no dataset knowledge
  losses.py       canonical losses and their links
  measure.py      neighbour-pair headroom diagnostic (reported, not applied)
  targets.py      hard-label and cross-mask retrieval targets
  certify.py      split-conformal LAC gate and harm accounting
  splits.py       data roles, with exchangeability made explicit
  pipeline.py     preparation -> Recalibrate -> Certify, in one function
hosts/            thin adapters that turn a published model into HostOutputs
experiments/      one file per table in the paper; they call run() and nothing else
scripts/          run_all.sh, build_tables.py
tests/            the guarantee checked on synthetic data
data/             download scripts only; no data is committed
docs/             reproduction tiers, protocol notes, upstream patches
results/          saved result tables; see results/preflight_screen/
```

If you want to know what the method does, read `pipeline.py`.  The experiment
files only load a frozen model's outputs and choose a split.

`HostOutputs` keeps its name for API compatibility.  Everywhere else this
document says *model*, which is the term the paper uses.

## More

`docs/REPRODUCING.md` gives the three tiers above with their real costs, which
row came from which rule, the two artefacts whose names mislead, and the pieces
that cannot be scripted because they sit behind a licence.
`docs/UPSTREAM_PATCHES.md` lists the changes made to third-party code.

## Citing

```bibtex
@inproceedings{guard,
  title     = {Risk-Controlled Correction for Frozen Multimodal Models
               under Missing Modalities},
  booktitle = {ICLR},
  year      = {2027}
}
```

Released under the MIT licence.  The datasets and model checkpoints keep their
own licences; see `data/README.md`.
