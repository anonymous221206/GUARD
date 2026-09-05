# Data

Nothing in this directory is committed.  Each script downloads one source into
`data/raw/` and prints where it landed.

| script | what it fetches | size | licence |
|---|---|---|---|
| `download_drugban.sh` | DrugBAN code + BindingDB/BioSNAP/Human with official random and cluster splits | ~40 MB | MIT (code), see upstream for data |
| `download_opportunity.sh` | OPPORTUNITY activity recognition | ~300 MB | CC BY 4.0 |
| `download_artifacts.sh` | our derived artefacts: checkpoints and saved model outputs, one directory per (dataset, model) pair | ~10 GB total, per-pair download | MIT |

## Datasets that cannot be scripted

The remaining benchmarks need a manual step, either because they sit behind a
click-through licence or because the release is a web form:

* **CMU-MOSEI** — via the CMU-MultimodalSDK; follow that project's instructions.
* **IEMOCAP** — <https://sail.usc.edu/iemocap/>, requires a signed academic
  licence agreement.
* **AVE** — the archive linked from the AVE (ECCV 2018) repository.
* **NinaPro DB5** — <https://ninapro.hevs.ch/>, requires registration and
  acceptance of the data-use terms.
* **PTB-XL** — <https://physionet.org/content/ptb-xl/>, open access.

Place each under `data/raw/` following the layout expected by the corresponding
adapter in `hosts/`.

## Where the model weights come from

This matters for reproduction, so it is stated exactly rather than summarised.

* **Released weights we use as published:** the AV-att checkpoint only.
* **Retrained by us from the authors' released code:** DrugBAN, DeepConvLSTM,
  CMAD, MoMKE, `resnet1d_wang`.  These projects publish code, and in some
  cases features, but not the weights this work needs.
* **Our own architecture:** the small sEMG CNN used on NinaPro DB5.  Its original
  checkpoints were not retained; the ones shipped here are a retrain, and the
  deviation that introduces is quantified in the top-level `README.md`.

We do not redistribute any third-party dataset or any third-party checkpoint.
What `download_artifacts.sh` fetches is our own output: trained weights we
produced, and the saved predictions and representations of frozen models.
