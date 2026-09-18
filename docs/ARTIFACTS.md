# External artifact dumps

The release commits compact JSON/CSV summaries under `results/`. Larger frozen
host outputs are distributed through the public artifact repository used by
`data/download_artifacts.sh`; they are never committed to Git.

| Benchmark | Release-relative directory | Main consumer |
| --- | --- | --- |
| CMU-MOSEI / CMAD | `artifacts/mosei_cmad/dumps` | `experiments/repro_check.py` |
| AVE | `artifacts/ave_av_att/dumps` | `experiments/gates_ave.py`, `experiments/ave_eta.py` |
| IEMOCAP | `artifacts/iemocap_momke/folds` | `experiments/gates_iemocap.py`, `experiments/iemocap_eta.py` |
| OPPORTUNITY DeepConvLSTM | `artifacts/opportunity_dcl_v2` | `experiments/opp_dcl.py`, alpha drivers |
| NinaPro | `artifacts/ninapro_cnn` | `experiments/nina_sev_dense.py`, `experiments/nina_alpha.py` |
| PTB-XL | `artifacts/ptbxl_resnet1d_wang`, `artifacts/ptbxl_dropladder` | severity and alpha drivers |
| DrugBAN | `artifacts/drugban_processed`, `artifacts/drugban_protladder_v2` | DrugBAN table and ladder drivers |

## OPPORTUNITY reproducibility correction

The comment in the older `GUARD/opp_dcl.py` claiming that a second DeepConvLSTM training run did not reproduce closely enough to mix outputs is incorrect. The retraining reproduced the stored outputs **bitwise**: `richer_deploy_s0.npy`, `deploy_y.npy`, and the specialist probability arrays have identical SHA-256 hashes. Thus this release supports a genuine training path, not only archived dumps.

## DrugBAN environment

The recorded DrugBAN installation manifest is in
[DRUGBAN_INSTALLED.md](DRUGBAN_INSTALLED.md). It uses Python 3.9, torch
2.1.2+cpu, DGL 2.2.1, and dgllife 0.3.2.

## AVE retrieval sidecars

`AV_att_<condition>.npz` holds posteriors only. The paper protocol also needs the
input to the frozen model's final `L2` layer, on the training pool and on the
deployment split, which `hosts/ave.py export` writes as
`AV_att_<condition>_retrieval_paper.npz`. All three conditions ship, `full`
included: without it the intact negative control and the AVE panel of Figure 4
cannot be run at all.

The exporter checks itself against the stored posteriors and refuses to write a
sidecar that does not reproduce them. The stored dumps were produced on a GPU, so
an exact match needs one (`--cuda-device N`); on CPU the same code reproduces
every decision but differs by about `5e-3` in probability, which the check
rejects on purpose.
