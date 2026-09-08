# External artifact dumps

The release contains the compact JSON/CSV summaries in `results/` and the requested model checkpoints in `checkpoints/`. The larger intermediate dumps below were deliberately **not** copied. Regenerate them at the stated release-relative `artifacts/` location, or request the preserved source directory.

| Benchmark | Preserved dump directory (size at packaging) | Producer | Release consumer |
| --- | --- | --- | --- |
| CMU-MOSEI / CMAD | `$B/GUARD/artifacts/mosei_cmad/dumps` (37 MB) | `guard_mosei_cmad.py` | `experiments/repro_check.py` |
| AVE | `artifacts/ave_av_att/dumps` (32 MB) | `ave2.py`, and `hosts/ave.py export` for the retrieval sidecars | `experiments/gates_ave.py`, `experiments/ave_eta.py`, `experiments/blanket_two.py` |
| IEMOCAP | `$B/GUARD/artifacts/iemocap_momke/folds` (452 MB) | `iemocap_all.py` | `experiments/blanket_two.py` |
| OPPORTUNITY DeepConvLSTM | `$B/GUARD/artifacts/opportunity_dcl_v2` (292 MB) | `dcl_hosts2.py` | `experiments/opp_dcl_v2.py`, `experiments/opp_alpha.py`, `experiments/opp_alpha_nx.py` |
| NinaPro host outputs | `$B/GUARD/artifacts/ninapro_cnn` (1.6 GB) | `scripts/train_ninapro_retrained.py` | `experiments/nina_ladder_dense.py`, `experiments/nina_sev_dense.py`, `experiments/nina_alpha.py` |
| NinaPro ladder | `$B/GUARD/artifacts/ninapro_ladder_v2` (2.7 MB) | `experiments/nina_ladder_dense.py` | `experiments/nina_sev_dense.py`, paper figure scripts |
| NinaPro specialists | `$B/GUARD/artifacts/ninapro_specialist` (34 MB) | `scripts/ninapro_dense_specialist_v3.py` | paper figure scripts (`mk9.py`) |
| PTB-XL host features | `$B/GUARD/artifacts/ptbxl_resnet1d_wang` (5.3 GB) | PTB-XL Wang host export | `experiments/ptbxl_sev_dense.py`, `experiments/ptbxl_sev_auc.py`, `experiments/ptbxl_shift.py`, `experiments/alpha_sweep_new.py` |
| PTB-XL lead ladder | `$B/GUARD/artifacts/ptbxl_dropladder` (8.5 GB) | `guard_leadladder2.py` | `experiments/ptbxl_sev_dense.py`, `experiments/ptbxl_sev_auc.py`, `experiments/ptbxl_shift.py`, `experiments/alpha_sweep_new.py` |
| DrugBAN protein ladder | `artifacts/drugban_protladder_v2` (269 MB of dumps) | `scripts/drugban_protladder_v2.py` | `scripts/drugban_protladder_guard.py`, `scripts/figures/make_fig_severity.py` |

`$B` above stands for the machine the dumps were preserved on; it is not part of this release. The drivers use `Path(__file__)` to locate the release root, so regenerated artifacts belong under `<release>/artifacts/`.

## OPPORTUNITY reproducibility correction

The comment in the older `GUARD/opp_dcl.py` claiming that a second DeepConvLSTM training run did not reproduce closely enough to mix outputs is incorrect. The retraining reproduced the stored outputs **bitwise**: `richer_deploy_s0.npy`, `deploy_y.npy`, and the specialist probability arrays have identical SHA-256 hashes. Thus this release supports a genuine training path, not only archived dumps.

## DrugBAN environment

The preserved CPU-only environment is `$B/GUARD/envs/drugban/`; its installation manifest has been copied verbatim to [DRUGBAN_INSTALLED.md](DRUGBAN_INSTALLED.md). It uses Python 3.9, torch 2.1.2+cpu, DGL 2.2.1, and dgllife 0.3.2.

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

