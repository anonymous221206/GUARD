# Additions made in response to external review

## Joint versus conditional harm

`tab:conditional-harm` reports exact applied and harmful occurrence counts for
all seven IEMOCAP masks over 25 fold-cut runs. The key sparse-intervention case
is audio+text: mean joint harm is 0.031, mean apply rate is 3.74%, and the
pooled conditional harm is 296/356 = 83.15%. This does not violate the joint
budget. It shows why a low marginal rate is not an operational promise that an
applied correction is usually safe.

`experiments/review_conditional.py` reconstructs the integer counts from the
unrounded saved rates and original held-out lengths, rejecting any non-integral
reconstruction. Repeated cuts overlap, so pooled counts are evaluated
occurrences rather than unique people; no iid confidence interval is claimed.

## Matched probe and efficiency studies

The probe extension uses the same labelled pool, fit, calibration and test
roles as retrieval, with separate fit-only hyperparameter selection and the
same certificate. It covers 30 CMU-MOSEI cells and six deployment-pool DrugBAN
cluster cells. The result does not support a general retrieval advantage under
domain shift: the probe has higher mean gain and lower joint harm on both
DrugBAN cluster datasets.

The efficiency table is a controlled single-thread CPU microbenchmark at fixed
settings. It includes feature standardisation, target construction, blending,
scoring and gating, while excluding frozen feature extraction, fitting, disk
I/O and calibration. It is not a timing of the per-cell selected winners.

## HME comparator

HME is included as experimental context, not as a like-for-like GUARD baseline.
HME retrains its representation and fusion model without deployment labels;
GUARD keeps CMAD frozen and uses labelled deployment pool/fit/calibration
splits. The run uses the official HME code at commit
`cdb1d60b30bddb5ba7fe7762dadaf7e21dc8619f`, one training seed (5576), and five
paired deployment partitions.

The released-code configuration uses hidden size 192, depths 2/3/3, batch size
256, learning rate 2e-5, and objective
`L_task + 0.8 L_aux + 0.01 L_info`. Checkpoint selection minimises mean
validation loss over the six incomplete masks. Zero-based epoch 53 is selected;
training stops after 64 epochs with patience 10. Test evaluation occurs only
after selection. The environment is recorded in `seed5576/protocol.json`.

Saved evidence includes the training history, protocol, 35 evaluation rows and
predictions. The large checkpoint and prepared source-data cache are excluded.
`scripts/tables/verify_hme_run.py` recomputes all 70 accuracy/F1 fields from the
saved predictions and pairs them with CMAD on the same held-out indices.
The `driver_sha256` in the stored protocol identifies the exact VUW runner; the
public copy changes only machine-specific paths into command-line arguments.

Only HME was completed. MissModal's official release did not provide an
executable implementation, and the examined MiDl adaptation selected no
LayerNorm parameters in the current AVE host. The repository and paper do not
claim that those two comparisons were run.
