#!/usr/bin/env python3
"""Prove the ported rule D matches the experiment-script original.

Compares guard.pipeline._select_beta(..., "crossfit") against pick_D from
ninapro41_gate.py on identical inputs.  Any disagreement is a porting bug.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

GR = Path("$WORKSPACE")
REBUTTAL = "$WORKSPACE/scripts/rebuttal"
ART = GR / "artifacts" / "ninapro_cnn"
RUNGS = (12, 8, 6, 4)
ALPHAS = (0.05, 0.20, 0.40)
N_CLS, K = 41, 50

sys.path.insert(0, str(GR)); sys.path.insert(0, str(GR / "src"))
sys.path.insert(0, REBUTTAL)
from guard import losses as guard_losses          # noqa: E402
from guard.pipeline import _select_beta           # noqa: E402
from ninapro41_gate import DELTA, pick_D          # noqa: E402
import torch                                      # noqa: E402


def knn_target(query, pool, pool_y):
    sq = (pool ** 2).sum(1)
    vals = np.eye(N_CLS)[pool_y]
    out = []
    for s in range(0, len(query), 2048):
        b = query[s:s + 2048]
        d = (b ** 2).sum(1)[:, None] + sq[None, :] - 2 * b @ pool.T
        out.append(vals[np.argpartition(d, K - 1, axis=1)[:, :K]].mean(1))
    return np.concatenate(out)


loss = guard_losses.get("cross_entropy")
worst, checked = 0.0, 0
for seed in (0, 1):
    for rung in RUNGS:
        for subj in range(1, 4):                  # three subjects is enough to prove it
            d = ART / f"seed{seed}" / f"subject{subj:02d}"
            pr, em = np.load(d / "preds.npz"), np.load(d / "masked_embeddings.npz")
            ck = torch.load(ART / "checkpoints" / f"seed{seed}"
                            / f"subject{subj:02d}_rung{rung:02d}.pt",
                            map_location="cpu", weights_only=False)
            tr_y, q_y = pr["train_y"], pr["sess1_y"]
            tr_e, q_e = em[f"train_{rung}"], em[f"sess1_{rung}"]
            pos = np.searchsorted(np.asarray(ck["train_index"]), np.asarray(ck["fit_index"]))
            mu = tr_e[pos].mean(0); sd = tr_e[pos].std(0) + 1e-8
            tgt = knn_target((q_e - mu) / sd, (tr_e - mu) / sd, tr_y)
            probs = pr[f"sess1_{rung}"].astype(np.float64)

            perm = np.random.default_rng(7).permutation(len(q_y))
            third = len(perm) // 3
            fit = perm[:third]
            for alpha in ALPHAS:
                b_orig = pick_D(probs, tgt, q_y, fit, loss, alpha)
                b_port = _select_beta(probs[fit], tgt[fit], q_y[fit], loss,
                                      "crossfit", alpha=alpha, delta=DELTA)
                gap = abs(b_orig - b_port)
                worst = max(worst, gap); checked += 1
                if gap > 0:
                    print(f"MISMATCH seed={seed} rung={rung} subj={subj} "
                          f"alpha={alpha}: orig={b_orig} port={b_port}", flush=True)

print(f"checked={checked}  max|beta_orig - beta_port|={worst}")
print("PORT_FAITHFUL" if worst == 0 else "PORT_DIFFERS")
