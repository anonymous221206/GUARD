#!/usr/bin/env python3
"""NinaPro DB5 on the published 41-class task: reproduction, then the ladder.

The 12-movement run had no benchmark, and that was recorded as its weakness. It
turns out one exists for a class set we were not using. Pizzolato et al. 2017
(PLOS ONE 12(10):e0186132) report **69.04%** for the double-Myo setup on 41
classes -- Ninapro exercises B and C plus rest -- with mDWT features and an SVM,
200 ms windows overlapping by 100 ms, repetitions 1/3/4/6 training and 2/5 test.

At the Myo's 200 Hz our existing window of 40 samples with step 20 is exactly
200 ms / 100 ms, and the repetition split is already theirs. Only the class set
differed. So this is not a new protocol, it is the same host read against the
class set the benchmark is defined on.

Declared before running (see ninapro41_prereg.md):

  * rest is 58% of this task -- 12,239 pure windows against 8,738 for all forty
    movements on subject 1 -- so 69.04% sits eleven points above always-rest, and
    the majority class is printed beside every rung
  * the bar is one-sided, mean >= 0.6654, because the host is a convolution and
    the target came from mDWT+SVM; landing above a 2017 shallow pipeline is
    expected, landing below it is disqualifying
  * miss and the domain is dropped from the paper, no retry

The channel permutation seed is unchanged from the 12-movement run, so the rungs
are the same electrode subsets and the two ladders can be read side by side.
"""
from __future__ import annotations

import glob
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = str(Path(__file__).resolve().parents[1] / "data/ninapro")
POOL_REPS, QUERY_REPS = [1, 3, 4, 6], [2, 5]
WINDOW, STEP, N_CH = 40, 20, 16
EXERCISES = ((2, 0), (3, 17))          # (file index, label offset); rest stays 0
N_CLS = 41
LADDER = [16, 12, 8, 6, 4]
KS = (1, 3, 5, 10, 20, 50)
PUBLISHED, FLOOR = 0.6904, 0.6904 - 0.025


def load_subject(s):
    """Windows, 41-way labels and repetition ids for one subject, E2 + E3."""
    xs, ys, rs = [], [], []
    for ex, off in EXERCISES:
        pat = f"{ROOT}/s{s}/**/S{s}_E{ex}_*.mat"
        for f in sorted(glob.glob(pat, recursive=True)):
            d = sio.loadmat(f)
            emg = d["emg"].astype(np.float32)
            lab = d["restimulus"].ravel().astype(int)
            rep = d["rerepetition"].ravel().astype(int)
            n = (len(emg) - WINDOW) // STEP + 1
            idx = np.arange(WINDOW)[None, :] + STEP * np.arange(n)[:, None]
            yl, rl = lab[idx[:, -1]], rep[idx[:, -1]]
            # a window counts only if it lies inside one movement and one
            # repetition; rest windows pass the same test rather than a looser one
            ok = ((lab[idx] == yl[:, None]).all(1)
                  & (rep[idx] == rl[:, None]).all(1)
                  & (rl > 0))
            xs.append(emg[idx][ok])
            ys.append(np.where(yl[ok] > 0, yl[ok] + off, 0))
            rs.append(rl[ok])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(rs)


def build(dev, n_out, seed):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Conv1d(N_CH, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
        nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten()).to(dev)
    head = nn.Linear(128, n_out).to(dev)
    return net, head


def main():
    import os
    import torch
    from torch import nn
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.checkpoints.exists():
        raise FileExistsError("refusing to overwrite output/checkpoint directory")
    args.out.mkdir(parents=True)
    args.checkpoints.mkdir(parents=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    order_ch = np.random.default_rng(1234).permutation(N_CH)

    rows = {ne: [] for ne in LADDER}
    full41, bal41, mov40, majs, ses = [], [], [], [], []

    for s in range(1, 11):
        X, Y, R = load_subject(s)
        tr = np.flatnonzero(np.isin(R, POOL_REPS))
        te = np.flatnonzero(np.isin(R, QUERY_REPS))
        rng = np.random.default_rng(s)
        p = rng.permutation(len(tr)); c = int(0.85 * len(p))
        fit, val = tr[p[:c]], tr[p[c:]]
        mu, sd = X[fit].mean((0, 1)), X[fit].std((0, 1)) + 1e-8
        Xn = ((X - mu) / sd).transpose(0, 2, 1)                # (n, 16, 40)
        maj = float(np.bincount(Y[te], minlength=N_CLS).max() / len(te))
        majs.append(maj); ses.append(float(np.sqrt(0.25 / len(te))))

        net, head = build(dev, N_CLS, args.seed)
        opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()),
                                lr=1e-3, weight_decay=1e-3)
        xt = torch.tensor(Xn[fit]); yt = torch.tensor(Y[fit], dtype=torch.long)
        xv = torch.tensor(Xn[val], device=dev)
        g = torch.Generator().manual_seed(args.seed)
        best, state = -1.0, None
        for _ in range(40):
            net.train(); head.train()
            o = torch.randperm(len(xt), generator=g)
            for i in range(0, len(o), 128):
                j = o[i:i + 128]
                opt.zero_grad()
                nn.functional.cross_entropy(head(net(xt[j].to(dev))),
                                            yt[j].to(dev)).backward()
                opt.step()
            net.eval(); head.eval()
            with torch.no_grad():
                a = float((head(net(xv)).argmax(1).cpu().numpy() == Y[val]).mean())
            if a > best:
                best = a
                state = ({k: v.clone() for k, v in net.state_dict().items()},
                         {k: v.clone() for k, v in head.state_dict().items()})
        net.load_state_dict(state[0]); head.load_state_dict(state[1])
        net.eval(); head.eval()

        def emb(x):
            with torch.no_grad():
                return np.concatenate([net(torch.tensor(x[i:i + 2048], device=dev))
                                       .cpu().numpy()
                                       for i in range(0, len(x), 2048)])

        def pred(x):
            with torch.no_grad():
                return np.concatenate([head(net(torch.tensor(x[i:i + 2048],
                                                             device=dev)))
                                       .argmax(1).cpu().numpy()
                                       for i in range(0, len(x), 2048)])

        # --- the reproduction number, on all sixteen channels ---
        pt = pred(Xn[te])
        a41 = float((pt == Y[te]).mean())
        rec = [float((pt[Y[te] == k] == k).mean())
               for k in range(N_CLS) if (Y[te] == k).any()]
        m = Y[te] > 0
        full41.append(a41); bal41.append(float(np.mean(rec)))
        mov40.append(float((pt[m] == Y[te][m]).mean()))
        print(f"  chu the {s}: 41 lop {a41:.4f} | can bang {np.mean(rec):.4f} | "
              f"40 cu dong {float((pt[m] == Y[te][m]).mean()):.4f} | "
              f"lop da so {maj:.4f} | {len(te)} cua so", flush=True)

        # --- the ladder, same electrode subsets as the 12-movement run ---
        for ne in LADDER:
            ch = np.sort(order_ch[:ne])
            Xm = np.zeros_like(Xn)
            Xm[:, ch] = Xn[:, ch]
            ha = float((pred(Xm[te]) == Y[te]).mean())
            ef, ev, et = emb(Xm[fit]), emb(Xm[val]), emb(Xm[te])
            m2, s2 = ef.mean(0), ef.std(0) + 1e-8
            ef, ev, et = (ef - m2) / s2, (ev - m2) / s2, (et - m2) / s2
            sq = (ef ** 2).sum(1); onehot = np.eye(N_CLS)[Y[fit]]

            def knn(q, k):
                out = []
                for i in range(0, len(q), 2048):
                    b = q[i:i + 2048]
                    d = (b ** 2).sum(1)[:, None] + sq[None, :] - 2 * b @ ef.T
                    out.append(onehot[np.argpartition(d, k - 1, axis=1)[:, :k]]
                               .mean(1).argmax(1))
                return np.concatenate(out)

            bk = max(KS, key=lambda kk: float((knn(ev, kk) == Y[val]).mean()))
            target_acc = float((knn(et, bk) == Y[te]).mean())
            rows[ne].append((ha, target_acc))
            torch.save({
                "artifact_status": "retrained_checkpoint_not_original_paper_checkpoint",
                "seed": args.seed,
                "subject": s,
                "rung": ne,
                "channels": ch,
                "net_state_dict": net.state_dict(),
                "head_state_dict": head.state_dict(),
                "normalization_mean": mu,
                "normalization_std": sd,
                "train_index": tr,
                "query_index": te,
                "fit_index": fit,
                "validation_index": val,
                "best_k": bk,
                "full_accuracy": a41,
                "base_accuracy": ha,
                "target_accuracy": target_acc,
            }, args.checkpoints / f"subject{s:02d}_rung{ne:02d}.pt")
        print(f"  chu the {s}: thang xong", flush=True)

    mean41 = float(np.mean(full41))
    print(f"\n=== tai tao tren 41 lop ===")
    print(f"trung binh 10 chu the : {mean41:.4f}")
    print(f"Pizzolato 2017        : {PUBLISHED:.4f}  (mDWT + SVM)")
    print(f"san mot phia          : {FLOOR:.4f}  -> "
          f"{'DAT' if mean41 >= FLOOR else 'TRUOT, BO NINAPRO'}")
    print(f"can bang (macro)      : {np.mean(bal41):.4f}")
    print(f"chi 40 cu dong        : {np.mean(mov40):.4f}")
    print(f"lop da so (rest)      : {np.mean(majs):.4f}   "
          f"<- 41 lop la bai de hon 12 cu dong (0.1181)")
    print(f"sai so chuan tb       : {np.mean(ses):.4f}")

    print(f"\n{'dien cuc':10s} {'host_A':>8s} {'target':>8s} {'margin':>9s} "
          f"{'tren da so':>11s}")
    for ne in LADDER:
        h = np.mean([r[0] for r in rows[ne]]); t = np.mean([r[1] for r in rows[ne]])
        print(f"{ne:>3d} kenh   {h:8.4f} {t:8.4f} {t - h:+9.4f} "
              f"{h - np.mean(majs):+11.4f}")
    np.savez_compressed(args.out / "ladder.npz",
                        ladder=np.array(LADDER),
                        host=np.array([[r[0] for r in rows[ne]] for ne in LADDER]),
                        target=np.array([[r[1] for r in rows[ne]] for ne in LADDER]),
                        maj=np.array(majs), full41=np.array(full41),
                        balanced41=np.array(bal41), movement40=np.array(mov40),
                        seed=np.array(args.seed))
    print("\nNINAPRO41_DONE")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
