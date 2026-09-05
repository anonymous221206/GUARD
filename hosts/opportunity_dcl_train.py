#!/usr/bin/env python3
"""One training pass that dumps everything the DeepConvLSTM host is asked for.

The first pass dumped only what was needed then: each design's predictions on
the masked deployment input.  Two things turned out to need more.  Cross-mask
retrieval averages the richer host's predictions on the *unmasked* input, and
the appendix comparisons against direct use of deployment labels run under
subject shift, calibrating on subject 3 -- neither of which that pass saved.

Filling the gaps by retraining and appending was tried and rejected: seed 0
retrained to validation accuracy 0.9431 against the original 0.9440, so a second
run does not reproduce the first closely enough to mix their outputs.  cuDNN's
LSTM kernels are not deterministic across runs even at a fixed seed.

So everything is dumped from a single pass instead, and determinism is requested
as well, since a reproduction that also holds across runs is worth having.  The
outputs of this script supersede the first pass entirely; they are not to be
mixed with it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch



def f1(pred, true, weighted=True):
    labs = np.unique(true)
    fs, sup = [], []
    for c in labs:
        tp = ((pred == c) & (true == c)).sum()
        fp = ((pred == c) & (true != c)).sum()
        fn = ((pred != c) & (true == c)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * p * r / (p + r) if p + r else 0.0)
        sup.append((true == c).sum())
    fs, sup = np.array(fs), np.array(sup)
    return float((fs * sup).sum() / sup.sum()) if weighted else float(fs.mean())


def train_host(xtr, ytr, xva, yva, n_classes, seed, epochs, device):
    import DeepConvLSTM_py3 as M
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net = M.DeepConvLSTM(n_channels=xtr.shape[-1], n_classes=n_classes,
                         dataset="opportunity").to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.1)
    xt = torch.tensor(xtr, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    xv = torch.tensor(xva, device=device)
    g = torch.Generator().manual_seed(seed)
    best, state = -1.0, None
    for _ in range(epochs):
        net.train()
        order = torch.randperm(len(xt), generator=g).to(device)
        for i in range(0, len(order), 256):
            j = order[i:i + 256]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(xt[j])[-1], yt[j]).backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(xv[i:i + 512])[-1].argmax(1)
                            for i in range(0, len(xv), 512)]).cpu().numpy()
        acc = float((pv == yva).mean())
        if acc > best:
            best, state = acc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()

    def predict(x):
        with torch.no_grad():
            xs = torch.tensor(x, device=device)
            return np.concatenate([
                torch.softmax(net(xs[i:i + 512])[-1], -1).cpu().numpy()
                for i in range(0, len(xs), 512)]).astype(np.float64)
    return predict, best, state


def fit_block_pca(x, blocks, n_comp=6):
    pca = {}
    for b, (lo, hi) in enumerate(blocks):
        w = x[:, :, lo:hi]
        f = np.concatenate([w.mean(1), w.std(1)], 1)
        f = f - f.mean(0)
        _, _, vt = np.linalg.svd(f, full_matrices=False)
        pca[b] = vt[:min(n_comp, vt.shape[0])].T
    return pca


def block_features(x, blocks, obs, pca):
    parts = []
    for b in obs:
        lo, hi = blocks[b]
        w = x[:, :, lo:hi]
        parts.append(np.concatenate([w.mean(1), w.std(1)], 1) @ pca[b])
    return np.concatenate(parts, 1)


def save_checkpoint(path, state, role, seed, n_classes, n_channels, blocks, cfgs, obs, pca, val_acc):
    """Save architecture and preprocessing metadata needed for inference."""
    torch.save({"format": "opportunity_dcl_v2", "role": role, "seed": seed,
                "n_classes": n_classes, "n_channels": n_channels, "dataset": "opportunity",
                "state_dict": state, "blocks": blocks, "configs": cfgs,
                "observed_blocks": obs, "pca": pca,
                "input_normalization": {"kind": "identity", "note": "prepared NPZ is already normalised"},
                "best_validation_accuracy": val_acc}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--configs", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    sys.path.insert(0, str(a.repo))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    z = np.load(a.data, allow_pickle=True)
    xtr_all, ytr_all = z["train_x"], z["train_y"]
    splits = {"deploy": (z["deploy_x"], z["deploy_y"]),
              "calib": (z["calib_x"], z["calib_y"])}
    blocks = [tuple(b) for b in z["blocks"]]
    n_classes = int(max(ytr_all.max(), z["deploy_y"].max(), z["calib_y"].max())) + 1
    cfgs = json.loads(str(np.load(a.configs, allow_pickle=True)["configs"]))
    print(f"train {xtr_all.shape}, deploy {splits['deploy'][0].shape}, "
          f"calib {splits['calib'][0].shape}, {n_classes} lop", flush=True)

    pca = fit_block_pca(xtr_all, blocks)
    a.out.mkdir(parents=True, exist_ok=True)
    all_blocks = list(range(len(blocks)))

    def mask(x, obs):
        out = np.zeros_like(x)
        for b in obs:
            lo, hi = blocks[b]
            out[:, :, lo:hi] = x[:, :, lo:hi]
        return out

    rows = []
    manifest = {"format": "opportunity_dcl_v2", "data": str(a.data), "epochs": a.epochs,
                "seeds": a.seeds, "input_normalization": {"kind": "identity", "note": "prepared input is already normalised"},
                "configs": {name: {"observed_blocks": obs, "seeds": {}} for name, obs in cfgs.items()}}
    run_started = time.time()
    for seed in a.seeds:
        rng = np.random.default_rng(500 + seed)
        p = rng.permutation(len(ytr_all))
        cut = int(0.85 * len(p))
        tr_i, va_i = p[:cut], p[cut:]

        t0 = time.time()
        full_pred, vacc, full_state = train_host(xtr_all[tr_i], ytr_all[tr_i], xtr_all[va_i],
                                                 ytr_all[va_i], n_classes, seed, a.epochs, a.device)
        save_checkpoint(a.out / f"checkpoint_full_s{seed}.pt", full_state, "full", seed,
                        n_classes, xtr_all.shape[-1], blocks, cfgs, all_blocks, pca, vacc)
        print(f"  seed {seed} host day du: val acc {vacc:.4f}, "
              f"{time.time() - t0:.0f}s", flush=True)
        # the richer outputs cross-mask needs: full host, unmasked input
        for sp, (x, _) in splits.items():
            np.save(a.out / f"richer_{sp}_s{seed}.npy", full_pred(x))

        for name, obs in cfgs.items():
            t0 = time.time()
            spec_pred, spec_vacc, spec_state = train_host(mask(xtr_all, obs)[tr_i], ytr_all[tr_i],
                                                           mask(xtr_all, obs)[va_i], ytr_all[va_i],
                                                           n_classes, seed, a.epochs, a.device)
            ckpt = a.out / f"checkpoint_condition_specialist_{name}_s{seed}.pt"
            save_checkpoint(ckpt, spec_state, "condition_specialist", seed, n_classes,
                            xtr_all.shape[-1], blocks, cfgs, obs, pca, spec_vacc)
            entry = {"checkpoint": ckpt.name, "best_validation_accuracy": spec_vacc, "dumps": {}}
            print(f"    {name} chuyen gia: {time.time() - t0:.0f}s", flush=True)
            for design, pr in (("condition_specialist", spec_pred),
                               ("full_masked", full_pred)):
                for sp, (x, y) in splits.items():
                    probs = pr(mask(x, obs))
                    filename = f"probs_{design}_{name}_{sp}_s{seed}.npy"
                    np.save(a.out / filename, probs)
                    entry["dumps"][filename] = list(probs.shape)
                    if sp == "deploy":
                        pred = probs.argmax(1)
                        acc, wf1, mf1 = float((pred == y).mean()), f1(pred, y), f1(pred, y, False)
                        rows.append(dict(design=design, config=name, seed=seed, acc=round(acc, 4),
                                         weighted_f1=round(wf1, 4), macro_f1=round(mf1, 4), n_deploy=len(y)))
                        if design == "condition_specialist":
                            entry.update(base_accuracy=acc, weighted_f1=wf1, macro_f1=mf1)
            manifest["configs"][name]["seeds"][str(seed)] = entry

    for name, obs in cfgs.items():
        for sp, (x, _) in splits.items():
            np.save(a.out / f"retfeat_{name}_{sp}.npy",
                    block_features(mask(x, obs), blocks, obs, pca))
    for sp, (_, y) in splits.items():
        np.save(a.out / f"{sp}_y.npy", y)

    with open(a.out / "host_quality.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    manifest["wall_clock_seconds"] = time.time() - run_started
    with open(a.out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True); fh.write("\n")
    print(f"\n{'design':21s} {'config':22s} {'acc':>7s} {'w-F1':>7s}")
    for design in ("condition_specialist", "full_masked"):
        for name in cfgs:
            v = [r for r in rows if r["design"] == design and r["config"] == name]
            g = lambda c: float(np.mean([x[c] for x in v]))
            print(f"{design:21s} {name:22s} {g('acc'):7.4f} {g('weighted_f1'):7.4f}")
    print("DCL_HOSTS2_DONE")


if __name__ == "__main__":
    main()
