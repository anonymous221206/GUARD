#!/usr/bin/env python3
"""Regenerate OPPORTUNITY DCL v2 per-sample dumps from a released checkpoint."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

def mask(x, blocks, obs):
    out = np.zeros_like(x)
    for b in obs:
        lo, hi = blocks[b]; out[:, :, lo:hi] = x[:, :, lo:hi]
    return out

def predict(net, x, device):
    with torch.no_grad():
        xs = torch.tensor(x, device=device)
        return np.concatenate([torch.softmax(net(xs[i:i+512])[-1], -1).cpu().numpy()
                               for i in range(0, len(xs), 512)]).astype(np.float64)

def features(x, blocks, obs, pca):
    return np.concatenate([np.concatenate([x[:, :, blocks[b][0]:blocks[b][1]].mean(1),
                                            x[:, :, blocks[b][0]:blocks[b][1]].std(1)], 1) @ pca[b]
                           for b in obs], 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); sys.path.insert(0, str(a.repo))
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("format") != "opportunity_dcl_v2": raise ValueError("not an opportunity_dcl_v2 checkpoint")
    import DeepConvLSTM_py3 as model
    net = model.DeepConvLSTM(n_channels=ckpt["n_channels"], n_classes=ckpt["n_classes"],
                             dataset=ckpt["dataset"]).to(a.device)
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    z = np.load(a.data, allow_pickle=True)
    splits = {"deploy": (z["deploy_x"], z["deploy_y"]), "calib": (z["calib_x"], z["calib_y"])}
    blocks = [tuple(b) for b in ckpt["blocks"]]; configs = ckpt["configs"]; seed = ckpt["seed"]
    a.out.mkdir(parents=True, exist_ok=True)
    if ckpt["role"] == "full":
        for sp, (x, _) in splits.items(): np.save(a.out / f"richer_{sp}_s{seed}.npy", predict(net, x, a.device))
        for name, obs in configs.items():
            for sp, (x, _) in splits.items():
                np.save(a.out / f"probs_full_masked_{name}_{sp}_s{seed}.npy", predict(net, mask(x, blocks, obs), a.device))
                np.save(a.out / f"retfeat_{name}_{sp}.npy", features(mask(x, blocks, obs), blocks, obs, ckpt["pca"]))
        for sp, (_, y) in splits.items(): np.save(a.out / f"{sp}_y.npy", y)
    elif ckpt["role"] == "condition_specialist":
        names = [n for n, obs in configs.items() if list(obs) == list(ckpt["observed_blocks"])]
        if len(names) != 1: raise ValueError("checkpoint observed block set does not identify one configuration")
        name = names[0]
        for sp, (x, _) in splits.items():
            np.save(a.out / f"probs_condition_specialist_{name}_{sp}_s{seed}.npy",
                    predict(net, mask(x, blocks, ckpt["observed_blocks"]), a.device))
    else: raise ValueError(f"unknown checkpoint role: {ckpt['role']}")
if __name__ == "__main__": main()

