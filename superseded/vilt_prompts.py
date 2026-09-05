#!/usr/bin/env python3
"""ViLT with missing-aware prompts (Lee et al., CVPR 2023) as a frozen host.

Runs the released checkpoint once under each modality condition and saves its
outputs, so every later step is pure numpy.  The model code is the authors';
we only read from it.

    # train with the authors' own entry point and task configuration
    python hosts/vilt_prompts.py train --task hateful_memes \\
        --repo data/raw/missing_aware_prompts \\
        --backbone data/raw/vilt_200k_mlm_itm.ckpt

    # then read its outputs under each condition
    python hosts/vilt_prompts.py dump --task hateful_memes \\
        --repo data/raw/missing_aware_prompts \\
        --ckpt checkpoints/vilt/hateful_memes.ckpt \\
        --out data/processed/hateful_memes

Two things here are not obvious and both cost us a day:

* ``torchmetrics`` must be imported before ``transformers``; otherwise
  ``transformers.__spec__`` is ``None`` and the import chain dies.
* the datamodule writes its missing-modality tables to a path relative to the
  current directory, so we chdir into the host repository regardless of where
  this script was launched from.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np

#: condition name -> (missing rate, which modality is dropped)
CONDITIONS = {
    "complete": (0.0, "text"),
    "textmiss": (1.0, "text"),      # text dropped, image observed
    "imgmiss": (1.0, "image"),      # image dropped, text observed
}

TASKS = {
    "food101": dict(head="food101_classifier", multilabel=False,
                    datamodule="vilt.datamodules.food101_datamodule:FOOD101DataModule"),
    "hateful_memes": dict(head="hatememes_classifier", multilabel=False,
                          datamodule="vilt.datamodules.hatememes_datamodule:HateMemesDataModule"),
    "mmimdb": dict(head="mmimdb_classifier", multilabel=True,
                   datamodule="vilt.datamodules.mmimdb_datamodule:MMIMDBDataModule"),
}


def _to_device(batch, device):
    out = {}
    for k, v in batch.items():
        import torch
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            out[k] = [t.to(device) for t in v]
        else:
            out[k] = v
    return out


def _forward(model, head, loader, device, multilabel):
    import torch
    probs, feats, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            out = model.infer(batch)
            logits = head(out["cls_feats"])
            p = torch.sigmoid(logits) if multilabel else torch.softmax(logits, -1)
            probs.append(p.float().cpu().numpy())
            feats.append(out["cls_feats"].float().cpu().numpy())
            y = batch["label"]
            labels.append(np.array(y if isinstance(y, list) else y.cpu().numpy()))
    return np.concatenate(probs), np.concatenate(feats), np.concatenate(labels)


def train(repo: Path, task: str, backbone: Path, out_ckpt: Path,
          batch_size: int, seed: int) -> None:
    """Run the host authors' training entry point unchanged.

    Their config already fixes the missing-modality regime (70% missing-both);
    we only pass the task, the pre-trained backbone and a seed.
    """
    import subprocess
    cmd = ["python", "run.py", "with", f"task_finetune_{task}",
           f"data_root={(repo / 'datasets' / task).resolve()}",
           "num_gpus=1", "num_nodes=1", f"per_gpu_batchsize={batch_size}",
           f"load_path={backbone.resolve()}", f"seed={seed}",
           f"exp_name=guard_{task}_s{seed}"]
    print("  " + " ".join(cmd) + f"   (cwd={repo})")
    subprocess.run(cmd, cwd=repo, check=True)
    found = sorted(repo.glob(f"result/guard_{task}_s{seed}*/version_*/checkpoints/epoch=*.ckpt"))
    if not found:
        raise SystemExit(f"training produced no checkpoint under {repo}/result")
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    out_ckpt.write_bytes(found[-1].read_bytes())
    print(f"  checkpoint -> {out_ckpt}")
    print("  NOTE  this benchmark has large seed variance; see docs/REPRODUCTION.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="dump", choices=["train", "dump"])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--backbone", type=Path, default=Path("data/raw/vilt_200k_mlm_itm.ckpt"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--pool-size", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ckpt = a.ckpt or Path("checkpoints/vilt") / f"{a.task}.ckpt"
    if a.mode == "train":
        train(a.repo.resolve(), a.task, a.backbone, ckpt, a.batch_size, a.seed)
        return
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} not found -- run `hosts/vilt_prompts.py train ...` "
                         "or fetch it with data/download_artifacts.sh")
    a.ckpt = ckpt
    a.out = a.out or Path("data/processed") / a.task

    repo = a.repo.resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)                     # the datamodule writes relative paths

    from torchmetrics.functional import f1_score  # noqa: F401  must precede transformers
    import importlib
    import torch
    from torch.utils.data import DataLoader, Subset
    from vilt.modules import ViLTransformerSS

    spec = TASKS[a.task]
    mod, cls = spec["datamodule"].split(":")
    DataModule = getattr(importlib.import_module(mod), cls)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu")
    base_cfg = dict(ck["hyper_parameters"]["config"] if "hyper_parameters" in ck else ck["config"])
    base_cfg.update(test_only=True, num_workers=8, per_gpu_batchsize=a.batch_size,
                    simulate_missing=False, load_path="")

    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = None
    summary = {}
    for name, (rate, kind) in CONDITIONS.items():
        cfg = copy.deepcopy(base_cfg)
        cfg["test_ratio"], cfg["test_type"] = rate, kind
        cfg["missing_ratio"] = {s: rate for s in ("train", "val", "test")}
        cfg["missing_type"] = {s: kind for s in ("train", "val", "test")}
        dm = DataModule(cfg)
        dm.setup("fit")
        dm.setup("test")
        if model is None:
            model = ViLTransformerSS(cfg)
            miss = model.load_state_dict(ck["state_dict"], strict=False)
            print(f"loaded checkpoint: {len(miss.missing_keys)} missing, "
                  f"{len(miss.unexpected_keys)} unexpected keys")
            model.to(device).eval()
        head = getattr(model, spec["head"])

        def loader(ds, idx=None):
            base = Subset(ds, idx) if idx is not None else ds
            return DataLoader(base, batch_size=a.batch_size, shuffle=False, num_workers=8,
                              collate_fn=partial(ds.collate, mlm_collator=dm.mlm_collator))

        train_ds = dm.train_dataset
        # random permutation, never the first N: these tables are class-sorted
        pool_idx = np.sort(np.random.default_rng(a.seed)
                           .permutation(len(train_ds))[: min(a.pool_size, len(train_ds))]).tolist()
        parts = {}
        for role, dl in (("val", loader(dm.val_dataset)),
                         ("test", loader(dm.test_dataset)),
                         ("src", loader(train_ds, pool_idx))):
            p, f, y = _forward(model, head, dl, device, spec["multilabel"])
            parts[f"{role}_probs"], parts[f"{role}_feats"], parts[f"{role}_labels"] = p, f, y
        np.savez_compressed(out / f"extract_{name}.npz", **parts)
        acc = float((parts["test_probs"] > 0.5).astype(int).__eq__(parts["test_labels"]).mean()
                    if spec["multilabel"] else
                    (parts["test_probs"].argmax(1) == parts["test_labels"]).mean())
        n_classes = int(len(np.unique(parts["src_labels"]))) if not spec["multilabel"] else -1
        summary[name] = {"test_accuracy": acc, "pool": len(pool_idx),
                         "pool_classes": n_classes}
        print(f"  {name:10s} test {acc:.4f}   pool {len(pool_idx)}"
              + (f" covering {n_classes} classes" if n_classes > 0 else ""))

    (out / "manifest.json").write_text(json.dumps(
        {"task": a.task, "checkpoint": str(a.ckpt), "seed": a.seed,
         "conditions": summary}, indent=2))
    print(f"dumps written to {out}")


if __name__ == "__main__":
    main()
