#!/usr/bin/env python3
"""DrugBAN adapter: a published drug--target host, frozen, under realistic degradation.

Drug--target prediction has no established missing-modality protocol, so we
define one from domain practice rather than zeroing a whole branch (which
collapses the host and would be a straw man):

``full``             unchanged
``prot50``/``prot25`` protein truncated to the first 50% / 25% of residues --
                     unresolved regions are routine in deposited structures
``scaffold``         drug reduced to its Murcko scaffold: the core is known, the
                     substituents are not, as in early-stage screening
``scaffold_prot50``  both

Roughly 39% of compounds are acyclic and have an empty Murcko scaffold; those
are left unchanged, which makes ``scaffold`` a *conservative* degradation.  The
exact fraction is printed and recorded.

Usage::

    # train the host with the authors' own code and configuration
    python hosts/drugban.py train --dataset biosnap --split random --seed 42

    # then read its outputs under each degradation
    python hosts/drugban.py dump --dataset biosnap --split random --seed 42 \
        --ckpt checkpoints/drugban/biosnap_s42.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CONDITIONS = ("full", "prot50", "prot25", "scaffold", "scaffold_prot50")


def _murcko(smiles: str, cache: dict) -> tuple[str, bool]:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    if smiles in cache:
        return cache[smiles]
    out, ok = smiles, False
    try:
        s = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
        if s and Chem.MolFromSmiles(s) is not None:
            out, ok = s, True
    except Exception:
        pass
    cache[smiles] = (out, ok)
    return out, ok


def degrade(df, condition: str, cache: dict | None = None):
    """Apply one deployment degradation; returns (dataframe, n_scaffold_fallbacks)."""
    cache = {} if cache is None else cache
    d = df.copy()
    fallbacks = 0
    if condition.startswith("scaffold"):
        outs = [_murcko(s, cache) for s in d["SMILES"]]
        d["SMILES"] = [o[0] for o in outs]
        fallbacks = sum(1 for o in outs if not o[1])
    if "prot50" in condition:
        d["Protein"] = [p[: max(1, len(p) // 2)] for p in d["Protein"]]
    if "prot25" in condition:
        d["Protein"] = [p[: max(1, len(p) // 4)] for p in d["Protein"]]
    return d.reset_index(drop=True), fallbacks


RANDOM_CFG = """SOLVER:
  BATCH_SIZE: 64
  MAX_EPOCH: 100
  LR: 5e-5
  SEED: {seed}
DA:
  TASK: False
  USE: False
DECODER:
  BINARY: 1
RESULT:
  OUTPUT_DIR: "{out}"
COMET:
  USE: False
"""


def write_config(repo: Path, split: str, seed: int, out_dir: Path, path: Path) -> Path:
    """The authors' configuration with only the seed and output path set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if split == "cluster":
        # their cross-domain configuration, verbatim apart from seed/output
        base = (repo / "configs" / "DrugBAN_Non_DA.yaml").read_text()
        base = "\n".join(f"  SEED: {seed}" if ln.strip().startswith("SEED:") else ln
                          for ln in base.splitlines())
        base += f'\n\nRESULT:\n  OUTPUT_DIR: "{out_dir}"\nCOMET:\n  USE: False\n'
        path.write_text(base)
    else:
        path.write_text(RANDOM_CFG.format(seed=seed, out=out_dir))
    return path


def train(repo: Path, dataset: str, split: str, seed: int, out_ckpt: Path) -> None:
    """Run the host authors' training entry point unchanged."""
    import subprocess
    work = out_ckpt.parent / f"train_{dataset}_{split}_s{seed}"
    cfg = write_config(repo, split, seed, work.resolve(),
                       out_ckpt.parent / f"{out_ckpt.stem}.yaml")
    cmd = ["python", "main.py", "--cfg", str(cfg.resolve()),
           "--data", dataset, "--split", split]
    print("  " + " ".join(cmd) + f"   (cwd={repo})")
    subprocess.run(cmd, cwd=repo, check=True)
    best = sorted(work.glob("best_model_epoch_*.pth"))
    if not best:
        raise SystemExit(f"training produced no checkpoint in {work}")
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    out_ckpt.write_bytes(best[-1].read_bytes())
    print(f"  checkpoint -> {out_ckpt}")


def _load_model(repo: Path, cfg_path: Path, ckpt: Path, device: str):
    sys.path.insert(0, str(repo))
    import torch
    from configs import get_cfg_defaults
    from models import DrugBAN
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(cfg_path))
    model = DrugBAN(**cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, cfg


def _forward(model, df, device: str, batch_size: int = 64):
    """Host probabilities and the fused representation used for retrieval."""
    import torch
    from dataloader import DTIDataset
    from torch.utils.data import DataLoader
    from utils import graph_collate_func
    ds = DTIDataset(df.index.values, df)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                    collate_fn=graph_collate_func)
    P, F, Y = [], [], []
    with torch.no_grad():
        for bg, vp, y in dl:
            _, _, f, score = model(bg.to(device), vp.to(device))
            if score.shape[-1] == 2:                     # cluster configs use 2 logits
                P.append(torch.softmax(score, -1).float().cpu().numpy())
            else:
                pr = torch.sigmoid(score).squeeze(-1).float().cpu().numpy()
                P.append(np.stack([1.0 - pr, pr], 1))
            F.append(f.float().cpu().numpy())
            Y.append(np.asarray(y))
    return (np.concatenate(P).astype(np.float64), np.concatenate(F),
            np.concatenate(Y).astype(np.int64))


def dump(repo: Path, dataset: str, split: str, seed: int, ckpt: Path,
         cfg_path: Path, out: Path, pool_size: int = 12000) -> None:
    """Write one ``.npz`` per condition, ready for ``experiments/exp_drugban.py``."""
    import pandas as pd
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = _load_model(repo, cfg_path, ckpt, device)
    folder = repo / "datasets" / dataset / split
    if split == "cluster":
        pool_df = pd.read_csv(folder / "source_train.csv")
        calib_df = pd.read_csv(folder / "target_train.csv")
        test_df = pd.read_csv(folder / "target_test.csv")
    else:
        pool_df = pd.read_csv(folder / "train.csv")
        calib_df = pd.read_csv(folder / "val.csv")
        test_df = pd.read_csv(folder / "test.csv")
    rng = np.random.default_rng(0)
    if len(pool_df) > pool_size:                       # random, never the first N
        pool_df = pool_df.iloc[rng.permutation(len(pool_df))[:pool_size]].reset_index(drop=True)

    out.mkdir(parents=True, exist_ok=True)
    meta = {"dataset": dataset, "split": split, "seed": seed,
            "checkpoint": str(ckpt), "scaffold_fallback": {}}
    cache: dict = {}
    for cond in CONDITIONS:
        parts = {}
        for name, df in (("pool", pool_df), ("calib", calib_df), ("test", test_df)):
            d, fb = degrade(df, cond, cache)
            p, f, y = _forward(model, d, device)
            parts[f"{name}_probs"], parts[f"{name}_feats"], parts[f"{name}_labels"] = p, f, y
            if name == "test":
                meta["scaffold_fallback"][cond] = int(fb)
        np.savez_compressed(out / f"{cond}.npz", **parts)
        acc = float((parts["test_probs"].argmax(1) == parts["test_labels"]).mean())
        print(f"  {cond:16s} test accuracy {acc:.4f}")
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(f"dumps written to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="dump", choices=["train", "dump"])
    ap.add_argument("--repo", type=Path, default=Path("data/raw/DrugBAN"))
    ap.add_argument("--dataset", required=True, choices=["human", "biosnap", "bindingdb"])
    ap.add_argument("--split", default="random", choices=["random", "cluster"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--cfg", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    tag = (f"cluster_{a.dataset}_s{a.seed}" if a.split == "cluster"
           else f"{a.dataset}_s{a.seed}")
    ckpt = a.ckpt or Path("checkpoints/drugban") / f"{tag}.pth"

    if a.mode == "train":
        train(a.repo, a.dataset, a.split, a.seed, ckpt)
        return

    cfg = a.cfg or ckpt.with_suffix(".yaml")
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} not found -- run `hosts/drugban.py train ...` "
                         "or fetch it with data/download_artifacts.sh")
    if not cfg.exists():
        cfg = write_config(a.repo, a.split, a.seed, Path("results/tmp").resolve(), cfg)
    out = a.out or Path("data/processed") / f"drugban_{a.dataset}_{a.split}_s{a.seed}"
    dump(a.repo, a.dataset, a.split, a.seed, ckpt, cfg, out)


if __name__ == "__main__":
    main()
