#!/usr/bin/env python3
"""Export frozen PTB-XL ``resnet1d_wang`` outputs into one GUARD artifact.

PTB-XL has twelve simultaneous ECG leads rather than native audio/text/video
modalities.  The explicit engineering adapter maps limb leads to ``a``, V1--V3
to ``t``, and V4--V6 to ``v``.  This script saves the frozen host's probability
outputs and pre-pooling embedding for every resulting mask.  The latter is the
retrieval representation used by GUARD; ``hosts.ptbxl`` must consume it instead
of concatenating the three compact per-group arrays in ``raw_features.npz``.
Those per-group arrays use the host head's adaptive max+mean pooling and are
only 256-dimensional, matching the compact feature style of other artifacts.

The source checkpoint and raw records are read-only.  Output is first written
to a new sibling temporary directory and atomically renamed only on success, so
an interrupted export never leaves a partial artifact at ``--out``.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path("$WORKSPACE/external/ecg_ptbxl_benchmarking/code")
ROOT = Path("$WORKSPACE/data/ptbxl")
CKPT = Path("$WORKSPACE/results/ptbxl_wang/resnet1d_wang.pt")
CLASSES = ("CD", "HYP", "MI", "NORM", "STTC")
LEAD_GROUPS = {"a": (0, 1, 2, 3, 4, 5), "t": (6, 7, 8), "v": (9, 10, 11)}
MASKS = ("a", "v", "t", "av", "at", "tv", "atv")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path,
                        help="new artifact directory; existing paths are refused")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def labels_inputs_and_folds() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd
    import wfdb

    database = pd.read_csv(ROOT / "ptbxl_database.csv", index_col="ecg_id")
    statements = pd.read_csv(ROOT / "scp_statements.csv", index_col=0)
    diagnostic = statements[statements.diagnostic == 1]
    superclass = database.scp_codes.apply(ast.literal_eval).apply(
        lambda codes: sorted({diagnostic.loc[key].diagnostic_class
                              for key in codes if key in diagnostic.index})
    )
    database = database.assign(superclass=superclass)
    database = database[database.superclass.apply(len) >= 1]
    labels = np.zeros((len(database), len(CLASSES)), dtype=np.float32)
    for row, classes in enumerate(database.superclass):
        for name in classes:
            labels[row, CLASSES.index(name)] = 1.0
    records_root = next(path.parent for path in ROOT.rglob("records100") if path.is_dir())
    signals = np.stack([wfdb.rdsamp(str(records_root / record))[0].astype(np.float32)
                        for record in database.filename_lr]).transpose(0, 2, 1)
    return signals, labels, database.strat_fold.to_numpy()


def channels(mask: str) -> list[int]:
    return sorted(channel for group in mask for channel in LEAD_GROUPS[group])


def macro_auc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(np.mean([roc_auc_score(labels[:, column], probabilities[:, column])
                          for column in range(labels.shape[1])
                          if labels[:, column].min() != labels[:, column].max()]))


def adaptive_concat_pool(values: np.ndarray) -> np.ndarray:
    """Apply the frozen host head's first pooling layer to 128 x 250 activations."""
    values = values.reshape(len(values), 128, 250)
    return np.concatenate([values.max(2), values.mean(2)], axis=1).astype(np.float32)


def main() -> None:
    args = arguments()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.out}")
    if not CKPT.is_file() or not ROOT.is_dir() or not REPO.is_dir():
        raise FileNotFoundError("checkpoint, PTB-XL root, or benchmark source is unavailable")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from torch import nn

    sys.path.insert(0, str(REPO))
    from models.resnet1d import resnet1d_wang

    checkpoint = torch.load(CKPT, map_location="cpu")
    signals, labels, folds = labels_inputs_and_folds()
    train, sess1 = folds <= 8, folds == 10
    signals = (signals - checkpoint["mu"]) / checkpoint["sd"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = resnet1d_wang(num_classes=5, input_channels=12, kernel_size=5,
                            ps_head=0.5, lin_ftrs_head=[128]).to(device)
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()
    trunk = nn.Sequential(*list(network.children())[:-1]).eval()

    def infer(array: np.ndarray, model) -> np.ndarray:
        values = []
        with torch.no_grad():
            for start in range(0, len(array), args.batch_size):
                batch = torch.as_tensor(array[start:start + args.batch_size], device=device)
                values.append(model(batch).float().cpu().numpy())
        return np.concatenate(values)

    predictions = {"train_y": labels[train], "sess1_y": labels[sess1]}
    embeddings: dict[str, np.ndarray] = {}
    for mask in MASKS:
        masked = np.zeros_like(signals)
        masked[:, channels(mask)] = signals[:, channels(mask)]
        for split, selected in (("train", train), ("sess1", sess1)):
            logits = infer(masked[selected], network)
            predictions[f"{split}_{mask}"] = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
            embeddings[f"{split}_{mask}"] = infer(masked[selected], trunk).reshape(int(selected.sum()), -1).astype(np.float32)
        print(f"exported mask {mask}", flush=True)

    raw = {"train_y": labels[train], "sess1_y": labels[sess1]}
    for split in ("train", "sess1"):
        for group in LEAD_GROUPS:
            raw[f"{split}_{group}"] = adaptive_concat_pool(embeddings[f"{split}_{group}"])

    temporary = Path(tempfile.mkdtemp(prefix=f".{args.out.name}.partial.",
                                      dir=args.out.parent))
    np.savez_compressed(temporary / "raw_features.npz", **raw)
    np.savez_compressed(temporary / "preds.npz", **predictions)
    np.savez_compressed(temporary / "masked_embeddings.npz", **embeddings)
    metadata = {
        "host": "ptbxl_resnet1d_wang",
        "checkpoint": str(CKPT),
        "classes": CLASSES,
        "labels": "five-dimensional multi-label diagnostic-superclass vector",
        "train_split": "PTB-XL stratified folds 1--8",
        "sess1_split": "PTB-XL stratified fold 10",
        "lead_groups": LEAD_GROUPS,
        "retrieval_features": "masked_embeddings.npz; exact frozen-host embedding per mask",
        "full_mask_test_macro_auc": macro_auc(predictions["sess1_atv"], labels[sess1]),
    }
    (temporary / "adapter.json").write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.rename(args.out)
    print(f"full-mask fold-10 macro AUC {metadata['full_mask_test_macro_auc']:.10f}")
    print(f"artifact written to {args.out}")


if __name__ == "__main__":
    main()
