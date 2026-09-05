"""Adapter and one-time exporter for the AVE AV-att artifacts.

The paper protocol retrieves from the AVE training split using the input to the
frozen model's final ``L2`` layer. The posterior dumps alone are insufficient:
``export`` adds one retrieval sidecar per condition. Experiments after that are
NumPy-only and never load a checkpoint or the AVE source data.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import weakref
from pathlib import Path

import numpy as np

CONDITIONS = ("audio_only", "visual_only")


def _single_match(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{directory}: expected exactly one {pattern!r}, found {matches}"
        )
    return matches[0]


def _dump_path(host_dir: Path, condition: str) -> Path:
    return _single_match(host_dir / "dumps", f"*_{condition}.npz")


def _retrieval_path(host_dir: Path, condition: str, variant: str = "") -> Path:
    dump = _dump_path(host_dir, condition)
    suffix = f"_{variant}" if variant else ""
    return dump.with_name(f"{dump.stem}_retrieval{suffix}.npz")


def build(host_dir: Path, condition: str):
    """Return paper-protocol arrays in one ``[train pool | deployment]`` space."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown AVE condition {condition!r}; choose from {CONDITIONS}")
    dump = np.load(_dump_path(host_dir, condition))
    paper_path = _retrieval_path(host_dir, condition, "paper")
    retrieval_path = paper_path if paper_path.is_file() else _retrieval_path(host_dir, condition)
    if not retrieval_path.is_file():
        raise FileNotFoundError(
            f"{retrieval_path} is required: posterior-only AVE dumps cannot reproduce "
            "the paper's train-pool/L2-feature protocol; run hosts/ave.py export first"
        )
    retrieval = np.load(retrieval_path)
    required = {"pool_probs", "pool_features", "pool_labels", "deploy_features"}
    missing = required.difference(retrieval.files)
    if missing:
        raise KeyError(f"{retrieval_path}: missing {sorted(missing)}")

    deploy_probs = dump["probs"].reshape(-1, dump["probs"].shape[-1]).astype(np.float64)
    deploy_labels = dump["labels"].reshape(-1, dump["labels"].shape[-1]).argmax(1)
    pool_probs = retrieval["pool_probs"].astype(np.float64)
    pool_features = retrieval["pool_features"].astype(np.float64)
    pool_labels = retrieval["pool_labels"].astype(np.int64)
    deploy_features = retrieval["deploy_features"].astype(np.float64)
    if len(deploy_probs) != len(deploy_features):
        raise ValueError(f"{retrieval_path}: deployment features and probabilities differ in size")
    n_video, n_segment = dump["probs"].shape[:2]
    return (
        np.concatenate([pool_probs, deploy_probs]),
        np.concatenate([pool_features, deploy_features]),
        np.concatenate([pool_labels, deploy_labels]),
        len(pool_labels),
        n_video,
        n_segment,
    )


def _patch_old_checkpoint(torch, cpu_only: bool) -> None:
    import torch.nn as nn

    def rnn_setstate(module, state):
        module.__dict__.update(state)
        module.__dict__.setdefault("proj_size", 0)
        module._flat_weights_names = [name for group in module._all_weights for name in group]
        module._flat_weights = [getattr(module, name, None) for name in module._flat_weights_names]
        module._flat_weight_refs = [weakref.ref(weight) if weight is not None else None
                                    for weight in module._flat_weights]

    nn.RNNBase.__setstate__ = rnn_setstate
    if cpu_only:
        # The 2018 forward hard-codes .cuda() for two zero hidden states.
        torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor


def _repair_modules(model) -> None:
    import torch.nn as nn

    template = nn.Module().__dict__
    for module in model.modules():
        for key, value in template.items():
            if key not in module.__dict__:
                module.__dict__[key] = type(value)() if isinstance(value, (dict, set, list)) else value


def export_retrieval(host_dir: Path, source: Path, conditions: list[str], batch_size: int,
                     cuda_device: int | None, variant: str) -> None:
    """Create missing retrieval sidecars without modifying existing files."""
    host_dir = host_dir.resolve()
    source = source.resolve()
    outputs = [_retrieval_path(host_dir, condition, variant) for condition in conditions]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing AVE sidecars: {existing}")

    import h5py
    import torch

    sys.path.insert(0, str(source))
    previous = Path.cwd()
    os.chdir(source)
    try:
        cpu_only = cuda_device is None
        if not cpu_only:
            torch.cuda.set_device(cuda_device)
        _patch_old_checkpoint(torch, cpu_only)
        checkpoint = _single_match(host_dir / "checkpoints", "*.pt")
        device = torch.device("cpu" if cpu_only else f"cuda:{cuda_device}")
        model = torch.load(checkpoint, map_location=device, weights_only=False)
        _repair_modules(model)
        model.to(device)
        model.eval()

        data = source / "data"
        with h5py.File(data / "train_order.h5", "r") as handle:
            train_order = handle["order"][:]
        with h5py.File(data / "test_order.h5", "r") as handle:
            test_order = handle["order"][:]
        with h5py.File(data / "audio_feature.h5", "r") as handle:
            audio_all = handle["avadataset"][:]
        with h5py.File(data / "labels.h5", "r") as handle:
            labels_all = handle["avadataset"][:]

        # Match ave_guard.py: 52 complete batches, dropping the final 11 train
        # videos because its loop uses n_pool_vid // 64.
        train_order = train_order[: (len(train_order) // batch_size) * batch_size]
        orders = {"pool": train_order, "deploy": test_order}
        visual_file = h5py.File(data / "visual_feature.h5", "r")
        visual_all = visual_file["avadataset"]
        try:
            for condition, output_path in zip(conditions, outputs):
                keep_audio = condition == "audio_only"
                keep_visual = condition == "visual_only"
                captured: dict[str, np.ndarray] = {}
                hook = model.L2.register_forward_hook(
                    lambda module, inputs, output: captured.__setitem__(
                        "features", inputs[0].detach().cpu().numpy()
                    )
                )
                exported: dict[str, np.ndarray] = {}
                try:
                    for split, order in orders.items():
                        probs, features, labels = [], [], []
                        step = batch_size if split == "pool" else len(order)
                        for start in range(0, len(order), step):
                            selected = order[start:start + step]
                            audio = audio_all[selected] if keep_audio else np.zeros(
                                (len(selected), 10, 128), dtype=np.float32
                            )
                            if keep_visual:
                                visual = np.stack([visual_all[int(index)] for index in selected])
                            else:
                                visual = np.zeros(
                                    (len(selected), 10, 7, 7, 512), dtype=np.float32
                                )
                            with torch.no_grad():
                                prediction = model(torch.from_numpy(audio).float().to(device),
                                                   torch.from_numpy(visual).float().to(device))
                            prediction = prediction.cpu()
                            probs.append(prediction.numpy().reshape(-1, prediction.shape[-1]))
                            features.append(captured["features"].reshape(
                                -1, captured["features"].shape[-1]
                            ))
                            labels.append(labels_all[selected].reshape(
                                -1, labels_all.shape[-1]
                            ).argmax(1))
                        exported[f"{split}_probs"] = np.concatenate(probs).astype(np.float32)
                        exported[f"{split}_features"] = np.concatenate(features).astype(np.float32)
                        exported[f"{split}_labels"] = np.concatenate(labels).astype(np.int16)
                finally:
                    hook.remove()

                saved = np.load(_dump_path(host_dir, condition))
                saved_probs = saved["probs"].reshape(-1, saved["probs"].shape[-1])
                difference = np.abs(exported["deploy_probs"] - saved_probs)
                agreement = float((exported["deploy_probs"].argmax(1)
                                   == saved_probs.argmax(1)).mean())
                # CPU and the original GPU inference differ at float32 rounding
                # scale, but must make exactly the same decisions.
                if difference.max() > 1e-4 or agreement != 1.0:
                    raise ValueError(
                        f"export does not reproduce saved {condition} posteriors: "
                        f"max_abs={difference.max():.8g}, argmax_agreement={agreement:.8f}"
                    )
                payload = {
                    "pool_probs": exported["pool_probs"],
                    "pool_features": exported["pool_features"],
                    "pool_labels": exported["pool_labels"],
                    "deploy_features": exported["deploy_features"],
                    "deploy_prob_max_abs": np.asarray(difference.max()),
                    "inference_backend": np.asarray(str(device)),
                }
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=output_path.parent, suffix=".npz", delete=False
                ) as handle:
                    temporary = Path(handle.name)
                np.savez_compressed(temporary, **payload)
                temporary.rename(output_path)
                print(f"wrote {output_path}", flush=True)
        finally:
            visual_file.close()
    finally:
        os.chdir(previous)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--host", type=Path, required=True)
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=list(CONDITIONS))
    export.add_argument("--batch-size", type=int, default=64)
    export.add_argument("--cuda-device", type=int)
    export.add_argument("--variant", default="")
    args = parser.parse_args()
    export_retrieval(args.host, args.source, args.conditions, args.batch_size,
                     args.cuda_device, args.variant)


if __name__ == "__main__":
    main()
