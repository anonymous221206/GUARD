#!/usr/bin/env python3
"""AVE: reproduce the paper's train-pool, video-split GUARD protocol.

    python experiments/exp_ave.py --hosts artifacts/ave_av_att
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guard import HostOutputs, run  # noqa: E402
from guard.splits import Split  # noqa: E402
from hosts.ave import CONDITIONS, build  # noqa: E402


def paper_split(n_pool: int, n_video: int, n_segment: int, seed: int) -> Split:
    video_ids = np.repeat(np.arange(n_video), n_segment)
    fit_videos, conf_videos, test_videos = np.array_split(
        np.random.default_rng(seed).permutation(n_video), 3
    )

    def segments(videos: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.isin(video_ids, videos)) + n_pool

    return Split(np.arange(n_pool), segments(fit_videos), segments(conf_videos),
                 segments(test_videos), origin={
                     "pool": "AVE train videos", "fit": "AVE test videos",
                     "conf": "AVE test videos", "test": "AVE test videos",
                 })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts", type=Path, nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=list(CONDITIONS))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    for host_dir in args.hosts:
        out_dir = args.out / f"ave_{host_dir.name}_pool-train"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        print(f"\n{host_dir.name}: AVE train pool, L2-input features, video split")
        print(f"{'condition':12s} {'seed':>4s} {'base':>7s} {'GUARD':>7s} {'gain':>8s} "
              f"{'harm':>6s} {'apply':>6s} {'blanket':>8s} {'bl.harm':>8s} {'beta':>5s}")
        for condition in args.conditions:
            probs, features, labels, n_pool, n_video, n_segment = build(host_dir, condition)
            host = HostOutputs(probs=probs, features=features, labels=labels)
            for seed in args.seeds:
                split = paper_split(n_pool, n_video, n_segment, seed)
                result = run(host, split, condition=condition, target="hard",
                             alpha=args.alpha, delta=args.delta, k=args.k,
                             beta_objective="crossfit")
                guard_accuracy = result.base_metric + result.gate_metric_delta
                rows.append({**result.as_row(), "seed": seed, "host": host_dir.name,
                             "split_unit": "video", "retrieval_pool": "AVE train",
                             "retrieval_features": "input to frozen L2"})
                print(f"{condition:12s} {seed:4d} {result.base_metric:7.4f} "
                      f"{guard_accuracy:7.4f} {result.gate_metric_delta:+8.4f} "
                      f"{result.joint_harm:6.3f} {result.apply_rate:6.3f} "
                      f"{result.blanket_metric_delta:+8.4f} "
                      f"{result.blanket_joint_harm:8.3f} {result.beta:5.2f}")
        with open(out_dir / "guard.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle,
                                    fieldnames=[key for key in rows[0] if key != "notes"])
            writer.writeheader()
            for row in rows:
                writer.writerow({key: value for key, value in row.items() if key != "notes"})
        (out_dir / "manifest.json").write_text(json.dumps({
            "host": str(host_dir), "pool": "AVE train", "split_unit": "video",
            "retrieval_features": "input to frozen L2", "target": "hard",
            "alpha": args.alpha, "delta": args.delta, "k": args.k,
            "seeds": args.seeds, "conditions": args.conditions,
            "beta_objective": "crossfit",
        }, indent=2) + "\n")
        print(f"wrote {out_dir / 'guard.csv'}")


if __name__ == "__main__":
    main()
