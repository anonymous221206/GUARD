#!/usr/bin/env python3
"""OPPORTUNITY MLP variant, not the DeepConvLSTM host reported in the paper's OPPORTUNITY table. Sensors drop out, and where the calibration set comes from matters.

The host is a small classifier trained on subjects 1--2 with the unavailable
sensor blocks mean-filled; deployment is subject 4.  Two calibration protocols
are available and they give completely different answers:

``cross_subject``  calibrate on subject 3, evaluate on subject 4.  This is what
                   an off-the-shelf train/val/test split gives you, and it
                   **violates exchangeability**: the certificate's assumption
                   does not hold across subjects, and the retrieval pool is
                   drawn from yet another population.
``deployment``     carve pool / fit / conf / test out of subject 4 itself.  The
                   host still never sees a subject-4 label during training; only
                   the small calibration sample comes from the deployment
                   distribution, which is exactly what the method assumes.

Switching from the first to the second takes the accuracy gain from ~0.004 to
~0.17.  ``--protocol`` selects which, and running both is the point.

    python experiments/exp_opportunity.py --features data/processed/opportunity.npz \\
        --protocol deployment --label-budget 6411
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guard import HostOutputs, run                       # noqa: E402
from guard.splits import Split                           # noqa: E402
from guard.targets import richer_is_richer               # noqa: E402


def train_mlp(x, y, x_val, y_val, n_out, seed, hidden=128, epochs=300):
    """A deliberately small host; the method never touches its weights."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, n_out)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt = torch.tensor(x, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.long, device=dev)
    best, best_state = -1.0, None
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        nn.functional.cross_entropy(net(xt), yt).backward(); opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                acc = (net(torch.tensor(x_val, dtype=torch.float32, device=dev))
                       .argmax(1).cpu().numpy() == y_val).mean()
            if acc > best:
                best, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()

    def predict(z):
        with torch.no_grad():
            u = net(torch.tensor(z, dtype=torch.float32, device=dev))
            return torch.softmax(u, -1).cpu().numpy().astype(np.float64)
    return predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True,
                    help="npz with H, y, subject, block_slices")
    ap.add_argument("--protocol", default="deployment",
                    choices=["deployment", "cross_subject"])
    ap.add_argument("--label-budget", type=int, default=0,
                    help="deployment labels for pool+fit+conf (0 = use all)")
    ap.add_argument("--fixed-host", action="store_true",
                    help="pick the host's epoch on the full fit split at every "
                         "budget, so the sweep varies only what GUARD is given")
    ap.add_argument("--decimate", type=int, default=1,
                    help="keep every d-th deployment window in temporal order. "
                         "Windows are length 24 at stride 12, so consecutive "
                         "windows share half their raw samples and a pool row "
                         "can answer a query with measurements the query "
                         "itself contains. d=2 makes the kept windows pairwise "
                         "disjoint while leaving the split a random permutation, "
                         "so exchangeability between conf and test survives")
    ap.add_argument("--decimate-offset", type=int, default=0,
                    help="which residue class to keep; offsets 0 and 1 give two "
                         "disjoint non-overlapping halves of the same data")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()

    z = np.load(a.features, allow_pickle=True)
    H, y, subject = z["H"], z["y"], z["subject"]
    blocks = list(z["block_slices"])
    configs = json.loads(str(z["configs"]))
    n_out = int(y.max()) + 1

    tag = (f"opportunity_{a.protocol}"
           + (f"_nL{a.label_budget}" if a.label_budget else "")
           + ("_fixedhost" if a.fixed_host else "")
           + (f"_dec{a.decimate}" if a.decimate > 1 else "")
           + (f"o{a.decimate_offset}" if a.decimate > 1 and a.decimate_offset else ""))
    out_dir = a.out / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    tr = np.where(np.isin(subject, [1, 2]))[0]
    val = np.where(subject == 3)[0]
    dep = np.where(subject == 4)[0]
    if a.decimate > 1:
        # rows are in temporal order, so every d-th row is a window that shares
        # no raw sample with any other kept window
        val = np.sort(val)[a.decimate_offset:: a.decimate]
        dep = np.sort(dep)[a.decimate_offset:: a.decimate]
        print(f"decimate {a.decimate} offset {a.decimate_offset}: "
              f"deploy {len(dep)}, cross-subject pool {len(val)}")

    for seed in a.seeds:
        rng = np.random.default_rng(700 + seed)
        if a.protocol == "deployment":
            perm = rng.permutation(len(dep))
            cut = 3 * (len(dep) // 4)
            avail_all, test_idx = dep[perm[:cut]], dep[perm[cut:]]
            avail = avail_all[: a.label_budget] if a.label_budget else avail_all
            t3 = len(avail) // 3
            pool, fit, conf = avail[:t3], avail[t3:2 * t3], avail[2 * t3:3 * t3]
            # the host keeps the epoch that scores best on its validation split,
            # so leaving that split inside the budget makes the frozen host a
            # function of the budget and the sweep moves two things at once
            t3_all = len(avail_all) // 3
            fit_host = avail_all[t3_all: 2 * t3_all] if a.fixed_host else fit
            origin = {k: "subject 4" for k in ("pool", "fit", "conf", "test")}
        else:
            perm = rng.permutation(len(val))
            t3 = len(val) // 3
            pool, fit, conf = val[perm[:t3]], val[perm[t3:2 * t3]], val[perm[2 * t3:3 * t3]]
            fit_host = fit
            test_idx = dep
            origin = {"pool": "subject 3", "fit": "subject 3",
                      "conf": "subject 3", "test": "subject 4"}
        split = Split(pool, fit, conf, test_idx, origin=origin)

        def masked(observed_blocks):
            keep = np.zeros(H.shape[1], dtype=bool)
            for b in observed_blocks:
                keep[blocks[b]] = True
            out = np.zeros_like(H)
            out[:, keep] = H[:, keep]
            out[:, ~keep] = H[tr][:, ~keep].mean(0)
            return out

        all_blocks = list(range(len(blocks)))
        h_rich = masked(all_blocks)
        pred_rich = train_mlp(h_rich[tr], y[tr], h_rich[fit_host], y[fit_host],
                              n_out, seed)
        richer = pred_rich(h_rich)

        for name, obs in configs.items():
            h = masked(obs)
            pred = train_mlp(h[tr], y[tr], h[fit_host], y[fit_host], n_out, seed)
            probs = pred(h)
            if seed == a.seeds[0] and name == list(configs)[0]:
                chk = richer_is_richer(richer[fit], probs[fit], y[fit])
                print(f"pre-flight on the deployment distribution: richer host "
                      f"{chk['richer_accuracy']:.4f} vs poorer {chk['poorer_accuracy']:.4f} "
                      f"-> cross-mask {'applicable' if chk['precondition_met'] else 'NOT applicable'}")
            host = HostOutputs(probs=probs, features=h[:, np.concatenate(
                [np.arange(blocks[b].start, blocks[b].stop) for b in obs])],
                labels=y, richer_probs=richer)
            for target in ("hard", "cross_mask"):
                r = run(host, split, condition=name, target=target,
                        alpha=a.alpha, delta=a.delta)
                rows.append({**r.as_row(), "seed": seed})
                print(f"seed{seed} {name:22s} {target:11s} base={r.base_metric:.4f} "
                      f"tgt={r.target_accuracy:.3f} gate={r.gate_metric_delta:+.4f} "
                      f"harm={r.joint_harm:.3f}")
                for note in r.notes:
                    print(f"    note: {note}")

    with open(out_dir / "guard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "notes"])
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != "notes"})
    viol = sum(1 for r in rows if r["joint_harm"] > a.alpha)
    print(f"\n{viol}/{len(rows)} cells exceed the harm budget -> {out_dir}/guard.csv")


if __name__ == "__main__":
    main()
