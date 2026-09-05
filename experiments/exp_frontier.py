#!/usr/bin/env python3
"""The alpha- and delta-frontiers behind the validity figure.

Two sweeps the paper reports: whether observed harm tracks alpha across the
whole range, and what the loss budget delta actually buys.

Both are pure re-gating.  Nothing is retrained -- not the hosts, not the
corrector -- and every number comes from :func:`guard.run`, the same entry point
the other experiments call, so these rows are directly comparable with the
tables.  The only thing this driver adds is that it builds each host once and
then loops the grid, instead of paying for retrieval seven times over.

    python experiments/exp_frontier.py --family drugban --grid alpha
    python experiments/exp_frontier.py --family opportunity --grid delta

One caveat when plotting the result.  The curve is monotone up to alpha ~ 0.4
and then turns over: at large alpha the conformal quantile shrinks until the
plausible set would be empty, and ``certify`` never returns an empty set, so
those points fall back to the worst case over *all* labels and are refused.
Apply rate and harm both fall as a consequence.  That is a property of the
convention, not of the guarantee, and it is better stated than left for a
reader to discover.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from guard import HostOutputs, run                    # noqa: E402
from guard.splits import Split                        # noqa: E402

ALPHAS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
DELTAS = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5)
TARGETS = ("hard", "cross_mask")

COLS = ["family", "dataset", "condition", "target", "seed", "alpha", "delta",
        "beta", "base_metric", "apply_rate", "joint_harm", "cond_harm",
        "acc_gain", "loss_gain", "blanket_acc_gain", "blanket_loss_gain",
        "blanket_joint_harm", "target_accuracy", "q_hat", "n_conf", "n_test",
        "exchangeable"]


def row_of(fam, ds, seed, r):
    return dict(family=fam, dataset=ds, condition=r.condition, target=r.target,
                seed=seed, alpha=r.alpha, delta=r.delta, beta=round(r.beta, 4),
                base_metric=round(r.base_metric, 6),
                apply_rate=round(r.apply_rate, 6),
                joint_harm=round(r.joint_harm, 6),
                cond_harm=round(r.cond_harm, 6),
                acc_gain=round(r.gate_metric_delta, 6),
                loss_gain=round(r.gate_loss_gain, 6),
                blanket_acc_gain=round(r.blanket_metric_delta, 6),
                blanket_loss_gain=round(r.blanket_loss_gain, 6),
                blanket_joint_harm=round(r.blanket_joint_harm, 6),
                target_accuracy=round(r.target_accuracy, 6),
                q_hat=round(r.q_hat, 6), n_conf=r.n_conf, n_test=r.n_test,
                exchangeable=r.exchangeable)


def cells_drugban(data, seed):
    from exp_drugban import CONDITIONS, build
    jobs = [("biosnap_random_s42", "source"), ("bindingdb_random_s42", "source"),
            ("human_random_s42", "source"),
            ("biosnap_cluster_s42", "source"), ("biosnap_cluster_s42", "deployment"),
            ("bindingdb_cluster_s42", "source"), ("bindingdb_cluster_s42", "deployment")]
    for tag, pool in jobs:
        dumps = data / f"drugban_{tag}"
        if not (dumps / "full.npz").exists():
            print(f"  skip {tag}: no dump"); continue
        for cond in CONDITIONS:
            host, split = build(dumps, cond, pool, seed)
            yield f"{tag}_pool-{pool}", cond, seed, host, split


def cells_vision(data, seed):
    from exp_vision_language import CONDITIONS, build_split, load
    for task in ("hateful_memes", "food101"):
        dumps = data / task
        if not (dumps / "extract_complete.npz").exists():
            print(f"  skip {task}: no dump"); continue
        rich = load(dumps, "complete")
        for cond in CONDITIONS:
            d = load(dumps, cond)
            n_val, n_test, n_pool = (len(d["val_labels"]), len(d["test_labels"]),
                                     len(d["src_labels"]))
            split = build_split(n_pool, n_val, n_test, seed)
            host = HostOutputs(
                probs=np.concatenate([d["src_probs"], d["val_probs"],
                                      d["test_probs"]]).astype(np.float64),
                features=np.concatenate([d["src_feats"], d["val_feats"], d["test_feats"]]),
                labels=np.concatenate([d["src_labels"], d["val_labels"], d["test_labels"]]),
                richer_probs=np.concatenate([rich["src_probs"], rich["val_probs"],
                                             rich["test_probs"]]).astype(np.float64))
            yield task, cond, seed, host, split


def cells_affective(hosts_dir, seeds):
    from hosts.dumps import MASKS, build
    for hdir in sorted(hosts_dir.glob("*/")):
        if not (hdir / "preds.npz").exists():
            continue
        for mask in MASKS:
            probs, feats, labels, richer, n_pool, n_dep = build(hdir, mask, "train")
            host = HostOutputs(probs=probs, features=feats, labels=labels,
                               richer_probs=richer)
            for seed in seeds:
                perm = np.random.default_rng(seed).permutation(n_dep) + n_pool
                th = np.array_split(perm, 3)
                split = Split(pool=np.arange(n_pool), fit=th[0], conf=th[1], test=th[2],
                              origin={"pool": "training session",
                                      "fit": "deployment session",
                                      "conf": "deployment session",
                                      "test": "deployment session"})
                yield hdir.name, mask, seed, host, split


def cells_opportunity(features, seeds):
    """Train each masked host once, then reuse it across the whole grid."""
    from exp_opportunity import train_mlp
    z = np.load(features, allow_pickle=True)
    H, y, subject = z["H"], z["y"], z["subject"]
    blocks = list(z["block_slices"])
    configs = json.loads(str(z["configs"]))
    n_out = int(y.max()) + 1
    tr = np.where(np.isin(subject, [1, 2]))[0]
    val = np.where(subject == 3)[0]
    dep = np.where(subject == 4)[0]

    def masked(obs):
        keep = np.zeros(H.shape[1], dtype=bool)
        for b in obs:
            keep[blocks[b]] = True
        out = np.zeros_like(H)
        out[:, keep] = H[:, keep]
        out[:, ~keep] = H[tr][:, ~keep].mean(0)
        return out

    for proto in ("deployment", "cross_subject"):
        for seed in seeds:
            rng = np.random.default_rng(700 + seed)
            if proto == "deployment":
                perm = rng.permutation(len(dep))
                cut = 3 * (len(dep) // 4)
                avail, test_idx = dep[perm[:cut]], dep[perm[cut:]]
                t3 = len(avail) // 3
                pool, fit, conf = avail[:t3], avail[t3:2 * t3], avail[2 * t3:3 * t3]
                origin = {k: "subject 4" for k in ("pool", "fit", "conf", "test")}
            else:
                perm = rng.permutation(len(val))
                t3 = len(val) // 3
                pool, fit, conf = (val[perm[:t3]], val[perm[t3:2 * t3]],
                                   val[perm[2 * t3:3 * t3]])
                test_idx = dep
                origin = {"pool": "subject 3", "fit": "subject 3",
                          "conf": "subject 3", "test": "subject 4"}
            split = Split(pool, fit, conf, test_idx, origin=origin)
            h_rich = masked(list(range(len(blocks))))
            richer = train_mlp(h_rich[tr], y[tr], h_rich[fit], y[fit], n_out, seed)(h_rich)
            for name, obs in configs.items():
                h = masked(obs)
                probs = train_mlp(h[tr], y[tr], h[fit], y[fit], n_out, seed)(h)
                cols = np.concatenate([np.arange(blocks[b].start, blocks[b].stop)
                                       for b in obs])
                host = HostOutputs(probs=probs, features=h[:, cols], labels=y,
                                   richer_probs=richer)
                yield f"opportunity_{proto}", name, seed, host, split


def cells_dcl(dcl_dir, features, seeds):
    """The DeepConvLSTM hosts, re-gated under the reported protocol.

    Nothing is retrained: the twenty-four saved probability dumps are read back
    and only the gate moves across the grid.  The split is the one every
    sliding-window dataset now uses -- non-overlapping windows drawn in a random
    permutation, so ``D_conf`` stays exchangeable with ``D_test`` while no pool
    row can share raw samples with a query it answers.

    Only the hard-label target is available here.  Cross-mask needs the richer
    host's outputs on the unmasked input, which the training run did not dump,
    so ``run`` raises and the caller skips it -- the same path other families
    without ``richer_probs`` already take.
    """
    y = np.load(dcl_dir / "deploy_y.npy")
    configs = json.loads(str(np.load(features, allow_pickle=True)["configs"]))
    universe = np.arange(len(y))[::2]
    for seed in seeds:
        rng = np.random.default_rng(700 + seed)
        perm = universe[rng.permutation(len(universe))]
        cut = 3 * (len(universe) // 4)
        avail, test_idx = perm[:cut], perm[cut:]
        t3 = len(avail) // 3
        split = Split(avail[:t3], avail[t3:2 * t3], avail[2 * t3:3 * t3], test_idx,
                      origin={k: "subject 4" for k in ("pool", "fit", "conf", "test")})
        for cond in configs:
            feat = np.load(dcl_dir / f"retfeat_{cond}.npy")
            for design in ("condition_specialist", "full_masked"):
                f = dcl_dir / f"probs_{design}_{cond}_s{seed}.npy"
                if not f.exists():
                    print(f"  skip {f.name}: no dump"); continue
                host = HostOutputs(probs=np.load(f), features=feat, labels=y)
                yield f"dcl_{design}", cond, seed, host, split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True,
                    choices=["drugban", "vision", "affective", "opportunity",
                             "dcl"])
    ap.add_argument("--grid", required=True, choices=["alpha", "delta"])
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed")
    ap.add_argument("--hosts", type=Path, default=ROOT / "data/raw/hosts")
    ap.add_argument("--dcl", type=Path,
                    default=ROOT / "artifacts/opportunity_dcl_v2",
                    help="directory holding the saved DeepConvLSTM dumps")
    ap.add_argument("--out", type=Path, default=ROOT / "results/frontier")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--k", type=int, default=50)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / f"{a.family}_{a.grid}.csv"
    fh = open(dest, "w", newline="")
    wr = csv.DictWriter(fh, fieldnames=COLS)
    wr.writeheader()

    if a.family == "drugban":
        cells = cells_drugban(a.data, (a.seeds or [0])[0])
    elif a.family == "vision":
        cells = cells_vision(a.data, (a.seeds or [0])[0])
    elif a.family == "affective":
        cells = cells_affective(a.hosts, a.seeds or [0, 1, 2])
    elif a.family == "dcl":
        cells = cells_dcl(a.dcl, a.data / "opportunity.npz", a.seeds or [0, 1, 2])
    else:
        cells = cells_opportunity(a.data / "opportunity.npz", a.seeds or [0, 1, 2])

    grid = ([(al, 0.05) for al in ALPHAS] if a.grid == "alpha"
            else [(0.2, dl) for dl in DELTAS])
    n = 0
    for ds, cond, seed, host, split in cells:
        for alpha, delta in grid:
            for target in TARGETS:
                try:
                    r = run(host, split, condition=cond, target=target,
                            alpha=alpha, delta=delta, k=a.k)
                except ValueError as exc:        # e.g. a family without richer_probs
                    print(f"    {ds}/{cond}/{target} alpha={alpha}: {exc}")
                    continue
                wr.writerow(row_of(a.family, ds, seed, r))
                n += 1
        fh.flush()
        print(f"  {ds:34s} {cond:22s} seed{seed}  ({n} rows)", flush=True)
    fh.close()

    rows = list(csv.DictReader(open(dest)))
    ex = [r for r in rows if r["exchangeable"] == "True"]
    bad = [r for r in ex if float(r["joint_harm"]) > float(r["alpha"])]
    print(f"{len(bad)}/{len(ex)} exchangeable rows exceed their alpha -> {dest}")


if __name__ == "__main__":
    main()
