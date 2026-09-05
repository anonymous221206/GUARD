#!/usr/bin/env python3
"""OPPORTUNITY: sensors drop out, and where the calibration set comes from matters.

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


def _weighted_f1(pred, true, n_class: int) -> float:
    """Support-weighted F1, the metric the OPPORTUNITY table is published under."""
    fs, sup = [], []
    for c in range(n_class):
        tp = ((pred == c) & (true == c)).sum()
        fp = ((pred == c) & (true != c)).sum()
        fn = ((pred != c) & (true == c)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        sup.append((true == c).sum())
    fs, sup = np.array(fs), np.array(sup)
    return float((fs * sup).sum() / sup.sum())


def artifact_route(a) -> None:
    """Reproduce the printed OPPORTUNITY rows from the released dumps.

    Two details decide whether the numbers come out right, and neither is
    guessable from the outputs alone.

    First, the printed rows are the ``condition_specialist`` design: a model
    refitted for each degraded condition.  The ``full_masked`` design is a second
    arm, kept here because of the second detail.

    Second, and easy to lose: a single generator serves the whole seed, and both
    designs draw from it, in the order (condition, design).  The ``full_masked``
    draw is never used by the printed rows, but removing it advances the generator
    differently and every later condition shifts.  The loop below therefore keeps
    the draw even when the row is discarded downstream.
    """
    d = a.artifact / "dumps" if (a.artifact / "dumps").is_dir() else a.artifact
    y_dep = np.load(d / "deploy_y.npy")
    y_cal = np.load(d / "calib_y.npy")
    cfgs = json.loads(str(np.load(d / "opportunity.npz", allow_pickle=True)["configs"]))

    # Non-overlapping windows: a window and its 50%-overlapping neighbour must not
    # land in different roles, or calibration sees its own evaluation rows.
    dep_u = np.arange(len(y_dep))[::2]
    cal_u = np.arange(len(y_cal))[::2]
    deployment = a.protocol == "deployment"

    rows = []
    for seed in a.seeds:
        rng = np.random.default_rng(700 + seed)
        fit_ref = None          # xem chu thich o phep kiem cross-mask ben duoi
        for name in cfgs:
            fd = np.load(d / f"retfeat_{name}_deploy.npy")
            fc = np.load(d / f"retfeat_{name}_calib.npy")
            for design in ("condition_specialist", "full_masked"):
                pd_ = np.load(d / f"probs_{design}_{name}_deploy_s{seed}.npy")
                pc = np.load(d / f"probs_{design}_{name}_calib_s{seed}.npy")
                rich_d = np.load(d / f"richer_deploy_s{seed}.npy")
                rich_c = np.load(d / f"richer_calib_s{seed}.npy")

                if deployment:
                    perm = dep_u[rng.permutation(len(dep_u))]
                    cut = 3 * (len(dep_u) // 4)
                    av, te = perm[:cut], perm[cut:]
                    t3 = len(av) // 3
                    idx = (av[:t3], av[t3:2 * t3], av[2 * t3:3 * t3], te)
                    probs, feat, y, rich = pd_, fd, y_dep, rich_d
                    origin = {r: "subject 4" for r in ("pool", "fit", "conf", "test")}
                else:
                    off = len(y_cal)
                    perm = cal_u[rng.permutation(len(cal_u))]
                    t3 = len(perm) // 3
                    idx = (perm[:t3], perm[t3:2 * t3], perm[2 * t3:3 * t3], dep_u + off)
                    probs = np.concatenate([pc, pd_])
                    feat = np.concatenate([fc, fd])
                    y = np.concatenate([y_cal, y_dep])
                    rich = np.concatenate([rich_c, rich_d])
                    origin = {"pool": "subject 3", "fit": "subject 3",
                              "conf": "subject 3", "test": "subject 4"}

                split = Split(*idx, origin=origin)
                host = HostOutputs(probs=probs, features=feat, labels=y,
                                   richer_probs=rich)
                if fit_ref is None:
                    fit_ref = split.fit
                if design == "condition_specialist":
                    # Route nay moi sinh ra cac hang bai in ra, nen phep kiem
                    # cross-mask phai chay o day chu khong chi o nhanh --features.
                    #
                    # Do tren MOT split co dinh cho ca seed, la fit cua cau hinh dau
                    # tien, chu khong phai fit rieng cua tung cau hinh. Split o day
                    # duoc rut lai theo tung (cau hinh, design) nen neu do rieng thi
                    # do chinh xac cua richer host se doi theo cau hinh va phep so
                    # giua cac cau hinh lan them nhieu cua split. Do chung mot split
                    # thi richer host la mot so duy nhat cho ca seed, dung nhu bang
                    # trong ablation_data/METRICS/crossmask_precondition.csv.
                    # Van la du lieu phan phoi trien khai, van khong dung nhan test.
                    chk = richer_is_richer(rich[fit_ref], probs[fit_ref], y[fit_ref])
                    print(f"pre-flight [{name} s{seed}]: richer host "
                          f"{chk['richer_accuracy']:.4f} vs poorer "
                          f"{chk['poorer_accuracy']:.4f} -> cross-mask "
                          f"{'applicable' if chk['precondition_met'] else 'NOT applicable'}",
                          flush=True)
                for target in ("hard", "cross_mask"):
                    r = run(host, split, condition=name, target=target,
                            alpha=a.alpha, delta=a.delta,
                            beta_objective=a.beta_objective)
                    A = r.test_arrays
                    yt = A["labels"]
                    n_class = int(max(y_dep.max(), y_cal.max())) + 1
                    extra = {"seed": seed, "design": design, "protocol": a.protocol}
                    for pol, key in (("base", "base_probs"),
                                     ("blanket", "blanket_probs"),
                                     ("gated", "gated_probs")):
                        pp = A[key].argmax(1)
                        extra[f"{pol}_acc"] = float((pp == yt).mean())
                        extra[f"{pol}_wf1"] = _weighted_f1(pp, yt, n_class)
                    rows.append({**r.as_row(), **extra})
                    if design == "condition_specialist" and target == "hard":
                        print(f"seed{seed} {name:24s} base={r.base_metric:.4f} "
                              f"guard={r.base_metric + r.gate_metric_delta:.4f} "
                              f"harm={r.joint_harm:.4f} apply={r.apply_rate:.3f} "
                              f"beta={r.beta:.3f}", flush=True)

    tag = f"opportunity_{a.protocol}_artifact"
    if a.beta_objective != "crossfit":
        tag += f"_{a.beta_objective}"
    out_dir = a.out / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "guard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_dir}/guard.csv  "
          f"({len(rows)} rows; the printed table is design=condition_specialist, "
          f"target=hard)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=None,
                    help="npz with H, y, subject, block_slices.  Required unless "
                         "--artifact is given.")
    ap.add_argument("--artifact", type=Path, default=None,
                    help="artifacts/opportunity_deepconvlstm: reproduce the rows the "
                         "paper prints, from the saved per-condition outputs.")
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
    ap.add_argument("--beta-objective", default="crossfit",
                    choices=["loss", "metric", "crossfit"],
                    help="beta selection rule: 'loss' is rule A, 'crossfit' is "
                         "rule D, the cross-fit rule the paper reports.")
    a = ap.parse_args()
    if a.artifact is not None:
        return artifact_route(a)
    if a.features is None:
        ap.error("one of --features or --artifact is required")

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
            # Moi cau hinh va moi seed: phep kiem la dai luong theo tung cell,
            # khong phai theo tung lan chay. Chay mot lan roi gan cho ca run se
            # gan nham so cua cau hinh dau tien sang cac cau hinh khac.
            chk = richer_is_richer(richer[fit], probs[fit], y[fit])
            print(f"pre-flight [{name} s{seed}] on the deployment distribution: richer host "
                  f"{chk['richer_accuracy']:.4f} vs poorer {chk['poorer_accuracy']:.4f} "
                  f"-> cross-mask {'applicable' if chk['precondition_met'] else 'NOT applicable'}")
            host = HostOutputs(probs=probs, features=h[:, np.concatenate(
                [np.arange(blocks[b].start, blocks[b].stop) for b in obs])],
                labels=y, richer_probs=richer)
            for target in ("hard", "cross_mask"):
                r = run(host, split, condition=name, target=target,
                        alpha=a.alpha, delta=a.delta,
                        beta_objective=a.beta_objective)
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
