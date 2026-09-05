#!/usr/bin/env python3
"""Run GUARD on frozen OPPORTUNITY DeepConvLSTM paper-host dumps."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from guard import HostOutputs, run
from guard.splits import Split
CONFIGS = ("low_cost_accels_only", "no_imu_family", "no_shoes", "severe_three_sensors")
def weighted_f1(y, pred):
    vals, counts = [], []
    for c in np.unique(y):
        tp=((pred==c)&(y==c)).sum(); fp=((pred==c)&(y!=c)).sum(); fn=((pred!=c)&(y==c)).sum()
        p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.
        vals.append(2*p*r/(p+r) if p+r else 0.); counts.append((y==c).sum())
    return float(np.average(vals, weights=counts))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dumps",type=Path,required=True,help="DCL_HOSTS2 directory")
    ap.add_argument("--out",type=Path,default=Path("results")); ap.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
    ap.add_argument("--alpha",type=float,default=.2); ap.add_argument("--delta",type=float,default=.05); ap.add_argument("--k",type=int,default=50)
    a=ap.parse_args(); y=np.load(a.dumps/"deploy_y.npy"); out=a.out/"opportunity_dcl"; out.mkdir(parents=True,exist_ok=True); rows=[]
    print(f"{'config':24s} {'frozen weighted F1':>18s}")
    for cfg in CONFIGS:
        f=np.load(a.dumps/f"retfeat_{cfg}_deploy.npy").astype(np.float64); scores=[]
        for seed in a.seeds:
            probs=np.load(a.dumps/f"probs_condition_specialist_{cfg}_deploy_s{seed}.npy").astype(np.float64)
            rich=np.load(a.dumps/f"richer_deploy_s{seed}.npy").astype(np.float64); scores.append(weighted_f1(y,probs.argmax(1)))
            perm=np.random.default_rng(seed).permutation(len(y)); q=len(y)//4
            split=Split(perm[:q],perm[q:2*q],perm[2*q:3*q],perm[3*q:],origin={k:"OPPORTUNITY deployment subject" for k in ("pool","fit","conf","test")})
            host=HostOutputs(probs=probs,features=f,labels=y,richer_probs=rich)
            for target in ("hard","cross_mask"):
                r=run(host,split,condition=cfg,target=target,alpha=a.alpha,delta=a.delta,k=a.k)
                rows.append({**r.as_row(),"seed":seed}); print(f"  seed {seed} {target:10s} base={r.base_metric:.4f} gate={r.gate_metric_delta:+.4f}")
        print(f"{cfg:24s} {np.mean(scores):18.4f}")
    with open(out/"guard.csv","w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=[k for k in rows[0] if k!="notes"]); w.writeheader()
        for row in rows: w.writerow({k:v for k,v in row.items() if k!="notes"})
if __name__ == "__main__": main()
