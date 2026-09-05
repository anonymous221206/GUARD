#!/usr/bin/env python3
"""Run GUARD on archived NinaPro DB5 subject-specific CNN dumps."""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from guard import HostOutputs,run
from guard.splits import Split
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--dumps",type=Path,required=True); ap.add_argument("--out",type=Path,default=Path("results")); ap.add_argument("--seed",type=int,default=0,help="calibration split seed"); a=ap.parse_args()
 rows=[]; out=a.out/"ninapro"; out.mkdir(parents=True,exist_ok=True)
 for sub in sorted(a.dumps.glob("seed*/subject*")):
  p=np.load(sub/"preds.npz",allow_pickle=True); e=np.load(sub/"masked_embeddings.npz",allow_pickle=True)
  for condition in ("12","8","6","4"):
   probs=np.concatenate([p[f"train_{condition}"],p[f"sess1_{condition}"]]).astype(np.float64); probs=np.clip(probs,1e-12,None); probs/=probs.sum(1,keepdims=True)
   features=np.concatenate([e[f"train_{condition}"],e[f"sess1_{condition}"]]).astype(np.float64); y=np.concatenate([p["train_y"],p["sess1_y"]]).astype(int); n=len(p["train_y"]); dep=np.random.default_rng(a.seed).permutation(len(y)-n)+n; q=len(dep)//3
   x=run(HostOutputs(probs,features,y),Split(np.arange(n),dep[:q],dep[q:2*q],dep[2*q:],origin={"pool":"NinaPro train","fit":"NinaPro deployment","conf":"NinaPro deployment","test":"NinaPro deployment"}),condition=condition)
   rows.append({**x.as_row(),"subject":sub.name,"host_seed":sub.parent.name,"seed":a.seed}); print(f"{sub.parent.name}/{sub.name} {condition} base={x.base_metric:.4f} gate={x.gate_metric_delta:+.4f}")
 with open(out/"guard.csv","w",newline="") as fh: w=csv.DictWriter(fh,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
if __name__=="__main__": main()
