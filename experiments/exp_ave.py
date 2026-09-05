#!/usr/bin/env python3
"""Run GUARD on archived AVE audio/visual-attention host dumps."""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from guard import HostOutputs,run
from guard.splits import Split
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--dumps",type=Path,required=True); ap.add_argument("--out",type=Path,default=Path("results")); ap.add_argument("--seeds",type=int,nargs="+",default=[0,1,2]); a=ap.parse_args()
 d=a.dumps/"dumps" if (a.dumps/"dumps").is_dir() else a.dumps; rows=[]; out=a.out/"ave"; out.mkdir(parents=True,exist_ok=True)
 for condition in ("audio_only","visual_only"):
  h=np.load(d/f"AV_att_{condition}.npz",allow_pickle=True); r=np.load(d/f"AV_att_{condition}_retrieval_paper.npz",allow_pickle=True)
  p=h["probs"].reshape(-1,h["probs"].shape[-1]).astype(np.float64); y=h["labels"].reshape(-1,h["labels"].shape[-1]).argmax(1); n=len(r["pool_labels"])
  host=HostOutputs(np.concatenate([r["pool_probs"],p]).astype(np.float64),np.concatenate([r["pool_features"],r["deploy_features"]]).astype(np.float64),np.concatenate([r["pool_labels"],y]))
  for seed in a.seeds:
   dep=np.random.default_rng(seed).permutation(len(y))+n; q=len(dep)//3; split=Split(np.arange(n),dep[:q],dep[q:2*q],dep[2*q:],origin={"pool":"training population","fit":"AVE deployment","conf":"AVE deployment","test":"AVE deployment"})
   for target in ("hard",):
    x=run(host,split,condition=condition,target=target); rows.append({**x.as_row(),"seed":seed}); print(f"{condition} seed{seed} base={x.base_metric:.4f} gate={x.gate_metric_delta:+.4f}")
 with open(out/"guard.csv","w",newline="") as fh: w=csv.DictWriter(fh,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
if __name__=="__main__": main()
