#!/usr/bin/env python3
"""Run GUARD on archived PTB-XL frozen-host dumps (multi-label Bernoulli loss)."""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from guard import HostOutputs,run
from guard.splits import Split
def split(n,m,seed):
 p=np.random.default_rng(seed).permutation(m)+n; q=len(p)//3
 return Split(np.arange(n),p[:q],p[q:2*q],p[2*q:],origin={"pool":"PTB-XL train","fit":"PTB-XL deployment","conf":"PTB-XL deployment","test":"PTB-XL deployment"})
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--dumps",type=Path,required=True); ap.add_argument("--variant",choices=("resnet1d_wang","leadladder"),default="resnet1d_wang"); ap.add_argument("--out",type=Path,default=Path("results")); ap.add_argument("--seeds",type=int,nargs="+",default=[0,1,2]); a=ap.parse_args()
 rows=[]; out=a.out/f"ptbxl_{a.variant}"; out.mkdir(parents=True,exist_ok=True)
 if a.variant=="resnet1d_wang":
  p=np.load(a.dumps/"preds.npz",allow_pickle=True); f=np.load(a.dumps/"raw_features.npz",allow_pickle=True); y=np.concatenate([f["train_y"],f["sess1_y"]]).astype(np.float64); n=len(f["train_y"])
  for condition,key in (("a","a"),("t","t"),("v","v"),("at","a"),("av","a"),("tv","t")):
   host=HostOutputs(np.concatenate([p[f"train_{condition}"],p[f"sess1_{condition}"]]).astype(np.float64),np.concatenate([f[f"train_{key}"],f[f"sess1_{key}"]]).astype(np.float64),y)
   for seed in a.seeds: x=run(host,split(n,len(y)-n,seed),condition=condition,loss_name="bernoulli"); rows.append({**x.as_row(),"seed":seed}); print(f"{condition} seed{seed} base={x.base_metric:.4f} gate={x.gate_metric_delta:+.4f}")
 else:
  rng=np.random.default_rng(0); projection=None
  for name in ("lead12","lead6","lead4","lead3","lead2"):
   d=np.load(a.dumps/f"{name}.npz",allow_pickle=True); emb=np.concatenate([d["train_emb"],d["sess1_emb"]]); projection=rng.normal(size=(emb.shape[1],256)).astype(np.float32)/np.sqrt(256) if projection is None else projection
   host=HostOutputs(np.concatenate([d["train_probs"],d["sess1_probs"]]).astype(np.float64),(emb@projection).astype(np.float64),np.concatenate([d["train_y"],d["sess1_y"]]).astype(np.float64)); n=len(d["train_y"])
   for seed in a.seeds: x=run(host,split(n,len(host.labels)-n,seed),condition=name,loss_name="bernoulli"); rows.append({**x.as_row(),"seed":seed}); print(f"{name} seed{seed} base={x.base_metric:.4f} gate={x.gate_metric_delta:+.4f}")
 with open(out/"guard.csv","w",newline="") as fh: w=csv.DictWriter(fh,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
if __name__=="__main__": main()
