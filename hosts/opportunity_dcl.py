#!/usr/bin/env python3
"""Train and dump the OPPORTUNITY DeepConvLSTM hosts used by the paper.

The DCL_HOSTS2 layout is emitted from one coherent run. A second training run
does not reproduce the first closely enough to mix outputs with it: cuDNN LSTM
kernels are not deterministic across runs, even at a fixed seed. Prefer the
archived dumps for reproduction.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
def train_host(xtr,ytr,xva,yva,classes,seed,epochs,device):
    import DeepConvLSTM_py3 as model
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    net=model.DeepConvLSTM(n_channels=xtr.shape[-1],n_classes=classes,dataset="opportunity").to(device)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3); sched=torch.optim.lr_scheduler.StepLR(opt,100,.1)
    xt,yt,xv=torch.tensor(xtr,device=device),torch.tensor(ytr,dtype=torch.long,device=device),torch.tensor(xva,device=device)
    gen=torch.Generator().manual_seed(seed); best,state=-1.,None
    for _ in range(epochs):
        net.train(); order=torch.randperm(len(xt),generator=gen).to(device)
        for i in range(0,len(order),256):
            j=order[i:i+256]; opt.zero_grad(); torch.nn.functional.cross_entropy(net(xt[j])[-1],yt[j]).backward(); opt.step()
        sched.step(); net.eval()
        with torch.no_grad(): pred=torch.cat([net(xv[i:i+512])[-1].argmax(1) for i in range(0,len(xv),512)]).cpu().numpy()
        if (acc:=float((pred==yva).mean()))>best: best,state=acc,{k:v.clone() for k,v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    def predict(x):
        with torch.no_grad():
            xs=torch.tensor(x,device=device)
            return np.concatenate([torch.softmax(net(xs[i:i+512])[-1],-1).cpu().numpy() for i in range(0,len(xs),512)]).astype(np.float64)
    return predict,best
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",type=Path,required=True); ap.add_argument("--configs",type=Path,required=True)
    ap.add_argument("--repo",type=Path,required=True,help="directory containing DeepConvLSTM_py3.py"); ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--epochs",type=int,default=300); ap.add_argument("--seeds",type=int,nargs="+",default=[0,1,2]); ap.add_argument("--device",default="cuda:0")
    a=ap.parse_args(); sys.path.insert(0,str(a.repo)); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    z=np.load(a.data,allow_pickle=True); xtr,ytr,blocks=z["train_x"],z["train_y"],[tuple(b) for b in z["blocks"]]
    splits={"deploy":(z["deploy_x"],z["deploy_y"]),"calib":(z["calib_x"],z["calib_y"])}
    cfgs=json.loads(str(np.load(a.configs,allow_pickle=True)["configs"])); classes=int(max(ytr.max(),z["deploy_y"].max(),z["calib_y"].max()))+1
    pca={}
    for b,(lo,hi) in enumerate(blocks):
        f=np.concatenate([xtr[:,:,lo:hi].mean(1),xtr[:,:,lo:hi].std(1)],1); _,_,vt=np.linalg.svd(f-f.mean(0),full_matrices=False); pca[b]=vt[:6].T
    def mask(x,obs):
        out=np.zeros_like(x)
        for b in obs: lo,hi=blocks[b]; out[:,:,lo:hi]=x[:,:,lo:hi]
        return out
    def features(x,obs): return np.concatenate([np.concatenate([x[:,:,blocks[b][0]:blocks[b][1]].mean(1),x[:,:,blocks[b][0]:blocks[b][1]].std(1)],1)@pca[b] for b in obs],1)
    a.out.mkdir(parents=True,exist_ok=True)
    for seed in a.seeds:
        p=np.random.default_rng(500+seed).permutation(len(ytr)); cut=int(.85*len(p))
        full,score=train_host(xtr[p[:cut]],ytr[p[:cut]],xtr[p[cut:]],ytr[p[cut:]],classes,seed,a.epochs,a.device); print(f"seed {seed}: full validation accuracy {score:.4f}",flush=True)
        for sp,(x,_) in splits.items(): np.save(a.out/f"richer_{sp}_s{seed}.npy",full(x))
        for name,obs in cfgs.items():
            specialist,_=train_host(mask(xtr,obs)[p[:cut]],ytr[p[:cut]],mask(xtr,obs)[p[cut:]],ytr[p[cut:]],classes,seed,a.epochs,a.device)
            for design,predictor in (("condition_specialist",specialist),("full_masked",full)):
                for sp,(x,_) in splits.items(): np.save(a.out/f"probs_{design}_{name}_{sp}_s{seed}.npy",predictor(mask(x,obs)))
    for name,obs in cfgs.items():
        for sp,(x,_) in splits.items(): np.save(a.out/f"retfeat_{name}_{sp}.npy",features(mask(x,obs)))
    for sp,(_,y) in splits.items(): np.save(a.out/f"{sp}_y.npy",y)
if __name__=="__main__": main()
