import sys, json, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
import os
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', ARTIFACTS))
sys.path.insert(0,str(ROOT / 'scripts'))
import train_ninapro_retrained as T
B=str(ROOT / 'experiments')
ART=str(ARTIFACTS / 'ninapro_cnn')
COUNTS=[16,14,12,10,8,7,6,5,4,3,2]
dev='cuda' if torch.cuda.is_available() else 'cpu'
order_ch=np.random.default_rng(1234).permutation(T.N_CH)
def emb(net,x):
    with torch.no_grad():
        return np.concatenate([net(torch.tensor(x[i:i+2048],device=dev)).cpu().numpy() for i in range(0,len(x),2048)])
def pred(net,head,x):
    with torch.no_grad():
        return np.concatenate([head(net(torch.tensor(x[i:i+2048],device=dev))).argmax(1).cpu().numpy() for i in range(0,len(x),2048)])
def knn(Q,A,ya,k):
    oh=np.eye(T.N_CLS)[ya]; sq=(A**2).sum(1); out=[]
    for i in range(0,len(Q),2048):
        q=Q[i:i+2048]; d=(q**2).sum(1)[:,None]+sq[None,:]-2*q@A.T
        out.append(oh[np.argpartition(d,k-1,axis=1)[:,:k]].mean(1).argmax(1))
    return np.concatenate(out)
def cell(seed,s,counts):
    X,Y,R=T.load_subject(s)
    tr=np.flatnonzero(np.isin(R,T.POOL_REPS)); te=np.flatnonzero(np.isin(R,T.QUERY_REPS))
    rng=np.random.default_rng(s); p=rng.permutation(len(tr)); c=int(0.85*len(p))
    fit,val=tr[p[:c]],tr[p[c:]]
    mu,sd=X[fit].mean((0,1)),X[fit].std((0,1))+1e-8
    Xn=((X-mu)/sd).transpose(0,2,1)
    ck=torch.load(f'{ART}/checkpoints/seed{seed}/subject{s:02d}_rung16.pt',map_location=dev,weights_only=False)
    net,head=T.build(dev,T.N_CLS,seed)
    net.load_state_dict(ck['net_state_dict']); head.load_state_dict(ck['head_state_dict'])
    net.eval(); head.eval()
    r={}
    for ne in counts:
        ch=np.sort(order_ch[:ne]); Xm=np.zeros_like(Xn); Xm[:,ch]=Xn[:,ch]
        base=float((pred(net,head,Xm[te])==Y[te]).mean())
        ef,ev,et=emb(net,Xm[fit]),emb(net,Xm[val]),emb(net,Xm[te])
        m2,s2=ef.mean(0),ef.std(0)+1e-8
        ef,ev,et=(ef-m2)/s2,(ev-m2)/s2,(et-m2)/s2
        bk=max(T.KS,key=lambda kk: float((knn(ev,ef,Y[fit],kk)==Y[val]).mean()))
        tp=knn(et,ef,Y[fit],bk)
        r[ne]=dict(base=base,target=float((tp==Y[te]).mean()),best_k=int(bk))
        r[ne]['_pred']=tp
    return r,Y[te]
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--gate',action='store_true'); ap.add_argument('--out')
    a=ap.parse_args()
    if a.gate:
        for seed,s in ((0,1),(1,1)):
            r,yte=cell(seed,s,[16,12,8,6,4])
            P=np.load(f'{ART}/seed{seed}/subject{s:02d}/preds.npz')
            Tg=np.load(f'{ART}/seed{seed}/subject{s:02d}/target_predictions.npz')
            print(f'--- seed{seed} subject{s:02d} ---')
            for ne in [16,12,8,6,4]:
                sb=float((P[f'sess1_{ne}']==P['sess1_y']).mean()) if P[f'sess1_{ne}'].ndim==1 else float((P[f'sess1_{ne}'].argmax(1)==P['sess1_y']).mean())
                st=float((Tg[f'sess1_{ne}']==P['sess1_y']).mean())
                ok=np.array_equal(r[ne]['_pred'],Tg[f'sess1_{ne}'])
                print(f'  ne={ne:2d} base {r[ne]["base"]:.10f} vs {sb:.10f} | target {r[ne]["target"]:.10f} vs {st:.10f} | khop mang {ok}')
    else:
        out={}
        for seed in (0,1):
            for s in range(1,11):
                r,_=cell(seed,s,COUNTS)
                out[f'seed{seed}_subject{s:02d}']={str(k):{kk:vv for kk,vv in v.items() if kk!='_pred'} for k,v in r.items()}
                print(f'seed{seed} subject{s:02d} xong',flush=True)
        json.dump(out,open(a.out,'w'),indent=1); print('DA GHI',a.out)
