import sys, json, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'src'))
sys.path.insert(0,str(ROOT / 'scripts'))
from guard import losses as _L, targets as _T, certify as _C
from guard.pipeline import _select_beta
import train_ninapro_retrained as T
B=str(ROOT / 'experiments')
ALPHA,DELTA=0.2,0.05; KS=(5,10,20,35,50); SPACES=('standardise','cosine'); WTS=('uniform','distance')
CE=_L.get('cross_entropy'); argmax=lambda P,Y: float((P.argmax(1)==Y).mean())
COUNTS=[16,14,12,10,8,7,6,5,4,3,2]
dev='cuda' if torch.cuda.is_available() else 'cpu'
order_ch=np.random.default_rng(1234).permutation(T.N_CH)
def run(P,F,Y,split,loss,score):
    pool,fit,conf,test=split
    SP={sp:{q:_T.retrieval_space(F[pool],sp)(F[i]) for q,i in (('pool',pool),('fit',fit),('conf',conf),('test',test))} for sp in SPACES}
    vals=_T.hard_label_values(Y[pool],P.shape[1],loss.simplex); best=None
    for sp in SPACES:
        for wt in WTS:
            for k in KS:
                tf=_T.knn_average(SP[sp]['fit'],SP[sp]['pool'],vals,k,weighting=wt)
                b=_select_beta(P[fit],tf,Y[fit],loss,'loss')
                sc=score((1-b)*P[fit]+b*tf,Y[fit])
                if best is None or sc>best[0]: best=(sc,sp,wt,k,b)
    _,sp,wt,k,b=best; f=SP[sp]
    tc=_T.knn_average(f['conf'],f['pool'],vals,k,weighting=wt); tt=_T.knn_average(f['test'],f['pool'],vals,k,weighting=wt)
    cc=(1-b)*P[conf]+b*tc; ct=(1-b)*P[test]+b*tt
    g=_C.certify(cc,Y[conf],ct,P[test],loss,ALPHA,DELTA); ap=g['apply']
    bl=loss(P[test],Y[test]); cl=loss(ct,Y[test]); gp=np.where(ap[:,None],ct,P[test])
    return (score(P[test],Y[test]),score(ct,Y[test]),score(gp,Y[test]),float(ap.mean()),
            float((ap&((cl-bl)>DELTA)).mean()),float((((cl-bl)>DELTA)).mean()))
def arrays(seed,s):
    X,Y,R=T.load_subject(s)
    tr=np.flatnonzero(np.isin(R,T.POOL_REPS)); te=np.flatnonzero(np.isin(R,T.QUERY_REPS))
    rng=np.random.default_rng(s); p=rng.permutation(len(tr)); c=int(0.85*len(p)); fit=tr[p[:c]]
    mu,sd=X[fit].mean((0,1)),X[fit].std((0,1))+1e-8
    Xn=((X-mu)/sd).transpose(0,2,1)
    ck=torch.load(f'{B}/artifacts/ninapro_cnn/checkpoints/seed{seed}/subject{s:02d}_rung16.pt',map_location=dev,weights_only=False)
    net,head=T.build(dev,T.N_CLS,seed)
    net.load_state_dict(ck['net_state_dict']); head.load_state_dict(ck['head_state_dict']); net.eval(); head.eval()
    out={}
    for ne in COUNTS:
        ch=np.sort(order_ch[:ne]); Xm=np.zeros_like(Xn); Xm[:,ch]=Xn[:,ch]
        eb,pb=[],[]
        with torch.no_grad():
            for i in range(0,len(Xm),2048):
                z=net(torch.tensor(Xm[i:i+2048],device=dev)); eb.append(z.cpu().numpy())
                pb.append(torch.softmax(head(z),1).cpu().numpy())
        out[ne]=(np.concatenate(pb),np.concatenate(eb))
    return out,Y,tr,te
res=[]
store={c:[] for c in COUNTS}
for s in range(1,11):
    A,Y,tr,te=arrays(0,s)
    n,m=len(tr),len(te)
    idx=np.concatenate([tr,te])
    for c in COUNTS:
        P,F=A[c]; P=np.clip(P[idx],1e-12,None); P=P/P.sum(1,keepdims=True); F=F[idx].astype(np.float64); Yc=Y[idx].astype(int)
        perm=np.random.default_rng(0).permutation(m)+n; k=m//3
        store[c].append(run(P,F,Yc,(np.arange(n),perm[:k],perm[k:2*k],perm[2*k:]),CE,argmax))
    print('subject',s,'xong',flush=True)
for c in COUNTS:
    a=np.mean(store[c],0)
    res.append(dict(level=str(c),frozen=a[0],blanket=a[1],guard=a[2],apply=a[3],harm=a[4],bharm=a[5]))
    print('ninapro',c,np.round(a[:3],4),flush=True)
json.dump(res,open(f'{B}/artifacts/ninapro_ladder_v2/severity_dense.json','w'),indent=1)
print('DA GHI')
