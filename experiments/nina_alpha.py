import sys, json, csv, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'src'))
sys.path.insert(0,str(ROOT / 'scripts'))
from guard import losses as _L, targets as _T, certify as _C
from guard.pipeline import _select_beta
import train_ninapro_retrained as T
B=str(ROOT / 'experiments')
ALPHAS=[0.05,0.10,0.20,0.30,0.50]; DELTA=0.05
KS=(5,10,20,35,50); SPACES=('standardise','cosine'); WTS=('uniform','distance')
CE=_L.get('cross_entropy'); amax=lambda P,Y: float((P.argmax(1)==Y).mean())
COUNTS=[14,12,10,8,7,6,5,4,3,2]
dev='cuda' if torch.cuda.is_available() else 'cpu'
order_ch=np.random.default_rng(1234).permutation(T.N_CH)
def cellrun(P,F,Y,split):
    pool,fit,conf,test=split
    SP={sp:{q:_T.retrieval_space(F[pool],sp)(F[i]) for q,i in (('pool',pool),('fit',fit),('conf',conf),('test',test))} for sp in SPACES}
    vals=_T.hard_label_values(Y[pool],P.shape[1],CE.simplex); best=None
    for sp in SPACES:
        for wt in WTS:
            for k in KS:
                tf=_T.knn_average(SP[sp]['fit'],SP[sp]['pool'],vals,k,weighting=wt)
                b=_select_beta(P[fit],tf,Y[fit],CE,'loss'); sc=amax((1-b)*P[fit]+b*tf,Y[fit])
                if best is None or sc>best[0]: best=(sc,sp,wt,k,b)
    _,sp,wt,k,b=best; f=SP[sp]
    tc=_T.knn_average(f['conf'],f['pool'],vals,k,weighting=wt); tt=_T.knn_average(f['test'],f['pool'],vals,k,weighting=wt)
    cc=(1-b)*P[conf]+b*tc; ct=(1-b)*P[test]+b*tt
    bl=CE(P[test],Y[test]); cl=CE(ct,Y[test]); hurt=(cl-bl)>DELTA; base=amax(P[test],Y[test]); out=[]
    for al in ALPHAS:
        g=_C.certify(cc,Y[conf],ct,P[test],CE,al,DELTA); ap=g['apply']
        gp=np.where(ap[:,None],ct,P[test])
        out.append(dict(alpha=al,apply_rate=float(ap.mean()),joint_harm=float((ap&hurt).mean()),
                        acc_gain=amax(gp,Y[test])-base,blanket_joint_harm=float(hurt.mean()),
                        blanket_acc_gain=amax(ct,Y[test])-base))
    return out
rows=[]
for s in range(1,11):
    X,Y,R=T.load_subject(s)
    tr=np.flatnonzero(np.isin(R,T.POOL_REPS)); te=np.flatnonzero(np.isin(R,T.QUERY_REPS))
    rng=np.random.default_rng(s); p=rng.permutation(len(tr)); c=int(0.85*len(p)); fit=tr[p[:c]]
    mu,sd=X[fit].mean((0,1)),X[fit].std((0,1))+1e-8
    Xn=((X-mu)/sd).transpose(0,2,1)
    ck=torch.load(f'{B}/artifacts/ninapro_cnn/checkpoints/seed0/subject{s:02d}_rung16.pt',map_location=dev,weights_only=False)
    net,head=T.build(dev,T.N_CLS,0)
    net.load_state_dict(ck['net_state_dict']); head.load_state_dict(ck['head_state_dict']); net.eval(); head.eval()
    idx=np.concatenate([tr,te]); n,m=len(tr),len(te)
    for ne in COUNTS:
        ch=np.sort(order_ch[:ne]); Xm=np.zeros_like(Xn); Xm[:,ch]=Xn[:,ch]
        eb,pb=[],[]
        with torch.no_grad():
            for i in range(0,len(Xm),2048):
                z=net(torch.tensor(Xm[i:i+2048],device=dev)); eb.append(z.cpu().numpy())
                pb.append(torch.softmax(head(z),1).cpu().numpy())
        P=np.concatenate(pb)[idx].astype(np.float64); P=np.clip(P,1e-12,None); P/=P.sum(1,keepdims=True)
        F=np.concatenate(eb)[idx].astype(np.float64); Yc=Y[idx].astype(int)
        perm=np.random.default_rng(0).permutation(m)+n; k=m//3
        for o in cellrun(P,F,Yc,(np.arange(n),perm[:k],perm[k:2*k],perm[2*k:])):
            rows.append(dict(family='ninapro',dataset='ninapro_db5',condition=str(ne),target='hard',seed=s,exchangeable=True,**o))
    print('subject',s,'xong',flush=True)
with open(f'{B}/alpha_ninapro.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('DA GHI',len(rows))
