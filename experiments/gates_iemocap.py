"""Gate comparison on the frozen MoMKE host, IEMOCAP, over its seven patterns."""
import numpy as np, sys, json, collections
from gates_core import gate_row
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))

D=f'{ARTIFACTS}/iemocap_momke/folds'
MS=['a','t','v','at','av','tv','atv']; RULES=['blanket','random','confidence','agreement','learned','GUARD']
def key(d):
    cnt={}; out=[]
    for v in d['vid']:
        cnt[v]=cnt.get(v,0)+1; out.append((v,cnt[v]-1))
    return out
acc=collections.defaultdict(list)
for f in range(5):
    ref=np.load(f'{D}/fold{f}_atv.npz',allow_pickle=True)
    kr={k:i for i,k in enumerate(key(ref))}; n=len(ref['labels'])
    rich=np.zeros((n,ref['probs'].shape[1])); rich[[kr[k] for k in key(ref)]]=ref['probs']
    for m in MS:
        d=np.load(f'{D}/fold{f}_{m}.npz',allow_pickle=True)
        idx=np.array([kr[k] for k in key(d)])
        pr=np.zeros((n,d['probs'].shape[1])); pr[idx]=d['probs']
        ft=np.zeros((n,d['feats'].shape[1])); ft[idx]=d['feats']
        pf,pl=d['pool_feats'].astype(np.float64),d['pool_labels']
        # pool rows are prepended so every index set lives in one array
        P=np.concatenate([d['pool_probs'].astype(np.float64),pr])
        F=np.concatenate([pf,ft]); Y=np.concatenate([pl,ref['labels']])
        Rh=np.concatenate([np.load(f'{D}/fold{f}_atv.npz',allow_pickle=True)['pool_probs'].astype(np.float64),rich])
        npool=len(pl)
        for cut in range(5):
            uv=np.unique(ref['vid']); perm=np.random.default_rng(cut).permutation(len(uv))
            parts=[np.where(np.isin(ref['vid'],g))[0]+npool for g in np.array_split(uv[perm],3)]
            split=(np.arange(npool),parts[0],parts[1],parts[2])
            r=gate_row(P,F,Y,split,targets=('hard','cross'),richer=Rh)
            for k_ in RULES: acc[k_].append(r[k_])
            acc['_apply'].append(r['_meta']['apply']); acc['_base'].append(r['_meta']['base'])
print(f"{'rule':11}{'gain':>9}{'harm':>8}")
res={}
for k_ in RULES:
    g=np.nanmean([x[0] for x in acc[k_]]); h=np.nanmean([x[1] for x in acc[k_]])
    res[k_]=(float(g),float(h)); print(f'{k_:11}{g:+9.4f}{h:8.3f}')
print(f"base {np.mean(acc['_base']):.4f}  apply {np.mean(acc['_apply']):.3f}")
json.dump(res,open(str(_ROOT / 'results/gates/gates_iemocap.json'),'w'))
