"""Gate comparison on AVE, NinaPro and PTB-XL."""
import numpy as np, sys, json, collections, glob, os
from gates_core import gate_row
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))
os.makedirs(_ROOT / 'results/gates', exist_ok=True)

A=str(ARTIFACTS)
RULES=['blanket','random','confidence','agreement','learned','GUARD']
res={}

def report(name,acc):
    out={}
    print(f'--- {name} ---')
    for k in RULES:
        g=np.nanmean([x[0] for x in acc[k]]); h=np.nanmean([x[1] for x in acc[k]])
        out[k]=(float(g),float(h)); print(f'  {k:11}{g:+9.4f}{h:8.3f}')
    print(f"  base {np.mean(acc['_b']):.4f}  apply {np.mean(acc['_a']):.3f}  cells {len(acc['GUARD'])}",flush=True)
    res[name]=out

# ---- AVE: probs and retrieval features live in separate dumps ----
acc=collections.defaultdict(list)
for cond in ('audio_only','visual_only'):
    d=np.load(f'{A}/ave_av_att/dumps/AV_att_{cond}.npz',allow_pickle=True)
    r=np.load(f'{A}/ave_av_att/dumps/AV_att_{cond}_retrieval_paper.npz',allow_pickle=True)
    P=d['probs'].reshape(-1,d['probs'].shape[-1]).astype(np.float64)
    Y=d['labels'].reshape(-1,d['labels'].shape[-1]).argmax(1)
    F=r['deploy_features'].astype(np.float64)
    pP,pF,pY=r['pool_probs'].astype(np.float64),r['pool_features'].astype(np.float64),r['pool_labels']
    n=len(pY)
    PP=np.concatenate([pP,P]); FF=np.concatenate([pF,F]); YY=np.concatenate([pY,Y])
    for seed in range(3):
        perm=np.random.default_rng(seed).permutation(len(Y))+n
        k=len(perm)//3
        split=(np.arange(n),perm[:k],perm[k:2*k],perm[2*k:])
        row=gate_row(PP,FF,YY,split)
        for r_ in RULES: acc[r_].append(row[r_])
        acc['_a'].append(row['_meta']['apply']); acc['_b'].append(row['_meta']['base'])
report('AVE',acc)

# ---- NinaPro: one host per subject, electrode subsets as conditions ----
acc=collections.defaultdict(list)
for sd in sorted(glob.glob(f'{A}/ninapro_cnn/seed*')):
    for sub in sorted(glob.glob(f'{sd}/subject*'))[:10]:
        p=np.load(f'{sub}/preds.npz',allow_pickle=True); e=np.load(f'{sub}/masked_embeddings.npz',allow_pickle=True)
        for c in ('12','8','6','4'):
            P=np.concatenate([p[f'train_{c}'],p[f'sess1_{c}']]).astype(np.float64)
            P=np.clip(P,1e-12,None); P/=P.sum(1,keepdims=True)
            F=np.concatenate([e[f'train_{c}'],e[f'sess1_{c}']]).astype(np.float64)
            Y=np.concatenate([p['train_y'],p['sess1_y']]).astype(int)
            n=len(p['train_y']); m=len(p['sess1_y'])
            perm=np.random.default_rng(0).permutation(m)+n; k=m//3
            split=(np.arange(n),perm[:k],perm[k:2*k],perm[2*k:])
            row=gate_row(P,F,Y,split)
            for r_ in RULES: acc[r_].append(row[r_])
            acc['_a'].append(row['_meta']['apply']); acc['_b'].append(row['_meta']['base'])
report('NinaPro',acc)

# ---- PTB-XL: multilabel, so the loss is Bernoulli and accuracy is per label ----
acc=collections.defaultdict(list)
p=np.load(f'{A}/ptbxl_resnet1d_wang/preds.npz',allow_pickle=True)
r=np.load(f'{A}/ptbxl_resnet1d_wang/raw_features.npz',allow_pickle=True)
for c in ('a','v','av'):
    P=np.concatenate([p[f'train_{c}'],p[f'sess1_{c}']]).astype(np.float64)
    F=np.concatenate([r[f'train_{c[0]}'],r[f'sess1_{c[0]}']]).astype(np.float64)
    Y=np.concatenate([r['train_y'],r['sess1_y']]).astype(np.float64)
    n=len(r['train_y']); m=len(r['sess1_y'])
    for seed in range(3):
        perm=np.random.default_rng(seed).permutation(m)+n; k=m//3
        split=(np.arange(n),perm[:k],perm[k:2*k],perm[2*k:])
        row=gate_row(P,F,Y,split,loss_name='bernoulli')
        for r_ in RULES: acc[r_].append(row[r_])
        acc['_a'].append(row['_meta']['apply']); acc['_b'].append(row['_meta']['base'])
report('PTB-XL',acc)
json.dump(res,open(str(_ROOT / 'results/gates/gates_rest.json'),'w'))
