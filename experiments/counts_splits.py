"""Exact per-role sample counts for the three benchmarks whose splits are not plain quarters."""
import numpy as np, glob
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))

A=str(ARTIFACTS)
# AVE
for cond in ('audio_only',):
    d=np.load(f'{A}/ave_av_att/dumps/AV_att_{cond}.npz',allow_pickle=True)
    r=np.load(f'{A}/ave_av_att/dumps/AV_att_{cond}_retrieval_paper.npz',allow_pickle=True)
    N=len(d['labels'].reshape(-1,d['labels'].shape[-1])); n=len(r['pool_labels']); k=N//3
    print(f'AVE      source {n}  fit {k}  conf {k}  test {N-2*k}')
# NinaPro
sub=sorted(glob.glob(f'{A}/ninapro_cnn/seed*/subject*'))[0]
p=np.load(f'{sub}/preds.npz',allow_pickle=True)
n=len(p['train_y']); m=len(p['sess1_y']); k=m//3
print(f'NinaPro  source {n}  fit {k}  conf {k}  test {m-2*k}   (per subject)')
# IEMOCAP
D=f'{A}/iemocap_momke/folds'
tot=[]
for f in range(5):
    ref=np.load(f'{D}/fold{f}_atv.npz',allow_pickle=True)
    d=np.load(f'{D}/fold{f}_a.npz',allow_pickle=True)
    uv=np.unique(ref['vid']); perm=np.random.default_rng(0).permutation(len(uv))
    parts=[np.where(np.isin(ref['vid'],g))[0] for g in np.array_split(uv[perm],3)]
    tot.append((len(d['pool_labels']),)+tuple(len(x) for x in parts))
t=np.array(tot); print('IEMOCAP  source %d  fit %d  conf %d  test %d   (mean over 5 folds)'%tuple(t.mean(0).round()))
print('IEMOCAP  eval fold size', t[:,1:].sum(1))
