"""Recover exact held-out counts from full-precision rates and original splits.
No refitting, no change to the saved policy. Reject non-integral reconstruction.
Repeated cuts overlap: pooled counts count evaluated occurrences, not unique people.
"""
import json, hashlib, os, platform
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/os.environ.get('GUARD_REVIEW_OUTPUT','results/review_20260917'); OUT.mkdir(parents=True,exist_ok=True)
p=ROOT/'results/gates/cells_gates_iemocap.json'
rows=json.loads(p.read_text()); runs=[]; groups=defaultdict(list)
for r in rows:
    c=r['ctx']; f,cut,m=c['f'],c['cut'],c['m']
    ref=np.load(ROOT/f'artifacts/iemocap_momke/folds/fold{f}_atv.npz',allow_pickle=True)
    uv=np.unique(ref['vid']); perm=np.random.default_rng(cut).permutation(len(uv))
    ids=np.where(np.isin(ref['vid'],np.array_split(uv[perm],3)[2]))[0]
    n=len(ids); a=int(round(n*r['rate'])); h=int(round(n*r['GUARD'][1]))
    assert abs(a-n*r['rate'])<1e-9 and abs(h-n*r['GUARD'][1])<1e-9
    assert 0<=h<=a<=n
    out=dict(fold=f,cut=cut,mask=m,n_test=n,n_apply=a,n_harm=h,
             joint_harm=h/n,apply_rate=a/n,conditional_harm=h/a if a else None,
             accuracy_gain=r['GUARD'][0],base_accuracy=r['base'],
             test_indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
    runs.append(out); groups[m].append(out)
summary={}
for m,rs in groups.items():
    n,a,h=[sum(r[k] for r in rs) for k in ('n_test','n_apply','n_harm')]
    j=float(np.mean([r['joint_harm'] for r in rs])); ap=float(np.mean([r['apply_rate'] for r in rs]))
    ch=[r['conditional_harm'] for r in rs if r['n_apply']]
    summary[m]=dict(n_runs=len(rs),zero_apply_runs=sum(r['n_apply']==0 for r in rs),
       test_occurrences=n,applied_occurrences=a,harmful_occurrences=h,
       pooled_conditional_harm=h/a if a else None,mean_run_conditional_harm=float(np.mean(ch)) if ch else None,
       mean_joint_harm=j,mean_apply_rate=ap,ratio_mean_rates=j/ap if ap else None,
       mean_accuracy_gain=float(np.mean([r['accuracy_gain'] for r in rs])),
       mean_base_accuracy=float(np.mean([r['base_accuracy'] for r in rs])))
report=dict(protocol='Counts reconstructed from unrounded saved rates, checked integral at 1e-9, with exact original fold/cut test lengths. Pooled estimand counts repeated held-out occurrences; no iid confidence interval is implied.',
            python=platform.python_version(),numpy=np.__version__,alpha=.2,delta=.05,loss='cross_entropy',reference='raw frozen output',
            source=str(p.relative_to(ROOT)),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),summary=summary,runs=runs)
(OUT/'conditional_iemocap.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
