"""Matched-label probe extension using canonical probe_core.rows, unchanged.
MOSEI: same 30 cells as gates_mosei2, with recomputed retrieval control.
DrugBAN: target-domain pool from exp_drugban.build, 2 cluster datasets, 3 cuts.
The latter uses shared-core raw baseline and fit accuracy (a new comparison).
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import sklearn,platform
from threadpoolctl import threadpool_limits
from probe_core import rows
from gates_core import gate_row
from exp_drugban import build
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/os.environ.get('GUARD_REVIEW_OUTPUT','results/review_20260917');OUT.mkdir(parents=True,exist_ok=True)
ART=Path(os.environ.get('GUARD_ARTIFACTS',ROOT/'artifacts'))
mode=sys.argv[1];allrows=[]
previous=OUT/f'probe_{mode}.json'
if previous.exists(): allrows=json.loads(previous.read_text())['cells']

def record(ctx,result,split):
    out=dict(context=ctx,results=result,split_sizes={k:len(v) for k,v in zip(('pool','fit','conf','test'),split)},
      split_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in zip(('pool','fit','conf','test'),split)})
    allrows.append(out)
    summary={kind:{key:float(np.mean([r['results'][kind][key] for r in allrows])) for key in ('base','gain','harm','apply','blanket_gain','blanket_harm')} for kind in ('knn','linear')}
    doc=dict(protocol='Canonical probe_core.rows, unchanged grid and gate. Raw frozen baseline, accuracy selection on fit only. No test-label selection.',mode=mode,python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,
       source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('experiments/probe_core.py','experiments/gates_core.py','src/guard/action.py')},
       summary=summary,cells=allrows)
    path=OUT/f'probe_{mode}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(doc,indent=2)+'\n');tmp.replace(path)
    print(len(allrows),ctx,result,flush=True)

with threadpool_limits(limits=1):
    if mode=='mosei':
        D=ROOT/'artifacts/mosei_cmad/dumps';p=np.load(D/'student_preds.npz',allow_pickle=True);r=np.load(D/'raw_features.npz',allow_pickle=True)
        raw=r['test_y'].reshape(-1);Y=(raw>0).astype(int);keep=raw!=0
        def probs(m):
            s=p[f'test_{m}'].astype(float);e=1/(1+np.exp(-s));return np.stack([1-e,e],1)
        FE={'a':r['test_ac'].astype(float),'v':r['test_vis'].astype(float)}
        saved=json.loads((ROOT/'results/gates/cells_gates_mosei2.json').read_text())
        for cond in ('a','v','av'):
            F=np.concatenate([FE[m] for m in cond],1)
            for seed in range(10):
                if any(x['context'].get('condition')==cond and x['context'].get('seed')==seed for x in allrows):continue
                split=tuple(np.array_split(np.random.default_rng(seed).permutation(len(Y)),4))
                result=rows(probs(cond),F,Y,split,keep=keep,targets=('hard','cross'),richer=probs('tav'))
                old=[x for x in saved if x['ctx'].get('cond')==cond and x['ctx'].get('seed')==seed]
                assert len(old)==1,(cond,seed,[x['ctx'] for x in saved[:2]])
                now=np.array([result['knn']['gain'],result['knn']['harm'],result['knn']['apply']])
                before=np.array([*old[0]['GUARD'],old[0]['rate']])
                matches=bool(np.allclose(now,before,atol=1e-12,rtol=0))
                if not matches:
                    control=gate_row(probs(cond),F,Y,split,keep=keep,targets=('hard','cross'),richer=probs('tav'))
                    np.testing.assert_allclose(now,[*control['GUARD'],control['_meta']['apply']],atol=1e-12,rtol=0)
                record(dict(condition=cond,seed=seed,control_matches_saved=matches,
                    historical_control=before.tolist(),difference_from_saved=(now-before).tolist(),
                    independent_current_core_checked=not matches),result,split)
    elif mode=='cluster':
        for dataset in ('biosnap','bindingdb'):
            for seed in range(3):
                if any(x['context'].get('dataset')==dataset and x['context'].get('seed')==seed for x in allrows):continue
                h,s=build(ART/f'drugban_processed/drugban_{dataset}_cluster_s42','prot50','deployment',seed)
                split=(s.pool,s.fit,s.conf,s.test)
                result=rows(h.probs,h.features,h.labels,split,targets=('hard','cross'),richer=h.richer_probs)
                record(dict(dataset=dataset,condition='prot50',seed=seed,pool='deployment'),result,split)
    else: raise ValueError(mode)
print('COMPLETE',mode,len(allrows),flush=True)
