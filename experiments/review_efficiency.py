"""Controlled correction-overhead benchmark; fixed settings, not Table 6 winners.
python experiments/review_efficiency.py CASE
Measures actual canonical exact-kNN and canonical linear probe on saved features.
Single CPU thread, float64, k=20 uniform hard targets, z-score, logistic C=1.
Includes feature transform, target, blend, score and gate; excludes frozen host.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,time,gc,tracemalloc,platform,hashlib
from pathlib import Path
import numpy as np
import sklearn
from threadpoolctl import threadpool_info,threadpool_limits
from guard import targets as T,action as A,losses as L
from guard.pipeline import _select_beta
from probe_core import _fit_hard
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'
OUT=ROOT/'results/review_20260917'; OUT.mkdir(exist_ok=True)
case=sys.argv[1]

def load(case):
    if case=='AVE':
        d=np.load(ART/'ave_av_att/dumps/AV_att_audio_only.npz',allow_pickle=True)
        r=np.load(ART/'ave_av_att/dumps/AV_att_audio_only_retrieval_paper.npz',allow_pickle=True)
        P=d['probs'].reshape(-1,d['probs'].shape[-1]); Y=d['labels'].reshape(-1,d['labels'].shape[-1]).argmax(1)
        F=r['deploy_features']; pP,pF,pY=r['pool_probs'],r['pool_features'],r['pool_labels']; name='audio_only'
    elif case=='PTB-XL':
        d=np.load(ART/'ptbxl_resnet1d_wang/preds.npz',allow_pickle=True); r=np.load(ART/'ptbxl_resnet1d_wang/raw_features.npz',allow_pickle=True)
        pP,P=d['train_a'],d['sess1_a']; pF,F=r['train_a'],r['sess1_a']; pY,Y=r['train_y'],r['sess1_y'];name='a'
    elif case=='IEMOCAP':
        d=np.load(ART/'iemocap_momke/folds/fold0_a.npz',allow_pickle=True)
        pP,P=d['pool_probs'],d['probs'];pF,F=d['pool_feats'],d['feats'];pY,Y=d['pool_labels'],d['labels'];name='fold0_audio'
    elif case=='NinaPro':
        sub=sorted(sorted((ART/'ninapro_cnn').glob('seed*'))[0].glob('subject*'))[0]
        d=np.load(sub/'preds.npz',allow_pickle=True);r=np.load(sub/'masked_embeddings.npz',allow_pickle=True)
        pP,P=d['train_4'],d['sess1_4'];pF,F=r['train_4'],r['sess1_4'];pY,Y=d['train_y'],d['sess1_y'];name=str(sub.relative_to(ART))+'/4'
    else: raise ValueError(case)
    P,pP,F,pF=np.asarray(P,float),np.asarray(pP,float),np.asarray(F,float),np.asarray(pF,float)
    if case=='NinaPro':
        P=np.clip(P,1e-12,None);P/=P.sum(1,keepdims=True);pP=np.clip(pP,1e-12,None);pP/=pP.sum(1,keepdims=True)
    perm=np.random.default_rng(0).permutation(len(Y)); n=len(Y)//3
    return pP,pF,pY,P,F,Y,(perm[:n],perm[n:2*n],perm[2*n:]),name

def array_bytes(obj,seen=None):
    # Numerical deploy-state: count arrays once, including any retained pool labels.
    if seen is None: seen=set()
    if id(obj) in seen: return 0
    seen.add(id(obj))
    if isinstance(obj,np.ndarray): return obj.nbytes
    if isinstance(obj,(list,tuple)): return sum(array_bytes(x,seen) for x in obj)
    if isinstance(obj,dict): return sum(array_bytes(x,seen) for x in obj.values())
    if callable(obj) and getattr(obj,'__closure__',None): return sum(array_bytes(c.cell_contents,seen) for c in obj.__closure__)
    if hasattr(obj,'__dict__'): return array_bytes(vars(obj),seen)
    return 0

def timing(fn,F,P,batch):
    rng=np.random.default_rng(49+batch); repeats=200 if batch==1 else 50
    indices=[rng.choice(len(F),batch,replace=False) for _ in range(repeats+10)]
    qs=[(np.ascontiguousarray(F[i]),np.ascontiguousarray(P[i])) for i in indices]
    for q,p in qs[:10]: fn(q,p)
    vals=[];gc.disable()
    try:
        for q,p in qs[10:]:
            start=time.perf_counter_ns();fn(q,p);vals.append((time.perf_counter_ns()-start)/1e6)
    finally:gc.enable()
    peaks=[]
    for q,p in qs[:3]:
        tracemalloc.start();fn(q,p);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop();peaks.append(peak)
    return dict(batch=batch,repeats=repeats,p50_batch_ms=float(np.median(vals)),p95_batch_ms=float(np.percentile(vals,95)),
                p50_per_query_ms=float(np.median(vals)/batch),peak_traced_allocation_bytes=max(peaks))

with threadpool_limits(limits=1):
    pP,pF,pY,P,F,Y,(fit,conf,test),name=load(case)
    loss=L.get('bernoulli' if Y.ndim==2 else 'cross_entropy')
    z=T.retrieval_space(pF,'standardise'); pool=np.ascontiguousarray(z(pF)); C=P.shape[1];k=min(20,len(pool)-1)
    vals=T.hard_label_values(pY,C,loss.simplex)
    build=time.perf_counter(); probe=_fit_hard('linear',1.,pool,pY,C,loss.simplex);probe_fit=time.perf_counter()-build
    result={}
    for kind in ('knn','linear'):
        target=(lambda x:T.knn_average(x,pool,vals,k,weighting='uniform')) if kind=='knn' else probe
        tt={s:target(z(F[idx])) for s,idx in [('fit',fit),('conf',conf),('test',test)]}
        b=_select_beta(P[fit],tt['fit'],Y[fit],loss,'loss')
        corrected=lambda idx,tag:(1-b)*P[idx]+b*tt[tag]
        sc=A.fit_action_score(P[fit],tt['fit'],corrected(fit,'fit'),Y[fit],loss)
        g=A.certify_action(sc,P[conf],tt['conf'],corrected(conf,'conf'),Y[conf],P[test],tt['test'],loss,.2,.05,
            fit=(P[fit],tt['fit'],corrected(fit,'fit'),Y[fit]))
        lam=g['lambda']
        def predict(q,base):
            t=target(z(q));c=(1-b)*base+b*t;s=A.score(sc,base,t,loss.simplex)
            ap=np.zeros(len(base),bool) if lam is None else s>lam
            return np.where(ap[:,None],c,base)
        # Verify the timed deployment path agrees with canonical batch decisions.
        expected=np.where(g['apply'][:,None],corrected(test,'test'),P[test])
        np.testing.assert_allclose(predict(F[test],P[test]),expected,rtol=1e-11,atol=1e-12)
        for j in range(min(20,len(test))):
            np.testing.assert_allclose(predict(F[test[j:j+1]],P[test[j:j+1]]),expected[j:j+1],rtol=1e-10,atol=1e-11)
        state=array_bytes([z,pool,vals,sc]) if kind=='knn' else array_bytes([z,probe,sc])
        result[kind]=dict(beta=b,scorer_constant=sc is None,numerical_state_bytes=state,
             timings=[timing(predict,F[test],P[test],batch) for batch in (1,64)])
        print(case,kind,result[kind],flush=True)
    cpuinfo=Path('/proc/cpuinfo')
    cpu=(next((x.split(':',1)[1].strip() for x in cpuinfo.read_text().splitlines()
               if x.startswith('model name')),'') if cpuinfo.exists() else platform.processor())
    affinity=(sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else [])
    pools=threadpool_info()
    for item in pools:
        if item.get('filepath'):
            item['filepath']=Path(item['filepath']).name
    out=dict(case=case,condition=name,n_pool=len(pool),feature_dimension=pool.shape[1],outputs=C,
        settings=dict(k=k,weighting='uniform',space='standardise',target='hard',temperature=1,linear_C=1,alpha=.2,delta=.05),
        scope='Controlled fixed-configuration microbenchmark, not timings of per-cell Table 6 selected winners. Frozen feature extraction, disk I/O, fitting and calibration excluded. Canonical kNN includes pool-norm recomputation. State bytes exclude Python/interpreter overhead; traced peak is temporary allocation, not process RSS.',
        probe_fit_seconds=probe_fit,python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,
        cpu=cpu,affinity=affinity,threadpools=pools,results=result)
    (OUT/f'efficiency_{case}.json').write_text(json.dumps(out,indent=2)+'\n')
