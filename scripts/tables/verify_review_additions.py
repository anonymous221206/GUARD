"""Validate new manuscript tables against completed VUW experiment outputs.
Usage: python check_reported.py MANUSCRIPT.tex RESULTS_DIRECTORY
"""
import json,re,sys,hashlib
from pathlib import Path
tex=Path(sys.argv[1]).read_text();root=Path(sys.argv[2]);checked=0

def table(label):
    start=tex.index('\\label{'+label+'}')
    return tex[start:tex.index('\\end{table}',start)]

def check_row(label,name,want):
    global checked
    lines=[x for x in table(label).splitlines() if x.startswith(name+' & ')]
    assert len(lines)==1,(label,name,lines)
    clean=re.sub(r'\\(?:bm|mathbf)\{([^}]+)\}',r'\1',lines[0])
    got=[x for x in re.findall(r'\$([^$]+)\$',clean) if re.search(r'\d',x)]
    assert got==want,(label,name,got,want)
    checked+=len(want)

d=json.loads((root/'conditional_iemocap.json').read_text());assert len(d['runs'])==175
names={'a':'audio','t':'text','v':'visual','at':'audio $+$ text','av':'audio $+$ visual','tv':'text $+$ visual','atv':'intact'}
for mask,name in names.items():
    a=d['summary'][mask];rr=[r for r in d['runs'] if r['mask']==mask];assert len(rr)==25
    assert a['test_occurrences']==9206
    assert a['applied_occurrences']==sum(r['n_apply'] for r in rr)
    assert a['harmful_occurrences']==sum(r['n_harm'] for r in rr)
    assert all(0<=r['n_harm']<=r['n_apply']<=r['n_test'] for r in rr)
    for r in rr:assert r['conditional_harm']==(r['n_harm']/r['n_apply'] if r['n_apply'] else None)
    check_row('tab:conditional-harm',name,[f"{a['mean_joint_harm']:.3f}",f"{100*a['mean_apply_rate']:.2f}",f"{a['harmful_occurrences']}/{a['applied_occurrences']}",f"{100*a['pooled_conditional_harm']:.2f}",f"{100*a['mean_run_conditional_harm']:.2f}",str(a['zero_apply_runs'])])
for case in ('AVE','PTB-XL','IEMOCAP','NinaPro'):
    d=json.loads((root/f'efficiency_{case}.json').read_text())
    assert d['numpy']=='2.5.2' and d['sklearn']=='1.9.0'
    want=[str(d[k]) for k in ('n_pool','feature_dimension','outputs')]
    for kind in ('knn','linear'):
        t=d['results'][kind]['timings'][0];assert t['batch']==1 and t['repeats']==200
        want.append(f"{t['p50_batch_ms']:.3f}/{t['p95_batch_ms']:.3f}")
    for kind in ('knn','linear'):want.append(f"{d['results'][kind]['numerical_state_bytes']/2**20:.3f}")
    check_row('tab:efficiency',case,want)
for mode,expected_n in (('mosei',30),('cluster',6)):
    d=json.loads((root/f'probe_{mode}.json').read_text());assert len(d['cells'])==expected_n
    assert d['numpy']=='2.5.2' and d['sklearn']=='1.9.0'
    if mode=='mosei':
        assert all(r['context']['control_matches_saved'] for r in d['cells'])
        groups={'CMU-MOSEI':d['cells']}
        assert len({(r['context']['condition'],r['context']['seed']) for r in d['cells']})==30
    else:
        groups={n:[r for r in d['cells'] if r['context']['dataset']==k] for n,k in [('BioSNAP cluster','biosnap'),('BindingDB cluster','bindingdb')]}
        assert all(len(x)==3 for x in groups.values())
    for name,rows in groups.items():
        want=[]
        for metric in ('gain','harm','apply'):
            for kind in ('knn','linear'):
                v=sum(r['results'][kind][metric] for r in rows)/len(rows)
                want.append(f'{v:+.3f}' if metric=='gain' else (f'{v:.3f}' if metric=='harm' else f'{100*v:.1f}'))
        check_row('tab:probe-extension',name,want)
assert 'language-absent masks' in table('tab:allbench') or 'language-absent masks' in tex[tex.index('\\caption{\\textbf{Frozen model and GUARD'):tex.index('\\label{tab:allbench}')]
print(f'PASS: {checked} new reported table fields; 175 conditional runs; 30/30 MOSEI controls match saved; 6 target-pool cluster runs; all four timing cases use the release environment.')
