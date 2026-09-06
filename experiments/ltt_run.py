"""Run every benchmark driver with the LTT-augmented gate, one cell per condition."""
import sys, json, os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))
sys.path.insert(0, str(_HERE))
import gates_core, gates_ltt

gates_core.gate_row = gates_ltt.gate_row
LOG=[]; CUR=['?']
_inner = gates_ltt.gate_row
def _patched(*a,**k):
    r=_inner(*a,**k)
    f=sys._getframe(1).f_locals
    ctx={kk:vv for kk,vv in f.items()
         if isinstance(vv,(str,int)) and not isinstance(vv,bool)
         and not kk.startswith('_') and len(str(vv))<40}
    LOG.append(dict(driver=CUR[0],ctx=ctx,
                    **{kk:(list(vv) if isinstance(vv,tuple) else vv)
                       for kk,vv in r.items() if kk!='_meta'},
                    rate=r['_meta']['apply']))
    return r
gates_core.gate_row=_patched
_real=json.dump; json.dump=lambda *a,**k: None
for drv in ('gates_mosei2','gates_iemocap','gates_rest','gates_drugban','opp_dcl'):
    CUR[0]=drv; print('#####',drv,flush=True)
    src=_HERE/(drv+'.py')
    g={'__name__':'__main__','__file__':str(src)}
    exec(compile(src.read_text(),str(src),'exec'),g)
    _real(LOG,open(str(_HERE.parent / 'results/gates/ltt_cells.json'),'w'))
json.dump=_real; _real(LOG,open(str(_HERE.parent / 'results/gates/ltt_cells.json'),'w'))
print('##### DONE',len(LOG),'cells')
