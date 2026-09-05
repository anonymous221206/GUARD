"""Run every benchmark driver with the LTT-augmented gate, one cell per condition."""
import sys, json, os
import gates_core, gates_ltt
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))

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
    g={'__name__':'__main__','__file__':drv+'.py'}
    exec(compile(open(drv+'.py').read(),drv+'.py','exec'),g)
    _real(LOG,open('ltt_cells.json','w'))
json.dump=_real; _real(LOG,open('ltt_cells.json','w'))
print('##### DONE',len(LOG),'cells')
