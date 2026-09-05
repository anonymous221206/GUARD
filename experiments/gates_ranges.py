"""One registry behind both rule tables.

The per-benchmark drivers average over conditions before printing, so the summary
table and the per-condition ranges could drift apart. This runs the same drivers
with gate_row wrapped, records every (condition, seed) cell, and writes one file
from which both the means and the ranges are read.
"""
import sys, json, os
import gates_core
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))

_orig=gates_core.gate_row
LOG=[]; CUR=['?']
def _patched(*a,**k):
    r=_orig(*a,**k)
    f=sys._getframe(1).f_locals
    ctx={kk:vv for kk,vv in f.items()
         if isinstance(vv,(str,int)) and not isinstance(vv,bool)
         and not kk.startswith('_') and len(str(vv))<40}
    LOG.append(dict(driver=CUR[0],ctx=ctx,
                    GUARD=list(r['GUARD']),blanket=list(r['blanket']),
                    apply=r['_meta']['base'],rate=r['_meta']['apply']))
    return r
gates_core.gate_row=_patched
_real_dump=json.dump
json.dump=lambda *a,**k: None   # drivers must not overwrite their own result files
for drv in ('gates_mosei2','gates_iemocap','gates_rest','gates_drugban','opp_dcl'):
    CUR[0]=drv; print('#####',drv,flush=True)
    g={'__name__':'__main__','__file__':drv+'.py'}
    src=open(drv+'.py').read()
    exec(compile(src,drv+'.py','exec'),g)
    _real_dump(LOG,open('gates_cells.json','w'))
json.dump=_real_dump
_real_dump(LOG,open('gates_cells.json','w'))
print('##### DONE',len(LOG),'cells')
