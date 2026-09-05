#!/usr/bin/env python3
"""Add the PTB-XL driver the paper actually used to the per-cell registry."""
import sys, json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / 'experiments'))
os.chdir(ROOT)
import gates_core, gates_ltt
LOG = []
_inner = gates_ltt.gate_row

def _patched(*a, **k):
    r = _inner(*a, **k)
    f = sys._getframe(1).f_locals
    ctx = {kk: vv for kk, vv in f.items()
           if isinstance(vv, (str, int)) and not isinstance(vv, bool)
           and not kk.startswith('_') and len(str(vv)) < 40}
    LOG.append(dict(driver='gates_ptbxl', ctx=ctx,
                    **{kk: (list(vv) if isinstance(vv, tuple) else vv)
                       for kk, vv in r.items() if not kk.startswith('_')},
                    rate=r['_meta']['apply']))
    return r

gates_core.gate_row = _patched
_real = json.dump
json.dump = lambda *a, **k: None
g = {'__name__': '__main__', '__file__': str(ROOT / 'experiments' / 'gates_ptbxl.py')}
exec(compile(open('experiments/gates_ptbxl.py').read(), 'gates_ptbxl.py', 'exec'), g)
json.dump = _real
_real(LOG, open('results/gates/ptbxl_cells.json', 'w'))
print('DONE', len(LOG), 'cells')
