#!/usr/bin/env python3
"""Run one benchmark driver under the LTT-augmented gate.

The five drivers are independent, so running them separately turns a four-hour
serial pass into one bounded by the slowest driver.
"""
import sys, json, os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))
sys.path.insert(0, str(_HERE))
import gates_core, gates_ltt

DRV = sys.argv[1]
LOG = []
_inner = gates_ltt.gate_row

def _patched(*a, **k):
    r = _inner(*a, **k)
    f = sys._getframe(1).f_locals
    ctx = {kk: vv for kk, vv in f.items()
           if isinstance(vv, (str, int)) and not isinstance(vv, bool)
           and not kk.startswith('_') and len(str(vv)) < 40}
    LOG.append(dict(driver=DRV, ctx=ctx,
                    **{kk: (list(vv) if isinstance(vv, tuple) else vv)
                       for kk, vv in r.items() if not kk.startswith('_')},
                    rate=r['_meta']['apply']))
    return r

gates_core.gate_row = _patched
_real = json.dump
json.dump = lambda *a, **k: None
src = _HERE / (DRV + '.py')
exec(compile(src.read_text(), str(src), 'exec'),
     {'__name__': '__main__', '__file__': str(src)})
json.dump = _real
out = _ROOT / 'results/gates' / f'cells_{DRV}.json'
out.parent.mkdir(parents=True, exist_ok=True)
_real(LOG, open(out, 'w'))
print(f'DONE {DRV} {len(LOG)} cells -> {out}', flush=True)
