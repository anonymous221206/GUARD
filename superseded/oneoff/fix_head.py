from pathlib import Path
p = Path('experiments/ltt_run.py'); s = p.read_text()
old = """import sys, json, os
_HERE = Path(__file__).resolve().parent
import gates_core, gates_ltt
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))
"""
new = """import sys, json, os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))
sys.path.insert(0, str(_HERE))
import gates_core, gates_ltt
"""
assert s.count(old) == 1, 'khong khop dau file'
p.write_text(s.replace(old, new))
print('sua thu tu import')
