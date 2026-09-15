"""LaTeX rows for the corrector-swap table, straight from probe_vs_knn.json.

Bold marks the better of the two correctors on each half; a star marks a joint
harm above the budget. Run after experiments/probe_vs_knn.py.
"""
import json, sys, glob, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALPHA = 0.2
ORDER = ['IEMOCAP', 'AVE', 'NinaPro', 'PTB-XL', 'DrugBAN']
NAME = {'NinaPro': 'NinaPro DB5'}

# Later files win, and a shadowed benchmark is announced rather than silently
# dropped: probe_vs_knn_b.json still holds a PTB-XL run made with the gates_rest
# loader, which is not the driver Table 3 reports.
res, src = {}, {}
for f in sorted(glob.glob(str(ROOT / 'results/gates/probe_vs_knn*.json'))):
    d = json.load(open(f))
    for k in d:
        if k in res:
            print(f'% {k}: {src[k]} bi ghi de boi {os.path.basename(f)}', file=sys.stderr)
        src[k] = os.path.basename(f)
    res.update(d)

def fmt(v, signed, best, star):
    s = f'{v:+.3f}' if signed else f'{v:.3f}'
    if best:
        s = f'\\bm{{{s}}}'
    if star:
        s += '^{\\ast}'
    return f'${s}$'

out = []
for b in ORDER:
    if b not in res:
        print(f'% thieu {b}', file=sys.stderr); continue
    k, l = res[b]['knn'], res[b]['linear']
    row = [NAME.get(b, b),
           fmt(k['gain'], True, k['gain'] > l['gain'], False),
           fmt(l['gain'], True, l['gain'] > k['gain'], False),
           fmt(k['harm'], False, k['harm'] < l['harm'], False),
           fmt(l['harm'], False, l['harm'] < k['harm'], False),
           fmt(k['blanket_harm'], False, False, k['blanket_harm'] > ALPHA),
           fmt(l['blanket_harm'], False, False, l['blanket_harm'] > ALPHA)]
    out.append(' & '.join(row) + r' \\')
print('\n'.join(out))

n = len([b for b in ORDER if b in res])
gk = sum(res[b]['knn']['gain'] for b in ORDER if b in res) / n
gl = sum(res[b]['linear']['gain'] for b in ORDER if b in res) / n
hk = sum(res[b]['knn']['harm'] for b in ORDER if b in res) / n
hl = sum(res[b]['linear']['harm'] for b in ORDER if b in res) / n
over = lambda w: sum(res[b][w]['blanket_harm'] > ALPHA for b in ORDER if b in res)
print(f'% trung binh: gain knn {gk:+.4f} probe {gl:+.4f} (chenh {gl-gk:+.4f}); '
      f'harm knn {hk:.4f} probe {hl:.4f}', file=sys.stderr)
print(f'% probe gain cao hon o {sum(res[b]["linear"]["gain"]>res[b]["knn"]["gain"] for b in ORDER if b in res)}/{n}; '
      f'probe harm cao hon o {sum(res[b]["linear"]["harm"]>res[b]["knn"]["harm"] for b in ORDER if b in res)}/{n}',
      file=sys.stderr)
print(f'% blanket vuot budget: knn {over("knn")}/{n}, probe {over("linear")}/{n}', file=sys.stderr)
