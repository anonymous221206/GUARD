#!/usr/bin/env python3
"""LaTeX rows for every table added or changed in the revision, from saved outputs only.

Run after the rebuttal drivers. With --check TEX every generated row must appear
verbatim in the manuscript, so a table cannot drift from these files.
"""
import json, collections, sys, re
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
O, G = ROOT / 'results/rebuttal', ROOT / 'results/gates'
BENCH = [('mosei', 'CMU-MOSEI'), ('iemocap', 'IEMOCAP'), ('opportunity', 'OPPORTUNITY'), ('ave', 'AVE'),
         ('ninapro', 'NinaPro DB5'), ('ptbxl', 'PTB-XL'), ('drugban', 'DrugBAN')]
KEY = dict(mosei=lambda r: r['ctx']['cond'], iemocap=lambda r: r['ctx']['m'], opportunity=lambda r: r['ctx']['cfg'],
           ave=lambda r: r['ctx']['cond'], ninapro=lambda r: r['ctx']['c'], ptbxl=lambda r: r['ctx']['c'],
           drugban=lambda r: r['ctx']['d'] + '/' + r['ctx']['cond'])
def jl(p): return [json.loads(l) for l in open(p)]
F = {b: jl(O / f'factorial_{b}.jsonl') for b, _ in BENCH}
for b, c in F.items(): assert not any('error' in r for r in c), b
OUT = collections.OrderedDict()
mean = lambda v: float(np.nanmean(np.array([np.nan if x is None else x for x in v], float)))
def fx(r, sc, po): return r['x']['fact'][sc][po]
def pm(c, sc, po, f): return mean([fx(r, sc, po).get(f) for r in c])
g3 = lambda v: f'{v:+.3f}'.replace('0.', '.', 1)
h3 = lambda v: f'{v:.3f}'.replace('0.', '.', 1)
# ---- Table 3
rows = []
for b, name in BENCH:
    c = F[b]; bl = (mean([r['x']['blanket']['gain'] for r in c]), mean([r['x']['blanket']['harm'] for r in c]))
    cols = [('confidence', 'crc'), ('learned', 'ltt'), ('learned', 'crc'), ('confidence', 'crc_t'), ('distance', 'crc_t'), ('random', 'crc_t'), ('learned', 'ltt_t'), ('learned', 'crc_t')]
    vals = [(pm(c, s, p, 'gain'), pm(c, s, p, 'harm')) for s, p in cols]
    low = min(range(len(vals)), key=lambda i: vals[i][1]); low_t = None
    cell = lambda v, star=True: f'${g3(v[0])}$/${h3(v[1])}$' + ('$^{\\ast}$' if star and round(v[1], 3) > .2 else '')
    rows.append(' & '.join([name, cell(bl)] + [cell(v) for v in vals]) + r' \\')
OUT['tab:gates'] = rows
# ---- all-bench row and blocks for IEMOCAP
c = F['iemocap']; inc = [r for r in c if r['ctx']['m'] != 'atv']
gm = collections.defaultdict(list)
for r in c: gm[r['ctx']['m']].append(r)
base = {m: mean([r['base'] for r in v]) for m, v in gm.items()}
gain = {m: mean([r['GUARD'][0] for r in v]) for m, v in gm.items()}
harm = {m: mean([r['GUARD'][1] for r in v]) for m, v in gm.items()}
app = {m: mean([r['rate'] for r in v]) for m, v in gm.items()}
six = ['a', 't', 'v', 'at', 'av', 'tv']
f6 = np.mean([base[m] for m in six]); g6 = np.mean([base[m] + gain[m] for m in six])
OUT['tab:allbench'] = [f"IEMOCAP (MoMKE)            & $4$-class WA  & $6$ & ${f6:.3f}$ & $\\mathbf{{{g6:.3f}}}$ & ${min(harm[m] for m in six):.3f}$--${max(harm[m] for m in six):.3f}$ & ${100*min(app[m] for m in six):.0f}$--${100*max(app[m] for m in six):.0f}$ \\\\"]
NM = dict(a='audio only', t='text only', v='visual only', at='audio $+$ text', av='audio $+$ visual', tv='text $+$ visual', atv='\\emph{intact}')
lines = []
for m in six + ['atv']:
    fb, gb = base[m], base[m] + gain[m]
    a, b_ = (f'${fb:.3f}$', f'$\\mathbf{{{gb:.3f}}}$') if round(gb, 3) > round(fb, 3) else ((f'$\\mathbf{{{fb:.3f}}}$', f'${gb:.3f}$') if round(gb, 3) < round(fb, 3) else (f'${fb:.3f}$', f'${gb:.3f}$'))
    lines.append(f"{NM[m]} & {a} & {b_} & ${harm[m]:.3f}$ & ${100*app[m]:.0f}$ \\\\")
OUT['tab:fourdomains:iemocap'] = lines
blh = {m: mean([r['blanket'][1] for r in v]) for m, v in gm.items()}
OUT['tab:blanket'] = [f"IEMOCAP      & $7$ & ${mean([r['blanket'][0] for r in c]):+.3f}$ & ${min(blh.values()):.3f}$--${max(blh.values()):.3f}$ & $\\mathbf{{{min(harm.values()):.3f}}}$--$\\mathbf{{{max(harm.values()):.3f}}}$ \\\\"]
ratios = {n: mean([r['blanket'][1] for r in F[b]]) / mean([r['GUARD'][1] for r in F[b]]) for b, n in BENCH}
OUT['stats:ratio'] = [f'% blanket/GUARD joint harm: ' + ', '.join(f'{k} {v:.2f}' for k, v in ratios.items())]
# ---- conditional IEMOCAP
s = json.load(open(ROOT / 'results/review_20260917/conditional_iemocap.json'))['summary']
NC = dict(a='audio', t='text', v='visual', at='audio $+$ text', av='audio $+$ visual', tv='text $+$ visual', atv='intact')
OUT['tab:conditional-harm'] = [f"{NC[m]} & ${v['mean_joint_harm']:.3f}$ & ${100*v['mean_apply_rate']:.2f}$ & ${v['harmful_occurrences']}/{v['applied_occurrences']}$ & ${100*v['pooled_conditional_harm']:.2f}$ & ${100*v['mean_run_conditional_harm']:.2f}$ & ${v['zero_apply_runs']}$ \\\\" for m in six + ['atv'] for v in [s[m]]]
# ---- reliability, all benchmarks
rel = []
for b, name in BENCH:
    c = F[b]; xs = [fx(r, 'learned', 'crc_t') for r in c]
    H, A_ = mean([x['harm'] for x in xs]), mean([x['eff_apply'] for x in xs])
    g = collections.defaultdict(list)
    for r in c: g[KEY[b](r)].append(fx(r, 'learned', 'crc_t'))
    wc = max(np.mean([y['harm'] for y in v]) / max(np.mean([y['eff_apply'] for y in v]), 1e-9) for v in g.values())
    fm, fp = mean([x['flip_m'] for x in xs]), mean([x['flip_p'] for x in xs])
    hr = mean([x.get('h_right') for x in xs]); hk = mean([x.get('harm_k') for x in xs]); er = mean([x.get('err_corr') for x in xs])
    over = sum(r['GUARD'][1] > .2 for r in c)
    aucs = [mean([r['x']['fact']['learned'].get(k) for r in c]) for k in ('auc_help', 'auc_harm')]
    aco = mean([r['x']['fact']['confidence'].get('auc_help') for r in c])
    rel.append(f"{name} & ${100*H/A_:.1f}$ & ${100*wc:.1f}$ & ${mean([x.get('mean_d') for x in xs]):+.2f}$ & ${fm:.3f}$ & ${fp:.3f}$ & " + ('---' if b == 'ptbxl' else f"${100*er:.1f}$") + ' & ' + ('---' if b == 'ptbxl' else f"${100*hr/hk:.0f}$") + f" & ${aucs[0]:.2f}$ & ${aco:.2f}$ & ${aucs[1]:.2f}$ & ${over}/{len(c)}$ \\\\")
OUT['tab:reliability'] = rel
mt = []; ck = lambda d: json.dumps(d, sort_keys=True)
ML = collections.defaultdict(dict)
for r in jl(O / 'matched_l1.jsonl'): ML[r['driver']][ck(r['ctx'])] = r
st = []
for b, name in BENCH:
    c = F[b]; R_ = mean([r['rate'] for r in c]); cells = []
    for k in ('GUARD', 'l1', 'confidence', 'agreement', 'random'):
        val = (lambda r: ML[b][ck(r['ctx'])]['l1']) if k == 'l1' else (lambda r: r[k])
        g, h = mean([val(r)[0] for r in c]), mean([val(r)[1] for r in c])
        cells.append(f'${g3(g)}$/${h3(h)}$/${100*h/R_:.1f}$')
    mt.append(f'{name} & ${100*R_:.1f}$ & ' + ' & '.join(cells) + r' \\')
    for lab, fa, fb, rate in (('guard rate', lambda r: r['GUARD'], lambda r: ML[b][ck(r['ctx'])]['l1'], R_),
                              ('l1 rate', lambda r: ML[b][ck(r['ctx'])]['at_l1']['learned'], lambda r: ML[b][ck(r['ctx'])]['at_l1']['l1'], mean([ML[b][ck(r['ctx'])]['rate_l1'] for r in c]))):
        g = collections.defaultdict(list)
        for r in c: g[KEY[b](r)].append(np.array(fa(r)[:2]) - np.array(fb(r)[:2]))
        v = np.array([np.mean(x, 0) for x in g.values()]); e = v.std(0, ddof=1) / np.sqrt(len(v))
        st.append(f'% {b} {lab}: learned - l1 gain {100*v[:, 0].mean():+.2f}+-{100*e[0]:.2f}, harm {100*v[:, 1].mean():+.2f}+-{100*e[1]:.2f}, conditional {100*v[:, 1].mean()/rate:+.2f}')
OUT['tab:matched'] = mt; OUT['stats:matched'] = st
# ---- paired
def paired(c, key, a, b, f):
    g = collections.defaultdict(list)
    for r in c: g[key(r)].append(fx(r, *a)[f] - fx(r, *b)[f])
    v = np.array([np.mean(x) for x in g.values()]); return v.mean(), v.std(ddof=1) / np.sqrt(len(v))
pr = []
G_ = ('learned', 'crc_t')
for b, name in BENCH:
    c = F[b]; cells = []
    for o in (('confidence', 'crc_t'), ('random', 'crc_t'), ('distance', 'crc_t')):
        d = paired(c, KEY[b], G_, o, 'gain'); e = paired(c, KEY[b], G_, o, 'harm')
        cells += [f'${d[0]:+.3f}\\pm{d[1]:.3f}$', f'${e[0]:+.3f}\\pm{e[1]:.3f}$']
    d = paired(c, KEY[b], ('learned', 'ltt_t'), G_, 'gain'); e = paired(c, KEY[b], ('learned', 'ltt_t'), G_, 'harm')
    cells += [f'${d[0]:+.3f}\\pm{d[1]:.3f}$', f'${e[0]:+.3f}\\pm{e[1]:.3f}$']
    pr.append(name + ' & ' + ' & '.join(cells).replace('0.', '.') + r' \\')
OUT['tab:paired'] = pr
# ---- budget
B = [r for r in jl(O / 'budget.jsonl') if r.get('r')]
def bm(sel, p, f):
    z = [r['r'][p][f] for r in B if sel(r)]; return float(np.mean(z)) if z else None
LV = (60, 120, 240, 480, 960, 'full')
main = []
cols = [('mosei', None), ('opportunity', None), ('drugban_deploy', 'biosnap')]
for L in LV:
    cells = []
    for bench, ds in cols:
        for kind, tg in (('knn', 'hard'), ('knn', 'cross'), ('linear', 'hard'), ('linear', 'cross')):
            v = bm(lambda r: r['bench'] == bench and r['B'] == L and r['kind'] == kind and r['target'] == tg and (ds is None or r['ctx'].get('dataset') == ds), 'crc_t', 'gain')
            cells.append(f'${g3(v)}$')
    main.append(('full' if L == 'full' else f'${L}$') + ' & ' + ' & '.join(cells) + r' \\')
OUT['tab:budget'] = main
full = []
for bench, ds, nm in (('mosei', None, 'CMU-MOSEI'), ('opportunity', None, 'OPPORTUNITY'), ('drugban_deploy', 'biosnap', 'BioSNAP cluster')):
    for L in (60, 120, 240, 480, 960, 1800, 'full'):
        cells = []
        for kind, tg in (('knn', 'hard'), ('knn', 'cross'), ('linear', 'hard'), ('linear', 'cross')):
            sel = lambda r: r['bench'] == bench and r['B'] == L and r['kind'] == kind and r['target'] == tg and (ds is None or r['ctx'].get('dataset') == ds)
            cells.append(f"${g3(bm(sel,'crc_t','gain'))}$/${h3(bm(sel,'crc_t','harm'))}$ (${g3(bm(sel,'ltt_t','gain'))}$)")
        full.append(f"{nm if L == 60 else ''} & " + ('full' if L == 'full' else f'${L}$') + ' & ' + ' & '.join(cells) + r' \\')
OUT['tab:budget-full'] = full
reg = []
for bench, nm in (('ptbxl', 'PTB-XL'), ('ave', 'AVE'), ('iemocap', 'IEMOCAP')):
    for L in (40, 80, 160, 320, 640, 'full'):
        sel = lambda r: r['bench'] == bench and r['B'] == L
        reg.append(f"{nm if L == 40 else ''} & " + ('full' if L == 'full' else f'${L}$') + f" & ${g3(bm(sel,'crc_t','gain'))}$ & ${h3(bm(sel,'crc_t','harm'))}$ & ${100*bm(sel,'crc_t','apply'):.0f}$ & ${g3(bm(sel,'ltt_t','gain'))}$ & ${h3(bm(sel,'ltt_t','harm'))}$ & ${100*bm(sel,'ltt_t','apply'):.0f}$ & ${h3(bm(sel,'crc','harm'))}$ \\\\")
OUT['tab:budget-pool'] = reg
# ---- shift
S = [r for r in jl(O / 'shift.jsonl') if 'error' not in r]; mix = [r for r in S if r['kind'] == 'mix']; pair = [r for r in S if r['kind'] == 'pair']
mx = []
for lab, sel in (('CMU-MOSEI', lambda r: r['bench'] == 'mosei'), ('NinaPro DB5', lambda r: r['bench'] == 'nina'), ('IEMOCAP, $\\eta=0.3$', lambda r: r['bench'] == 'iem' and r['ctx']['eta'] == .3), ('IEMOCAP, $\\eta=0.5$', lambda r: r['bench'] == 'iem' and r['ctx']['eta'] == .5)):
    x = [r for r in mix if sel(r)]; cells = []
    for po in ('pooled_crc', 'pooled_crc_t', 'mask_crc', 'mask_crc_t'):
        pats = sorted({p for r in x for p, v in r['out'][po]['per'].items() if v['n'] >= 20})
        pmn = {p: float(np.mean([r['out'][po]['per'][p]['harm'] for r in x if p in r['out'][po]['per'] and r['out'][po]['per'][p]['n'] >= 20])) for p in pats}
        cells += [f"${np.mean([r['out'][po]['gain'] for r in x]):+.3f}$", f"${np.mean([r['out'][po]['harm'] for r in x]):.3f}$", f"${max(pmn.values()):.3f}$"]
    mx.append(lab + ' & ' + ' & '.join(cells) + r' \\')
OUT['tab:mixed'] = mx
ORD = dict(mosei={'a': 1, 'v': 1, 'av': 2}, iem={'a': 1, 't': 1, 'v': 1, 'at': 2, 'av': 2, 'tv': 2}, nina={'12': 3, '8': 2, '6': 2, '4': 1})
def d(b, p): return 'same' if p[0] == p[1] else ('rich' if ORD[b][p[0]] > ORD[b][p[1]] else 'poor')
ru = []
for b, nm in (('mosei', 'CMU-MOSEI'), ('iem', 'IEMOCAP'), ('nina', 'NinaPro DB5')):
    for dr, lab in (('same', 'same condition'), ('rich', 'richer to poorer'), ('poor', 'poorer to richer')):
        cells = []
        for sc in ('s1', 's2'):
            x = [r for r in pair if r['bench'] == b and r['scen'] == sc and d(b, r['pair']) == dr]
            cells += [f"${np.mean([r['out']['crc_t']['gain'] for r in x]):+.3f}$", f"${np.mean([r['out']['crc_t']['harm'] for r in x]):.3f}$", f"${sum(r['out']['crc_t']['harm'] > .2 for r in x)}/{len(x)}$"]
        ru.append(f"{nm if dr == 'same' else ''} & {lab} & " + ' & '.join(cells) + r' \\')
OUT['tab:reuse'] = ru
# ---- IEMOCAP missing rate
e = json.load(open(ROOT / 'results/iemocap_eta.json')); ET = sorted({r['eta'] for r in e})
fz = [mean([r['base'] for r in e if r['eta'] == t]) for t in ET]; gd = [mean([r['base'] + r['GUARD'][0] for r in e if r['eta'] == t]) for t in ET]
dots = lambda v: f'${v:.3f}$'.replace('0.', '.', 1)
OUT['tab:iemocap'] = ['MoMKE, our reproduction (frozen) & ' + ' & '.join(dots(v) for v in fz) + f' & {dots(np.mean(fz))} & --- \\\\',
                      '\\quad $+$ GUARD          & ' + ' & '.join(dots(v) for v in gd) + f' & {dots(np.mean(gd))} & ${(np.mean(gd)-np.mean(fz)):+.3f}$'.replace('0.', '.', 1) + ' \\\\']
rep = collections.defaultdict(list)
for r in e: rep[(r['fold'], r['draw'])].append(r['GUARD'][0])
v = np.array([np.mean(x) for x in rep.values()])
OUT['stats:eta'] = [f'% eta gain {v.mean():+.4f} SE {v.std(ddof=1)/np.sqrt(len(v)):.4f}; blanket harm by eta ' + ', '.join(f"{t}:{mean([r['blanket'][1] for r in e if r['eta']==t]):.3f}" for t in ET) + '; GUARD harm ' + ', '.join(f"{t}:{mean([r['GUARD'][1] for r in e if r['eta']==t]):.3f}" for t in ET)]
rep = collections.defaultdict(list)
for r in F['iemocap']:
    if r['ctx']['m'] != 'atv': rep[(r['ctx']['f'], r['ctx']['cut'])].append(r['GUARD'][0])
v = np.array([np.mean(x) for x in rep.values()]); OUT['stats:iem6'] = [f'% six masks gain {v.mean():+.4f} SE {v.std(ddof=1)/np.sqrt(len(v)):.4f} negative {(v<0).sum()}/{len(v)}']
OUT['stats:mosei'] = ['% keep-set conditional harm and error after correction: ' + ', '.join(f"{b} {mean([fx(r,'learned','crc_t').get('harm_k') for r in F[b]]):.3f}/{mean([fx(r,'learned','crc_t').get('err_corr') for r in F[b]]):.3f}" for b in ('mosei', 'drugban'))]
A_ = json.load(open(O / 'action_iemocap.json'))['summary']
OUT['tab:actionscore:iemocap'] = [f"{k}: ${A_[k]['gain']:+.3f}$/${A_[k]['harm']:.2f}$ auroc {A_[k]['auroc']:.3f}" for k in A_ if k != 'cells']
if len(sys.argv) > 2 and sys.argv[1] == '--check':
    tex = Path(sys.argv[2]).read_text(); norm = lambda s: re.sub(r'\s+', ' ', s).strip(); T = norm(tex); bad = 0; n = 0
    for k, ls in OUT.items():
        if k.startswith('stats') or k.endswith(':iemocap') and 'action' in k: continue
        for l in ls:
            n += 1
            if norm(l) not in T: bad += 1; print('MISSING', k, l[:110])
    print(f'checked {n} generated rows, {bad} missing'); sys.exit(1 if bad else 0)
for k, ls in OUT.items():
    print('%%%%', k); print('\n'.join(ls))
