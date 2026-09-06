#!/usr/bin/env python3
"""How much labelled fit data does each rule need?

Both rules read labels in the same two places: the fit third, and the calibration
third. The difference is what they do with the fit third. GUARD computes its
score from the model outputs and uses the labels only to pick a blend weight; the
learned gate fits a classifier there and has to generalise. Shrinking the fit
third should therefore cost the learned gate more.
"""
import sys, os, collections, statistics as st
import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _HERE.parent / 'artifacts'))
from gates_ltt import gate_row

D = ARTIFACTS / 'mosei_cmad/dumps'
p = np.load(D / 'student_preds.npz', allow_pickle=True)
r = np.load(D / 'raw_features.npz', allow_pickle=True)
raw = r['test_y'].reshape(-1); y = (raw > 0).astype(int); keep = raw != 0
FE = {'v': r['test_vis'].astype(np.float64), 'a': r['test_ac'].astype(np.float64)}

def probs(m):
    s = p[f'test_{m}'].astype(np.float64); e = 1 / (1 + np.exp(-s))
    return np.stack([1 - e, e], 1)

FRACS = (0.05, 0.1, 0.25, 0.5, 1.0)
RULES = ('GUARD', 'CRC-guard', 'CRC-learned', 'CRC-confidence')
acc = collections.defaultdict(lambda: collections.defaultdict(list))
for cond in ('a', 'v', 'av'):
    F = np.concatenate([FE[m] for m in cond], 1)
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(len(y))
        pool, fit, conf, test = np.array_split(perm, 4)
        for fr in FRACS:
            k = max(30, int(len(fit) * fr))
            row = gate_row(probs(cond), F, y, (pool, fit[:k], conf, test), keep=keep,
                           targets=('hard', 'cross'), richer=probs('tav'))
            for rl in RULES:
                if rl in row and row[rl][0] == row[rl][0]:
                    acc[fr][rl].append(row[rl])

print(f'{"|D_fit|":>9}' + ''.join(f'{rl:>18}' for rl in RULES))
for fr in FRACS:
    n = int(1160 * fr)
    out = []
    for rl in RULES:
        v = acc[fr][rl]
        out.append(f'{st.fmean(x[0] for x in v):+.4f}/{st.fmean(x[1] for x in v):.3f}' if v else '---')
    print(f'{n:9d}' + ''.join(f'{c:>18}' for c in out))
