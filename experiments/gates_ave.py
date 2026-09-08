"""AVE per condition, the same six cells gates_rest runs, reported one by one."""
import numpy as np, sys, json, os, collections
from pathlib import Path
sys.path.insert(0, 'experiments')
from gates_ltt import gate_row
A = Path(os.environ.get('GUARD_ARTIFACTS', 'artifacts'))
out = {}
for cond in ('audio_only', 'visual_only', 'full'):
    d = np.load(f'{A}/ave_av_att/dumps/AV_att_{cond}.npz', allow_pickle=True)
    r = np.load(f'{A}/ave_av_att/dumps/AV_att_{cond}_retrieval_paper.npz', allow_pickle=True)
    P = d['probs'].reshape(-1, d['probs'].shape[-1]).astype(np.float64)
    Y = d['labels'].reshape(-1, d['labels'].shape[-1]).argmax(1)
    F = r['deploy_features'].astype(np.float64)
    pP, pF, pY = r['pool_probs'].astype(np.float64), r['pool_features'].astype(np.float64), r['pool_labels']
    n = len(pY)
    PP = np.concatenate([pP, P]); FF = np.concatenate([pF, F]); YY = np.concatenate([pY, Y])
    acc = collections.defaultdict(list)
    for seed in range(3):
        perm = np.random.default_rng(seed).permutation(len(Y)) + n
        k = len(perm) // 3
        row = gate_row(PP, FF, YY, (np.arange(n), perm[:k], perm[k:2*k], perm[2*k:]))
        acc['base'].append(row['_meta']['base']); acc['apply'].append(row['_meta']['apply'])
        for r_ in ('GUARD', 'blanket'):
            acc[r_].append(row[r_])
    b = float(np.mean(acc['base']))
    out[cond] = dict(frozen=b, guard=b + float(np.mean([x[0] for x in acc['GUARD']])),
                     blanket=b + float(np.mean([x[0] for x in acc['blanket']])),
                     harm=float(np.mean([x[1] for x in acc['GUARD']])),
                     bharm=float(np.mean([x[1] for x in acc['blanket']])),
                     apply=float(np.mean(acc['apply'])))
    print(cond, {k: round(v, 4) for k, v in out[cond].items()}, flush=True)
json.dump(out, open('results/gates/ave_conditions.json', 'w'), indent=1)
print('wrote results/gates/ave_conditions.json')
