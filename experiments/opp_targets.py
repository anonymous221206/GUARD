"""OPPORTUNITY under each retrieval target separately.

The main driver lets the fit split choose between the two targets, which is the
deployed policy. The cross-mask comparison needs them apart: one uses deployment
labels in the retrieval bank, the other uses none, and the question is what the
labels are worth. Same hosts, same splits, same budget as opp_dcl.py.
"""
import numpy as np, sys, json, os, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gates_ltt import gate_row
from opp_split import deploy_split

_ROOT = Path(__file__).resolve().parents[1]
A = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))
D = A / 'opportunity_dcl_v2'
CFG = ['low_cost_accels_only', 'no_imu_family', 'no_shoes', 'severe_three_sensors']
OUT = _ROOT / 'results/gates/opp_targets.json'

y = np.load(D / 'deploy_y.npy')
acc = collections.defaultdict(list)
cells = []          # one row per (configuration, seed), for the cross-mask screen
for cfg in CFG:
    F = np.load(D / f'retfeat_{cfg}_deploy.npy').astype(np.float64)
    for s in (0, 1, 2):
        P = np.load(D / f'probs_condition_specialist_{cfg}_deploy_s{s}.npy').astype(np.float64)
        rich = np.load(D / f'richer_deploy_s{s}.npy').astype(np.float64)
        split = deploy_split(len(y), s)
        for tg in ('hard', 'cross'):
            r = gate_row(P, F, y, split, targets=(tg,), richer=rich)
            acc[tg].append(r['GUARD'][0])
        acc['richer'].append(float((rich[split[3]].argmax(1) == y[split[3]]).mean()))
        # the screen reads the fit split only, never test labels
        fit = split[1]
        ra, pa = (float((Q[fit].argmax(1) == y[fit]).mean()) for Q in (rich, P))
        cells.append(dict(dataset='OPPORTUNITY', row=cfg, seed=s, richer_acc=ra, poorer_acc=pa,
                          precondition_met=ra > pa, gain_hard=acc['hard'][-1],
                          gain_cross=acc['cross'][-1]))
        print(f"{cfg:24} s{s} hard {acc['hard'][-1]:+.4f} cross {acc['cross'][-1]:+.4f} "
              f"richer {acc['richer'][-1]:.4f}", flush=True)
out = {k: float(np.mean(v)) for k, v in acc.items()}
json.dump(out, open(OUT, 'w'), indent=1)
json.dump(cells, open(OUT.with_name('opp_targets_cells.json'), 'w'), indent=1)
print(f"OPPORTUNITY & ${out['richer']:.3f}$ & ${out['hard']:+.3f}$ & ${out['cross']:+.3f}$ \\\\")
