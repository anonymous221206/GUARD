"""Compare the two calibrators across the label-budget study, same tightening for both.

Reads results/rebuttal/budget.jsonl and shift.jsonl and writes crc_ltt.json, the source
of the aggregate numbers in Sec. 5.6 and App. B and of the worst-pattern standard errors.
"""
import collections, json
from pathlib import Path
import numpy as np

O = Path(__file__).resolve().parents[1] / 'results/rebuttal'


def budget_arms():
    rows = [json.loads(l) for l in open(O / 'budget.jsonl')]
    groups = collections.defaultdict(list)
    for r in rows:
        if not r.get('r'):
            continue
        name = r['bench'] + ('/' + r['ctx'].get('dataset', '') if r['bench'] == 'drugban_deploy' else '')
        groups[(name, r['kind'], r['target'], str(r['B']))].append(r['r'])
    arms = []
    for key, runs in groups.items():
        mean = lambda p, f: float(np.mean([x[p][f] for x in runs]))
        harm = np.array([x['crc']['harm'] for x in runs])
        arms.append(dict(key=key, n=len(runs), n_conf=float(np.mean([x['meta']['n_conf'] for x in runs])),
                         gain_crc=mean('crc_t', 'gain'), gain_ltt=mean('ltt_t', 'gain'),
                         harm_crc=mean('crc_t', 'harm'), harm_ltt=mean('ltt_t', 'harm'),
                         harm_plain=float(harm.mean()),
                         se_plain=float(harm.std(ddof=1) / np.sqrt(len(harm))) if len(harm) > 1 else None))
    return arms


def summary(arms):
    dg = np.array([a['gain_crc'] - a['gain_ltt'] for a in arms])
    dh = np.array([a['harm_crc'] - a['harm_ltt'] for a in arms])
    return dict(arms=len(arms), crc_higher=int((dg > 1e-4).sum()), equal=int((abs(dg) <= 1e-4).sum()),
                ltt_higher=int((dg < -1e-4).sum()), mean_gain_diff=float(dg.mean()),
                mean_harm_diff=float(dh.mean()), max_harm_guard=max(a['harm_crc'] for a in arms))


def worst_patterns():
    rows = [json.loads(l) for l in open(O / 'shift.jsonl')]
    mix = [r for r in rows if r.get('kind') == 'mix' and 'error' not in r]
    sets = dict(mosei=lambda r: r['bench'] == 'mosei', ninapro=lambda r: r['bench'] == 'nina',
                iemocap_03=lambda r: r['bench'] == 'iem' and r['ctx']['eta'] == .3,
                iemocap_05=lambda r: r['bench'] == 'iem' and r['ctx']['eta'] == .5)
    out = {}
    for name, sel in sets.items():
        x = [r for r in mix if sel(r)]
        for policy in ('pooled_crc', 'mask_crc'):
            per = collections.defaultdict(list)
            for r in x:
                for p, v in r['out'][policy]['per'].items():
                    if v['n'] >= 20:
                        per[p].append(v['harm'])
            worst = max(per, key=lambda p: np.mean(per[p]))
            v = np.array(per[worst])
            out[f'{name}/{policy}'] = dict(pattern=worst, mean=float(v.mean()), runs=len(v),
                                           se=float(v.std(ddof=1) / np.sqrt(len(v))))
    return out


if __name__ == '__main__':
    arms = budget_arms()
    bins = [('n_conf<=25', 0, 25), ('25<n_conf<=60', 25, 60), ('60<n_conf<=130', 60, 130),
            ('130<n_conf<=350', 130, 350), ('n_conf>350', 350, 1e9)]
    res = dict(all=summary(arms), by_n_conf={n: summary([a for a in arms if lo < a['n_conf'] <= hi]) for n, lo, hi in bins},
               plain_over_budget=sorted(([*a['key'], a['n'], round(a['harm_plain'], 3), a['se_plain'] and round(a['se_plain'], 3)]
                                         for a in arms if a['harm_plain'] > .2), key=lambda z: -z[5]),
               worst_patterns=worst_patterns())
    json.dump(res, open(O / 'crc_ltt.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))
