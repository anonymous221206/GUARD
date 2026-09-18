#!/usr/bin/env python3
"""Recompute the saved HME metrics and paired CMAD comparison."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=root)
    p.add_argument('--seed', type=int, default=5576)
    a = p.parse_args()
    root = a.root.resolve()
    directory = root/'results/external_review_20260917/hme'
    saved = directory/f'seed{a.seed}'

    required = [saved/'metrics.json', saved/'history.json', saved/'predictions.npz',
                root/'artifacts/mosei_cmad/dumps/raw_features.npz',
                root/'artifacts/mosei_cmad/dumps/student_preds.npz',
                root/'results/gates/mosei_hosts.json', directory/'source_feature_alignment.json']
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        p.error('missing required file(s):\n  '+'\n  '.join(missing))

    metrics = json.loads((saved/'metrics.json').read_text())
    history = json.loads((saved/'history.json').read_text())
    preds = np.load(saved/'predictions.npz')
    raw = np.load(root/'artifacts/mosei_cmad/dumps/raw_features.npz')['test_y'].reshape(-1)
    base = np.load(root/'artifacts/mosei_cmad/dumps/student_preds.npz')
    guard = json.loads((root/'results/gates/mosei_hosts.json').read_text())
    align = json.loads((directory/'source_feature_alignment.json').read_text())
    assert all(v == 0 for v in align.values())
    assert metrics['selected_epoch'] == min(history, key=lambda row: row['validation_loss'])['epoch']
    assert len(metrics['rows']) == 35
    assert len({(row['split_seed'], row['mask']) for row in metrics['rows']}) == 35
    assert history[-1]['epoch'] - metrics['selected_epoch'] == 10

    masks = ['t', 'a', 'v', 'ta', 'tv', 'av', 'tav']
    out = {}
    checked = 0
    for mask in masks:
        hme_rows, base_rows = [], []
        for split_seed in range(5):
            idx = np.array_split(np.random.default_rng(split_seed).permutation(len(raw)), 4)[-1]
            assert np.array_equal(idx, preds[f's{split_seed}_indices'])
            keep = raw[idx] != 0
            gold = raw[idx][keep] > 0
            prediction = preds[f's{split_seed}_{mask}']
            assert np.isfinite(prediction).all()
            decision = prediction[keep] > 0
            row = next(r for r in metrics['rows']
                       if r['split_seed'] == split_seed and r['mask'] == mask)
            accuracy = float(np.mean(decision == gold))
            f1 = float(f1_score(gold, decision, average='weighted'))
            assert row['n'] == len(idx) and row['n_nonzero'] == int(keep.sum())
            assert abs(row['accuracy']-accuracy) < 1e-12
            assert abs(row['f1_weighted']-f1) < 1e-12
            hme_rows.append([accuracy*100, f1*100])
            checked += 2

            base_decision = base[f'test_{mask}'].reshape(-1)[idx][keep] > 0
            base_rows.append([float(np.mean(base_decision == gold))*100,
                              float(f1_score(gold, base_decision, average='weighted'))*100])
        hme = np.mean(hme_rows, axis=0)
        cmad = np.mean(base_rows, axis=0)
        if mask != 't':
            assert np.allclose(cmad, guard['CMAD|'+mask][2:], atol=1e-10, rtol=0)
            corrected = guard['CMAD|'+mask][:2]
        else:
            corrected = cmad.tolist()
        out[mask] = {'HME': hme.tolist(), 'CMAD': cmad.tolist(),
                     'CMAD_GUARD': corrected}

    for name, selected in [('incomplete_six', masks[:-1]),
                           ('language_absent', ['a', 'v', 'av'])]:
        out[name] = {method: np.mean([out[m][method] for m in selected], axis=0).tolist()
                     for method in ('HME', 'CMAD', 'CMAD_GUARD')}
    result = {
        'status': 'PASS', 'verified_metric_fields': checked,
        'selected_epoch_zero_based': metrics['selected_epoch'], 'epochs_run': len(history),
        'train_validation_seconds': sum(row['seconds'] for row in history),
        'feature_alignment': 'all six source mean/std arrays match CMAD dumps exactly',
        'cmad_controls': ('six masks match saved CMAD means; t is recomputed on paired '
                          'quarters for both CMAD and GUARD'),
        'training_seeds': 1, 'deployment_partitions': 5, 'rows': out,
    }
    target = directory/'paired_comparison.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(f'PASS: {checked} HME metric fields; wrote {target}')


if __name__ == '__main__':
    main()

