#!/usr/bin/env python3
"""Audit HME split construction and masked-input invariance.

Run after ``review_hme.py prepare`` in the same HME environment.
"""
import argparse
import json
from pathlib import Path

import review_hme as h


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=5576)
    p.add_argument('--hme-repo', type=Path, required=True)
    p.add_argument('--bert-dir', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output-root', type=Path,
                   default=h.ROOT/'results/external_review_20260917/hme')
    a = p.parse_args()
    import numpy as np
    import torch
    h.REPO, h.BERT, h.DATA = a.hme_repo.resolve(), a.bert_dir.resolve(), a.dataset.resolve()
    m, versions = h.init(a.seed)
    out = a.output_root.resolve()
    data = torch.load(out/'prepared.pt', map_location='cpu')
    model, _ = h.load_model(m)
    model.eval()

    raw = np.load(h.ROOT/'artifacts/mosei_cmad/dumps/raw_features.npz')
    checks = {}
    for split in ('train', 'dev', 'test'):
        tensors = data[split]
        assert all(torch.isfinite(t).all() for t in tensors)
        assert np.array_equal(tensors[-1].numpy().reshape(-1), raw[split+'_y'])
        checks[split+'_label_order_matches_cmad'] = True

    batch = tuple(t[:8].clone() for t in data['dev'])
    with torch.no_grad():
        for mask, bits in h.MASKS.items():
            if mask == 'tav':
                continue
            modified = [t.clone() for t in batch]
            if not bits[0]: modified[0] = torch.remainder(modified[0] + 179, 30522)
            if not bits[1]: modified[2] = modified[2] * -2 + 71
            if not bits[2]: modified[1] = modified[1] * -3 - 19
            torch.manual_seed(915)
            p0, _ = h.run_batch(m, model, batch, bits)
            torch.manual_seed(915)
            p1, _ = h.run_batch(m, model, tuple(modified), bits)
            error = float(torch.max(torch.abs(p0['predictions']-p1['predictions'])))
            assert error == 0, ('unobserved modality influences prediction', mask, error)
            checks[mask+'_masked_input_invariance'] = error

    for seed in range(5):
        parts = np.array_split(np.random.default_rng(seed).permutation(len(data['test'][0])), 4)
        assert len(np.unique(np.concatenate(parts))) == len(data['test'][0])
        checks[f'split{seed}_sizes'] = [len(x) for x in parts]

    report = {'checks': checks, 'versions': versions, 'status': 'PASS'}
    (out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
