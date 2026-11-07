"""Row alignment for the MoMKE fold dumps.

Each mask dump stores the pool in its own row order, so rows of two dumps cannot
be paired by position. `(video, occurrence)` identifies a row in every dump.
"""
import numpy as np


def row_keys(vid):
    seen, out = {}, []
    for v in vid:
        seen[v] = seen.get(v, 0) + 1
        out.append((v, seen[v] - 1))
    return out


def align_pool(ref, d):
    """Indices that put `ref`'s pool rows in `d`'s order."""
    at = {k: i for i, k in enumerate(row_keys(ref['pool_vid']))}
    idx = np.array([at[k] for k in row_keys(d['pool_vid'])])
    assert np.array_equal(ref['pool_labels'][idx], d['pool_labels']), 'pool rows do not align'
    return idx
