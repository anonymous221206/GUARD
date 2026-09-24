"""The deployment split every OPPORTUNITY driver uses.

Deployment windows are stored in temporal order and consecutive windows overlap,
so a split by row puts the window right next to a test query into the retrieval
pool. The deployment stream is cut instead into contiguous blocks of 200 windows,
far longer than one activity segment, and whole blocks are assigned at random to
pool, fit, conf and test. The last `gap` windows of each block are dropped, so no
window of one role overlaps a window of another. Roles stay exchangeable at the
block level, which is what the certificate needs.
"""
import numpy as np

BLOCK, GAP = 200, 2


def deploy_split(n, seed, block=BLOCK, gap=GAP):
    rng = np.random.default_rng(seed)
    blocks = [np.arange(i, min(i + block, n) - gap) for i in range(0, n, block)]
    blocks = [b for b in blocks if len(b)]
    roles = (np.arange(len(blocks)) % 4)[rng.permutation(len(blocks))]
    return tuple(np.concatenate([blocks[j] for j in np.flatnonzero(roles == r)])
                 for r in range(4))
