"""GUARD: certified post-hoc correction for a frozen model under missing modalities.

Three steps, one function::

    from guard import HostOutputs, random_split, run

    split  = random_split(np.arange(n), seed=0)
    result = run(HostOutputs(probs, features, labels), split, target="cross_mask")

``result.joint_harm <= alpha`` is the guarantee; ``result.gate_metric_delta`` is
what you gained.  See :mod:`guard.pipeline` for the whole method in one file.
"""

from .losses import CROSS_ENTROPY, BERNOULLI, SQUARED, accuracy, auroc, f1_macro, get
from .measure import potential_headroom, nearest_neighbour_index
from .pipeline import HostOutputs, Result, run
from .splits import Split, random_split, split_with_external_pool
from .targets import knn_average, richer_is_richer, standardise

__version__ = "1.0.0"
__all__ = [
    "HostOutputs", "Result", "run",
    "Split", "random_split", "split_with_external_pool",
    "potential_headroom", "nearest_neighbour_index",
    "knn_average", "richer_is_richer", "standardise",
    "CROSS_ENTROPY", "BERNOULLI", "SQUARED",
    "accuracy", "auroc", "f1_macro", "get",
]
