"""Data splits, with the exchangeability requirement made explicit.

The conformal gate in :mod:`guard.certify` is only valid when the calibration
set and the deployment set are exchangeable.  During this project we found a
real experiment where they were not -- calibration came from one subject and
evaluation from another -- and the harm budget silently broke.  The split is
therefore a first-class object that records *where each part came from* and
refuses to build an invalid one by accident.

Roles
-----
``pool``  unlabelled (or labelled) neighbours used to build the retrieval target
``fit``   used only to choose the blend weight beta
``conf``  used only to calibrate the conformal quantile
``test``  held out; never touched until the final metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Split:
    """Index sets for the four roles, plus provenance for each."""

    pool: np.ndarray
    fit: np.ndarray
    conf: np.ndarray
    test: np.ndarray
    #: human-readable origin of each role, e.g. {"conf": "subject 3", ...}
    origin: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        parts = {"pool": self.pool, "fit": self.fit, "conf": self.conf, "test": self.test}
        for a, ia in parts.items():
            for b, ib in parts.items():
                if a < b and np.intersect1d(ia, ib).size:
                    raise ValueError(
                        f"split roles {a!r} and {b!r} overlap in "
                        f"{np.intersect1d(ia, ib).size} indices; "
                        "the certificate assumes disjoint roles"
                    )

    @property
    def exchangeable(self) -> bool:
        """True when calibration and evaluation are declared to share an origin.

        This is a *declaration*, not a test: exchangeability cannot be verified
        from data.  Experiments must set ``origin`` honestly.
        """
        return self.origin.get("conf") == self.origin.get("test")

    def warn_if_not_exchangeable(self) -> str | None:
        if self.exchangeable:
            return None
        return (
            f"calibration comes from {self.origin.get('conf', 'an undeclared source')!r} "
            f"but evaluation from {self.origin.get('test', 'an undeclared source')!r}; "
            "the coverage guarantee does not apply across that shift"
        )

    def sizes(self) -> dict:
        return {k: int(len(v)) for k, v in
                (("pool", self.pool), ("fit", self.fit),
                 ("conf", self.conf), ("test", self.test))}


def random_split(
    index: Sequence[int],
    seed: int,
    fractions: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    origin: str = "same population",
) -> Split:
    """Split one exchangeable population into the four roles.

    All four roles come from a single random permutation, so calibration and
    evaluation are exchangeable by construction.  This is the split every
    experiment should use unless it has a documented reason not to.
    """
    index = np.asarray(index)
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {sum(fractions)}")
    perm = np.random.default_rng(seed).permutation(len(index))
    cuts = np.cumsum([int(round(f * len(index))) for f in fractions[:-1]])
    pool, fit, conf, test = np.split(index[perm], cuts)
    return Split(pool, fit, conf, test,
                 origin={k: origin for k in ("pool", "fit", "conf", "test")})


def split_with_external_pool(
    deployment_index: Sequence[int],
    pool_index: Sequence[int],
    seed: int,
    pool_origin: str,
    deployment_origin: str = "deployment population",
) -> Split:
    """Retrieval pool from one population, calibration and test from another.

    Calibration and evaluation still share an origin, so the certificate holds;
    only the *quality* of the retrieval target is affected by the pool's origin.
    """
    dep = np.asarray(deployment_index)
    perm = np.random.default_rng(seed).permutation(len(dep))
    thirds = np.array_split(dep[perm], 3)
    return Split(
        pool=np.asarray(pool_index), fit=thirds[0], conf=thirds[1], test=thirds[2],
        origin={"pool": pool_origin, "fit": deployment_origin,
                "conf": deployment_origin, "test": deployment_origin},
    )
