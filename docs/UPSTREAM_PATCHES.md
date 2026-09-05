# Upstream patches

The published repository below does not run as released on a current stack.
The compatibility change does not alter results; we record it so that anyone
reproducing our numbers uses the same environment.

## DrugBAN (Bai et al., Nature Machine Intelligence 2023)

The code runs unchanged.  The pinned environment (torch 1.7.1 / CUDA 10.2 /
dgl 0.7.1) does not build on current drivers; we use torch 2.1.2 + cu121 with
dgl 2.2.1 + cu121 and dgllife 0.3.2.  Installing DGL from PyPI silently gives
the CPU build, which fails at the first graph transfer -- install from
`https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html`.

## Our own scripts

For completeness, two defects we found in our earlier experiment code and fixed
before release:

* A seed derived from `hash(scenario_name)`.  String hashing is salted per
  process in Python 3, so results were not reproducible across runs -- two
  cells changed the sign of their loss gain.  Seeds are now derived from an
  explicit index.
* A retrieval pool built from the first *N* rows of a class-sorted table, which
  covered only 20 of 101 classes.  Pools are now drawn by random permutation,
  and `guard.splits.Split` refuses overlapping roles.
