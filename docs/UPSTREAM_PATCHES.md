# Upstream patches

Two published repositories do not run as released on a current stack.  Both
fixes are mechanical and change no result; we record them so that anyone
reproducing our numbers hits the same code we did.

## missing-aware prompts (Lee et al., CVPR 2023)

The Hateful Memes preparation path has never been runnable:

1. `vilt/utils/write_hatememes.py` reads an undefined name:

   ```python
   text_aug = text_aug_dir['{}.png'.format(data['id'])]   # NameError
   ```

   `text_aug` is assigned but not used on the next line
   (`data = (binary, text, label, split)`), so the statement is dead code.  We
   delete that one line.

2. The same file writes the text column as `"text"`, but
   `vilt/datasets/hatememes_dataset.py` reads `text_column_name="plots"`,
   raising `KeyError: Field "plots" does not exist in table schema` three
   seconds into training.  We rename the column in the *writer*, leaving the
   model-path code exactly as released.  For comparison, Food-101 uses `"text"`
   on both sides and MM-IMDb uses `"plots"` on both sides; only Hateful Memes
   is inconsistent.

The dataset's own `test.jsonl` carries no labels; as the repository's `DATA.md`
notes, the labelled test set is `test_seen.jsonl` from a different mirror.

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
