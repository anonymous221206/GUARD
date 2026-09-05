# Superseded

Kept for provenance, not for running. Nothing here produces a number in the paper.

## vilt_prompts.py, exp_vision_language.py, download_vilt.sh

A ViLT host on Food-101 and Hateful Memes. The submitted paper reports no
vision-language experiment, so these were retired on 2026-09-05. They still ran,
which is what made them dangerous: `scripts/run_all.sh` invoked them and
`docs/REPRODUCING.md` listed their tables, so a reader could produce output that
corresponds to nothing in the paper.
