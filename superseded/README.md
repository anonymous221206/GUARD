# Superseded

Kept for provenance, not for running. Nothing here produces a number in the paper.

## vilt_prompts.py, exp_vision_language.py, download_vilt.sh

A ViLT host on Food-101 and Hateful Memes. The submitted paper reports no
vision-language experiment, so these were retired on 2026-09-05. They still ran,
which is what made them dangerous: `scripts/run_all.sh` invoked them and
`docs/REPRODUCING.md` listed their tables, so a reader could produce output that
corresponds to nothing in the paper.

## exp_ablations, exp_groupwise, exp_modules, exp_frontier

Studies from an earlier draft. The submitted paper reports none of them, and three
still called the retired plausible-label gate, so running them would have produced
numbers from a mechanism the paper no longer describes.

## finite_sample, fs_mosei

The calibration-resampling study of the old appendix. It analysed the Beta law of
split-conformal miscoverage, which is not the mechanism any more.

## certify_plausible_label.py

The retired gate itself: a conformal plausible-label set and a worst-case loss
over it. Replaced by a learned action score with a conformal-risk-control
threshold; see `src/guard/action.py`.
