#!/usr/bin/env bash
# Derived artefacts: trained model checkpoints and the saved model outputs.
#
# These are OUR outputs, not the original datasets or the third-party model
# weights, so they can be redistributed.  They exist so that the certification
# results can be reproduced without a GPU: every experiment downstream of a
# dump is pure numpy.
#
# The artefacts are ~10 GB in total but are laid out one directory per
# (dataset, model) pair -- one pair per row of the paper's tables -- so you only
# need the pair you want to check:
#
#   bash data/download_artifacts.sh                     # list the pairs
#   bash data/download_artifacts.sh ninapro_cnn         # one pair (~1.6 GB)
#   bash data/download_artifacts.sh biosnap_drugban     # one pair (~493 MB)
#   bash data/download_artifacts.sh all                 # everything (~10 GB)
#
# To rebuild a pair yourself instead:
#   python hosts/drugban.py train --dataset biosnap --split random --seed 42
set -euo pipefail

REPO="${GUARD_ARTIFACT_REPO:-anonymous221206/GUARD_checkpoint}"
WHAT="${1:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/artifacts"

PAIRS="ave_av_att bindingdb_drugban biosnap_drugban human_drugban
iemocap_momke mosei_cmad ninapro_cnn opportunity_deepconvlstm
ptbxl_resnet1d_wang"

if [ -z "$REPO" ]; then
  cat <<'MSG'
No artefact repository configured.

Set GUARD_ARTIFACT_REPO to the Hugging Face dataset repository, then re-run:

    export GUARD_ARTIFACT_REPO=anonymous221206/GUARD_checkpoint
    bash data/download_artifacts.sh ninapro_cnn

Without it, reproduce from scratch instead:

  1. bash data/download_drugban.sh
     python hosts/drugban.py train --dataset biosnap --split random --seed 42
     python experiments/exp_drugban.py --dumps data/processed/drugban_biosnap_random_s42

  2. bash data/download_opportunity.sh
     python hosts/opportunity_prepare.py
     python hosts/opportunity.py --source data/processed/opportunity_features.npz \
         --configs configs/opportunity.json
     python experiments/exp_opportunity.py --features data/processed/opportunity.npz

The synthetic study and the unit tests need no downloads at all:

     pytest -q
     python experiments/exp_synthetic.py
MSG
  exit 1
fi

if [ -z "$WHAT" ]; then
  echo "Available pairs (pass one, or 'all'):"
  for p in $PAIRS; do echo "  $p"; done
  exit 0
fi

# Prefer huggingface_hub: it resumes, verifies, and fetches only what is asked.
have_hf() { python3 -c 'import huggingface_hub' >/dev/null 2>&1; }

pull() {                       # pair-name-or-empty-for-all
  local pattern="$1"
  mkdir -p "$DEST"
  if have_hf; then
    python3 - "$REPO" "$DEST" "$pattern" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, dest, pattern = sys.argv[1], sys.argv[2], sys.argv[3]
kwargs = dict(repo_id=repo, repo_type="dataset", local_dir=dest)
if pattern:
    kwargs["allow_patterns"] = [f"{pattern}/*"]
snapshot_download(**kwargs)
PY
  else
    echo "huggingface_hub not installed. Install it and re-run:" >&2
    echo "    pip install -e '.[artifacts]'" >&2
    exit 1
  fi
}

case "$WHAT" in
  all) pull "" ;;
  *)
    found=0
    for p in $PAIRS; do [ "$p" = "$WHAT" ] && found=1; done
    if [ "$found" -eq 0 ]; then
      echo "unknown pair: $WHAT" >&2
      echo "run without arguments to list the pairs" >&2
      exit 1
    fi
    pull "$WHAT"
    ;;
esac

cat <<'MSG'

Artefacts ready under artifacts/.

Check one number before going further:

    python experiments/repro_check.py

It should print 63.1 -> 68.6, 63.7 -> 68.3 and 64.5 -> 70.3, the CMU-MOSEI row
of Table 1. A few tenths of a point of drift across platforms is expected.
MSG
