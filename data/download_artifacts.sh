#!/usr/bin/env bash
# Derived artefacts: the saved host outputs, and the host checkpoints behind them.
#
# These are OUR outputs, not the original datasets or third-party model weights,
# so they can be redistributed.  They exist so that every certification result can
# be reproduced without a GPU: everything downstream of a dump is pure numpy.
#
#   bash data/download_artifacts.sh              # everything (~2.5 GB)
#   bash data/download_artifacts.sh dumps        # host outputs only (~2.2 GB)
#   bash data/download_artifacts.sh checkpoints  # host weights only (~330 MB)
#
# Files land in ./artifacts, which is where every driver in experiments/ looks.
# Set GUARD_ARTIFACTS to put them elsewhere.
set -euo pipefail
set -f            # patterns below are for the hub, not for the local shell
REPO="${GUARD_ARTIFACT_REPO:-anonymous221206/GUARD_checkpoint}"
WHAT="${1:-all}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${GUARD_ARTIFACTS:-$ROOT/artifacts}"

case "$WHAT" in
  dumps)       PATTERNS="mosei_cmad/* iemocap_momke/* ave_av_att/* ptbxl_dropladder/* ptbxl_resnet1d_wang/* ninapro_cnn/seed*/* opportunity_dcl_v2/* drugban_processed/*" ;;
  checkpoints) PATTERNS="ninapro_cnn/checkpoints/* ninapro_specialist/* ptbxl_resnet1d_wang.pt" ;;
  all)         PATTERNS="*" ;;
  *) echo "unknown selection: $WHAT (use all | dumps | checkpoints)" >&2; exit 1 ;;
esac

"${PYTHON:-python3}" - "$REPO" "$DEST" $PATTERNS <<'PY'
import sys
from huggingface_hub import snapshot_download
try:
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
except ImportError:                       # older hub versions
    GatedRepoError = RepositoryNotFoundError = Exception

repo, dest, *patterns = sys.argv[1:]
try:
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest,
                      allow_patterns=patterns or None)
except (GatedRepoError, RepositoryNotFoundError) as exc:
    sys.exit(f"cannot reach {repo}: {exc}\n"
             "If the repository is private, run `huggingface-cli login` first, "
             "or set GUARD_ARTIFACT_REPO to a mirror you can read.")
print(f"artefacts in {dest}")
PY

echo "next: python experiments/repro_check.py"
