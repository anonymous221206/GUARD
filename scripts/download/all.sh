#!/usr/bin/env bash
set -euo pipefail
h=$(cd "$(dirname "$0")"&&pwd);m="${1:-}"
for s in ptb_xl.sh opportunity.sh drugban.sh guard_checkpoints.sh ninapro_db5.sh cmu_mosei.sh ave.sh iemocap.sh; do "$h/$s" "$m"; done
echo "Manual licence steps: NinaPro DB5 (NinaPro/HEIA-FR), CMU-MOSEI (CMU MultiComp Lab), AVE (AVE-ECCV18 authors), and IEMOCAP (USC SAIL). Once supplied, each script continues with its released pipeline."
