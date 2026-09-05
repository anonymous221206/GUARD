#!/usr/bin/env bash
# verified 200: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK
set -euo pipefail
u=https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK; raw=data/raw/mosei/mosei.pkl
c=$(curl -sIL -o /dev/null -w '%{http_code}' "$u"); [ "${1:-}" = --dry-run ] && { echo "cmu_mosei.sh $c $u"; exit; }
echo "CMU-MOSEI is distributed under the CMU MultiComp Lab/CMU Multimodal SDK terms. Request access through the SDK and place its host-ready train/valid/test text, audio, visual and label mapping at $raw."
[ -s "$raw" ] || { echo "Waiting for licensed CMU-MOSEI at $raw; nothing was downloaded."; exit 0; }
echo "Licensed CMU-MOSEI found. Running the released affective host-output/GUARD pipeline."
bash scripts/run_all.sh affective
