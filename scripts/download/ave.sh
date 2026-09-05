#!/usr/bin/env bash
# verified 200: https://github.com/YapengTian/AVE-ECCV18
set -euo pipefail
u=https://github.com/YapengTian/AVE-ECCV18; raw=data/raw/AVE
c=$(curl -sIL -o /dev/null -w '%{http_code}' "$u"); [ "${1:-}" = --dry-run ] && { echo "ave.sh $c $u"; exit; }
echo "AVE is distributed by the AVE-ECCV18 authors (Yapeng Tian et al.) through their access-controlled Drive links. While signed in, obtain it there and place videos in $raw/videos, released features in $raw/features, and DMRN_model in $raw/DMRN_model."
[ -d "$raw/videos" ] && [ -d "$raw/features" ] && [ -e "$raw/DMRN_model" ] || { echo "Waiting for licensed AVE files at $raw; nothing was downloaded."; exit 0; }
echo "Licensed AVE found. Running the released archived-host GUARD driver."
[ -d artifacts/ave_av_att ] || { echo "Expected prepared AV-att outputs at artifacts/ave_av_att; see docs/REPRODUCING.md for the published-host layout."; exit 1; }
python experiments/exp_ave.py --dumps artifacts/ave_av_att
