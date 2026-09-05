#!/usr/bin/env bash
# verified 200: https://sail.usc.edu/iemocap/
set -euo pipefail
u=https://sail.usc.edu/iemocap/; raw=data/raw/iemocap/IEMOCAP_full_release
c=$(curl -sIL -o /dev/null -w '%{http_code}' "$u"); [ "${1:-}" = --dry-run ] && { echo "iemocap.sh $c $u"; exit; }
echo "IEMOCAP is licensed by USC SAIL. Obtain it by signing USC's release agreement, then unpack the official IEMOCAP_full_release (Session1/ ... Session5/) at $raw."
[ -d "$raw/Session1" ] || { echo "Waiting for licensed data at $raw; nothing was downloaded."; exit 0; }
echo "Licensed IEMOCAP found. Running the released host-output/GUARD pipeline; it uses prepared host directories in data/raw/hosts/*iemocap*/."
bash scripts/run_all.sh affective
