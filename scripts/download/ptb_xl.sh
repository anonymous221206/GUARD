#!/usr/bin/env bash
# verified 200: https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
set -euo pipefail
u=https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
c=$(curl -sIL -o /dev/null -w '%{http_code}' "$u"); [ "${1:-}" = --dry-run ] && { echo "PTB-XL $c $u"; exit; }
d=data/raw/ptb-xl-1.0.3; [ -f "$d/SHA256SUMS.txt" ] && (cd "$d" && sha256sum -c SHA256SUMS.txt >/dev/null) && exit
mkdir -p data/raw; curl -fL --retry 3 -o "$d.zip" "$u"; unzip -q "$d.zip" -d data/raw; mv data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 "$d"; rm -f "$d.zip"; (cd "$d" && sha256sum -c SHA256SUMS.txt >/dev/null)
