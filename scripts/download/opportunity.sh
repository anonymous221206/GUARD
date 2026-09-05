#!/usr/bin/env bash
# verified 200: https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip
set -euo pipefail
u=https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip;c=$(curl -sIL -o /dev/null -w '%{http_code}' "$u");[ "${1:-}" = --dry-run ]&&{ echo "OPPORTUNITY $c $u";exit;};d=data/raw/OpportunityUCIDataset;[ "$(find "$d" -type f 2>/dev/null|wc -l)" = 88 ]&&exit;mkdir -p data/raw;curl -fL --retry 3 -o "$d.zip" "$u";unzip -q "$d.zip" -d data/raw;rm -f "$d.zip";[ "$(find "$d" -type f|wc -l)" = 88 ]
