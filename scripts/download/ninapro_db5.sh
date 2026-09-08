#!/usr/bin/env bash
# verified 200: https://ninapro.hevs.ch/files/DB5_Preproc/s1.zip (application/zip)
#
# DB5 is distributed by the NinaPro project at HEIA-FR/HES-SO under its own
# terms; read them at https://ninapro.hevs.ch/instructions/DB5.html before
# running this. The ten subject archives are served directly, so the download
# needs no account, but the licence still applies to what it fetches.
set -euo pipefail
base=https://ninapro.hevs.ch/files/DB5_Preproc
raw=${GUARD_NINAPRO:-data/ninapro}

if [ "${1:-}" = --dry-run ]; then
  c=$(curl -sIL -o /dev/null -w '%{http_code}' --max-time 30 "$base/s1.zip")
  echo "ninapro_db5.sh $c $base/s1.zip"
  exit
fi

mkdir -p "$raw"
for i in $(seq 1 10); do
  d="$raw/s$i"
  if [ -d "$d" ] && ls "$d"/S*_E2_*.mat >/dev/null 2>&1; then
    echo "s$i already unpacked, skipping"
    continue
  fi
  echo "fetching s$i"
  curl -fL --retry 3 -o "$raw/s$i.zip" "$base/s$i.zip"
  # the archives carry their own top-level folder name, which varies by subject
  tmp=$(mktemp -d); unzip -q "$raw/s$i.zip" -d "$tmp"
  mkdir -p "$d"; find "$tmp" -name '*.mat' -exec mv {} "$d"/ \;
  rm -rf "$tmp" "$raw/s$i.zip"
done

missing=$(for i in $(seq 1 10); do ls "$raw/s$i"/S*_E2_*.mat >/dev/null 2>&1 || echo "s$i"; done)
if [ -n "$missing" ]; then
  echo "incomplete, missing exercise-2 files for:$missing" >&2
  exit 1
fi

echo "DB5 unpacked under $raw. Preprocessing windows and training the released CNN reproduction."
python scripts/train_ninapro_retrained.py --seed 0 --out results/ninapro_db5_raw \
       --checkpoints checkpoints/ninapro_db5_raw
