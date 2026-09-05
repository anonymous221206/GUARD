#!/usr/bin/env bash
# verified 200: https://ninapro.hevs.ch/instructions/DB5.html
set -euo pipefail
u=https://ninapro.hevs.ch/instructions/DB5.html; raw=data/ninapro
c=$(curl -sIL -o /dev/null -w '%{http_code}' "$u"); [ "${1:-}" = --dry-run ] && { echo "ninapro_db5.sh $c $u"; exit; }
echo "NinaPro DB5 is distributed by the NinaPro project at HEIA-FR/HES-SO. Create an account, request DB5, accept its terms, and unpack S1/.../S10 with S*_E2_*.mat and S*_E3_*.mat below $raw."
[ -d "$raw/s1" ] || { echo "Waiting for licensed DB5 files at $raw; nothing was downloaded."; exit 0; }
echo "Licensed DB5 found. Preprocessing windows and training the released CNN reproduction."
python scripts/train_ninapro_retrained.py --seed 0 --out results/ninapro_db5_raw --checkpoints checkpoints/ninapro_db5_raw
