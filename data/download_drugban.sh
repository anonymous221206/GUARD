#!/usr/bin/env bash
# DrugBAN (Bai et al., Nature Machine Intelligence 2023) -- code, data and the
# official random / cluster splits.  ~40 MB.
set -euo pipefail
ROOT="${1:-$(dirname "$0")/raw}"
mkdir -p "$ROOT"
cd "$ROOT"
if [ ! -d DrugBAN ]; then
  git clone --depth 1 https://github.com/peizhenbai/DrugBAN.git
fi
echo "DrugBAN ready at $ROOT/DrugBAN"
echo "  datasets: $(ls "$ROOT"/DrugBAN/datasets | tr '\n' ' ')"
echo
echo "NOTE  Two upstream fixes are needed to run the released code on a modern"
echo "      stack; see docs/UPSTREAM_PATCHES.md.  They do not change any result."
