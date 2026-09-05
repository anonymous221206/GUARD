#!/usr/bin/env bash
# ViLT with missing-aware prompts (Lee et al., CVPR 2023) plus the pre-trained
# ViLT backbone.  The three vision-language datasets must be fetched separately
# because each has its own licence -- see data/README.md.
set -euo pipefail
ROOT="${1:-$(dirname "$0")/raw}"
mkdir -p "$ROOT"
cd "$ROOT"
if [ ! -d missing_aware_prompts ]; then
  git clone --depth 1 https://github.com/YiLunLee/missing_aware_prompts.git
fi
if [ ! -f vilt_200k_mlm_itm.ckpt ]; then
  curl -L -o vilt_200k_mlm_itm.ckpt \
    https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt
fi
echo "host code + backbone ready at $ROOT"
