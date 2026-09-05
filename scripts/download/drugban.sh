#!/usr/bin/env bash
# verified 200: https://github.com/peizhenbai/DrugBAN
set -euo pipefail
u=https://github.com/peizhenbai/DrugBAN;v=9923f8c99959e00263103ff9ac61ba0eaccc8e02;c=$(curl -sIL -o /dev/null -w '%{http_code}' "$u");[ "${1:-}" = --dry-run ]&&{ echo "DrugBAN $c $u";exit;};d=data/raw/DrugBAN;[ -d "$d/.git" ]&&[ "$(git -C "$d" rev-parse HEAD)" = "$v" ]&&exit;rm -rf "$d";mkdir -p data/raw;git clone "$u" "$d";git -C "$d" fetch --depth 1 origin "$v";git -C "$d" checkout --detach "$v";[ "$(git -C "$d" ls-files|wc -l)" = 42 ]
