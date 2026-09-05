#!/usr/bin/env bash
# verified 200: https://huggingface.co/api/datasets/anonymous221206/GUARD_checkpoint
set -euo pipefail
a=https://huggingface.co/api/datasets/anonymous221206/GUARD_checkpoint
base=https://huggingface.co/datasets/anonymous221206/GUARD_checkpoint/resolve/main
f(){ curl -fsSL "$a" | python3 -c 'import json,sys; print("\n".join(x["rfilename"] for x in json.load(sys.stdin)["siblings"] if x["rfilename"].endswith(".pt")))'; }
cmad(){ printf "%s\n" mosei_cmad/raw_features.npz mosei_cmad/student_preds.npz; }
fetch(){ local remote=$1 dest=$2; [ -s "$dest" ] && return; mkdir -p "$(dirname "$dest")"; curl -fL --retry 3 -o "$dest" "$base/$remote"; }
if [ "${1:-}" = --dry-run ]; then
  c=$(curl -sIL -o /dev/null -w '%{http_code}' "$a"); echo "GUARD API $c $a"
  cmad | while read -r x; do c=$(curl -sIL -o /dev/null -w '%{http_code}' "$base/$x"); echo "GUARD CMAD dump $c $base/$x"; done
  echo "GUARD checkpoints: $(f | wc -l) files listed by the API"
  exit
fi
f | while read -r x; do fetch "$x" "checkpoints/$x"; done
cmad | while read -r x; do fetch "$x" "artifacts/mosei_cmad/dumps/${x##*/}"; done
[ "$(f | wc -l)" = 176 ]
[ -s artifacts/mosei_cmad/dumps/raw_features.npz ]
[ -s artifacts/mosei_cmad/dumps/student_preds.npz ]
