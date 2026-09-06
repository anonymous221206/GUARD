#!/usr/bin/env bash
# Reproduce every table in the paper.
#
#   bash scripts/run_all.sh            # everything that has a checkpoint
#   bash scripts/run_all.sh drugban    # one family only
#
# Hosts are never retrained when a checkpoint is present; delete checkpoints/
# to train from scratch.  Each stage appends to logs/ and is skipped when its
# output already exists, so the script is safe to re-run.
set -uo pipefail
G="$(cd "$(dirname "$0")/.." && pwd)"
cd "$G"
mkdir -p logs results data/processed
WHICH="${1:-all}"
want() { [ "$WHICH" = all ] || [ "$WHICH" = "$1" ]; }

say() { printf '\n== %s\n' "$*"; }
ok()  { printf '   %s\n' "$*"; }

# ---------------------------------------------------------------- DrugBAN ----
drugban_cell() {                       # dataset split seed pool
  local ds=$1 sp=$2 sd=$3 pool=$4 tag dumps
  tag=$([ "$sp" = cluster ] && echo "cluster_${ds}_s${sd}" || echo "${ds}_s${sd}")
  local ck="checkpoints/drugban/${tag}.pth" cfg="checkpoints/drugban/${tag}.yaml"
  [ -f "$ck" ] || { ok "skip $tag (no checkpoint)"; return; }
  dumps="data/processed/drugban_${ds}_${sp}_s${sd}"
  if [ ! -f "$dumps/full.npz" ]; then
    python hosts/drugban.py --dataset "$ds" --split "$sp" --seed "$sd" \
        --ckpt "$ck" --cfg "$cfg" --out "$dumps" >> "logs/${tag}.log" 2>&1 \
      || { ok "$tag dump FAILED -> logs/${tag}.log"; return; }
  fi
  python experiments/exp_drugban.py --dumps "$dumps" --pool "$pool" \
      >> "logs/${tag}_${pool}.log" 2>&1 \
    && ok "$(tail -1 "logs/${tag}_${pool}.log")" \
    || ok "$tag guard FAILED -> logs/${tag}_${pool}.log"
}

if want drugban; then
  say "DrugBAN, in-domain (random) splits"
  for sd in 42 1 2; do for ds in human biosnap bindingdb; do
    drugban_cell "$ds" random "$sd" source
  done; done
  say "DrugBAN, cross-domain (cluster) splits, both retrieval pools"
  for sd in 42 1 2; do for ds in bindingdb biosnap; do
    drugban_cell "$ds" cluster "$sd" source
    drugban_cell "$ds" cluster "$sd" deployment
  done; done
fi

# ------------------------------------------------------------ affective ------
if want affective; then
  say "Affective hosts (published checkpoints, consumed as saved predictions)"
  for h in data/raw/hosts/*/; do
    [ -f "$h/preds.npz" ] || continue
    n=$(basename "$h")
    python experiments/exp_affective.py --host "$h" >> "logs/affective_${n}.log" 2>&1 \
      && ok "$n: $(grep -c . /dev/null; tail -2 "logs/affective_${n}.log" | head -1)" \
      || ok "$n FAILED -> logs/affective_${n}.log"
  done
fi


# ---------------------------------------------------------- OPPORTUNITY ------
if want opportunity; then
  say "OPPORTUNITY, both calibration protocols"
  feat=data/processed/opportunity.npz
  if [ -f "$feat" ]; then
    for proto in deployment cross_subject; do
      python experiments/exp_opportunity.py --features "$feat" --protocol "$proto" \
          >> "logs/opportunity_${proto}.log" 2>&1 \
        && ok "$proto: $(tail -2 "logs/opportunity_${proto}.log" | head -1)" \
        || ok "$proto FAILED -> logs/opportunity_${proto}.log"
    done
    say "OPPORTUNITY, deployment-label budget sweep"
    # --fixed-host picks the host's epoch on the full fit split at every budget.
    # Without it the budget also sizes the host's validation split, so the
    # frozen host improves with n_L and the sweep moves two things at once.
    for nl in 150 300 600 1200 2400 4800; do
      python experiments/exp_opportunity.py --features "$feat" --protocol deployment \
          --fixed-host --label-budget "$nl" >> "logs/opportunity_nL${nl}.log" 2>&1 \
        && ok "n_L=${nl}: $(tail -2 "logs/opportunity_nL${nl}.log" | head -1)"
    done
    python experiments/exp_opportunity.py --features "$feat" --protocol deployment \
        --fixed-host >> logs/opportunity_fixedhost.log 2>&1 \
      && ok "n_L=all: $(tail -2 logs/opportunity_fixedhost.log | head -1)"
  else
    ok "skip (run hosts/opportunity.py first)"
  fi
fi

if want synthetic; then
  say "Synthetic study (no download required)"
  python experiments/exp_synthetic.py >> logs/synthetic.log 2>&1 \
    && ok "$(tail -3 logs/synthetic.log | head -1)" \
    || ok "FAILED -> logs/synthetic.log"
fi
