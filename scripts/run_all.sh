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

if want ablations; then
  say "Ablations: selectors at matched coverage, and plug-in estimators"
  for d in data/processed/drugban_*_random_s42; do
    [ -f "$d/full.npz" ] || continue
    b=$(basename "$d")
    for study in selectors plugins; do
      python experiments/exp_ablations.py --study "$study" --kind drugban --dumps "$d" \
          >> "logs/ablation_${study}_${b}.log" 2>&1 \
        && ok "${study} on ${b} -> results/ablation_${study}_${b}/" \
        || ok "${study} on ${b} FAILED -> logs/ablation_${study}_${b}.log"
    done
  done
fi

if want labeleff; then
  say "Label efficiency and the contraction bound (regression host)"
  for h in data/raw/hosts/*/; do
    [ -f "$h/student_preds.npz" ] || continue
    n=$(basename "$h")
    python experiments/exp_label_efficiency.py --host "$h" >> "logs/labeleff_${n}.log" 2>&1 \
      && ok "$n -> results/label_efficiency/" \
      || ok "$n FAILED -> logs/labeleff_${n}.log"
  done
fi

if want frontier; then
  say "alpha- and delta-frontiers behind the validity figure"
  for fam in drugban vision affective opportunity; do
    for grid in alpha delta; do
      python experiments/exp_frontier.py --family "$fam" --grid "$grid" \
          >> "logs/frontier_${fam}_${grid}.log" 2>&1 \
        && ok "${fam}/${grid}: $(tail -1 "logs/frontier_${fam}_${grid}.log")" \
        || ok "${fam}/${grid} FAILED -> logs/frontier_${fam}_${grid}.log"
    done
  done
fi

if want modules; then
  say "Measure x Certify, with the null conditions as the deciding cells"
  python experiments/exp_modules.py >> logs/modules.log 2>&1 \
    && ok "$(tail -1 logs/modules.log)" || ok "FAILED -> logs/modules.log"
fi

if want groupwise; then
  say "class-conditional harm, and harm spread against calibration size"
  for s in class_harm calib_size; do
    python experiments/exp_groupwise.py --study "$s" >> "logs/groupwise_${s}.log" 2>&1 \
      && ok "${s} -> results/groupwise/" || ok "${s} FAILED -> logs/groupwise_${s}.log"
  done
fi

say "tables"
python scripts/build_tables.py results/

# Archived frozen paper hosts (dump-backed; see docs/REPRODUCING.md).
if want archived; then
  say "Archived frozen paper hosts"
  archive_root="${GUARD_ARCHIVED_DUMPS:-}"
  [ -n "$archive_root" ] || { ok "set GUARD_ARCHIVED_DUMPS to run archived hosts"; archive_root=""; }
  if [ -n "$archive_root" ]; then
    py="${PYTHON:-python}"
    [ -d "$archive_root/DCL_HOSTS2" ] && "$py" experiments/exp_opportunity_dcl.py --dumps "$archive_root/DCL_HOSTS2" >> logs/opportunity_dcl.log 2>&1 && ok "OPPORTUNITY DeepConvLSTM -> logs/opportunity_dcl.log"
    [ -d "$archive_root/ave_av_att" ] && "$py" experiments/exp_ave.py --dumps "$archive_root/ave_av_att" >> logs/ave.log 2>&1 && ok "AVE -> logs/ave.log"
    [ -d "$archive_root/ninapro_cnn" ] && "$py" experiments/exp_ninapro.py --dumps "$archive_root/ninapro_cnn" >> logs/ninapro.log 2>&1 && ok "NinaPro DB5 -> logs/ninapro.log"
    [ -d "$archive_root/ptbxl_resnet1d_wang" ] && "$py" experiments/exp_ptbxl.py --dumps "$archive_root/ptbxl_resnet1d_wang" >> logs/ptbxl_resnet.log 2>&1 && ok "PTB-XL ResNet-1D -> logs/ptbxl_resnet.log"
    [ -d "$archive_root/ptbxl_leadladder" ] && "$py" experiments/exp_ptbxl.py --variant leadladder --dumps "$archive_root/ptbxl_leadladder" >> logs/ptbxl_leads.log 2>&1 && ok "PTB-XL lead ladder -> logs/ptbxl_leads.log"
  fi
fi
