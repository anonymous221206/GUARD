#!/usr/bin/env bash
# Re-run the CPU-side drivers behind the reported GUARD results.
# Requires the saved host-output dumps downloaded by data/download_artifacts.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
L=${GUARD_LOG_DIR:-logs/rerun}; mkdir -p "$L"

run() {
  local tag=$1; shift
  echo "START $tag"
  "$PY" -u "$@" > "$L/$tag.log" 2>&1
  echo "DONE  $tag"
}

for driver in gates_iemocap gates_rest gates_drugban gates_mosei2 opp_dcl gates_ptbxl; do
  tag=$driver; [ "$driver" = opp_dcl ] && tag=opp_dcl_cells
  run "$tag" experiments/ltt_one.py "$driver"
done
for driver in mosei_hosts mosei_full iemocap_eta ave_eta ptbxl_sev_dense nina_sev_dense opp_dcl opp_targets gates_ave; do
  run "$driver" "experiments/$driver.py"
done
for item in \
  alpha_affective:experiments/affective_alpha.py \
  alpha_ave:experiments/ave_alpha.py \
  alpha_drugban:experiments/drugban_alpha.py \
  alpha_nina:experiments/nina_alpha.py \
  alpha_opp:experiments/opp_alpha.py \
  alpha_opp_nx:experiments/opp_alpha_nx.py \
  alpha_ptbxl:experiments/alpha_sweep_new.py \
  ptbxl_shift:experiments/ptbxl_shift.py \
  blanket_two:experiments/blanket_two.py \
  drugban_ladder:scripts/drugban_protladder_guard.py; do
  run "${item%%:*}" "${item#*:}"
done

OUT=results PY="$PY" bash scripts/run_drugban_all.sh

run review_conditional experiments/review_conditional.py
run review_probe_mosei experiments/review_probe_extension.py mosei
run review_probe_cluster experiments/review_probe_extension.py cluster
if [ "${RUN_TIMINGS:-0}" = 1 ]; then
  for case in AVE PTB-XL IEMOCAP NinaPro; do
    run "review_efficiency_${case//-/_}" experiments/review_efficiency.py "$case"
  done
else
  echo "SKIP hardware-dependent timing reruns (set RUN_TIMINGS=1 to enable)"
fi

echo "All CPU-side result drivers completed."
