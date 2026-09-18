#!/usr/bin/env bash
# Every DrugBAN cell behind the per-condition and summary tables.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-${PYTHON:-python3}}
OUT=${OUT:-results}
DUMP_ROOT=${GUARD_DRUGBAN_DUMPS:-${GUARD_ARTIFACTS:-artifacts}/drugban_processed}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
mkdir -p logs

run() {
  local name=$1 pool=$2 dumps="$DUMP_ROOT/$1"
  if [ ! -d "$dumps" ]; then
    echo "SKIP $name (missing $dumps)"
    return
  fi
  echo "START $name $pool"
  "$PY" experiments/exp_drugban.py --dumps "$dumps" --pool "$pool" --out "$OUT" \
    > "logs/${name}_${pool}.log" 2>&1
  echo "DONE  $name $pool"
}

for seed in s1 s2 s42; do
  for dataset in biosnap bindingdb human; do
    run "drugban_${dataset}_random_${seed}" source
  done
done
for seed in s1 s2 s42; do
  run "drugban_biosnap_cluster_${seed}" source
  run "drugban_biosnap_cluster_${seed}" deployment
done
run drugban_bindingdb_cluster_s42 source
run drugban_bindingdb_cluster_s42 deployment

echo "All available DrugBAN cells completed."
