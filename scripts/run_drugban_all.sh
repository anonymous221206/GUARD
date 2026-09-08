#!/usr/bin/env bash
# Every DrugBAN cell behind Table 9 and the DrugBAN row of Table 2.
# Random splits draw the retrieval pool from the source population; the cluster
# splits are run both ways, because which pool a cross-domain deployment can
# reach is one of the paper's findings.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-../venv-release/bin/python}
JOBS=${JOBS:-3}
OUT=${OUT:-results}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

run() {
  d=$1; pool=$2
  echo "START $d $pool"
  $PY experiments/exp_drugban.py --dumps "data/processed/$d" --pool "$pool" --out "$OUT" \
      > "logs/drugban_${d}_${pool}.log" 2>&1 && echo "DONE $d $pool" || echo "FAIL $d $pool"
}

mkdir -p logs
for s in s1 s2 s42; do
  for ds in biosnap bindingdb human; do
    echo "${ds}_random_$s source"
  done
done > /tmp/guard_db_jobs.$$
for s in s1 s2 s42; do echo "biosnap_cluster_$s source"; echo "biosnap_cluster_$s deployment"; done >> /tmp/guard_db_jobs.$$
echo "bindingdb_cluster_s42 source" >> /tmp/guard_db_jobs.$$
echo "bindingdb_cluster_s42 deployment" >> /tmp/guard_db_jobs.$$

while read -r name pool; do
  [ -d "data/processed/drugban_$name" ] || { echo "SKIP drugban_$name (khong co dump)"; continue; }
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  run "drugban_$name" "$pool" &
done < /tmp/guard_db_jobs.$$
wait
rm -f /tmp/guard_db_jobs.$$
echo "ALL DONE"
