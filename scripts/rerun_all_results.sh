#!/usr/bin/env bash
# One pass over every driver behind a reported number, after the review fixes:
# C1 tightening floor, C2 pool-scaled kernel, C3 raw-host baseline, C11 AUROC ties,
# C12 multi-label temperature, C13 CRC infimum, C14 screen, C15 exchangeable,
# D6 LTT grid from the fit split, D7 stale fit target, D15 multi-label entropy.
set -u
cd <workspace>/guard-release
PY=../venv-release/bin/python
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
L=logs/rerun2; mkdir -p $L
run() { local tag=$1 cores=$2; shift 2
        echo "START $tag $(date +%H:%M)"
        taskset -c "$cores" $PY -u "$@" > "$L/$tag.log" 2>&1 \
          && echo "DONE $tag $(date +%H:%M)" || echo "FAIL $tag $(date +%H:%M)"; }

# longest first so they overlap with everything else
run gates_iemocap 2-4   experiments/ltt_one.py gates_iemocap &
run gates_rest    5-7   experiments/ltt_one.py gates_rest &
run gates_drugban 8-10  experiments/ltt_one.py gates_drugban &
run gates_mosei2  11-13 experiments/ltt_one.py gates_mosei2 &
run opp_dcl_cells 14-16 experiments/ltt_one.py opp_dcl &
wait
run gates_ptbxl   2-4   experiments/ltt_one.py gates_ptbxl &
run mosei_hosts   5-7   experiments/mosei_hosts.py &
run mosei_full    8-10  experiments/mosei_full.py &
run iemocap_eta   11-13 experiments/iemocap_eta.py &
run ave_eta       14-16 experiments/ave_eta.py &
wait
run ptbxl_sev_dense 2-4   experiments/ptbxl_sev_dense.py &
run nina_sev_dense  5-7   experiments/nina_sev_dense.py &
run opp_dcl         8-10  experiments/opp_dcl.py &
run opp_targets     11-13 experiments/opp_targets.py &
run gates_ave       14-16 experiments/gates_ave.py &
wait
run alpha_affective 2-4   experiments/affective_alpha.py &
run alpha_ave       5-7   experiments/ave_alpha.py &
run alpha_drugban   8-10  experiments/drugban_alpha.py &
run alpha_nina      11-13 experiments/nina_alpha.py &
run alpha_opp       14-16 experiments/opp_alpha.py &
wait
run alpha_opp_nx    2-4   experiments/opp_alpha_nx.py &
run alpha_ptbxl     5-7   experiments/alpha_sweep_new.py &
run ptbxl_shift     8-10  experiments/ptbxl_shift.py &
run blanket_two     11-13 experiments/blanket_two.py &
run drugban_ladder  14-16 scripts/drugban_protladder_guard.py &
wait
echo "PHAN NON-DRUGBAN XONG $(date +%H:%M)"
JOBS=4 OUT=results PY=$PY bash scripts/run_drugban_all.sh
echo "TAT CA XONG $(date +%H:%M)"
