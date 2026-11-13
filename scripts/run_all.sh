#!/usr/bin/env bash
# run_all.sh -- run experiment(s) in the CURRENT allocation (inside a PBS job, an interactive job,
# or locally). The PBS templates in scripts/pbs/ just call this.
#   bash scripts/run_all.sh <atlas|vanda|local> <serial|strong|weak|decomp_halo|hybrid|gpu_sweep|gpu_multi|cpu|gpu|all>
#   SMOKE=1 bash scripts/run_all.sh local all     # tiny local pipeline test (output in results/local_smoke)
# Environment: NP_LIST_JOB=1:2:4 restricts the rank counts; RESULTS_DIR overrides ./results
set -u
cd "$(dirname "$0")/.."
source scripts/lib.sh
[ $# -ge 2 ] || { sed -n '2,6p' "$0"; exit 2; }
load_config "$1"
source scripts/experiments.sh
what="$2"
case "$what" in
  serial|strong|weak|decomp_halo|hybrid|gpu_sweep|gpu_multi) "exp_$what" ;;
  cpu) exp_serial; exp_strong; exp_weak; exp_decomp_halo; exp_hybrid ;;
  gpu) exp_gpu_sweep; exp_gpu_multi ;;
  all) # policy: CPU experiments on Atlas; Vanda (limited core-hours) only runs the GPU experiments
       if [ "$CLUSTER" = vanda ] && [ "${ALLOW_VANDA_CPU:-0}" != 1 ]; then exp_gpu_sweep; exp_gpu_multi
       else exp_serial; exp_strong; exp_weak; exp_decomp_halo; exp_hybrid
            [ "$CLUSTER" = vanda ] && { exp_gpu_sweep; exp_gpu_multi; }; fi ;;
  *) echo "unknown experiment $what" >&2; exit 2 ;;
esac
echo "done: $what ($CLUSTER) -> ${RESDIR:-results}"
[ -s "${FAILS:-/dev/null}" ] && { echo "WARNING: some runs failed, see $FAILS" >&2; exit 1; }
exit 0
