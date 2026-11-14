#!/usr/bin/env bash
# validate_full.sh -- the FULL validation grid (all rank counts, thread counts and the negative control)
# inside the current allocation (batch job: scripts/pbs/validatefull.pbs, 32 cores are enough for 8 ranks x 4 threads).
#   bash scripts/validate_full.sh <atlas|vanda|local> [all|negative]     (default: all)
# Correctness only: nothing here is timed or used for results.
set -u
cd "$(dirname "$0")/.."
source scripts/lib.sh
load_config "${1:?usage: validate_full.sh <atlas|vanda|local>}"
init="$(cfg ENV_INIT)"; [ -n "$init" ] && env_eval "$init"
mods="$(cfg MODULES_CPU)"; [ -n "$mods" ] && env_eval "$mods"
make serial mpi hybrid $(cfg MAKE_FLAGS) || exit 1
MPIRUN="$(mpirun_cmd)"
echo ">>> full validation on $(hostname -s), $(date -u +%FT%TZ)"
env MPIRUN="$MPIRUN" MPIRUN_FLAGS="--bind-to none" python3 scripts/validate.py "${2:-all}"
rc=$?
echo ">>> exit status $rc"
exit $rc
