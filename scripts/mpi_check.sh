#!/usr/bin/env bash
# mpi_check.sh -- run INSIDE a (multi-node) allocation: does mpirun, the default transport, MPI_Init_thread
# (hybrid build) and the binding flags from config/cluster.env work? Every launch has a timeout.
#   bash scripts/mpi_check.sh <atlas|vanda>          (PBS: scripts/pbs/mpicheck.pbs)
# Only correctness / environment information; nothing here is timed or reported.
set -u
cd "$(dirname "$0")/.."
source scripts/lib.sh
load_config "${1:?usage: mpi_check.sh <atlas|vanda>}"
init="$(cfg ENV_INIT)"; [ -n "$init" ] && env_eval "$init"
mods="$(cfg MODULES_CPU)"; [ -n "$mods" ] && [ "$mods" != CHANGE_ME ] && env_eval "$mods"
MPIRUN="$(mpirun_cmd)"; BIND="$(cfg MPI_BIND)"; HBIND="$(cfg HYBRID_BIND)"; PIN="$(cfg PIN)"
TMO=${CHECK_TIMEOUT:-90}
FAIL=0
say() { echo; echo ">>> $*"; }
try() { # try <label> <cmd...> : run with a timeout in its own process group
  local label="$1"; shift
  echo "+ $*"
  timeout -k 5 "$TMO" "$@"; local rc=$?
  if [ $rc -eq 0 ]; then echo "[PASS] $label"; else echo "[FAIL rc=$rc] $label"; FAIL=1; fi
}

say "environment"
echo "host=$(hostname) pbs_jobid=${PBS_JOBID:-none}"
[ -n "${PBS_NODEFILE:-}" ] && { echo "nodes:"; sort "$PBS_NODEFILE" | uniq -c; }
lscpu | grep -E "Model name|^CPU\(s\)|Socket|Thread|NUMA node\(s\)|L3" 
command -v nvidia-smi >/dev/null && nvidia-smi -L
echo "mpirun=$(command -v mpirun)  $(mpirun --version 2>&1 | head -1)"
ompi_info 2>/dev/null | grep -E "MCA (pml|mtl)|Thread support|plm: tm|ras: tm" | head -8
command -v ucx_info >/dev/null && { echo "UCX transports:"; ucx_info -d 2>/dev/null | grep -E "Transport:|Device:" | sort -u | head -10; }

say "build (skipped if already built)"
[ -x build/mpi ] && [ -x build/hybrid ] || { make serial mpi hybrid $(cfg MAKE_FLAGS) 2>&1 | tail -3; }
[ -x build/mpi ] && [ -x build/hybrid ] || { echo "build failed"; exit 1; }

NP=${PBS_NODEFILE:+$(wc -l < "$PBS_NODEFILE")}; NP=${NP:-4}
NNODES=${PBS_NODEFILE:+$(sort -u "$PBS_NODEFILE" | wc -l)}; NNODES=${NNODES:-1}
say "launch: $NP ranks on $NNODES node(s), binding '$BIND'"
echo "ranks per host:"; timeout -k 5 "$TMO" $MPIRUN -np "$NP" $BIND hostname 2>&1 | sort | uniq -c
NH=$(timeout -k 5 "$TMO" $MPIRUN -np "$NP" $BIND hostname 2>/dev/null | sort -u | wc -l)
if [ "$NH" = "$NNODES" ]; then echo "[PASS] ranks are spread over all $NNODES node(s)"; else echo "[FAIL] ranks run on $NH host(s), the job has $NNODES"; FAIL=1; fi
PIN_VERBOSE=1 try "mpirun + pinning: every rank starts" $MPIRUN -np "$NP" $BIND -x PIN_VERBOSE -x PIN_THREADS $PIN hostname
try "pure MPI, default transport"   env PIN_THREADS=1 $MPIRUN -np "$NP" $BIND -x PIN_THREADS $PIN build/mpi    --dim 3 --N 64 --iters 20
try "hybrid build (MPI_Init_thread), 1 thread, default transport" env PIN_THREADS=1 OMP_NUM_THREADS=1 $MPIRUN -np "$NP" $BIND -x OMP_NUM_THREADS -x PIN_THREADS $PIN build/hybrid --dim 3 --N 64 --iters 20
if [ "$NP" -ge 4 ]; then
  R=$(( NP / 2 )); T=2
  HB="${HBIND//\{T\}/$T}"
  try "hybrid ${R}x${T} (pinned, $T CPUs per rank)" env PIN_THREADS=$T PIN_VERBOSE=1 OMP_NUM_THREADS=$T OMP_PLACES="$(cfg OMP_PLACES)" OMP_PROC_BIND="$(cfg OMP_PROC_BIND)" $MPIRUN -np "$R" $HB -x OMP_NUM_THREADS -x OMP_PLACES -x OMP_PROC_BIND -x PIN_THREADS -x PIN_VERBOSE $PIN build/hybrid --dim 3 --N 64 --iters 20
fi
say "result"; [ $FAIL = 0 ] && echo "ALL MPI CHECKS PASSED" || echo "SOME MPI CHECKS FAILED"
exit $FAIL
