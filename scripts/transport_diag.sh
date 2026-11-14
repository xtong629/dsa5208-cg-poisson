#!/usr/bin/env bash
# transport_diag.sh <cluster> -- inside a MULTI-node allocation: which Open MPI transport settings work between nodes?
# Every candidate runs the solver (8+ ranks across the nodes) and the Allreduce latency benchmark, with a timeout.
set -u
cd "$(dirname "$0")/.."
source scripts/lib.sh
load_config "${1:?usage: transport_diag.sh <atlas|vanda>}"
init="$(cfg ENV_INIT)"; [ -n "$init" ] && env_eval "$init"
mods="$(cfg MODULES_CPU)"; [ -n "$mods" ] && [ "$mods" != CHANGE_ME ] && env_eval "$mods"
M="$(mpirun_cmd)"; PIN="$(cfg PIN)"
NP=$(wc -l < "$PBS_NODEFILE"); NN=$(sort -u "$PBS_NODEFILE" | wc -l)
echo "job ${PBS_JOBID:-}: $NP slots on $NN nodes: $(sort -u "$PBS_NODEFILE" | tr '\n' ' ')"
echo "--- do the nodes look like the same machine to UCX?"
for n in $(sort -u "$PBS_NODEFILE"); do ssh -o BatchMode=yes "$n" 'echo "$(hostname): machine-id=$(cat /etc/machine-id 2>/dev/null) boot_id=$(cat /proc/sys/kernel/random/boot_id)"'; done
ip -br addr 2>/dev/null | grep -v "^lo" | head -6
ibv_devinfo 2>/dev/null | grep -E "hca_id|transport|state|active_speed" | head -8
T=${CHECK_TIMEOUT:-30}
cand() { # cand <label> <mpirun options...>   (binary: $BIN)
  local label="$1"; shift
  echo; echo "################ $label"
  echo "+ mpirun options: $*"
  PIN_THREADS=1 timeout -k 5 "$T" $M -np "$NP" --bind-to none -x PIN_THREADS "$@" $PIN $BIN --dim 3 --N 64 --iters 20 > /tmp/td_$$.out 2>&1; rc=$?
  grep -vE "^\[.*(UCX|LOG_CAT|TL_UCP|UCC)" /tmp/td_$$.out | tail -2 | cut -c1-200; echo "solver rc=$rc"
  if [ $rc = 0 ]; then
    PIN_THREADS=1 timeout -k 5 "$T" $M -np "$NP" --bind-to none -x PIN_THREADS "$@" $PIN $BIN --bench-allreduce 2000 --csv /tmp/ar_$$.csv --tag diag 2>&1 | grep -iE "allreduce|latency" | tail -2 | cut -c1-200
  fi
}
NOUCX="--mca coll ^hcoll,ucc --mca osc ^ucx"
BIN=build/mpi
echo "######## stack 1: NVHPC HPC-X Open MPI (the build in build/)"
cand "1 ob1+tcp, hcoll/ucc/ucx-osc disabled"          $NOUCX --mca pml ob1 --mca btl self,vader,tcp
cand "2 ob1+tcp on ens6f1, same"                       $NOUCX --mca pml ob1 --mca btl self,vader,tcp --mca btl_tcp_if_include ens6f1
cand "3 UCX pml, tcp+shm, hcoll/ucc off"               $NOUCX -x UCX_TLS=tcp,self,sm -x UCX_NET_DEVICES=ens6f1
cand "4 UCX pml, tcp only (no shared memory)"          $NOUCX -x UCX_TLS=tcp,self -x UCX_NET_DEVICES=ens6f1

echo; echo "######## stack 2: cluster Open MPI 4.1.8 + gcc 13.1"
type module >/dev/null 2>&1 || source /etc/profile
module unload nvhpc-hpcx-cuda12/26.1 hpcx >/dev/null 2>&1      # (not piped: a pipe would lose the environment change)
module load gcc/13.1.0 openmpi/gcc/64/4.1.8
echo "mpirun=$(command -v mpirun)  $(mpirun --version 2>&1 | head -1)"; ompi_info 2>/dev/null | grep -E "plm: tm|ras: tm" 
make mpi BUILD=build_b CXX=g++ MPICXX=mpicxx 2>&1 | tail -1
BIN=build_b/mpi
M="mpirun --hostfile $PBS_NODEFILE"
cand "5 Bright OpenMPI, default transport"            
cand "6 Bright OpenMPI, ob1+tcp"                        --mca pml ob1 --mca btl self,vader,tcp
rm -f /tmp/ar_$$.csv /tmp/td_$$.out
