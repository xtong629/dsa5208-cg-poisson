#!/usr/bin/env bash
# bind_diag.sh <cluster> -- inside an allocation: what may this job use, and which mpirun binding works?
set -u
cd "$(dirname "$0")/.."
source scripts/lib.sh
load_config "${1:?usage: bind_diag.sh <atlas|vanda>}"
init="$(cfg ENV_INIT)"; [ -n "$init" ] && env_eval "$init"
mods="$(cfg MODULES_CPU)"; [ -n "$mods" ] && [ "$mods" != CHANGE_ME ] && env_eval "$mods"
echo "host=$(hostname) job=${PBS_JOBID:-none}"; lscpu | grep -E "Model name|NUMA node[0-9]"
echo "--- what this job may use"
taskset -pc $$
grep -E "Cpus_allowed_list|Mems_allowed_list" /proc/self/status
cat /proc/self/cgroup | head -3
numactl -s 2>/dev/null | head -4
NP=${NP:-4}
t() { echo; echo "### $*"; timeout -k 5 60 "$@" 2>&1 | sed -e 's/^/    /' | head -${HEADN:-14}; echo "    rc=${PIPESTATUS[0]}"; }
t mpirun -np $NP --bind-to none --report-bindings hostname
t mpirun -np $NP --bind-to core --report-bindings hostname
t mpirun -np $NP --bind-to hwthread --report-bindings hostname
t mpirun -np $NP --bind-to numa --report-bindings hostname
t mpirun -np $NP --bind-to core --use-hwthread-cpus --report-bindings hostname
echo; echo "=== taskset wrapper (scripts/pin.sh), one CPU per rank"
PIN_VERBOSE=1 t mpirun -np $NP --bind-to none -x PIN_VERBOSE bash scripts/pin.sh hostname
echo "=== wrapper, 2 CPUs per rank (hybrid)"
PIN_THREADS=2 PIN_VERBOSE=1 t mpirun -np $((NP/2)) --bind-to none -x PIN_VERBOSE -x PIN_THREADS bash scripts/pin.sh hostname
