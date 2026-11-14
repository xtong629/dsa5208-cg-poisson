#!/usr/bin/env bash
# map_diag.sh -- where do the ranks of a hybrid launch land? Prints ranks per node for the old (slot-by-slot) and the new
# (ppr) mapping, inside a multi-node allocation. Correctness/placement only; nothing is timed.
set -u
cd "$(dirname "$0")/.."
source scripts/lib.sh
load_config "${1:?usage: map_diag.sh <atlas|vanda|local>}"
init="$(cfg ENV_INIT)"; [ -n "$init" ] && env_eval "$init"
mods="$(cfg MODULES_CPU)"; [ -n "$mods" ] && env_eval "$mods"
MPIRUN="$(mpirun_cmd)"; CPN="$(cfg CORES_PER_NODE)"
NODES=$(sort -u "$PBS_NODEFILE" | wc -l)
echo "nodes in allocation: $NODES, cores per node used: $CPN"
for th in 1 2 4 16; do
  np=$(( NODES * CPN / th ))
  echo "--- np=$np threads=$th, OLD mapping (--bind-to none):"
  $MPIRUN -np "$np" --bind-to none hostname | sort | uniq -c
  echo "--- np=$np threads=$th, NEW mapping (--map-by ppr:$(( CPN / th )):node):"
  $MPIRUN -np "$np" --bind-to none --map-by ppr:$(( CPN / th )):node hostname | sort | uniq -c
done
