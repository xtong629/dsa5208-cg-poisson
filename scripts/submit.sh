#!/usr/bin/env bash
# submit.sh -- turn config/cluster.env into qsub commands (one PBS job per experiment tier).
#   bash scripts/submit.sh <atlas|vanda> <experiment|essential|comparisons|cpu|gpu|all> [--dry-run] [--preview]
#     essential   = serial strong weak gpu_sweep gpu_multi   (what the grading needs first: baseline, scaling, GPU)
#     comparisons = decomp_halo hybrid                       (submit after the essential results are in)
#   experiments: serial strong weak decomp_halo hybrid gpu_sweep gpu_multi
#   --dry-run : print the qsub commands and the core-hour upper bound (cores x walltime), submit nothing
#   --no-vanda-cpu : on Vanda submit only the GPU experiments (default: CPU experiments run on Vanda too)
#   --preview : like --dry-run but with assumed cluster numbers (64 cores/node ...) while
#               config/cluster.env still holds CHANGE_ME -- to look at the job plan before the cluster is known
# JOB_SPLIT=tiered (default): one job per node count so a 1-node run does not hold a 4-node allocation.
set -u
cd "$(dirname "$0")/.."
source scripts/lib.sh
[ $# -ge 2 ] || { sed -n '2,9p' "$0"; exit 2; }
load_config "$1"; WHAT="$2"; shift 2
DRY=0; PREVIEW=0; ALLOW_VANDA_CPU=1
for a in "$@"; do case "$a" in --dry-run) DRY=1 ;; --preview) DRY=1; PREVIEW=1 ;; --allow-vanda-cpu) ALLOW_VANDA_CPU=1 ;; --no-vanda-cpu) ALLOW_VANDA_CPU=0 ;; *) echo "unknown option $a" >&2; exit 2 ;; esac; done

if [ "$PREVIEW" = 1 ]; then
  # assumed numbers, only for the preview
  eval "${CLUSTER_U}_CORES_PER_NODE=64; ${CLUSTER_U}_MAX_CORES_PER_JOB=1024; ${CLUSTER_U}_QUEUE=QUEUE_PLACEHOLDER"
  eval "${CLUSTER_U}_GPU_QUEUE=GPU_QUEUE_PLACEHOLDER; ${CLUSTER_U}_GPUS_PER_NODE=4; ${CLUSTER_U}_GPU_NODE_CPUS=16"
  echo "(PREVIEW: assuming 64 cores/node, 4 GPUs/node, 16 CPUs with a GPU job -- NOT real values)"
fi

TOTAL_CH=0; TOTAL_GH=0
sub() {   # sub <name> <queue> <select-string> <walltime> <cores-charged> <vars> [gpus]
  local name="$1" queue="$2" sel="$3" wall="$4" cores="$5" vars="$6" gpus="${7:-0}" proj cmd h place
  proj="$(cfg PROJECT)"; place="$(cfg PLACE)"; [ "$gpus" -gt 0 ] && place=""   # exclusive nodes only for CPU jobs
  cmd="qsub -N p3_$name ${queue:+-q $queue }${proj:+-P $proj }-l $sel ${place:+-l place=$place }-l walltime=$wall -v $vars scripts/pbs/${name%%@*}.pbs"
  h="$(echo "$wall" | awk -F: '{ print $1 + $2 / 60 + $3 / 3600 }')"
  TOTAL_CH="$(awk -v t="$TOTAL_CH" -v c="$cores" -v h="$h" 'BEGIN { printf "%.1f", t + c * h }')"
  TOTAL_GH="$(awk -v t="$TOTAL_GH" -v g="$gpus" -v h="$h" 'BEGIN { printf "%.1f", t + g * h }')"
  printf '%s   # <= %s core-hours%s\n' "$cmd" "$(awk -v c="$cores" -v h="$h" 'BEGIN { printf "%.1f", c * h }')" \
    "$([ "$gpus" -gt 0 ] && awk -v g="$gpus" -v h="$h" 'BEGIN { printf ", %.1f GPU-hours", g * h }')"
  if [ "$DRY" = 0 ]; then eval "$cmd"; fi
}

fill() { local s="$1"; s="${s//\{NODES\}/$2}"; s="${s//\{CPN\}/$3}"; s="${s//\{CPUS\}/$2}"; s="${s//\{GPUS\}/$3}"; echo "$s"; }

# tiers <np list>  -> lines "<nodes> <np1:np2:...>" (np grouped by the number of nodes they need)
tiers() {
  local cpn="$1"; shift
  echo "$@" | tr ' ' '\n' | awk -v cpn="$cpn" 'NF { n = int(($1 + cpn - 1) / cpn); if (JOB == "single") n = -1; g[n] = g[n] (g[n] ? ":" : "") $1; if (n > maxn) maxn = n }
    END { if (JOB == "single") { printf "%d %s\n", maxn, g[-1]; exit } for (k in g) printf "%d %s\n", k, g[k] }' JOB="$JOB_SPLIT" | sort -rn      # largest node count first: it waits longest in the queue
}

cpu_job() {   # cpu_job <exp> <wall> <np list>
  local exp="$1" wall="$2" nps="$3" cpn maxc queue nodes list line sel
  cpn="$(cfg CORES_PER_NODE)"; maxc="$(cfg MAX_CORES_PER_JOB)"; queue="$(cfg QUEUE)"
  local w
  while read -r nodes list; do      # (here-string, not a pipe: TOTAL_CH must survive the loop)
    # a "single" job must be sized by the largest np
    if [ "$JOB_SPLIT" = single ]; then nodes="$(ceil_div "$(echo "$list" | tr ':' '\n' | sort -n | tail -1)" "$cpn")"; fi
    if [ $(( nodes * cpn )) -gt "$maxc" ]; then echo "SKIP $exp: needs $(( nodes * cpn )) cores > max $maxc per job (np: $list)"; continue; fi
    sel="$(fill "$(cfg SELECT_CPU)" "$nodes" "$cpn")"
    w="$wall"; [ "$nodes" -gt 1 ] && w="$WALL_MULTINODE"     # big allocations: short walltime
    sub "$exp" "$queue" "$sel" "$w" $(( nodes * cpn )) "CLUSTER=$CLUSTER,NP_LIST_JOB=$list"
  done <<< "$(tiers "$cpn" $nps)"
}

do_exp() {
  local e="$1" cpn
  case "$e" in serial|strong|weak|decomp_halo|hybrid)
    if [ "$CLUSTER" = vanda ] && [ "$ALLOW_VANDA_CPU" = 0 ]; then
      echo "SKIP $e on vanda (--no-vanda-cpu given)"
      return
    fi ;;
  esac
  case "$e" in
    serial)
      require_cfg QUEUE SELECT_SERIAL || exit 1
      sub serial "$(cfg QUEUE)" "$(cfg SELECT_SERIAL)" "$WALL_SERIAL" 1 "CLUSTER=$CLUSTER" ;;
    strong) cpu_job strong "$WALL_STRONG" "$NP_LIST $NP_EXTRA" ;;
    weak)   cpu_job weak "$WALL_WEAK" "$NP_LIST $NP_EXTRA" ;;
    decomp_halo) cpu_job decomp_halo "$WALL_DECOMP_HALO" "$DH_NP_LIST" ;;
    hybrid) cpu_job hybrid "$WALL_HYBRID" "$HYB_TOTAL_CORES" ;;
    gpu_sweep|gpu_multi)
      [ "$CLUSTER" = vanda ] || { echo "GPU experiments run on Vanda only" >&2; return; }
      local cpus gpus sel w
      if [ "$e" = gpu_sweep ]; then gpus=1; w="$WALL_GPU_SWEEP"; else gpus="$(cfg GPUS_PER_NODE)"; w="$WALL_GPU_MULTI"; fi
      cpus=$(( $(cfg GPU_NODE_CPUS) * gpus ))       # GPU_NODE_CPUS = CPU cores that come with ONE GPU
      sel="$(fill "$(cfg SELECT_GPU)" "$cpus" "$gpus")"
      sub "$e" "$(cfg GPU_QUEUE)" "$sel" "$w" "$cpus" "CLUSTER=$CLUSTER" "$gpus" ;;
    *) echo "unknown experiment $e" >&2; exit 2 ;;
  esac
}

if [ "$PREVIEW" = 0 ]; then
  req="CORES_PER_NODE MAX_CORES_PER_JOB QUEUE"
  if [ "$CLUSTER" = vanda ] && [ "$ALLOW_VANDA_CPU" = 0 ]; then req="GPU_QUEUE GPUS_PER_NODE GPU_NODE_CPUS"; fi
  case "$WHAT" in gpu*|all) [ "$CLUSTER" = vanda ] && req="$req GPU_QUEUE GPUS_PER_NODE GPU_NODE_CPUS" ;; esac
  require_cfg $req || { echo "fill in config/cluster.env (or use --preview to see the plan)" >&2; exit 1; }
fi
case "$WHAT" in
  essential)   for e in serial strong weak; do do_exp $e; done
               [ "$CLUSTER" = vanda ] && for e in gpu_sweep gpu_multi; do do_exp $e; done ;;
  comparisons) for e in decomp_halo hybrid; do do_exp $e; done ;;
  cpu) for e in serial strong weak decomp_halo hybrid; do do_exp $e; done ;;
  gpu) for e in gpu_sweep gpu_multi; do do_exp $e; done ;;
  all) for e in serial strong weak decomp_halo hybrid; do do_exp $e; done
       [ "$CLUSTER" = vanda ] && for e in gpu_sweep gpu_multi; do do_exp $e; done ;;
  *) do_exp "$WHAT" ;;
esac
echo "TOTAL upper bound if every job ran to its walltime limit: $TOTAL_CH core-hours (CPU cores requested, incl. the CPUs that come with GPU jobs), $TOTAL_GH GPU-hours"
