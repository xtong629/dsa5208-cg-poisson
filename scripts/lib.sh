# lib.sh -- helpers shared by run_all.sh, experiments.sh, submit.sh and the first-login scripts.
# Written for bash 3.2+ (macOS default) -- no associative arrays, no ${var^^}.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

load_config() {            # load_config <cluster>   (atlas | vanda | local)
  CLUSTER="$1"
  CLUSTER_U="$(echo "$CLUSTER" | tr '[:lower:]' '[:upper:]')"
  # shellcheck disable=SC1091
  source "$REPO_ROOT/config/cluster.env"
  if [ -n "${SMOKE:-}" ]; then SMOKE_OVERRIDES; fi
}

cfg() {                    # cfg NAME -> value of <CLUSTER>_NAME
  local v="${CLUSTER_U}_$1"
  echo "${!v}"
}

require_cfg() {            # require_cfg NAME...  -> abort if empty or still CHANGE_ME
  local bad=0 n val
  for n in "$@"; do
    val="$(cfg "$n")"
    if [ -z "$val" ] || [ "$val" = "CHANGE_ME" ]; then
      echo "config/cluster.env: ${CLUSTER_U}_$n is not set (CHANGE_ME)" >&2
      bad=1
    fi
  done
  return $bad
}

ceil_div() { echo $(( ($1 + $2 - 1) / $2 )); }

# N(n) for weak scaling: round(base * n^(1/dim))
weak_N() { awk -v b="$1" -v n="$2" -v d="$3" 'BEGIN { printf "%d", b * exp(log(n) / d) + 0.5 }'; }

# unknowns = N^dim
unknowns() { awk -v n="$1" -v d="$2" 'BEGIN { printf "%.0f", n ^ d }'; }

# prepare the environment for running jobs on a cluster: modules, directories, CSV path
init_run() {               # init_run <experiment> [cpu|gpu]
  EXP="$1"; KIND="${2:-cpu}"
  cd "${PBS_O_WORKDIR:-$REPO_ROOT}" || exit 1
  local init mods
  init="$(cfg ENV_INIT)"
  [ -n "$init" ] && eval "$init"
  if [ "$KIND" = "gpu" ]; then mods="$(cfg MODULES_GPU)"; else mods="$(cfg MODULES_CPU)"; fi
  [ -n "$mods" ] && [ "$mods" != "CHANGE_ME" ] && eval "$mods"
  RESDIR="${RESULTS_DIR:-results}/$CLUSTER"
  [ -n "${SMOKE:-}" ] && RESDIR="${RESULTS_DIR:-results}/local_smoke"
  mkdir -p "$RESDIR"
  CSV="$RESDIR/$EXP.csv"
  MPIRUN="$(cfg MPIRUN)"
  MPI_BIND="$(cfg MPI_BIND)"
  HYBRID_BIND="$(cfg HYBRID_BIND)"
  FAILS="$RESDIR/$EXP.failures.txt"
  # provenance of this run (tiny text file next to the CSV; results/ is git-ignored)
  {
    echo "experiment=$EXP cluster=$CLUSTER date=$(date -u +%FT%TZ) host=$(hostname)"
    echo "git=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "pbs_jobid=${PBS_JOBID:-none} np_list=${NP_LIST_JOB:-}"
    [ -n "${PBS_NODEFILE:-}" ] && sort -u "$PBS_NODEFILE" | tr '\n' ' ' && echo
    echo "OMP/MPI: $($MPIRUN --version 2>&1 | head -1)"
  } >> "$RESDIR/$EXP.meta.txt" 2>/dev/null
}

need_bin() {               # need_bin build/mpi ...
  local b
  for b in "$@"; do
    [ -x "$REPO_ROOT/build/$b" ] || { echo "missing build/$b -- build it first (see README / scripts/first_login.sh)" >&2; exit 2; }
  done
}

# run one command, never abort the whole job if a single run fails; log the failure
run_logged() {
  "$@" || echo "FAILED: $*" >> "$FAILS"
}

# MPI launch: run_mpi <np> <bin> args...   (pure MPI, 1 thread per rank)
run_mpi() {
  local np="$1"; shift
  OMP_NUM_THREADS=1 run_logged $MPIRUN -np "$np" $MPI_BIND "$REPO_ROOT/build/$1" "${@:2}"
}

# hybrid launch: run_hybrid <ranks> <threads> <bin> args...
run_hybrid() {
  local np="$1" th="$2"; shift 2
  local bind="${HYBRID_BIND//\{T\}/$th}"
  OMP_NUM_THREADS="$th" OMP_PLACES="$(cfg OMP_PLACES)" OMP_PROC_BIND="$(cfg OMP_PROC_BIND)" \
    run_logged $MPIRUN -np "$np" $bind "$REPO_ROOT/build/$1" "${@:2}"
}

# is a run worth doing? (skip subdomains that are tiny or too large for the memory/time budget)
pts_ok() {                 # pts_ok <dim> <N> <np> [max_per_rank]
  local u pr
  u="$(unknowns "$2" "$1")"
  pr="$(awk -v u="$u" -v n="$3" 'BEGIN { printf "%.0f", u / n }')"
  [ "$pr" -ge "$MIN_PTS_PER_RANK" ] || return 1
  if [ -n "${4:-}" ]; then [ "$pr" -le "$4" ] || return 1; fi
  return 0
}
