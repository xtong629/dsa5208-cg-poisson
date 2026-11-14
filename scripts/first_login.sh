#!/usr/bin/env bash
# first_login.sh -- Monday checklist, step 1: collect cluster information into files (login node).
#   bash scripts/first_login.sh atlas          # info, then (if modules are configured) build + tiny validation
#   bash scripts/first_login.sh vanda          # info, then prints the interactive-job commands (step 2)
#   bash scripts/first_login.sh vanda --batch  # ... and submits step 2 as a batch job instead of interactive
#   bash scripts/first_login.sh atlas info     # info only
# Output files: results/first_login_<cluster>/{queues,hpc_cpu,hpc_gpu,modules,system}.txt  (git-ignored;
# read them, then fill the CHANGE_ME values in config/cluster.env).
set -u
cd "$(dirname "$0")/.."
source scripts/lib.sh
[ $# -ge 1 ] || { sed -n '2,9p' "$0"; exit 2; }
load_config "$1"; shift
MODE=default; BATCH=0
for a in "$@"; do case "$a" in info) MODE=info ;; --batch) BATCH=1 ;; esac; done
OUT="results/first_login_$CLUSTER"; mkdir -p "$OUT"

init="$(cfg ENV_INIT)"; [ -n "$init" ] && { echo "+ $init"; env_eval "$init"; }

echo "== system -> $OUT/system.txt"
{ hostname; date; uname -a; echo; (lscpu 2>/dev/null || sysctl -a 2>/dev/null | grep -E 'machdep.cpu|hw.ncpu') ; echo; nproc 2>/dev/null; free -g 2>/dev/null; command -v nvidia-smi >/dev/null && nvidia-smi; } > "$OUT/system.txt" 2>&1

echo "== cluster status (gstat) -> $OUT/queues.txt"
{
  echo '$ gstat'; (command -v gstat >/dev/null && gstat) 2>&1 || echo "(gstat not available here)"
  echo; echo '$ qstat -Q'; (command -v qstat >/dev/null && qstat -Q) 2>&1 || echo "(qstat not available)"
  echo
  # queue limits: the configured queues, or ALL queues if still CHANGE_ME
  qs=""
  for n in QUEUE GPU_QUEUE; do v="$(cfg $n 2>/dev/null)"; [ -n "$v" ] && [ "$v" != CHANGE_ME ] && qs="$qs $v"; done
  if [ -z "$qs" ] && command -v qstat >/dev/null; then qs="$(qstat -Q 2>/dev/null | awk 'NR > 2 { print $1 }')"; fi
  for q in $qs; do echo "\$ qstat -Qf $q"; qstat -Qf "$q" 2>&1; echo; done
} > "$OUT/queues.txt" 2>&1

echo "== sample job scripts (hpc cpu / hpc gpu) -> $OUT/hpc_cpu.txt, hpc_gpu.txt"
if command -v hpc >/dev/null 2>&1; then
  hpc cpu > "$OUT/hpc_cpu.txt" 2>&1; hpc gpu > "$OUT/hpc_gpu.txt" 2>&1
else echo "(hpc command not available here)" | tee "$OUT/hpc_cpu.txt" > "$OUT/hpc_gpu.txt"; fi

# remaining core-hour allocation: we do not guess a command. Save the help text of `hpc`, look for
# accounting-related subcommands in it, and run the ones that exist (read-only queries only).
echo "== core-hour allocation / quota -> $OUT/allocation.txt"
{
  if command -v hpc >/dev/null 2>&1; then
    for h in "hpc" "hpc help" "hpc --help" "hpc -h"; do echo "\$ $h"; $h 2>&1 | head -60; echo; done
    cands="$(( hpc help; hpc --help; hpc ) 2>&1 | grep -o -i -w -E 'quota|balance|usage|allocation|account|project|credit|credits' | tr 'A-Z' 'a-z' | sort -u)"
    for c in $cands; do echo "\$ hpc $c"; timeout 20 hpc $c 2>&1 | head -40; echo; done
  else
    echo "(hpc command not available on this machine)"
  fi
} > "$OUT/allocation.txt" 2>&1
echo "   (look for your remaining core-hours in $OUT/allocation.txt; if nothing there, ask the cluster admins which command reports it)"
grep -i -E 'core.?hour|remaining|balance|quota|allocation|used|credit' "$OUT/allocation.txt" | head -10

echo "== module avail -> $OUT/modules.txt   (module avail prints to stderr)"
if type module >/dev/null 2>&1; then
  { echo '$ module avail'; module avail 2>&1; echo; echo '$ module list'; module list 2>&1; } > "$OUT/modules.txt"
else echo "(module command not available: set ${CLUSTER_U}_ENV_INIT in config/cluster.env)" | tee "$OUT/modules.txt"; fi
echo "   MPI-ish modules:"; grep -i -E 'mpi|nvhpc|cuda|gcc|intel' "$OUT/modules.txt" | head -20

echo
echo "POLICY: ALL experiments (serial T_s, strong/weak, decomposition/halo, hybrid, GPU) run on VANDA, so that"
echo "        T_s and T_p(n) come from the same hardware (Atlas: 96 cores per user < the 256 required)."
echo
echo "Now read $OUT/*.txt and fill CHANGE_ME in config/cluster.env:"
echo "  queue names, cores per node, max cores per job (qstat -Qf), modules (module avail), GPUs per node (Vanda)."
[ "$MODE" = info ] && exit 0

if [ "$CLUSTER" = atlas ]; then
  echo; echo "== Atlas: build mpi/hybrid on the login node + tiny validation"
  if [ "$(cfg MODULES_CPU)" = CHANGE_ME ]; then
    echo "ATLAS_MODULES_CPU is still CHANGE_ME: pick compiler + MPI from $OUT/modules.txt, edit config/cluster.env, then run:"
    echo "  bash scripts/first_login_build.sh atlas"
  else
    bash scripts/first_login_build.sh atlas cpu
  fi
else
  echo
  echo "== Vanda: login nodes are VMs -- compile inside an interactive job (step 2)."
  q="$(cfg QUEUE)"; gq="$(cfg GPU_QUEUE)"
  echo "   GPU build + self-test (needs a GPU in the job):"
  echo "     qsub -I ${gq:+-q $gq }-l select=1:ncpus=4:ngpus=1 -l walltime=00:20:00     # exact select string: see $OUT/hpc_gpu.txt"
  echo "     cd $(pwd) && bash scripts/first_login_build.sh vanda both"
  echo "   CPU-only build check:"
  echo "     qsub -I ${q:+-q $q }-l select=1:ncpus=1 -l walltime=00:10:00"
  echo "     cd $(pwd) && bash scripts/first_login_build.sh vanda cpu"
  if [ "$BATCH" = 1 ]; then
    require_cfg GPU_QUEUE || exit 1
    echo "   submitting step 2 as a batch job ..."
    qsub -N p3_firstlogin -q "$(cfg GPU_QUEUE)" -l "select=1:ncpus=4:ngpus=1" -l walltime=00:20:00 -j oe -v "CLUSTER=vanda" scripts/pbs/firstlogin.pbs
  fi
fi
