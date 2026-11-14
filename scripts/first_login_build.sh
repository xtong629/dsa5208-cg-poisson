#!/usr/bin/env bash
# first_login_build.sh -- Monday checklist, step 2: build + test inside the current allocation.
#   bash scripts/first_login_build.sh <atlas|vanda|local> [cpu|gpu|both]      (default: both on vanda, cpu otherwise)
# CPU part : load CPU modules, make serial mpi hybrid, run `validate.py --quick`.
# GPU part : load GPU modules, `make cuda` (falls back to `make cuda-alt`), run `build/cuda --selftest`
#            with 1 and 2 ranks. Needs a visible GPU; run it inside a GPU interactive job.
# Prints a PASS/FAIL summary at the end. Nothing here is timed or used for results.
set -u
cd "$(dirname "$0")/.."
source scripts/lib.sh
load_config "${1:?usage: first_login_build.sh <atlas|vanda|local> [cpu|gpu|both]}"
PART="${2:-}"; [ -z "$PART" ] && { [ "$CLUSTER" = vanda ] && PART=both || PART=cpu; }
init="$(cfg ENV_INIT)"; [ -n "$init" ] && env_eval "$init"
MPIRUN="$(mpirun_cmd)"
SUMMARY=""
step() { # step <label> <command...>
  local label="$1"; shift
  echo; echo ">>> $label"
  if "$@"; then SUMMARY="$SUMMARY\n  PASS  $label"; return 0; else SUMMARY="$SUMMARY\n  FAIL  $label"; return 1; fi
}

if [ "$PART" = cpu ] || [ "$PART" = both ]; then
  mods="$(cfg MODULES_CPU)"
  if [ -n "$mods" ] && [ "$mods" != CHANGE_ME ]; then env_eval "$mods"; else echo "note: ${CLUSTER_U}_MODULES_CPU not set; using the current environment"; fi
  step "CPU: mpicxx found ($(command -v mpicxx || echo none))" command -v mpicxx
  step "CPU: make serial mpi" make serial mpi $(cfg MAKE_FLAGS)
  step "CPU: make hybrid (OpenMP)" make hybrid $(cfg MAKE_FLAGS)
  [ -x build/hybrid ] || echo "   (hybrid failed: try OMPFLAGS=-mp for nvc++, or put the right flags in ${CLUSTER_U}_MAKE_FLAGS)"
  # correctness only: no pinning (a PBS cgroup only allows the job's own CPUs; mpirun's default socket binding fails there)
  step "CPU: tiny validation (serial, mpi$([ -x build/hybrid ] && echo ', hybrid'))" env MPIRUN="$MPIRUN" MPIRUN_FLAGS="--bind-to none" python3 scripts/validate.py all --quick
fi

if [ "$PART" = gpu ] || [ "$PART" = both ]; then
  if [ "$CLUSTER" != vanda ]; then echo "GPU part only on vanda"; else
    mods="$(cfg MODULES_GPU)"
    env_eval "$mods"
    MPIRUN="$(mpirun_cmd gpu)"
    echo; echo "GPU environment: nvcc=$(command -v nvcc || echo none)  mpicxx=$(command -v mpicxx || echo none)"
    nvidia-smi -L 2>&1 | head -4 || true
    ARCH="CUDA_ARCH=$CUDA_ARCH $(cfg CUDA_MAKE_FLAGS)"
    rm -f build/cuda
    if ! step "GPU: make cuda  (nvcc -ccbin mpicxx)" make cuda $ARCH; then
      echo "   -> falling back to make cuda-alt (system g++ as nvcc host compiler, MPI -I flags from mpicxx)"
      step "GPU: make cuda-alt (fallback)" make cuda-alt $ARCH
    fi
    if [ -x build/cuda ]; then
      step "GPU: selftest, 1 rank"  timeout 180 $MPIRUN --bind-to none -np 1 build/cuda --selftest
      step "GPU: selftest, 2 ranks (CUDA-aware MPI halo exchange between ranks)" timeout 180 $MPIRUN --bind-to none -np 2 build/cuda --selftest
      echo "   if the device-pointer runs crash: CUDA-aware MPI may be off -> rerun with staging only:  mpirun -np 2 build/cuda --selftest   (it skips device paths when MPI reports no support)"
      echo "   or rebuild with  make cuda CUDA_EXTRA=-DHOST_STAGING   (default to host-staged halos)"
    fi
  fi
fi
echo; echo "================ summary ================"; printf "$SUMMARY\n"
case "$SUMMARY" in *FAIL*) exit 1 ;; *) exit 0 ;; esac
