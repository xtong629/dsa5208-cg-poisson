# Matrix-free Conjugate Gradient for the Poisson equation: MPI, MPI+OpenMP and MPI+CUDA

Project 3 of NUS DSA5208 *Scalable Distributed Computing for Data Science*.

Authors: CHEN XINGTONG (A0357556N), QIAN LONG (A0255547W).

We solve the Poisson equation `-Δu = f` on the unit square (2D, 5-point stencil) and unit cube
(3D, 7-point stencil) with homogeneous Dirichlet boundary conditions, using a **matrix-free
Conjugate Gradient (CG)** method, and study how different ways of parallelising it affect
scaling and efficiency. Everything is C++17; the kernels (stencil, dot product, axpy) are written
by hand (no BLAS/LAPACK).

The problem data are **synthetic**: a manufactured analytic solution `u(x) = Π_d g(x_d)`,
`g(t) = t(1-t)eᵗ`, gives an analytic right-hand side `f`, so the exact error is computable and no
input files exist (see `src/common/problem.hpp` for why this is not a product of sines).

## What is implemented

| Component | Binary | Notes |
|---|---|---|
| Serial CG (provides `T_s`) | `build/serial` | 2D and 3D |
| MPI | `build/mpi` | `MPI_Cart_create`; **slab (1D)** vs **block (2D/3D)** decomposition; **blocking** (`MPI_Sendrecv`) vs **non-blocking** halo exchange, with or without **computation/communication overlap** (inner points first); derived datatypes for faces; 2 `MPI_Allreduce` per iteration; `MPI_Allreduce` latency micro-benchmark |
| MPI+OpenMP | `build/hybrid` | same source as `mpi`, `-DUSE_OPENMP`; `MPI_THREAD_FUNNELED`, static `parallel for`, reductions, first-touch initialisation |
| MPI+CUDA | `build/cuda` | one rank per GPU; stencil / axpy / dot / pack / unpack kernels; **CUDA-aware MPI** halo exchange from device buffers, or host staging (`--staging 1`); stream overlap; `--selftest` |

Not implemented: one-sided RMA halo exchange, communication-reducing CG (Chronopoulos–Gear), Kokkos.

Every program writes **one CSV row per run** (timers: stencil, halo exchange, Allreduce, vector
updates, other; iterations; errors). The report is in [`report.md`](report.md).

## Repository layout

```
Makefile                      builds: serial | mpi | hybrid | cuda | cuda-alt | cuda-check | test
config/cluster.env            ALL cluster-specific settings + experiment sizes (placeholders: CHANGE_ME)
src/common/                   grid layout, kernels, manufactured problem, CG loop, timers/CSV, inner/shell split
src/mpi/                      Cartesian decomposition, halo exchange, MPI operators
src/cuda/                     CUDA kernels, compatibility macros, stub header for `make cuda-check`
src/serial_main.cpp  src/mpi_main.cpp  src/cuda_main.cu
scripts/validate.py           automated correctness tests (see "Validation")
scripts/pbs/*.pbs             PBS job templates (one per experiment)
scripts/submit.sh             builds the qsub commands from config/cluster.env (dry-run supported)
scripts/run_all.sh            runs experiments in the current allocation (also used by the PBS templates)
scripts/experiments.sh        the experiment bodies
scripts/first_login*.sh       first-login checklist (info collection; build + tiny validation + GPU self-test)
scripts/plot.py               figures and summary tables from the CSVs
report.md                     report draft
```

## Requirements

* a C++17 compiler and an MPI library (tested with Open MPI);
* OpenMP for the hybrid build (on macOS: Homebrew `libomp`);
* CUDA toolkit and a CUDA-aware MPI for the GPU build (on Vanda: module `nvhpc-hpcx-cuda12/26.1`);
* Python 3.9+ with `numpy` and `matplotlib` for the plots (`pip install -r requirements.txt`).

## Build and run

### macOS (development only; timings from a laptop are for correctness, not for the report)

```bash
brew install open-mpi libomp          # once
make serial mpi hybrid                # build/serial  build/mpi  build/hybrid
make test                             # full automated validation (about a minute)
```

The Makefile detects macOS and (a) adds Homebrew's `libomp` flags for the hybrid build and (b) puts
the SDK's libc++ headers first (some Command Line Tools installs have a stale header directory;
this workaround is applied on macOS only). There is no CUDA toolkit on a Mac: `make cuda-check`
type-checks the CUDA sources against a stub header (it cannot run anything).

Examples:

```bash
build/serial --dim 3 --N 64 --tol 1e-10                       # run to convergence
mpirun -np 4 build/mpi --dim 2 --N 512 --iters 100 \
       --decomp block --halo nonblocking --overlap 1 --csv out.csv
OMP_NUM_THREADS=2 mpirun -np 2 build/hybrid --dim 3 --N 64 --iters 100
mpirun -np 4 build/mpi --bench-allreduce 5000                 # Allreduce latency (1 double)
```

Options of `mpi`/`hybrid`: `--dim 2|3 --N <points per axis> --decomp slab|block --halo blocking|nonblocking
--overlap 0|1 --tol <t> | --iters <K> --maxit <M> --csv <file> --tag <label> --rep <r>
--compare <serial dump>` (relative difference to a serial solution written by `build/serial --dump`).
`--iters K` runs exactly K CG iterations (timing runs); otherwise CG runs until `‖r‖/‖b‖ <= tol`.

### Atlas (CPU only; not used for the reported experiments)

```bash
ssh <user>@atlas8.nus.edu.sg          # NUS VPN needed off campus
source /app1/ebenv
module avail                          # choose a compiler + MPI, `module load ...`
git clone <repository-url> && cd <repository>
make serial mpi hybrid                # compile on the login node
```

### Vanda (CPU + A40 GPUs; 72-core CPU nodes, 2 GPUs per GPU node)

Login nodes are VMs: **compile inside an interactive job**.

```bash
qsub -I -q <gpu-queue> -l select=1:ncpus=4:ngpus=1 -l walltime=00:20:00   # exact syntax: `hpc gpu`
module load nvhpc-hpcx-cuda12/26.1    # recommended; alternative: OpenMPI/4.1.4-NVHPC-22.7-CUDA-11.7.0
make cuda                             # nvcc -ccbin mpicxx, -arch=sm_86 (A40)
make cuda-alt                         # fallback if `make cuda` fails (system g++ host compiler)
mpirun -np 2 build/cuda --selftest    # GPU result vs a CPU reference, 2D+3D, all halo variants
```

If CUDA-aware MPI is not available, use `--staging 1` or build with `make cuda CUDA_EXTRA=-DHOST_STAGING`.

## First-login checklist (do this first on each cluster)

1. On the login node: `bash scripts/first_login.sh atlas` (or `vanda`). It writes, into
   `results/first_login_<cluster>/`, the output of `gstat`, `qstat -Qf <queue>`, `hpc cpu`/`hpc gpu`,
   `module avail`, the `hpc` help text and any allocation/quota subcommand it finds (your remaining
   core-hours), and basic system information. On Atlas it then builds and runs a tiny validation.
2. Read those files and fill every `CHANGE_ME` in `config/cluster.env` (queues, cores per node,
   maximum cores per job, modules, GPUs per node).
3. On Vanda, inside an interactive GPU job: `bash scripts/first_login_build.sh vanda both`
   (builds serial/mpi/hybrid and runs the quick validation, then builds CUDA — falling back to
   `make cuda-alt` — and runs `--selftest` with 1 and 2 ranks; prints a PASS/FAIL summary).
   `bash scripts/first_login.sh vanda --batch` submits this step as a batch job instead.

**Policy.** Everything runs on **Vanda** (serial `T_s`, MPI strong/weak scaling to 256 ranks, the
decomposition/halo/hybrid comparisons and the GPU runs), so that `T_s` and `T_p(n)` come from the same
hardware. Atlas is not used for the experiments: its per-user limit (96 cores) is below the 256-core
requirement. `scripts/submit.sh vanda ... --no-vanda-cpu` restricts a submission to the GPU experiments.
Your PBS project name goes into `config/cluster.local.env` (git-ignored, see `config/cluster.env`).

`bash scripts/mpi_check.sh vanda` (PBS template `scripts/pbs/mpicheck.pbs`, run it on 2 nodes) checks
`mpirun`, the transport, `MPI_Init_thread` and the binding flags, with a timeout on every launch.

## Running the experiments

All sizes, repetition counts (3, the plots show the median), iteration counts, walltimes and
cluster parameters live in `config/cluster.env`. Timing runs use a fixed number of CG iterations
(`ITERS`, default 100) because CG's iteration count grows with the grid size (see the report);
convergence runs are separate.

```bash
bash scripts/submit.sh vanda all --dry-run     # show the qsub commands and the core-/GPU-hour upper bounds
bash scripts/submit.sh vanda serial            # submit one experiment
bash scripts/submit.sh vanda cpu               # serial strong weak decomp_halo hybrid
bash scripts/submit.sh vanda gpu               # gpu_sweep gpu_multi
```

`submit.sh` creates one PBS job per node count (so small runs do not occupy a 256-core
allocation) and packs many `mpirun` calls into each job. Results land in
`results/<cluster>/<experiment>.csv` (git-ignored). Inside an existing allocation you can also run
`bash scripts/run_all.sh <atlas|vanda> <experiment>`. A tiny local pipeline test:
`SMOKE=1 bash scripts/run_all.sh local all` (output in `results/local_smoke`, for testing the scripts
only).

| Experiment | Cluster | What it runs |
|---|---|---|
| `serial` | Vanda | fixed-iteration runtime vs N (and `T_s`), and runs to tolerance, 2D+3D |
| `strong` | Vanda | fixed N, 1…256 (…512) ranks, block + non-blocking overlap; Allreduce latency |
| `weak` | Vanda | fixed points per rank, N grows as `n^(1/d)` |
| `decomp_halo` | Vanda | slab vs block × blocking / non-blocking / overlap |
| `hybrid` | Vanda | ranks × threads = fixed total cores |
| `gpu_sweep` | Vanda | 1 GPU vs one CPU node (same allocation), size sweep |
| `gpu_multi` | Vanda | 1…G GPUs; overlap on/off; CUDA-aware vs host-staged |

## Reproducing the figures

```bash
python3 scripts/plot.py --results results/vanda --out figures
```

| Figure (in `figures/`) | Experiment | Quantity |
|---|---|---|
| `fig_serial_scaling` | `serial` | time per iteration and time to solution vs unknowns |
| `fig_strong_{2D,3D}` | `strong` (+ `serial` for `T_s`) | runtime, speedup `T_s/T_p(n)`, efficiency `T_s/(n T_p(n))`, Amdahl fit |
| `fig_weak_{2D,3D}` | `weak` (+ `serial`) | time per iteration, weak-scaling efficiency |
| `fig_comm_breakdown_{2D,3D}` | `strong` | time per section (stencil, halo, Allreduce, vector, other) and fractions |
| `fig_allreduce_latency` | `strong` (`allreduce.csv`) | Allreduce time vs ranks |
| `fig_decomposition_{2D,3D}`, `fig_halo_variants_{2D,3D}` | `decomp_halo` | slab vs block, halo surface; halo variants |
| `fig_hybrid_{2D,3D}` | `hybrid` | time at fixed total cores vs threads per rank |
| `fig_gpu_vs_cpu`, `fig_gpu_multi_{2D,3D}` | `gpu_sweep`, `gpu_multi` | GPU vs CPU node; multi-GPU scaling |

Tables with `T_s`, `T_p`, speedup and efficiency are written to `figures/tables/`. Time, problem
size and rank-count axes are logarithmic. Use `--label "LOCAL TEST DATA"` to watermark figures made
from test data.

**Python note.** `plot.py` needs only `numpy` and `matplotlib`. Some Anaconda installations ship a
matplotlib built against NumPy 1.x that fails with NumPy 2.x; use a clean environment
(`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`) or another interpreter.
`validate.py` uses only the standard library.

## Validation

`make test` (or `python3 scripts/validate.py all`; add `--quick` for a ~20-second version) runs five
kinds of checks: (1) discretisation error converges as O(h²) in 2D and 3D, and the CG recurrence
residual matches the explicitly recomputed `‖b − Ax‖`; (2) the parallel solution matches the serial
one to tight tolerance for several rank counts and thread counts; (3) iteration counts agree with
serial within ±2; (4) slab/block and all halo variants agree; (5) awkward sizes (N not divisible by
the process grid, subdomains thinner than three points). On the GPU: `build/cuda --selftest`.

Floating-point note: the order of summation in the global dot products changes with the number of
ranks, so parallel and serial residuals differ at rounding level; all comparisons use tolerances.

## Repository hygiene

Result CSVs, logs, PBS output files and build artefacts are git-ignored; the repository contains no
data files (all data are generated analytically).
