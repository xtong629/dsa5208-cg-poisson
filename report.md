---
title: "Scalable matrix-free Conjugate Gradient for the Poisson equation in 2D and 3D: MPI, MPI+OpenMP and MPI+CUDA"
subtitle: "DSA5208 Project 3 — DRAFT (sections marked TODO depend on cluster results)"
---

## Group members

| Name | Student ID |
|---|---|
| TODO: name 1 | TODO: ID 1 |
| TODO: name 2 | TODO: ID 2 |
| TODO: name 3 | TODO: ID 3 |

Source code: TODO: link to the public GitHub repository (the repository README explains how to build, run and reproduce every figure).

---

## 1 Introduction to the algorithm

### 1.1 The problem

We solve the Poisson equation with homogeneous Dirichlet boundary conditions on the unit square ($d=2$) and the unit cube ($d=3$),

$$-\Delta u = f \ \text{ in } \Omega=(0,1)^d, \qquad u = 0 \ \text{ on } \partial\Omega .$$

The Poisson equation is a standard model problem (electrostatics, steady heat conduction, the pressure
solve of incompressible flow). Both dimensions are implemented, with the same code templated on $d$.

### 1.2 Discretisation

We use a uniform grid with $N$ interior points per axis and spacing $h = 1/(N+1)$, so there are
$W = N^d$ unknowns (the problem size). The standard second-order finite-difference approximation of
$-\Delta$ gives, at interior point $i$,

$$(A u)_i = \frac{1}{h^2}\Big(2d\,u_i - \sum_{j \in \text{neighbours}(i)} u_j\Big),$$

with the 5-point stencil in 2D and the 7-point stencil in 3D; neighbours on the boundary contribute
$u=0$. $A$ is symmetric positive definite. The discrete problem is $A u_h = b$ with $b_i = f(x_i)$.
The discretisation error $u_h - u$ is $O(h^2)$.

**Matrix-free.** $A$ is never stored: a product $A p$ is computed by applying the stencil directly
to the array $p$. This needs $O(W)$ memory and, because the stencil only reads a handful of
neighbouring values, the product is memory-bandwidth bound rather than compute bound — which makes
communication and memory traffic, not arithmetic, the interesting part of the parallelisation.

### 1.3 Conjugate Gradient

CG (Hestenes & Stiefel, 1952) solves $A x = b$ for symmetric positive definite $A$ using only
products with $A$ and vector operations. We start from $x_0 = 0$, so $r_0 = b - A x_0 = b$, and
$p_0 = r_0$. Then, while $\|r_k\|$ is large:

$$\alpha_k = \frac{r_k^\top r_k}{p_k^\top A p_k},\quad
x_{k+1} = x_k + \alpha_k p_k,\quad
r_{k+1} = r_k - \alpha_k A p_k,\quad
\beta_k = \frac{r_{k+1}^\top r_{k+1}}{r_k^\top r_k},\quad
p_{k+1} = r_{k+1} + \beta_k p_k .$$

We stop when $\|r_k\|/\|b\| \le \text{tol}$. One iteration consists of **one product $A p_k$**,
**two dot products** ($p_k^\top A p_k$ and $r_{k+1}^\top r_{k+1}$) and three vector updates (two
"axpy" operations $y \leftarrow y + a x$ and one "xpay" $p \leftarrow r + \beta p$). The two dot
products are the only operations that need information from *all* parts of the vectors — this is
the origin of the global communication discussed in Section 2.

### 1.4 Why the iteration count grows with the grid size

The condition number of $A$ is $\kappa(A) = \lambda_{\max}/\lambda_{\min} \approx 4/(\pi^2 h^2) = O(h^{-2})$
(the smallest eigenvalue stays near $d\pi^2$, the largest grows like $4d/h^2$). CG's convergence
bound depends on $\sqrt{\kappa}$, so the number of iterations needed for a fixed tolerance grows
**linearly in $N$** (as $h^{-1}$). In the serial correctness runs at tolerance $10^{-10}$ we observe about $3.4\,N$
iterations in 2D and $4.3\,N$ in 3D. Consequences for the experiments:

* the time to *solution* grows like $N \cdot N^d$ (iterations × cost per iteration), faster than the problem size;
* when $N$ is increased with the number of ranks (weak scaling) the *number of iterations also grows*,
  so the time to solution is not constant even with perfect parallel efficiency.

For this reason, **all timing and scaling runs use a fixed number of iterations** (100) and we report time per
iteration; convergence to a tolerance is only used for validation and for the serial "time to solution" study.

### 1.5 Synthetic data and the manufactured solution

There are no input files; all data are generated analytically (which makes weak scaling easy: every rank
generates its own part of $f$ from the global grid indices). We use a *manufactured solution*

$$u(x) = \prod_{k=1}^{d} g(x_k),\qquad g(t) = t(1-t)e^{t},\qquad g''(t) = -t(3+t)e^{t},$$

which vanishes on the boundary, so $f = -\Delta u = -\sum_k g''(x_k)\prod_{l\ne k} g(x_l)$ is known in closed
form and the exact error $u_h - u$ is computable at every grid point. We report the discrete $L^2$ norm
$\sqrt{h^d \sum_i e_i^2}$ and the maximum norm.

A product of sines, $u = \prod \sin(\pi x_k)$, is the obvious first choice but is a **poor test problem**: the sampled
sine product is an exact eigenvector of the discrete operator $A$, so the Krylov space CG builds is
one-dimensional and CG converges in **one iteration**, making iteration counts and timings meaningless. (We
observed exactly this in our first implementation.) The function $g$ has broad spectral content, so CG needs
$O(N)$ iterations as it would for generic data.

---

## 2 Parallelization design decisions

*Terminology.* A **rank** is an MPI process. The grid is split into **subdomains**, one per rank. A **halo**
(or ghost layer) is a copy, stored on one rank, of the boundary values owned by a neighbouring rank. A **halo
exchange** updates these copies. $T_s$ is the runtime of the serial program and $T_p(n)$ the runtime on $n$
ranks (Lecture 7).

The algorithm is simple, so — in the words of Lecture 7 — the parallelisation has to be sophisticated. All of the
arithmetic (stencil, axpy, local part of the dot products) is perfectly parallel; the difficulty is entirely in
the two kinds of communication in each iteration:

* **(a) Halo exchange for $A p$.** Point-to-point messages with the neighbouring ranks. The message size is
  proportional to the *surface* of the subdomain while the computation is proportional to its *volume*, so this
  cost is bandwidth-bound and shrinks relative to computation as subdomains grow.
* **(b) Two global reductions** (`MPI_Allreduce` of one double each). A collective over all $n$ ranks; the message is tiny,
  so the cost is *latency*-bound, grows with $\log n$, and does not shrink when the subdomains shrink.

A simple model of one iteration on $n$ ranks, for problem size $W$, is
$t_\text{iter}(n) \approx c\,W/n + t_\text{halo} + 2\,t_\text{ar}(n)$, where $c$ is the per-unknown cost of the
local kernels, $t_\text{halo}$ the halo cost (set by surface, latency and bandwidth) and
$t_\text{ar}(n) \approx a + b\log_2 n$ the Allreduce time. In strong scaling ($W$ fixed) the first term shrinks
as $1/n$ while the other two do not, so efficiency must eventually fall; in weak scaling ($W/n$ fixed) only the
Allreduce term grows. We test these expectations in Section 4.

### 2.1 Data layout and domain decomposition

Each rank stores its box of $n_x \times n_y \,(\times n_z)$ interior points in a contiguous array with $x$ fastest,
surrounded by a one-cell ghost layer on every active side (`src/common/grid.hpp`). Ghost cells on the physical
boundary stay zero (the Dirichlet condition) and ghost cells on inter-rank boundaries receive neighbour data, so
the stencil kernel needs **no boundary special cases**. If $N$ is not divisible by the process grid, the first
$N \bmod p$ ranks along an axis get one extra row.

The process grid is created with `MPI_Dims_create` and `MPI_Cart_create`. We compare two decompositions:

* **Slab (1D):** all splits along the slowest axis. Every rank exchanges two full planes of $N^{d-1}$ points,
  **independent of $n$**; slabs cannot use more than $N$ ranks.
* **Block (2D in 2D, 3D in 3D):** a balanced process grid with $n^{1/d}$ ranks per axis. A rank has $2d$ faces
  of $\approx (N/n^{1/d})^{d-1}$ points: the halo volume per rank shrinks as $2d\,N^{d-1} n^{-(d-1)/d}$.

The surface-to-volume ratio (halo points per owned point) is $2n/N$ for slabs and $2d\,n^{1/d}/N$ for blocks, so blocks
communicate less at large $n$, at the price of more (smaller) messages per iteration ($2d$ instead of $2$).
`MPI_Dims_create` returns counts in non-increasing order; we reverse them so that the *slowest* axis receives the
largest count, which keeps the contiguous $x$ loops long.

### 2.2 Halo exchange: three variants

Because the stencil uses only face neighbours, no edge or corner data are needed: a rank exchanges at most $2d$
faces of one layer. Non-contiguous faces are described by MPI derived datatypes
(`MPI_Type_create_subarray`), so no manual packing is needed on the CPU.

1. **Blocking:** `MPI_Sendrecv`, axis by axis (2 phases per axis); the computation starts only after all exchanges are complete.
2. **Non-blocking:** `MPI_Irecv`/`MPI_Isend` for all faces at once, then `MPI_Waitall`, then the full stencil (no overlap).
3. **Non-blocking with overlap:** post the messages, apply the stencil to the **inner box** (points that do not touch
   a ghost cell) while the messages are in flight, call `MPI_Waitall`, then apply the stencil to the
   **boundary shell** (the remaining points, split into disjoint thin boxes).

*Caveat on overlap.* Posting a non-blocking operation does not guarantee that data move while the application
computes: many MPI implementations only progress large (rendezvous-protocol) messages inside MPI calls unless an
asynchronous progress thread or hardware offload is available. Overlap can therefore yield little or no benefit,
and we measure it (variant 3 against variant 2) instead of assuming it. In our timers the halo time is the time
spent inside MPI (posting plus waiting), i.e. the *exposed* communication; hidden communication appears as part of the stencil time.
(Periodically calling `MPI_Test` during the inner computation to drive progress is a possible refinement that we did not implement;
a one-sided variant with `MPI_Put`/`MPI_Win_fence` is also not implemented.)

### 2.3 Global reductions

Each dot product is a local sum followed by one `MPI_Allreduce` of a single `double`: two per iteration. Their cost is measured separately with a
micro-benchmark (`--bench-allreduce`) and inside the solver. Note that the time measured inside the solver includes **waiting for the slowest rank**
(load imbalance or noise), since a collective cannot finish before every rank has arrived. Because the order of
summation changes with the number of ranks, parallel and serial results differ at rounding level; we therefore
validate with tolerances and not with exact equality (Section 6).

### 2.4 Hybrid MPI+OpenMP

Each rank runs several OpenMP threads, so a fixed number of cores is covered by fewer ranks: smaller halo surfaces in total,
fewer participants in the Allreduce, but fork–join overhead and NUMA effects. Design:

* `MPI_Init_thread` with `MPI_THREAD_FUNNELED` (only the master thread calls MPI); the provided level is checked and recorded in the output.
* Stencil, axpy, xpay and the local dot products are `parallel for` loops over the two outer loops with **static** scheduling; dot products use
  `reduction(+:...)` followed by a single `MPI_Allreduce`.
* **Overlap:** the master thread posts `Isend`/`Irecv` outside any parallel region, all threads compute the inner box, the master calls `Waitall`, all threads compute the boundary shell
  (in a single parallel region to avoid repeated fork–join).
* **First-touch initialisation:** Linux places a memory page on the NUMA node of the thread that first writes it. Vectors are allocated uninitialised and zeroed by the same static
  loop distribution that the kernels use, so each thread's data stays on its own NUMA node.
* Threads are pinned (`OMP_PROC_BIND`, `OMP_PLACES`, and the launcher's binding options).

### 2.5 MPI+CUDA on GPUs

Since $X$ = OpenMP, the grading scheme also requires a GPU test; we implemented MPI+CUDA. One rank drives one GPU; the vectors live in GPU memory
in the same padded layout, and only scalars (dot products) go to the host.

* **Kernels:** one stencil kernel that works on a box of points (used for the inner box and for the thin shell boxes, so there is no boundary
  branch and no warp divergence from boundary handling); axpy/xpay kernels; a dot product as a two-pass tree reduction in a fixed order without atomics (deterministic); pack/unpack kernels that copy a face between the padded array and a contiguous message buffer.
* **Halo exchange:** the faces are packed into device buffers and sent with MPI. With **CUDA-aware MPI** the MPI calls receive device pointers, so data can move GPU-to-GPU without a copy through host memory where the hardware and library allow it;
  the alternative **host staging** copies the packed faces through pinned host buffers. Both paths are implemented and selectable at run time so that they can be compared.
* **Overlap:** packing and unpacking run on a high-priority stream; the inner-box kernel runs on another stream while the messages are in flight; the shell kernels run after the unpack.

### 2.6 Measurement methodology

* **Timers:** each run records the time in the stencil, the halo exchange, `MPI_Allreduce`, the vector updates, and "other" (the remainder). Section times are the mean over ranks and the total is the maximum over ranks.
* **Fixed iterations:** timing runs execute exactly 100 CG iterations; each point is repeated 3 times and the **median** is reported.
* **Definitions (Lecture 7).** Speedup $= T_s/T_p(n)$; parallel efficiency $= T_s/(n\,T_p(n))$, where $T_s$ is measured with the *serial program* on the same machine
  (MPI with one rank is shown only as an additional reference). **Amdahl's law:** speedup $= 1/(S + P/n)$ with serial fraction $S = 1-P$; we fit $S$ to the measured strong-scaling speedup. **Gustafson's law** for weak scaling:
  scaled speedup $= S + P\,n$.
  **Strong scaling:** fixed $N$, increasing $n$. **Weak scaling:** $N$ and $n$ increased together so that the number of points per rank is (approximately) constant; because $N(n)=\mathrm{round}(N_1 n^{1/d})$ is an integer, the efficiency is normalised by the exact points per rank
  (definition in `scripts/plot.py`).
* **Machine:** all experiments (CPU and GPU) on the NUS Vanda cluster, so that $T_s$ and $T_p(n)$ come from the same hardware. TODO: CPU model, cores per node (72 per node; 64 ranks per node are used), memory, interconnect, MPI library and version (NVHPC 26.1 / HPC-X Open MPI), compiler and flags, GPU model (A40) and CUDA version.
* Timings from the development laptop are used only to check correctness and never appear in this report.

---

## 3 Serial scaling

**TODO (insert after the `serial` experiment):**

* **Figure `fig_serial_scaling`** (left: time per CG iteration vs number of unknowns; right: time to reach tolerance vs number of unknowns; both axes logarithmic; 2D and 3D).
* **Table** of the serial runtimes $T_s$ used as baselines for the speedup/efficiency calculations (the $N$ used in Sections 4 and 5).
* Discussion to write: time per iteration should grow linearly with the number of unknowns once the arrays exceed the caches (a departure at small sizes indicates cache effects); time to solution grows like unknowns$^{1+1/d}$ because the iteration count grows
  like $N$ (Section 1.4); 3D has a more expensive iteration per unknown (7-point stencil, more memory traffic) but fewer unknowns for the same $N$.

---

## 4 MPI scaling and efficiency

**TODO: experimental setup paragraph** — rank counts (1…256, and 512 if available), problem sizes (strong: 2D $N=$ TODO, 3D $N=$ TODO; weak: TODO points per rank), process placement, how many ranks per node.

### 4.1 Strong scaling

**TODO (`strong`):** **Figure `fig_strong_2D`**, **Figure `fig_strong_3D`** (runtime, speedup $T_s/T_p(n)$ and efficiency $T_s/(n T_p(n))$ vs $n$ on logarithmic axes) and **Table `figures/tables/strong_scaling.csv`**.
Discuss: where efficiency drops and why (subdomain size at that $n$, share of communication from the breakdown in Section 4.3), the fitted Amdahl serial fraction $S$ and what it does and does not capture (it lumps communication and memory-bandwidth saturation into a single number),
the effect of the node boundary (intra-node vs inter-node messages), and the difference between the small and the large problem.

### 4.2 Weak scaling

**TODO (`weak`):** **Figure `fig_weak_2D`**, **Figure `fig_weak_3D`** (time per iteration and weak-scaling efficiency vs $n$) and **Table `figures/tables/weak_scaling.csv`** (including the measured scaled speedup to compare with Gustafson's law).
Discuss: which term grows with $n$ (expected: the Allreduce, $\propto \log n$, plus node-boundary effects), reminder that the number of iterations also grows with $N$ in a real run (Section 1.4), so we compare time per iteration.

### 4.3 Impact of communication

**TODO (`strong`, `decomp_halo`):**

* **Figure `fig_comm_breakdown_2D`**, **Figure `fig_comm_breakdown_3D`**: time per iteration by section (stencil, halo exchange, Allreduce, vector updates, other) and as a fraction of the total.
* **Figure `fig_allreduce_latency`**: time of one `MPI_Allreduce` of one double vs $n$ (micro-benchmark), and the time per Allreduce inside the solver.
* **Figure `fig_decomposition_2D`**, **`fig_decomposition_3D`**: slab vs block — doubles sent per rank per halo exchange (the `halo_doubles` column), halo time and total time vs $n$; compare with the surface formulas of Section 2.1.
* **Figure `fig_halo_variants_2D`**, **`fig_halo_variants_3D`**: blocking vs non-blocking vs non-blocking with overlap, for both decompositions; state whether overlap helped and relate to the progress caveat of Section 2.2.

Discuss which of the two communication types dominates at which $n$, and why (halo: bandwidth/surface; Allreduce: latency, independent of subdomain size).

---

## 5 MPI+X: OpenMP and GPU

### 5.1 MPI+OpenMP

**TODO (`hybrid`):** **Figure `fig_hybrid_2D`**, **Figure `fig_hybrid_3D`** — time per iteration and communication fraction at fixed total core counts (TODO: 64 and 256) for ranks × threads configurations, against the pure-MPI binary on the same cores.
Discuss: trade-off between smaller halo surface/fewer Allreduce participants and thread overheads and NUMA effects; best configuration and why; effect of pinning and first touch (if a configuration with and without them was run, otherwise state that this was not measured).

### 5.2 MPI+CUDA (GPU)

**TODO (Vanda `gpu_sweep`, `gpu_multi`):**

* **Figure `fig_gpu_vs_cpu`**: one A40 GPU against one CPU node (same allocation) for a sweep of problem sizes, time per iteration and GPU speedup, 2D and 3D.
* **Figure `fig_gpu_multi_2D`**, **`fig_gpu_multi_3D`**: strong scaling on 1…G GPUs; variants: overlap on/off, blocking, CUDA-aware vs host-staged halos.
* Result of the GPU self-test (`build/cuda --selftest`) — TODO: pass/fail and the date/cluster it was run on.

Discuss: below which problem size the GPU is slower than the CPU node (kernel-launch and host-synchronisation overhead per iteration — each dot product needs a device-to-host copy), memory-bandwidth bound behaviour at large sizes,
cost of the Allreduce relative to the (very fast) GPU compute, benefit of CUDA-aware MPI over staging, benefit of overlap.

---

## 6 Validation

We validate with a small number of well-chosen test categories (implemented in `scripts/validate.py`; the GPU version is `build/cuda --selftest`). Each category sweeps a small parameter grid. The numbers
quoted below come from the development runs and **are to be regenerated on the cluster before submission** (TODO).

1. **Convergence of the discretisation error.** The error against the exact manufactured solution decreases as $O(h^2)$: the observed order $\log(e_{N_1}/e_{N_2})/\log(h_1/h_2)$ between consecutive grids is $2.00$ in the $L^2$ norm
   (e.g. 2D $N=127\to255$, 3D $N=31\to63$) in both dimensions, for serial and parallel runs. In the same runs, the residual from the CG recurrence agrees with the explicitly recomputed $\|b - Ax\|/\|b\|$, which checks that the recurrences have not drifted.
2. **Parallel equals serial.** The relative difference $\|x_\text{par} - x_\text{ser}\|/\|x_\text{ser}\|$ between the distributed solution and the serial one is at rounding level — at most $2.5\times10^{-15}$ over all runs — for 1, 2, 4 and 8 ranks and for several thread counts in the hybrid
   code (tolerance used in the test: $10^{-9}$). Iteration counts agree with the serial count (identical in every run we made; the test allows $\pm 2$ because rounding differs with the order of summation).
3. **Decomposition and halo-variant equivalence.** Slab and block decompositions, and blocking, non-blocking and overlapped halo exchange, give the same iteration count and the same error norm to $10^{-10}$ relative (e.g. 159 iterations for every variant of the 3D $N=31$ problem).
4. **Awkward sizes.** $N$ not divisible by the process grid (2D $N=50$ on 6 ranks, 3D $N=29$ on 8 ranks) and subdomains thinner than three points so that the inner box is empty and the boundary shell covers everything (3D $N=4$ and $N=5$ on 8 ranks): differences from serial below $10^{-15}$ (largest: $7\times10^{-16}$).
5. **Negative control.** To confirm that the comparison can fail, we compared a tightly converged parallel solution with a deliberately loosely converged serial solution (tolerance $10^{-4}$): the test reports a difference of $3\times10^{-6}$, as it should.

TODO: GPU self-test result; one sentence stating that the same tests were run on the cluster with the cluster's MPI library.

---

## 7 Discussion

**TODO: write after the results are in.** Points to cover (all supported by the figures above):

* *How the problem was split.* Cartesian block decomposition of the grid, ghost layers, the matrix never stored; what each rank owns and what it exchanges (numbers from the `halo_doubles` and `local_points` columns).
* *Which parts need the most communication.* Compare the two communication types: halo exchange (point-to-point, $\propto$ surface, $2d$ neighbours) and Allreduce (collective, $\propto \log n$, 2 per iteration) — which dominates at 256 ranks for the small and for the large problem, and in which regime the answer changes.
* *How different parts used different techniques.* Blocking vs non-blocking vs overlap for the halo; OpenMP for node-level parallelism; CUDA kernels and CUDA-aware MPI on the GPU; deterministic tree reduction for the GPU dot product.
* *Slab vs block*, *hybrid vs pure MPI*, *CPU vs GPU*: the main findings in one paragraph each, with the quantitative evidence.
* *Limitations and threats to validity.* Shared nodes and network (noise; hence 3 repetitions and medians), fixed iteration counts instead of convergence runs, memory-bandwidth saturation within a node (which limits intra-node strong scaling regardless of communication), overlap depending on the MPI library's progress behaviour,
  rounded $N$ in weak scaling.
* *Not done.* One-sided (RMA) halo exchange, a communication-reducing CG variant that fuses the two Allreduces into one per iteration, Kokkos, and any preconditioner (a multigrid preconditioner would reduce the iteration count from $O(N)$ to $O(1)$ but is a different algorithm).

---

## 8 AI usage

**TODO — to be written by the group, honestly and specifically.** The course requires AI use to be declared and the report to make sense to a reader. Suggested content: which AI tools were used; for which parts (code, scripts, report text, debugging); what the group did to check
the output (the automated validation, review of the code, running the experiments themselves); and which parts were written without AI assistance.

---

## References

1. M. R. Hestenes and E. Stiefel, "Methods of conjugate gradients for solving linear systems", *Journal of Research of the National Bureau of Standards* 49(6), 409–436, 1952.
2. Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
3. G. M. Amdahl, "Validity of the single processor approach to achieving large scale computing capabilities", *AFIPS Spring Joint Computer Conference*, 483–485, 1967.
4. J. L. Gustafson, "Reevaluating Amdahl's law", *Communications of the ACM* 31(5), 532–533, 1988.
5. Message Passing Interface Forum, *MPI: A Message-Passing Interface Standard, Version 4.0*, 2021.
6. OpenMP Architecture Review Board, *OpenMP Application Programming Interface*.
7. NVIDIA, *CUDA C++ Programming Guide*.
8. DSA5208 lecture notes, Lectures 7, 8 and 12 (A. Chen), National University of Singapore.
