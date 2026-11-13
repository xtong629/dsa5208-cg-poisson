// cuda_main.cu -- MPI + CUDA matrix-free CG for -Laplace(u)=f in 2D (5-point) and 3D (7-point).
//
// One MPI rank per GPU (device = node-local rank modulo the number of visible GPUs). The same
// Cartesian decomposition, padded layout and CG loop (cg.hpp) as the CPU codes; the vectors live
// in GPU memory and only scalars (dot products) travel to the host.
//
// Halo exchange (per matvec):
//   1. pack the faces of p that neighbours need into contiguous device buffers (pack kernels,
//      high-priority stream "halo");
//   2. send/receive those buffers with MPI. Default: CUDA-aware MPI, i.e. the MPI calls receive
//      DEVICE pointers. --staging 1 (or -DHOST_STAGING): copy through pinned host buffers instead;
//   3. unpack the received faces into the ghost layers of p.
//   --halo blocking     : MPI_Sendrecv axis by axis, then compute everything.
//   --halo nonblocking  : MPI_Irecv/MPI_Isend for all faces. With --overlap 1 the INNER-box stencil
//                         kernel (stream "main") runs on the GPU while the messages are in flight;
//                         after MPI_Waitall + unpack the thin boundary-shell kernels run.
//
// usage: mpirun -np P cuda --dim 2|3 --N n [--decomp block|slab] [--halo nonblocking|blocking]
//        [--overlap 1] [--staging 0|1] [--tol 1e-10] [--maxit M] [--iters K] [--csv f] [--tag t]
//        [--rep r] [--gpu-timers 1]
//        mpirun -np P cuda --selftest        # correctness check against a CPU reference
#include "common/cg.hpp"
#include "common/ops_local.hpp"
#include "common/problem.hpp"
#include "common/split.hpp"
#include "common/timers.hpp"
#include "cuda/cuda_compat.hpp"
#include "cuda/gpu_kernels.cuh"
#include "mpi/decomp.hpp"
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#if defined(OPEN_MPI) && !defined(NO_MPI_EXT)
#include <mpi-ext.h>   // MPIX_Query_cuda_support (Open MPI / HPC-X)
#endif

#ifdef HOST_STAGING
constexpr bool kDefaultStaging = true;    // build default: go through pinned host buffers
#else
constexpr bool kDefaultStaging = false;   // build default: CUDA-aware MPI on device pointers
#endif

struct Config {
    int N = 64;
    bool slab = false, blocking = false, overlap = true, staging = kDefaultStaging;
    bool gpu_timers = true;   // synchronise after each kernel so the section times are attributable
    double tol = 1e-10;
    int maxit = 100000, iters = 0;
};

// ---------------------------------------------------------------------------------------------
template <int DIM>
class GpuSolver {   // also the "Ops" object of cg_solve
public:
    Decomp<DIM> D;
    Timers T;

    GpuSolver(const Config& c, MPI_Comm comm) : D(comm, c.N, c.slab), c_(c), split_(D.B) {
        const Block<DIM>& B = D.B;
        for (int e = 0; e < 3; ++e) { gb_.n[e] = B.n[e]; gb_.g[e] = Block<DIM>::ghost(e); gb_.str[e] = B.str[e]; }
        nint_ = B.interior_count();
        const double h = 1.0 / (c.N + 1);
        h2inv_ = 1.0 / (h * h);
        bytes_ = sizeof(double) * (size_t)B.size;

        int lo_pri, hi_pri;   // numerically smaller = higher priority
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo_pri, &hi_pri));
        CUDA_CHECK(cudaStreamCreateWithPriority(&main_, cudaStreamNonBlocking, lo_pri));
        CUDA_CHECK(cudaStreamCreateWithPriority(&halo_, cudaStreamNonBlocking, hi_pri));
        CUDA_CHECK(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming));

        for (double** v : {&x, &b, &r, &p, &Ap, &Ax}) {
            CUDA_CHECK(cudaMalloc((void**)v, bytes_));
            CUDA_CHECK(cudaMemset(*v, 0, bytes_));      // ghost layers = 0 (Dirichlet boundary)
        }
        dot_blocks_ = (int)std::min<long>(1024, (nint_ + kBlock - 1) / kBlock);
        if (dot_blocks_ < 1) dot_blocks_ = 1;
        CUDA_CHECK(cudaMalloc((void**)&d_partial_, sizeof(double) * dot_blocks_));
        CUDA_CHECK(cudaMalloc((void**)&d_dot_, sizeof(double)));

        // faces: [axis][side], side 0 = lower neighbour, 1 = upper neighbour
        for (int a = 0; a < DIM; ++a)
            for (int s = 0; s < 2; ++s) {
                Face& f = face_[a][s];
                f.axis = a;
                f.side = s;
                f.nbr = D.nbr[a][s];
                // the two other axes, in increasing order: a=0 -> (1,2), a=1 -> (0,2), a=2 -> (0,1)
                f.e1 = (a == 0) ? 1 : 0;
                f.e2 = (a == 2) ? 1 : 2;
                f.count = (long)B.n[f.e1] * B.n[f.e2];
                f.pos_send = (s == 0) ? 1 : B.n[a];        // first / last interior layer
                f.pos_recv = (s == 0) ? 0 : B.n[a] + 1;    // lower / upper ghost layer
                const size_t fb = sizeof(double) * (size_t)f.count;
                CUDA_CHECK(cudaMalloc((void**)&f.d_send, fb));
                CUDA_CHECK(cudaMalloc((void**)&f.d_recv, fb));
                if (c.staging) {
                    CUDA_CHECK(cudaMallocHost((void**)&f.h_send, fb));
                    CUDA_CHECK(cudaMallocHost((void**)&f.h_recv, fb));
                }
            }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    ~GpuSolver() {
        for (int a = 0; a < DIM; ++a)
            for (int s = 0; s < 2; ++s) {
                Face& f = face_[a][s];
                cudaFree(f.d_send); cudaFree(f.d_recv);
                if (c_.staging) { cudaFreeHost(f.h_send); cudaFreeHost(f.h_recv); }
            }
        for (double* v : {x, b, r, p, Ap, Ax}) cudaFree(v);
        cudaFree(d_partial_); cudaFree(d_dot_);
        cudaEventDestroy(ready_); cudaStreamDestroy(main_); cudaStreamDestroy(halo_);
    }
    GpuSolver(const GpuSolver&) = delete;
    GpuSolver& operator=(const GpuSolver&) = delete;

    double *x = nullptr, *b = nullptr, *r = nullptr, *p = nullptr, *Ap = nullptr, *Ax = nullptr;

    void upload(double* dev, const double* host) {
        CUDA_CHECK(cudaMemcpy(dev, host, bytes_, cudaMemcpyHostToDevice));
    }
    void download(std::vector<double>& host, const double* dev) {
        host.resize((size_t)D.B.size);
        CUDA_CHECK(cudaMemcpy(host.data(), dev, bytes_, cudaMemcpyDeviceToHost));
    }
    void finish() { CUDA_CHECK(cudaDeviceSynchronize()); }

    // ---------------- Ops interface used by cg_solve ----------------
    void zero(double* v) {
        ScopedTimer t(T, SEC_VECTOR);
        CUDA_CHECK(cudaMemsetAsync(v, 0, bytes_, main_));
        sync_main();
    }
    void copy(const double* src, double* dst) {
        ScopedTimer t(T, SEC_VECTOR);
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes_, cudaMemcpyDeviceToDevice, main_));
        sync_main();
    }
    void axpy(double alpha, const double* xv, double* yv) {
        ScopedTimer t(T, SEC_VECTOR);
        LAUNCH(axpy_kernel, grid_for(nint_), kBlock, main_, gb_, nint_, alpha, xv, yv);
        sync_main();
    }
    void xpay(double beta, const double* rv, double* pv) {
        ScopedTimer t(T, SEC_VECTOR);
        LAUNCH(xpay_kernel, grid_for(nint_), kBlock, main_, gb_, nint_, beta, rv, pv);
        sync_main();
    }
    double dot(const double* a, const double* bv) {
        double loc = 0.0, glob = 0.0;
        {
            ScopedTimer t(T, SEC_VECTOR);   // local reduction on the GPU + copy of one double to the host
            LAUNCH(dot_pass1, dot_blocks_, kBlock, main_, gb_, nint_, a, bv, d_partial_);
            LAUNCH(dot_pass2, 1, kBlock, main_, d_partial_, dot_blocks_, d_dot_);
            CUDA_CHECK(cudaMemcpyAsync(&loc, d_dot_, sizeof(double), cudaMemcpyDeviceToHost, main_));
            CUDA_CHECK(cudaStreamSynchronize(main_));   // the value is needed on the host now
        }
        {
            ScopedTimer t(T, SEC_ALLREDUCE);
            MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, D.cart);
        }
        return glob;
    }
    void matvec(double* pv, double* Apv) {
        if (c_.blocking) {
            { ScopedTimer t(T, SEC_HALO); exchange_blocking(pv); }
            { ScopedTimer t(T, SEC_STENCIL); launch_stencil(full_box(), pv, Apv); sync_main(); }
        } else if (!c_.overlap) {
            { ScopedTimer t(T, SEC_HALO); halo_post(pv); halo_finish(pv); }
            { ScopedTimer t(T, SEC_STENCIL); launch_stencil(full_box(), pv, Apv); sync_main(); }
        } else {
            { ScopedTimer t(T, SEC_HALO); halo_post(pv); }    // pack, (stage), sync, Irecv/Isend
            {   // launch the inner-box kernel; it runs on the GPU while the host waits for MPI below
                ScopedTimer t(T, SEC_STENCIL);
                if (box_size(split_.inner) > 0) launch_stencil(split_.inner, pv, Apv);
            }
            { ScopedTimer t(T, SEC_HALO); halo_finish(pv); }  // Waitall, (stage), unpack, sync
            {   // shell kernels need the ghost layers just unpacked; the time still spent waiting for
                // the inner kernel to drain is attributed to the stencil section
                ScopedTimer t(T, SEC_STENCIL);
                for (const Box& bx : split_.shell) launch_stencil(bx, pv, Apv);
                sync_main();
            }
        }
    }

private:
    struct Face {
        int axis = 0, side = 0, nbr = MPI_PROC_NULL, e1 = 0, e2 = 0, pos_send = 0, pos_recv = 0;
        long count = 0;
        double *d_send = nullptr, *d_recv = nullptr, *h_send = nullptr, *h_recv = nullptr;
    };
    Config c_;
    SplitBoxes<DIM> split_;
    GB gb_;
    long nint_ = 0;
    double h2inv_ = 1;
    size_t bytes_ = 0;
    cudaStream_t main_ = 0, halo_ = 0;
    cudaEvent_t ready_ = 0;
    double *d_partial_ = nullptr, *d_dot_ = nullptr;
    int dot_blocks_ = 1;
    Face face_[3][2];
    MPI_Request req_[12];
    int nreq_ = 0;

    static long box_size(const Box& bx) { return (long)(bx.hi[0] - bx.lo[0]) * (bx.hi[1] - bx.lo[1]) * (bx.hi[2] - bx.lo[2]); }
    static int grid_for(long total) { return (int)((total + kBlock - 1) / kBlock); }
    void sync_main() { if (c_.gpu_timers) CUDA_CHECK(cudaStreamSynchronize(main_)); }
    Box full_box() const {
        Box bx;
        for (int e = 0; e < 3; ++e) { bx.lo[e] = 0; bx.hi[e] = D.B.n[e]; }
        return bx;
    }
    void launch_stencil(const Box& bx, const double* pv, double* Apv) {
        const long total = box_size(bx);
        if (total <= 0) return;
        LAUNCH(stencil_kernel<DIM>, grid_for(total), kBlock, main_, gb_, bx, pv, Apv, h2inv_);
    }
    double* send_buf(const Face& f) const { return c_.staging ? f.h_send : f.d_send; }
    double* recv_buf(const Face& f) const { return c_.staging ? f.h_recv : f.d_recv; }

    // pack faces of p (stream halo, after p is complete on stream main), optionally copy to host
    void pack_all(const double* pv) {
        CUDA_CHECK(cudaEventRecord(ready_, main_));
        CUDA_CHECK(cudaStreamWaitEvent(halo_, ready_, 0));
        for (int a = 0; a < DIM; ++a)
            for (int s = 0; s < 2; ++s) {
                const Face& f = face_[a][s];
                if (f.nbr == MPI_PROC_NULL) continue;
                LAUNCH(pack_kernel, grid_for(f.count), kBlock, halo_, gb_, f.axis, f.e1, f.e2, f.pos_send, f.count, pv, f.d_send);
                if (c_.staging)
                    CUDA_CHECK(cudaMemcpyAsync(f.h_send, f.d_send, sizeof(double) * (size_t)f.count, cudaMemcpyDeviceToHost, halo_));
            }
        CUDA_CHECK(cudaStreamSynchronize(halo_));   // buffers must be complete before MPI reads them
    }
    void unpack_all(double* pv) {
        for (int a = 0; a < DIM; ++a)
            for (int s = 0; s < 2; ++s) {
                const Face& f = face_[a][s];
                if (f.nbr == MPI_PROC_NULL) continue;
                if (c_.staging)
                    CUDA_CHECK(cudaMemcpyAsync(f.d_recv, f.h_recv, sizeof(double) * (size_t)f.count, cudaMemcpyHostToDevice, halo_));
                LAUNCH(unpack_kernel, grid_for(f.count), kBlock, halo_, gb_, f.axis, f.e1, f.e2, f.pos_recv, f.count, f.d_recv, pv);
            }
        CUDA_CHECK(cudaStreamSynchronize(halo_));
    }
    // tags as in the CPU code: data moving "up" (to higher coordinate) uses 2a, "down" 2a+1
    void exchange_blocking(double* pv) {
        pack_all(pv);
        for (int a = 0; a < DIM; ++a) {
            const Face& lo = face_[a][0];
            const Face& hi = face_[a][1];
            MPI_Sendrecv(send_buf(hi), (int)hi.count, MPI_DOUBLE, hi.nbr, 2 * a,
                         recv_buf(lo), (int)lo.count, MPI_DOUBLE, lo.nbr, 2 * a, D.cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(send_buf(lo), (int)lo.count, MPI_DOUBLE, lo.nbr, 2 * a + 1,
                         recv_buf(hi), (int)hi.count, MPI_DOUBLE, hi.nbr, 2 * a + 1, D.cart, MPI_STATUS_IGNORE);
        }
        unpack_all(pv);
    }
    void halo_post(double* pv) {
        pack_all(pv);
        nreq_ = 0;
        for (int a = 0; a < DIM; ++a)
            for (int s = 0; s < 2; ++s) {
                const Face& f = face_[a][s];
                if (f.nbr == MPI_PROC_NULL) continue;
                MPI_Irecv(recv_buf(f), (int)f.count, MPI_DOUBLE, f.nbr, 2 * a + (s == 0 ? 0 : 1), D.cart, &req_[nreq_++]);
            }
        for (int a = 0; a < DIM; ++a)
            for (int s = 0; s < 2; ++s) {
                const Face& f = face_[a][s];
                if (f.nbr == MPI_PROC_NULL) continue;
                MPI_Isend(send_buf(f), (int)f.count, MPI_DOUBLE, f.nbr, 2 * a + (s == 1 ? 0 : 1), D.cart, &req_[nreq_++]);
            }
    }
    void halo_finish(double* pv) {
        MPI_Waitall(nreq_, req_, MPI_STATUSES_IGNORE);
        nreq_ = 0;
        unpack_all(pv);
    }
};

// ---------------------------------------------------------------------------------------------
struct Report {   // what a solve returns besides the record
    std::vector<double> xh;   // local solution (padded layout), for the self-test
    RunRecord rec;
};

// Solve once on the GPUs; fills rec (complete on rank 0) and the local host copy of the solution.
template <int DIM>
static void solve_gpu(const Config& c, Report& rep, Block<DIM>& Bout) {
    GpuSolver<DIM> S(c, MPI_COMM_WORLD);
    const Block<DIM>& B = S.D.B;
    Bout = B;
    const int N = c.N;

    std::vector<double> bh((size_t)B.size, 0.0);
    fill_rhs<DIM>(B, N, bh.data());
    S.upload(S.b, bh.data());
    S.finish();

    MPI_Barrier(S.D.cart);
    const double t0 = MPI_Wtime();
    CGResult res = cg_solve(S, S.x, S.b, S.r, S.p, S.Ap, c.tol, c.iters > 0 ? c.iters : c.maxit, c.iters > 0);
    S.finish();
    const double t_mine = MPI_Wtime() - t0;
    Timers solve_T = S.T;

    // ---- verification (untimed) ----
    S.matvec(S.x, S.Ax);   // explicit residual b - A x
    S.finish();
    std::vector<double> Axh;
    S.download(Axh, S.Ax);
    S.download(rep.xh, S.x);
    double loc[3] = {0, 0, 0}, sq = 0, mx = 0;
    for (int k = 0; k < B.n[2]; ++k)
        for (int j = 0; j < B.n[1]; ++j) {
            const long base = B.idx(0, j, k);
            for (int i = 0; i < B.n[0]; ++i) {
                const double d = bh[base + i] - Axh[base + i];
                loc[0] += d * d;
                loc[1] += bh[base + i] * bh[base + i];
            }
        }
    error_local<DIM>(B, N, rep.xh.data(), sq, mx);
    loc[2] = sq;
    double glob[3], gmax;
    MPI_Allreduce(loc, glob, 3, MPI_DOUBLE, MPI_SUM, S.D.cart);
    MPI_Allreduce(&mx, &gmax, 1, MPI_DOUBLE, MPI_MAX, S.D.cart);
    const double h = 1.0 / (N + 1);
    double hd = 1.0;
    for (int d = 0; d < DIM; ++d) hd *= h;

    double tsec[SEC_COUNT], tmean[SEC_COUNT], t_total;
    for (int s = 0; s < SEC_COUNT; ++s) tsec[s] = solve_T.t[s];
    MPI_Reduce(tsec, tmean, SEC_COUNT, MPI_DOUBLE, MPI_SUM, 0, S.D.cart);
    MPI_Reduce(&t_mine, &t_total, 1, MPI_DOUBLE, MPI_MAX, 0, S.D.cart);
    double hl = (double)S.D.halo_doubles(), lp = (double)B.interior_count(), hl_max, lp_max;
    MPI_Reduce(&hl, &hl_max, 1, MPI_DOUBLE, MPI_MAX, 0, S.D.cart);
    MPI_Reduce(&lp, &lp_max, 1, MPI_DOUBLE, MPI_MAX, 0, S.D.cart);

    RunRecord& rec = rep.rec;
    rec.program = "cuda"; rec.dim = DIM; rec.N = N; rec.ranks = S.D.size; rec.threads = 1;
    rec.thread_level = "single";
    rec.decomp = c.slab ? "slab" : "block";
    rec.halo = c.blocking ? "blocking" : "nonblocking";
    rec.overlap = (!c.blocking && c.overlap) ? 1 : 0;
    rec.comm_path = c.staging ? "staged" : "device";
    rec.mode = c.iters > 0 ? "iters" : "tol";
    rec.iters = res.iters; rec.converged = res.converged;
    rec.rel_resid = res.rel_resid; rec.true_rel_resid = std::sqrt(glob[0] / glob[1]);
    rec.err_l2 = std::sqrt(hd * glob[2]); rec.err_max = gmax;
    rec.halo_doubles = hl_max; rec.local_points = lp_max;
    rec.t_total = t_total;
    rec.t_stencil = tmean[SEC_STENCIL] / S.D.size; rec.t_halo = tmean[SEC_HALO] / S.D.size;
    rec.t_allreduce = tmean[SEC_ALLREDUCE] / S.D.size; rec.t_vector = tmean[SEC_VECTOR] / S.D.size;
}

// CPU reference of the whole global problem (serial CG), used by the self-test (tiny N only).
template <int DIM>
static void cpu_reference(int N, double tol, std::vector<double>& xs, Block<DIM>& B0, int& iters) {
    int n[3] = {N, N, DIM == 3 ? N : 1}, off[3] = {0, 0, 0};
    B0.init(n, off);
    SerialOps<DIM> ops(B0, N);
    std::vector<double> b((size_t)B0.size, 0.0), r(b), p(b), Ap(b);
    xs.assign((size_t)B0.size, 0.0);
    fill_rhs<DIM>(B0, N, b.data());
    CGResult res = cg_solve(ops, xs.data(), b.data(), r.data(), p.data(), Ap.data(), tol, 100000, false);
    iters = res.iters;
}

static bool cuda_aware_mpi_known_no() {
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    return MPIX_Query_cuda_support() == 0;
#else
    return false;
#endif
}

template <int DIM>
static int selftest_dim(int& nchecks, int& nfail) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const std::vector<int> sizes = (DIM == 2) ? std::vector<int>{63, 10} : std::vector<int>{31, 5};
    const bool device_ok = !cuda_aware_mpi_known_no();
    for (int N : sizes) {
        if (N < size) continue;   // too few points per rank for this process grid
        std::vector<double> xs; Block<DIM> B0; int it_ref = 0;
        cpu_reference<DIM>(N, 1e-13, xs, B0, it_ref);
        for (int slab = 0; slab < 2; ++slab)
            for (int staging = 0; staging < 2; ++staging) {
                if (!staging && !device_ok) continue;
                for (int variant = 0; variant < 3; ++variant) {   // blocking | nonblocking | +overlap
                    Config c;
                    c.N = N; c.slab = slab; c.staging = staging; c.tol = 1e-13;
                    c.blocking = (variant == 0); c.overlap = (variant == 2);
                    Report rep; Block<DIM> B;
                    solve_gpu<DIM>(c, rep, B);
                    double loc[2] = {0, 0};
                    for (int k = 0; k < B.n[2]; ++k)
                        for (int j = 0; j < B.n[1]; ++j)
                            for (int i = 0; i < B.n[0]; ++i) {
                                const double xg = xs[(size_t)B0.idx(i + B.off[0], j + B.off[1], k + B.off[2])];
                                const double d = rep.xh[(size_t)B.idx(i, j, k)] - xg;
                                loc[0] += d * d; loc[1] += xg * xg;
                            }
                    double glob[2];
                    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    const double diff = std::sqrt(glob[0] / glob[1]);
                    // iterations: recorded on every rank in rec.iters (same value everywhere)
                    const bool ok = diff < 1e-9 && std::abs(rep.rec.iters - it_ref) <= 2 && rep.rec.converged;
                    ++nchecks;
                    if (!ok) ++nfail;
                    if (rank == 0)
                        std::printf("  %s %dD N=%-3d ranks=%d %-5s %-8s %-12s diff=%.2e iters=%d (cpu %d)\n",
                                    ok ? "ok  " : "FAIL", DIM, N, size, slab ? "slab" : "block",
                                    staging ? "staged" : "device",
                                    variant == 0 ? "blocking" : variant == 1 ? "nonblocking" : "overlap",
                                    diff, rep.rec.iters, it_ref);
                }
            }
    }
    return 0;
}

static int selftest() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nchecks = 0, nfail = 0;
    if (rank == 0) std::printf("GPU self-test: CUDA solution vs CPU serial CG on the same problem\n");
    selftest_dim<2>(nchecks, nfail);
    selftest_dim<3>(nchecks, nfail);
    if (rank == 0) {
        if (cuda_aware_mpi_known_no()) std::printf("  note: MPI reports NO CUDA-aware support; device-pointer runs were skipped\n");
        if (nfail == 0) std::printf("SELFTEST PASSED (%d checks)\n", nchecks);
        else std::printf("SELFTEST FAILED (%d of %d checks failed)\n", nfail, nchecks);
    }
    return nfail == 0 ? 0 : 1;
}

struct Options {
    int dim = 2, rep = 0;
    bool selftest = false;
    Config c;
    std::string csv, tag;
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // one rank per GPU: device = node-local rank modulo number of visible devices
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int lrank, lsize, ndev = 0;
    MPI_Comm_rank(local, &lrank);
    MPI_Comm_size(local, &lsize);
    MPI_Comm_free(&local);
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) { if (rank == 0) std::fprintf(stderr, "no CUDA device visible\n"); MPI_Abort(MPI_COMM_WORLD, 8); }
    CUDA_CHECK(cudaSetDevice(lrank % ndev));
    if (lsize > ndev && rank == 0)
        std::fprintf(stderr, "warning: %d ranks per node share %d GPU(s)\n", lsize, ndev);

    Options o;
    for (int a = 1; a < argc; ++a) {
        auto is = [&](const char* s) { return std::strcmp(argv[a], s) == 0 && a + 1 < argc; };
        if (std::strcmp(argv[a], "--selftest") == 0) o.selftest = true;
        else if (is("--dim")) o.dim = std::atoi(argv[++a]);
        else if (is("--N")) o.c.N = std::atoi(argv[++a]);
        else if (is("--decomp")) o.c.slab = (std::strcmp(argv[++a], "slab") == 0);
        else if (is("--halo")) o.c.blocking = (std::strcmp(argv[++a], "blocking") == 0);
        else if (is("--overlap")) o.c.overlap = std::atoi(argv[++a]) != 0;
        else if (is("--staging")) o.c.staging = std::atoi(argv[++a]) != 0;
        else if (is("--gpu-timers")) o.c.gpu_timers = std::atoi(argv[++a]) != 0;
        else if (is("--tol")) o.c.tol = std::atof(argv[++a]);
        else if (is("--maxit")) o.c.maxit = std::atoi(argv[++a]);
        else if (is("--iters")) o.c.iters = std::atoi(argv[++a]);
        else if (is("--csv")) o.csv = argv[++a];
        else if (is("--tag")) o.tag = argv[++a];
        else if (is("--rep")) o.rep = std::atoi(argv[++a]);
        else { if (rank == 0) std::fprintf(stderr, "unknown/incomplete option %s\n", argv[a]); MPI_Abort(MPI_COMM_WORLD, 2); }
    }

    if (rank == 0) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, lrank % ndev));
        const char* aware = "unknown";
#if defined(MPIX_CUDA_AWARE_SUPPORT)
        aware = MPIX_Query_cuda_support() ? "yes" : "NO";
#endif
        std::printf("cuda: %d rank(s), %d GPU(s) visible on rank 0's node (%s); CUDA-aware MPI: %s\n",
                    size, ndev, prop.name, aware);
    }

    int rc = 0;
    if (o.selftest) {
        rc = selftest();
    } else {
        if ((o.dim != 2 && o.dim != 3) || o.c.N < 1) { if (rank == 0) std::fprintf(stderr, "bad --dim/--N\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
        Report rep;
        if (o.dim == 2) { Block<2> B; solve_gpu<2>(o.c, rep, B); }
        else            { Block<3> B; solve_gpu<3>(o.c, rep, B); }
        if (rank == 0) {
            rep.rec.tag = o.tag; rep.rec.rep = o.rep;
            const RunRecord& r = rep.rec;
            std::printf("cuda %dD N=%d ranks=%d %s/%s ov=%d %s iters=%d rel_resid=%.3e true=%.3e err_L2=%.6e time=%.4fs\n",
                        r.dim, r.N, r.ranks, r.decomp.c_str(), r.halo.c_str(), r.overlap, r.comm_path.c_str(),
                        r.iters, r.rel_resid, r.true_rel_resid, r.err_l2, r.t_total);
            if (!o.csv.empty()) write_csv(o.csv, r);
        }
    }
    MPI_Finalize();
    return rc;
}
