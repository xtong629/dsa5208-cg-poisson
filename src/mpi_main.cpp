// mpi_main.cpp -- distributed matrix-free CG for -Laplace(u)=f in 2D (5-point) / 3D (7-point).
// One binary; the parallelisation variant is chosen at run time:
//   --decomp slab|block        1D slabs vs DIM-dimensional blocks (MPI_Cart_create)
//   --halo blocking|nonblocking
//   --overlap 0|1              (nonblocking only) hide the halo exchange behind the inner stencil
// Built as "mpi" (pure MPI) or "hybrid" (-DUSE_OPENMP: MPI + OpenMP threads inside each rank).
//
// usage: mpirun -np P mpi --dim 2|3 --N n [--decomp block] [--halo nonblocking] [--overlap 1]
//        [--tol 1e-10] [--maxit M] [--iters K] [--csv file] [--tag t] [--rep r]
//        [--compare serial_dump_file]      relative difference to a serial solution (validation)
//        [--bench-allreduce reps]          time MPI_Allreduce of 1 double (latency) and exit
#include "common/cg.hpp"
#include "common/ops_local.hpp"
#include "common/problem.hpp"
#include "common/vec.hpp"
#include "mpi/ops_mpi.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef USE_OPENMP
#include <omp.h>
#endif

struct Options {
    int dim = 2, N = 64, maxit = 100000, iters = 0, rep = 0, overlap = 1, bench_reps = 0;
    double tol = 1e-10;
    std::string decomp = "block", halo = "nonblocking", csv, tag, compare;
};

static int g_provided = 0;   // MPI thread level actually provided
static const char* level_name(int l) {
    return l == MPI_THREAD_SINGLE ? "single" : l == MPI_THREAD_FUNNELED ? "funneled" :
           l == MPI_THREAD_SERIALIZED ? "serialized" : "multiple";
}

static int num_threads() {
#ifdef USE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Latency of one MPI_Allreduce of a single double (what each dot product costs at best).
static void bench_allreduce(MPI_Comm comm, const Options& o) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    double in = 1.0 + rank, out;
    for (int i = 0; i < 200; ++i) MPI_Allreduce(&in, &out, 1, MPI_DOUBLE, MPI_SUM, comm);  // warm-up
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int i = 0; i < o.bench_reps; ++i) MPI_Allreduce(&in, &out, 1, MPI_DOUBLE, MPI_SUM, comm);
    double t = (MPI_Wtime() - t0) / o.bench_reps, tmax;
    MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        std::printf("allreduce(1 double) ranks=%d  %.3f us/call\n", size, tmax * 1e6);
        if (!o.csv.empty()) {
            bool header = true;
            if (FILE* f = std::fopen(o.csv.c_str(), "r")) { std::fseek(f, 0, SEEK_END); header = std::ftell(f) == 0; std::fclose(f); }
            if (FILE* f = std::fopen(o.csv.c_str(), "a")) {
                if (header) std::fprintf(f, "tag,ranks,threads,reps,t_allreduce_us,rep\n");
                std::fprintf(f, "%s,%d,%d,%d,%.4f,%d\n", o.tag.c_str(), size, num_threads(), o.bench_reps, tmax * 1e6, o.rep);
                std::fclose(f);
            }
        }
    }
}

// Relative difference of the distributed solution to a raw dump written by the serial program.
template <int DIM>
static double compare_to_dump(const Decomp<DIM>& D, int N, const double* x, const std::string& file) {
    FILE* f = std::fopen(file.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", file.c_str()); MPI_Abort(D.cart, 3); }
    const Block<DIM>& B = D.B;
    std::vector<double> row(B.n[0]);
    double loc[2] = {0, 0};   // sum (x - xs)^2, sum xs^2
    for (int k = 0; k < B.n[2]; ++k)
        for (int j = 0; j < B.n[1]; ++j) {
            long g = ((long)(B.off[2] + k) * (DIM >= 2 ? N : 1) + (B.off[1] + j)) * N + B.off[0];
            std::fseek(f, g * (long)sizeof(double), SEEK_SET);
            if (std::fread(row.data(), sizeof(double), B.n[0], f) != (size_t)B.n[0]) { std::fprintf(stderr, "short read\n"); MPI_Abort(D.cart, 3); }
            const long base = B.idx(0, j, k);
            for (int i = 0; i < B.n[0]; ++i) {
                double d = x[base + i] - row[i];
                loc[0] += d * d;
                loc[1] += row[i] * row[i];
            }
        }
    std::fclose(f);
    double glob[2];
    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, D.cart);
    return std::sqrt(glob[0] / glob[1]);
}

template <int DIM>
static int run(const Options& o) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    const int N = o.N;
    Decomp<DIM> D(MPI_COMM_WORLD, N, o.decomp == "slab");
    Halo<DIM> H(D);
    const HaloMode mode = (o.halo == "blocking") ? HaloMode::Blocking : HaloMode::NonBlocking;
    MpiOps<DIM> ops(D, H, N, mode, o.overlap != 0);
    const Block<DIM>& B = D.B;

    if (ops.split.covered() != B.interior_count()) {
        std::fprintf(stderr, "rank %d: inner+shell boxes cover %ld of %ld points\n", D.rank, ops.split.covered(), B.interior_count());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    // aligned, first-touch initialised (parallel under OpenMP)
    AlignedVec<DIM> x(B), b(B), r(B), p(B), Ap(B);
    fill_rhs<DIM>(B, N, b.data());

    MPI_Barrier(D.cart);
    const double t0 = MPI_Wtime();
    CGResult res = cg_solve(ops, x.data(), b.data(), r.data(), p.data(), Ap.data(), o.tol,
                            o.iters > 0 ? o.iters : o.maxit, o.iters > 0);
    const double t_mine = MPI_Wtime() - t0;
    Timers solve_T = ops.T;   // snapshot: verification below must not pollute the timers

    // ---- verification (untimed) ----
    // true residual: explicit b - A x (matvec exchanges the halo of x)
    AlignedVec<DIM> Ax(B);
    ops.matvec(x.data(), Ax.data());
    double loc[3] = {0, 0, 0}, sq = 0, mx = 0;
    for (int k = 0; k < B.n[2]; ++k)
        for (int j = 0; j < B.n[1]; ++j) {
            const long base = B.idx(0, j, k);
            for (int i = 0; i < B.n[0]; ++i) {
                double d = b[base + i] - Ax[base + i];
                loc[0] += d * d;
                loc[1] += b[base + i] * b[base + i];
            }
        }
    error_local<DIM>(B, N, x.data(), sq, mx);
    loc[2] = sq;
    double glob[3], gmax;
    MPI_Allreduce(loc, glob, 3, MPI_DOUBLE, MPI_SUM, D.cart);
    MPI_Allreduce(&mx, &gmax, 1, MPI_DOUBLE, MPI_MAX, D.cart);
    const double h = 1.0 / (N + 1);
    double hd = 1.0;
    for (int d = 0; d < DIM; ++d) hd *= h;

    // timers: sections = mean over ranks, total = max over ranks; "other" = total - sum(sections)
    double tsec[SEC_COUNT], tsec_mean[SEC_COUNT], t_total;
    for (int s = 0; s < SEC_COUNT; ++s) tsec[s] = solve_T.t[s];
    MPI_Reduce(tsec, tsec_mean, SEC_COUNT, MPI_DOUBLE, MPI_SUM, 0, D.cart);
    MPI_Reduce(&t_mine, &t_total, 1, MPI_DOUBLE, MPI_MAX, 0, D.cart);
    double hd_max, lp_max, hl = (double)D.halo_doubles(), lp = (double)B.interior_count();
    MPI_Reduce(&hl, &hd_max, 1, MPI_DOUBLE, MPI_MAX, 0, D.cart);
    MPI_Reduce(&lp, &lp_max, 1, MPI_DOUBLE, MPI_MAX, 0, D.cart);

    double diff = -1;
    if (!o.compare.empty()) diff = compare_to_dump<DIM>(D, N, x.data(), o.compare);

    if (D.rank == 0) {
        RunRecord rec;
#ifdef USE_OPENMP
        rec.program = "hybrid";
#else
        rec.program = "mpi";
#endif
        rec.tag = o.tag; rec.dim = DIM; rec.N = N; rec.rep = o.rep;
        rec.ranks = D.size; rec.threads = num_threads(); rec.thread_level = level_name(g_provided);
        rec.decomp = o.decomp; rec.halo = o.halo;
        rec.overlap = (mode == HaloMode::NonBlocking && o.overlap) ? 1 : 0;
        rec.mode = o.iters > 0 ? "iters" : "tol";
        rec.iters = res.iters; rec.converged = res.converged;
        rec.rel_resid = res.rel_resid; rec.true_rel_resid = std::sqrt(glob[0] / glob[1]);
        rec.err_l2 = std::sqrt(hd * glob[2]); rec.err_max = gmax;
        rec.diff_serial = diff; rec.halo_doubles = hd_max; rec.local_points = lp_max;
        rec.t_total = t_total;
        rec.t_stencil = tsec_mean[SEC_STENCIL] / D.size; rec.t_halo = tsec_mean[SEC_HALO] / D.size;
        rec.t_allreduce = tsec_mean[SEC_ALLREDUCE] / D.size; rec.t_vector = tsec_mean[SEC_VECTOR] / D.size;
        std::printf("%s %dD N=%d ranks=%d (%dx%dx%d) %s/%s ov=%d iters=%d rel_resid=%.3e true=%.3e "
                    "err_L2=%.6e diff_serial=%.3e time=%.4fs\n",
                    rec.program.c_str(), DIM, N, D.size, D.dims[0], D.dims[1], D.dims[2], o.decomp.c_str(),
                    o.halo.c_str(), rec.overlap, res.iters, rec.rel_resid, rec.true_rel_resid, rec.err_l2,
                    diff, t_total);
        if (!o.csv.empty()) write_csv(o.csv, rec);
    }
    return 0;
}

int main(int argc, char** argv) {
#ifdef USE_OPENMP
    const int required = MPI_THREAD_FUNNELED;   // only the main (master) thread calls MPI
#else
    const int required = MPI_THREAD_SINGLE;
#endif
    MPI_Init_thread(&argc, &argv, required, &g_provided);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (g_provided < required) {   // the MPI library cannot give the level the hybrid design needs
        if (rank == 0) std::fprintf(stderr, "MPI provides thread level %s, need %s\n", level_name(g_provided), level_name(required));
        MPI_Abort(MPI_COMM_WORLD, 6);
    }
    Options o;
    for (int a = 1; a < argc; ++a) {
        auto is = [&](const char* s) { return std::strcmp(argv[a], s) == 0 && a + 1 < argc; };
        if (is("--dim")) o.dim = std::atoi(argv[++a]);
        else if (is("--N")) o.N = std::atoi(argv[++a]);
        else if (is("--decomp")) o.decomp = argv[++a];
        else if (is("--halo")) o.halo = argv[++a];
        else if (is("--overlap")) o.overlap = std::atoi(argv[++a]);
        else if (is("--tol")) o.tol = std::atof(argv[++a]);
        else if (is("--maxit")) o.maxit = std::atoi(argv[++a]);
        else if (is("--iters")) o.iters = std::atoi(argv[++a]);
        else if (is("--csv")) o.csv = argv[++a];
        else if (is("--tag")) o.tag = argv[++a];
        else if (is("--rep")) o.rep = std::atoi(argv[++a]);
        else if (is("--compare")) o.compare = argv[++a];
        else if (is("--bench-allreduce")) o.bench_reps = std::atoi(argv[++a]);
        else {
            if (rank == 0) std::fprintf(stderr, "unknown/incomplete option %s\n", argv[a]);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
    bool ok = (o.dim == 2 || o.dim == 3) && o.N >= 1 && (o.decomp == "slab" || o.decomp == "block") &&
              (o.halo == "blocking" || o.halo == "nonblocking");
    if (!ok) {
        if (rank == 0) std::fprintf(stderr, "bad option value (dim 2|3, decomp slab|block, halo blocking|nonblocking)\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    int rc = 0;
    if (o.bench_reps > 0) bench_allreduce(MPI_COMM_WORLD, o);
    else rc = (o.dim == 2) ? run<2>(o) : run<3>(o);
    MPI_Finalize();
    return rc;
}
