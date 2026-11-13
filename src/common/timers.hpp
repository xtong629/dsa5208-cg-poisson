// timers.hpp -- section timers and the CSV record (one row per run).
#pragma once
#include <chrono>
#include <cstdio>
#include <string>
#ifdef USE_MPI
#include <mpi.h>
#endif

inline double wtime() {
#ifdef USE_MPI
    return MPI_Wtime();
#else
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
#endif
}

// Sections of one CG iteration. "Other" = total - sum of the named sections (loop overhead, etc.)
enum Sec { SEC_STENCIL = 0, SEC_HALO, SEC_ALLREDUCE, SEC_VECTOR, SEC_COUNT };

struct Timers {
    double t[SEC_COUNT] = {0, 0, 0, 0};
    void reset() { for (double& x : t) x = 0; }
};

struct ScopedTimer {
    Timers& T; Sec s; double t0;
    ScopedTimer(Timers& T_, Sec s_) : T(T_), s(s_), t0(wtime()) {}
    ~ScopedTimer() { T.t[s] += wtime() - t0; }
};

// Everything that goes into one CSV row.
struct RunRecord {
    std::string program = "serial";   // serial | mpi | hybrid | cuda
    std::string tag;                  // free-text label set by the job script (e.g. "strong")
    int dim = 2, N = 0;
    int ranks = 1, threads = 1;
    std::string thread_level = "none"; // MPI thread level provided: none (serial) | single | funneled | ...
    std::string decomp = "none";      // none | slab | block
    std::string halo = "none";        // none | blocking | nonblocking | rma
    int overlap = 0;
    std::string comm_path = "host";   // host (CPU buffers) | device (CUDA-aware MPI) | staged (GPU via pinned host)
    std::string mode = "tol";         // tol | iters
    int iters = 0;
    int converged = 0;
    double rel_resid = 0, true_rel_resid = 0;   // recurrence residual / explicitly recomputed
    double err_l2 = 0, err_max = 0;             // vs the manufactured exact solution
    double diff_serial = -1;                    // ||x - x_serial|| / ||x_serial|| (-1 = not computed)
    double halo_doubles = 0, local_points = 0;  // max over ranks: values sent per halo exchange / owned points
    double t_total = 0, t_stencil = 0, t_halo = 0, t_allreduce = 0, t_vector = 0;
    int rep = 0;
};

inline void write_csv(const std::string& path, const RunRecord& r) {
    bool need_header = true;
    if (FILE* f = std::fopen(path.c_str(), "r")) {
        std::fseek(f, 0, SEEK_END);
        need_header = (std::ftell(f) == 0);
        std::fclose(f);
    }
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) { std::perror(("cannot open " + path).c_str()); return; }
    if (need_header)
        std::fprintf(f, "program,tag,dim,N,unknowns,ranks,threads,thread_level,decomp,halo,overlap,comm_path,mode,iters,"
                        "converged,rel_resid,true_rel_resid,err_l2,err_max,diff_serial,halo_doubles,local_points,"
                        "t_total,t_per_iter,"
                        "t_stencil,t_halo,t_allreduce,t_vector,t_other,rep\n");
    double unknowns = 1;
    for (int d = 0; d < r.dim; ++d) unknowns *= r.N;
    double other = r.t_total - r.t_stencil - r.t_halo - r.t_allreduce - r.t_vector;
    std::fprintf(f, "%s,%s,%d,%d,%.0f,%d,%d,%s,%s,%s,%d,%s,%s,%d,%d,%.6e,%.6e,%.10e,%.10e,%.6e,%.0f,%.0f,"
                    "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d\n",
                 r.program.c_str(), r.tag.c_str(), r.dim, r.N, unknowns, r.ranks, r.threads,
                 r.thread_level.c_str(), r.decomp.c_str(), r.halo.c_str(), r.overlap, r.comm_path.c_str(), r.mode.c_str(), r.iters, r.converged,
                 r.rel_resid, r.true_rel_resid, r.err_l2, r.err_max, r.diff_serial, r.halo_doubles,
                 r.local_points, r.t_total,
                 r.iters ? r.t_total / r.iters : 0.0, r.t_stencil, r.t_halo, r.t_allreduce,
                 r.t_vector, other, r.rep);
    std::fclose(f);
}
