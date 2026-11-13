// serial_main.cpp -- serial matrix-free CG for -Laplace(u)=f, 2D (5-point) and 3D (7-point).
// This program provides T_s (serial runtime) for speedup/efficiency, per Lecture 7.
//
// usage: serial --dim 2|3 --N <points per axis> [--tol 1e-10] [--maxit M] [--iters K]
//               [--csv file] [--tag label] [--rep r] [--dump file]
//   --iters K : run exactly K CG iterations (timing runs); otherwise iterate until
//               ||r||/||b|| <= tol.
#include "common/cg.hpp"
#include "common/ops_local.hpp"
#include "common/problem.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

template <int DIM>
int run(int N, double tol, int maxit, int fixed_iters, const std::string& csv, const std::string& tag,
        int rep, const std::string& dump) {
    int n[3] = {N, DIM >= 2 ? N : 1, DIM == 3 ? N : 1};
    int off[3] = {0, 0, 0};
    Block<DIM> B(n, off);
    SerialOps<DIM> ops(B, N);

    // first-touch/allocation outside the timed region
    std::vector<double> x(B.size, 0.0), b(B.size, 0.0), r(B.size, 0.0), p(B.size, 0.0), Ap(B.size, 0.0);
    fill_rhs<DIM>(B, N, b.data());

    double t0 = wtime();
    CGResult res = cg_solve(ops, x.data(), b.data(), r.data(), p.data(), Ap.data(), tol,
                            fixed_iters > 0 ? fixed_iters : maxit, fixed_iters > 0);
    double t_total = wtime() - t0;

    // --- verification (outside the timed region) ---
    // (1) true residual ||b - A x|| / ||b||, recomputed explicitly (checks CG recurrences)
    std::vector<double> Ax(B.size, 0.0);
    stencil_all<DIM>(B, x.data(), Ax.data(), ops.h2inv);
    double num = 0, den = 0;
    for (int k = 0; k < B.n[2]; ++k)
        for (int j = 0; j < B.n[1]; ++j) {
            long base = B.idx(0, j, k);
            for (int i = 0; i < B.n[0]; ++i) {
                double d = b[base + i] - Ax[base + i];
                num += d * d;
                den += b[base + i] * b[base + i];
            }
        }
    // (2) error vs the manufactured solution, discrete L2 norm sqrt(h^DIM sum e^2) and max norm
    double ssq, emax;
    error_local<DIM>(B, N, x.data(), ssq, emax);
    const double h = 1.0 / (N + 1);
    double hd = 1.0;
    for (int d = 0; d < DIM; ++d) hd *= h;

    RunRecord rec;
    rec.program = "serial"; rec.tag = tag; rec.dim = DIM; rec.N = N; rec.rep = rep;
    rec.mode = fixed_iters > 0 ? "iters" : "tol";
    rec.iters = res.iters; rec.converged = res.converged;
    rec.rel_resid = res.rel_resid; rec.true_rel_resid = std::sqrt(num / den);
    rec.err_l2 = std::sqrt(hd * ssq); rec.err_max = emax;
    rec.t_total = t_total;
    rec.t_stencil = ops.T.t[SEC_STENCIL]; rec.t_vector = ops.T.t[SEC_VECTOR];

    std::printf("serial %dD N=%d iters=%d rel_resid=%.3e true_resid=%.3e err_L2=%.6e err_max=%.6e "
                "time=%.4fs (%.3e s/iter)\n",
                DIM, N, res.iters, rec.rel_resid, rec.true_rel_resid, rec.err_l2, rec.err_max,
                t_total, res.iters ? t_total / res.iters : 0.0);
    rec.local_points = (double)B.interior_count();
    if (!csv.empty()) write_csv(csv, rec);
    // optional: raw dump of the interior solution (doubles, x fastest, global order) so the MPI
    // program can compare its distributed solution against this one (validation test 2)
    if (!dump.empty()) {
        FILE* f = std::fopen(dump.c_str(), "wb");
        if (!f) { std::perror("dump"); return 1; }
        for (int k = 0; k < B.n[2]; ++k)
            for (int j = 0; j < B.n[1]; ++j)
                std::fwrite(&x[B.idx(0, j, k)], sizeof(double), B.n[0], f);
        std::fclose(f);
    }
    return 0;
}

int main(int argc, char** argv) {
    int dim = 2, N = 63, maxit = 100000, iters = 0, rep = 0;
    double tol = 1e-10;
    std::string csv, tag, dump;
    for (int a = 1; a < argc; ++a) {
        auto is = [&](const char* s) { return std::strcmp(argv[a], s) == 0 && a + 1 < argc; };
        if (is("--dim")) dim = std::atoi(argv[++a]);
        else if (is("--N")) N = std::atoi(argv[++a]);
        else if (is("--tol")) tol = std::atof(argv[++a]);
        else if (is("--maxit")) maxit = std::atoi(argv[++a]);
        else if (is("--iters")) iters = std::atoi(argv[++a]);
        else if (is("--csv")) csv = argv[++a];
        else if (is("--tag")) tag = argv[++a];
        else if (is("--rep")) rep = std::atoi(argv[++a]);
        else if (is("--dump")) dump = argv[++a];
        else { std::fprintf(stderr, "unknown/incomplete option %s\n", argv[a]); return 2; }
    }
    if (N < 1 || (dim != 2 && dim != 3)) { std::fprintf(stderr, "bad --dim/--N\n"); return 2; }
    return dim == 2 ? run<2>(N, tol, maxit, iters, csv, tag, rep, dump)
                    : run<3>(N, tol, maxit, iters, csv, tag, rep, dump);
}
