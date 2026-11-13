// problem.hpp -- manufactured solution (synthetic data) for  -Laplace(u) = f  on (0,1)^DIM.
//
// Exact solution (product of 1D factors)   u(x) = prod_d g(x_d),   g(t) = t (1-t) e^t.
// g vanishes at t=0,1 so u = 0 on the boundary (homogeneous Dirichlet). Since
// g''(t) = -t (3+t) e^t,   f = -Laplace(u) = - sum_d g''(x_d) prod_{e != d} g(x_e).
// All data are generated analytically from these formulas; there is no input file.
// b_i = f(x_i) is sampled at the grid points; the numerical solution of A u_h = b
// (A = 5/7-point finite-difference -Laplacian) is compared with u(x_i) to measure the
// discretisation error, expected O(h^2).
//
// Why not u = prod sin(pi x_d)?  That function is an exact eigenvector of the discrete
// Laplacian, so CG would converge in ONE iteration and the timings/iteration counts would be
// meaningless. g above has a broad spectral content, so CG needs O(N) iterations as for
// generic data.
#pragma once
#include "grid.hpp"
#include <cmath>
#include <vector>

// 1D tables of g(x) and g''(x) at the local grid points of each axis (avoids exp() in 3D loops)
template <int DIM>
struct FieldTables {
    std::vector<double> g[3], gpp[3];
    FieldTables(const Block<DIM>& B, int N) {
        const double h = 1.0 / (N + 1);
        for (int d = 0; d < 3; ++d) {
            g[d].assign(B.n[d], 1.0);
            gpp[d].assign(B.n[d], 0.0);
            if (d < DIM)
                for (int i = 0; i < B.n[d]; ++i) {
                    const double t = (B.off[d] + i + 1) * h, e = std::exp(t);
                    g[d][i] = t * (1.0 - t) * e;
                    gpp[d][i] = -t * (3.0 + t) * e;
                }
        }
    }
    inline double u(int i, int j, int k) const { return g[0][i] * g[1][j] * g[2][k]; }
    inline double f(int i, int j, int k) const {
        double v = -gpp[0][i] * g[1][j] * g[2][k] - g[0][i] * gpp[1][j] * g[2][k];
        if (DIM == 3) v -= g[0][i] * g[1][j] * gpp[2][k];
        return v;
    }
};

// b = f at the local grid points (ghost cells left untouched)
template <int DIM>
void fill_rhs(const Block<DIM>& B, int N, double* b) {
    FieldTables<DIM> T(B, N);
    for (int k = 0; k < B.n[2]; ++k)
        for (int j = 0; j < B.n[1]; ++j) {
            long base = B.idx(0, j, k);
            for (int i = 0; i < B.n[0]; ++i) b[base + i] = T.f(i, j, k);
        }
}

// Local sums for the error norms: sum (x-u)^2 and max |x-u| over the local points.
template <int DIM>
void error_local(const Block<DIM>& B, int N, const double* x, double& sum_sq, double& max_abs) {
    FieldTables<DIM> T(B, N);
    sum_sq = 0.0;
    max_abs = 0.0;
    for (int k = 0; k < B.n[2]; ++k)
        for (int j = 0; j < B.n[1]; ++j) {
            long base = B.idx(0, j, k);
            for (int i = 0; i < B.n[0]; ++i) {
                double e = x[base + i] - T.u(i, j, k);
                sum_sq += e * e;
                if (std::fabs(e) > max_abs) max_abs = std::fabs(e);
            }
        }
}
