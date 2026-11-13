// cg.hpp -- matrix-free Conjugate Gradient, written exactly as in the lecture:
//   x0 = 0, r0 = b - A x0 = b, p0 = r0
//   while ||r_k|| large:
//     alpha_k = (r_k^T r_k) / (p_k^T A p_k)
//     x_{k+1} = x_k + alpha_k p_k
//     r_{k+1} = r_k - alpha_k A p_k
//     beta_k  = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
//     p_{k+1} = r_{k+1} + beta_k p_k
// The loop is shared by the serial, MPI, OpenMP and CUDA drivers. Each driver supplies an
// `Ops` object with the matrix-free product (halo exchange + stencil in the parallel codes)
// and the dot product (MPI_Allreduce in the parallel codes). Two dot products per iteration:
// p^T A p and r^T r.
#pragma once
#include <cmath>

struct CGResult {
    int iters = 0;
    double rel_resid = 0;   // ||r_k|| / ||b|| from the recurrence
    bool converged = false;
};

// Ops requirements:
//   void   matvec(double* p, double* Ap);          // Ap = A p (may exchange halos of p)
//   double dot(const double* a, const double* b);  // GLOBAL dot product
//   void   axpy(double alpha, const double* x, double* y);   // y += alpha x
//   void   xpay(double beta, const double* r, double* p);    // p = r + beta p
//   void   copy(const double* src, double* dst);
//   void   zero(double* v);
// fixed_iters: run exactly maxit iterations (timing runs) ignoring the tolerance.
template <class Ops>
CGResult cg_solve(Ops& ops, double* x, const double* b, double* r, double* p, double* Ap,
                  double tol, int maxit, bool fixed_iters) {
    CGResult res;
    ops.zero(x);
    ops.copy(b, r);   // r0 = b - A*0 = b
    ops.copy(b, p);   // p0 = r0
    double rr = ops.dot(r, r);
    const double bnorm = std::sqrt(rr);
    int k = 0;
    for (; k < maxit; ++k) {
        if (!fixed_iters && std::sqrt(rr) <= tol * bnorm) { res.converged = true; break; }
        ops.matvec(p, Ap);
        const double pAp = ops.dot(p, Ap);     // global reduction #1
        const double alpha = rr / pAp;
        ops.axpy(alpha, p, x);
        ops.axpy(-alpha, Ap, r);
        const double rr_new = ops.dot(r, r);   // global reduction #2
        const double beta = rr_new / rr;
        ops.xpay(beta, r, p);
        rr = rr_new;
    }
    if (!fixed_iters && std::sqrt(rr) <= tol * bnorm) res.converged = true;
    res.iters = k;
    res.rel_resid = std::sqrt(rr) / bnorm;
    return res;
}
