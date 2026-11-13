// ops_local.hpp -- vector operations on one local block, with section timers.
// SerialOps uses these directly; the MPI/OpenMP drivers derive from LocalOps and override
// matvec() (halo exchange + stencil) and dot() (MPI_Allreduce).
#pragma once
#include "grid.hpp"
#include "kernels.hpp"
#include "timers.hpp"

template <int DIM>
struct LocalOps {
    Block<DIM> B;
    Timers T;
    double h2inv;

    LocalOps(const Block<DIM>& B_, int N) : B(B_) {
        const double h = 1.0 / (N + 1);
        h2inv = 1.0 / (h * h);
    }
    void axpy(double a, const double* x, double* y) {
        ScopedTimer t(T, SEC_VECTOR);
        ::axpy<DIM>(B, a, x, y);
    }
    void xpay(double beta, const double* r, double* p) {
        ScopedTimer t(T, SEC_VECTOR);
        ::xpay<DIM>(B, beta, r, p);
    }
    void copy(const double* s, double* d) {
        ScopedTimer t(T, SEC_VECTOR);
        copy_vec<DIM>(B, s, d);
    }
    void zero(double* v) { fill_zero<DIM>(B, v); }   // parallel, same distribution as the kernels
};

// Serial operator: ghost cells of p stay 0 (Dirichlet), so no halo step at all.
template <int DIM>
struct SerialOps : LocalOps<DIM> {
    using LocalOps<DIM>::LocalOps;
    void matvec(double* p, double* Ap) {
        ScopedTimer t(this->T, SEC_STENCIL);
        stencil_all<DIM>(this->B, p, Ap, this->h2inv);
    }
    double dot(const double* a, const double* b) {
        ScopedTimer t(this->T, SEC_VECTOR);   // serial: dot products are plain vector work
        return dot_local<DIM>(this->B, a, b);
    }
};
