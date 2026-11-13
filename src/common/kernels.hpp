// kernels.hpp -- hand-written compute kernels (no BLAS/LAPACK, as required by the course).
// All operate on the padded Block layout and touch interior points only (except fill_zero).
//
// OpenMP (compiled in only with -DUSE_OPENMP, i.e. the "hybrid" build): every kernel is a
// `parallel for` over the two outer loops (k,j) with schedule(static). Static scheduling gives
// each thread the same rows in every kernel and every iteration, so the memory a thread first
// touched (see AlignedVec in vec.hpp) stays on its NUMA node. The inner loop over i is contiguous
// and left to the compiler to vectorise. In 2D the k loop has one iteration, so collapse(2)
// effectively parallelises over rows j. Dot products use reduction(+:s).
#pragma once
#include "grid.hpp"

#ifdef USE_OPENMP
#define PRAGMA_OMP(x) _Pragma(#x)
#else
#define PRAGMA_OMP(x)
#endif

// A box [lo,hi) of interior points in local interior coordinates.
struct Box { int lo[3], hi[3]; };

// Ap = A p on the box, where A = -Laplacian_h : (2*DIM*p_c - sum of 2*DIM neighbours) / h^2.
// Needs valid ghost cells of p (halo or zero Dirichlet boundary). Body of one box, to be called
// inside an existing parallel region (orphaned `omp for`) or from stencil_range below.
template <int DIM>
inline void stencil_box_body(const Block<DIM>& B, const double* __restrict__ p, double* __restrict__ Ap,
                             const int lo[3], const int hi[3], double h2inv) {
    const long sy = B.str[1], sz = B.str[2];
    PRAGMA_OMP(omp for collapse(2) schedule(static) nowait)
    for (int k = lo[2]; k < hi[2]; ++k)
        for (int j = lo[1]; j < hi[1]; ++j) {
            const long base = B.idx(0, j, k);
            for (int i = lo[0]; i < hi[0]; ++i) {
                const long c = base + i;
                double s = 2.0 * DIM * p[c] - p[c - 1] - p[c + 1] - p[c - sy] - p[c + sy];
                if constexpr (DIM == 3) s -= p[c - sz] + p[c + sz];
                Ap[c] = h2inv * s;
            }
        }
}

// Ap = A p on a single box (one parallel region).
template <int DIM>
void stencil_range(const Block<DIM>& B, const double* p, double* Ap, const int lo[3], const int hi[3],
                   double h2inv) {
    PRAGMA_OMP(omp parallel)
    stencil_box_body<DIM>(B, p, Ap, lo, hi, h2inv);
}

// Ap = A p on several disjoint boxes inside ONE parallel region (the boundary shell: up to 2*DIM
// thin boxes; a region per box would pay the fork/join cost 2*DIM times per iteration).
// The boxes are disjoint and p is only read, so the `nowait` loops cannot conflict.
template <int DIM>
void stencil_boxes(const Block<DIM>& B, const double* p, double* Ap, const Box* boxes, int nboxes,
                   double h2inv) {
    PRAGMA_OMP(omp parallel)
    for (int b = 0; b < nboxes; ++b) stencil_box_body<DIM>(B, p, Ap, boxes[b].lo, boxes[b].hi, h2inv);
}

template <int DIM>
void stencil_all(const Block<DIM>& B, const double* p, double* Ap, double h2inv) {
    int lo[3] = {0, 0, 0};
    stencil_range<DIM>(B, p, Ap, lo, B.n, h2inv);
}

// local part of a dot product over interior points
template <int DIM>
double dot_local(const Block<DIM>& B, const double* a, const double* b) {
    double s = 0.0;
    PRAGMA_OMP(omp parallel for collapse(2) schedule(static) reduction(+ : s))
    for (int k = 0; k < B.n[2]; ++k)
        for (int j = 0; j < B.n[1]; ++j) {
            const long base = B.idx(0, j, k);
            for (int i = 0; i < B.n[0]; ++i) s += a[base + i] * b[base + i];
        }
    return s;
}

// y += alpha * x
template <int DIM>
void axpy(const Block<DIM>& B, double alpha, const double* x, double* y) {
    PRAGMA_OMP(omp parallel for collapse(2) schedule(static))
    for (int k = 0; k < B.n[2]; ++k)
        for (int j = 0; j < B.n[1]; ++j) {
            const long base = B.idx(0, j, k);
            for (int i = 0; i < B.n[0]; ++i) y[base + i] += alpha * x[base + i];
        }
}

// p = r + beta * p
template <int DIM>
void xpay(const Block<DIM>& B, double beta, const double* r, double* p) {
    PRAGMA_OMP(omp parallel for collapse(2) schedule(static))
    for (int k = 0; k < B.n[2]; ++k)
        for (int j = 0; j < B.n[1]; ++j) {
            const long base = B.idx(0, j, k);
            for (int i = 0; i < B.n[0]; ++i) p[base + i] = r[base + i] + beta * p[base + i];
        }
}

// dst = src (interior only)
template <int DIM>
void copy_vec(const Block<DIM>& B, const double* src, double* dst) {
    PRAGMA_OMP(omp parallel for collapse(2) schedule(static))
    for (int k = 0; k < B.n[2]; ++k)
        for (int j = 0; j < B.n[1]; ++j) {
            const long base = B.idx(0, j, k);
            for (int i = 0; i < B.n[0]; ++i) dst[base + i] = src[base + i];
        }
}

// v = 0 on the WHOLE padded array (interior + ghost cells), row by row with the same (k,j)
// static distribution as the kernels above. Used for first-touch initialisation and for x0 = 0.
template <int DIM>
void fill_zero(const Block<DIM>& B, double* v) {
    PRAGMA_OMP(omp parallel for collapse(2) schedule(static))
    for (int k = 0; k < B.ext[2]; ++k)
        for (int j = 0; j < B.ext[1]; ++j) {
            double* row = v + (long)j * B.str[1] + (long)k * B.str[2];
            for (int i = 0; i < B.ext[0]; ++i) row[i] = 0.0;
        }
}
