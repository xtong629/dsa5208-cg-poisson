// gpu_kernels.cuh -- CUDA kernels. Same padded memory layout as the CPU code (grid.hpp):
// x fastest, one ghost layer per active axis. One thread handles one point; consecutive threads
// take consecutive x (i fastest) so global-memory accesses are coalesced.
//
// Design points
//  * stencil_kernel works on an arbitrary box [lo,hi) of interior points. The inner box (points
//    that need no ghost value) and the thin boundary-shell boxes use the SAME kernel; there is no
//    per-thread "is this a boundary point?" branch, hence no warp divergence from boundary
//    handling (the only branch is the tail guard t >= total).
//  * dot products: two-pass tree reduction with a fixed order and no atomics, so the result is
//    deterministic for a given problem size and launch configuration.
//  * pack/unpack kernels move a face between the padded array and a contiguous message buffer.
#pragma once
#include "common/kernels.hpp"   // Box
#include "cuda_compat.hpp"

constexpr int kBlock = 256;     // threads per block; must be a power of two (tree reductions)

// Device copy of the geometry of a Block (plain data, passed to kernels by value).
struct GB {
    int n[3];      // local interior points
    int g[3];      // ghost width per axis (1 for active axes, 0 otherwise)
    long str[3];   // strides of the padded array
};

// padded index of the t-th interior point (x fastest)
__device__ __forceinline__ long interior_index(const GB& b, long t) {
    const int i = (int)(t % b.n[0]);
    const long u = t / b.n[0];
    const int j = (int)(u % b.n[1]);
    const int k = (int)(u / b.n[1]);
    return (long)(i + b.g[0]) + (long)(j + b.g[1]) * b.str[1] + (long)(k + b.g[2]) * b.str[2];
}

// Ap = A p on the box: (2*DIM*p_c - sum of neighbours) * h2inv
template <int DIM>
__global__ void stencil_kernel(GB b, Box box, const double* __restrict__ p, double* __restrict__ Ap,
                               double h2inv) {
    const int dx = box.hi[0] - box.lo[0], dy = box.hi[1] - box.lo[1], dz = box.hi[2] - box.lo[2];
    const long total = (long)dx * dy * dz;
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const int i = box.lo[0] + (int)(t % dx);
    const long u = t / dx;
    const int j = box.lo[1] + (int)(u % dy);
    const int k = box.lo[2] + (int)(u / dy);
    const long c = (long)(i + b.g[0]) + (long)(j + b.g[1]) * b.str[1] + (long)(k + b.g[2]) * b.str[2];
    double s = 2.0 * DIM * p[c] - p[c - 1] - p[c + 1] - p[c - b.str[1]] - p[c + b.str[1]];
    if (DIM == 3) s -= p[c - b.str[2]] + p[c + b.str[2]];
    Ap[c] = h2inv * s;
}

// y += alpha x   (interior points)
__global__ void axpy_kernel(GB b, long n, double alpha, const double* __restrict__ x, double* __restrict__ y) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const long c = interior_index(b, t);
    y[c] += alpha * x[c];
}

// p = r + beta p   (interior points)
__global__ void xpay_kernel(GB b, long n, double beta, const double* __restrict__ r, double* __restrict__ p) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const long c = interior_index(b, t);
    p[c] = r[c] + beta * p[c];
}

// Dot product, pass 1: each thread sums a strided subset, each block tree-reduces its 256 partial
// sums in shared memory and writes one value partial[blockIdx.x].
__global__ void dot_pass1(GB b, long n, const double* __restrict__ x, const double* __restrict__ y,
                          double* __restrict__ partial) {
    __shared__ double s[kBlock];
    double acc = 0.0;
    const long stride = (long)gridDim.x * blockDim.x;
    for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        const long c = interior_index(b, t);
        acc += x[c] * y[c];
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int off = kBlock / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}

// Pass 2 (single block): sum the nb block partials in a fixed order -> out[0].
__global__ void dot_pass2(const double* __restrict__ partial, int nb, double* __restrict__ out) {
    __shared__ double s[kBlock];
    double acc = 0.0;
    for (int i = threadIdx.x; i < nb; i += kBlock) acc += partial[i];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int off = kBlock / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = s[0];
}

// Face addressing. A face is the layer at padded position `pos` along `axis`; e1<e2 are the two
// other axes (in 2D the unused third axis has n=1, ghost 0, so it contributes nothing).
// Point t of the face -> padded index. Both sides of a message use the same ordering (their
// extents along e1,e2 are identical because they are neighbours in a Cartesian grid).
__device__ __forceinline__ long face_index(const GB& b, int axis, int e1, int e2, int pos, long t) {
    const int c1 = (int)(t % b.n[e1]);
    const int c2 = (int)(t / b.n[e1]);
    return (long)(c1 + b.g[e1]) * b.str[e1] + (long)(c2 + b.g[e2]) * b.str[e2] + (long)pos * b.str[axis];
}

__global__ void pack_kernel(GB b, int axis, int e1, int e2, int pos, long count, const double* __restrict__ p,
                            double* __restrict__ buf) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    buf[t] = p[face_index(b, axis, e1, e2, pos, t)];
}

__global__ void unpack_kernel(GB b, int axis, int e1, int e2, int pos, long count, const double* __restrict__ buf,
                              double* __restrict__ p) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    p[face_index(b, axis, e1, e2, pos, t)] = buf[t];
}
