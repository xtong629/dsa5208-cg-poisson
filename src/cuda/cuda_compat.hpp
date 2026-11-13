// cuda_compat.hpp -- CUDA includes, error checking and the kernel-launch macro.
//   Real build : nvcc (or nvc++ -cuda) with <cuda_runtime.h>; LAUNCH expands to kernel<<<...>>>.
//   -DCUDA_STUB: type-check only on a machine without CUDA (see `make cuda-check`); kernels are
//                called as plain functions and nothing can run.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#ifdef CUDA_STUB
#include "cuda/stub/cuda_runtime.h"
#define LAUNCH(kernel, grid, block, stream, ...) kernel(__VA_ARGS__)
#else
#include <cuda_runtime.h>
// kernel<<<grid, block, dynamic shared mem = 0, stream>>>(args); the macro keeps template kernels
// with template arguments (commas) out of the argument list, so only single-parameter templates.
#define LAUNCH(kernel, grid, block, stream, ...) \
    do { kernel<<<(grid), (block), 0, (stream)>>>(__VA_ARGS__); CUDA_CHECK(cudaGetLastError()); } while (0)
#endif

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t err_ = (call);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,   \
                         cudaGetErrorString(err_));                                           \
            MPI_Abort(MPI_COMM_WORLD, 7);                                                     \
        }                                                                                     \
    } while (0)
