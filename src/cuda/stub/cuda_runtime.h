// stub/cuda_runtime.h -- NOT the real CUDA header. Declares just enough of the CUDA runtime API
// (same names and signatures as the real one) so that `make cuda-check` can type-check
// src/cuda_main.cu with an ordinary C++ compiler on a machine without CUDA. It cannot run
// anything. The real build never includes this file (it is only used with -DCUDA_STUB).
#pragma once
#include <cstddef>

#define __global__
#define __device__
#define __host__
#define __shared__
#define __forceinline__ inline

struct uint3 { unsigned x, y, z; };
struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
extern uint3 threadIdx, blockIdx;
extern dim3 blockDim, gridDim;
inline void __syncthreads() {}

typedef int cudaError_t;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
enum { cudaSuccess = 0 };
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault };
enum { cudaStreamNonBlocking = 1, cudaEventDisableTiming = 2 };
struct cudaDeviceProp { char name[256]; size_t totalGlobalMem; };

cudaError_t cudaGetDeviceCount(int*);
cudaError_t cudaSetDevice(int);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int);
cudaError_t cudaMalloc(void**, size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaMallocHost(void**, size_t);
cudaError_t cudaFreeHost(void*);
cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t = 0);
cudaError_t cudaMemset(void*, int, size_t);
cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t = 0);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaStreamCreateWithPriority(cudaStream_t*, unsigned, int);
cudaError_t cudaDeviceGetStreamPriorityRange(int*, int*);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = 0);
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t);
