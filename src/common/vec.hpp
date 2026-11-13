// vec.hpp -- aligned vector for one local block with parallel first-touch initialisation.
//
// std::vector<double>(n, 0.0) would have the main thread write every page, placing all pages on
// that thread's NUMA domain. Here the memory is allocated uninitialised and then zeroed by the
// same static (k,j) loop distribution that the compute kernels use (fill_zero), so each thread's
// part of the array is physically allocated near that thread ("first touch").
#pragma once
#include "kernels.hpp"
#include <cstdio>
#include <cstdlib>

template <int DIM>
class AlignedVec {
public:
    explicit AlignedVec(const Block<DIM>& B) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 64, sizeof(double) * (size_t)B.size) != 0) {
            std::fprintf(stderr, "out of memory (%ld doubles)\n", B.size);
            std::exit(5);
        }
        a_ = static_cast<double*>(ptr);
        fill_zero<DIM>(B, a_);
    }
    ~AlignedVec() { std::free(a_); }
    AlignedVec(const AlignedVec&) = delete;
    AlignedVec& operator=(const AlignedVec&) = delete;
    double* data() { return a_; }
    const double* data() const { return a_; }
    double& operator[](long i) { return a_[i]; }
    const double& operator[](long i) const { return a_[i]; }
private:
    double* a_ = nullptr;
};
