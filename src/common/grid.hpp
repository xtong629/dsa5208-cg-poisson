// grid.hpp -- local block of the global N^DIM grid, stored with a ghost layer.
//
// Global problem: N interior points per axis on the unit square/cube, spacing h = 1/(N+1),
// homogeneous Dirichlet boundary (u = 0 on the boundary), so N^DIM unknowns.
//
// Each process (rank) owns a box of n[0] x n[1] x n[2] interior points starting at global
// index off[]. In 2D the third axis has n[2] = 1 and no ghost layer.
// Storage is row-major with axis 0 (x) fastest: contiguous inner loops over i.
// Every vector uses the same padded layout (1 ghost cell on each side of each active axis);
// the ghost cells of the vector p hold either neighbour data (halo, MPI) or 0 (the Dirichlet
// boundary), so the stencil needs no boundary special-casing.
#pragma once
#include <cstddef>

template <int DIM>
struct Block {
    static_assert(DIM == 2 || DIM == 3, "DIM must be 2 or 3");
    int n[3];      // local interior points per axis (1 for unused axes)
    int off[3];    // global index (0-based, interior numbering) of the first local point
    int ext[3];    // padded extents = n + 2*ghost
    long str[3];   // strides in elements: str[0]=1
    long size;     // total padded element count
    static constexpr int ghost(int d) { return d < DIM ? 1 : 0; }

    Block() = default;
    Block(const int n_[3], const int off_[3]) { init(n_, off_); }
    void init(const int n_[3], const int off_[3]) {
        for (int d = 0; d < 3; ++d) {
            n[d] = n_[d];
            off[d] = off_[d];
            ext[d] = n[d] + 2 * ghost(d);
        }
        str[0] = 1;
        str[1] = ext[0];
        str[2] = (long)ext[0] * ext[1];
        size = str[2] * ext[2];
    }
    // linear index of interior point (i,j,k), 0-based local interior coordinates
    inline long idx(int i, int j, int k) const {
        return (long)(i + ghost(0)) + (long)(j + ghost(1)) * str[1] + (long)(k + ghost(2)) * str[2];
    }
    long interior_count() const { return (long)n[0] * n[1] * n[2]; }
};
