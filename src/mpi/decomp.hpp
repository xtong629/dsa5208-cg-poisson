// decomp.hpp -- Cartesian domain decomposition of the global N^DIM grid over the ranks.
//
//   block : balanced DIM-dimensional process grid from MPI_Dims_create (2D blocks in 2D,
//           3D blocks in 3D). Halo surface per rank ~ N^(DIM-1) * n^(-1/DIM).
//   slab  : 1D decomposition, all splits along the slowest axis (y in 2D, z in 3D); faces are
//           whole planes of N^(DIM-1) points whatever the rank count -> surface does not shrink.
// Comparing the two measures the surface-to-volume effect on the halo exchange.
//
// Process-grid axes follow the array axes (axis 0 = x = fastest). MPI_Dims_create returns the
// counts in non-increasing order; we reverse them so the largest count lies on the slowest
// axis, which keeps the inner (x) loops long and the x-faces small.
#pragma once
#include "common/grid.hpp"
#include <algorithm>
#include <cstdio>
#include <mpi.h>

template <int DIM>
struct Decomp {
    MPI_Comm cart = MPI_COMM_NULL;
    int rank = 0, size = 1;
    int dims[3] = {1, 1, 1};
    int coords[3] = {0, 0, 0};
    int nbr[3][2] = {{MPI_PROC_NULL, MPI_PROC_NULL}, {MPI_PROC_NULL, MPI_PROC_NULL},
                     {MPI_PROC_NULL, MPI_PROC_NULL}};   // [axis][0=lower,1=upper]
    Block<DIM> B;                                         // this rank's local block

    Decomp(MPI_Comm world, int N, bool slab) {
        int sz;
        MPI_Comm_size(world, &sz);
        int d[3] = {0, 0, 0};
        if (slab) {
            for (int a = 0; a < DIM; ++a) d[a] = 1;
            d[DIM - 1] = sz;
        } else {
            MPI_Dims_create(sz, DIM, d);                 // non-increasing
            std::reverse(d, d + DIM);                    // largest count on the slowest axis
        }
        for (int a = 0; a < DIM; ++a) dims[a] = d[a];
        int periods[3] = {0, 0, 0};                      // Dirichlet boundary: not periodic
        MPI_Cart_create(world, DIM, dims, periods, /*reorder=*/0, &cart);
        MPI_Comm_rank(cart, &rank);
        MPI_Comm_size(cart, &size);
        MPI_Cart_coords(cart, rank, DIM, coords);
        for (int a = 0; a < DIM; ++a) MPI_Cart_shift(cart, a, 1, &nbr[a][0], &nbr[a][1]);

        // Uneven split: the first (N % dims) coordinates along an axis get one extra point.
        int n[3] = {1, 1, 1}, off[3] = {0, 0, 0};
        for (int a = 0; a < DIM; ++a) {
            if (N < dims[a]) {
                if (rank == 0) std::fprintf(stderr, "N=%d too small for %d ranks along axis %d\n", N, dims[a], a);
                MPI_Abort(world, 2);
            }
            const int q = N / dims[a], r = N % dims[a];
            n[a] = q + (coords[a] < r ? 1 : 0);
            off[a] = coords[a] * q + std::min(coords[a], r);
        }
        B.init(n, off);
    }
    ~Decomp() { if (cart != MPI_COMM_NULL) MPI_Comm_free(&cart); }
    Decomp(const Decomp&) = delete;
    Decomp& operator=(const Decomp&) = delete;

    // number of doubles this rank sends per halo exchange (one face per existing neighbour)
    long halo_doubles() const {
        long s = 0;
        for (int a = 0; a < DIM; ++a) {
            long face = 1;
            for (int e = 0; e < DIM; ++e) if (e != a) face *= B.n[e];
            for (int s2 = 0; s2 < 2; ++s2) if (nbr[a][s2] != MPI_PROC_NULL) s += face;
        }
        return s;
    }
};
