// halo.hpp -- halo (ghost-layer) exchange of the vector p between neighbouring ranks.
//
// The 5/7-point stencil needs only the face neighbours (no edges/corners), so each rank
// exchanges at most 2*DIM faces of one layer each. Faces are described with
// MPI_Type_create_subarray derived datatypes on the padded local array, so the
// non-contiguous x/y faces are sent directly without manual packing.
//
// Variants (runtime choice):
//   blocking     : MPI_Sendrecv, one axis after the other.
//   nonblocking  : MPI_Irecv/MPI_Isend for all faces at once, then MPI_Waitall. The caller can
//                  compute the interior (points not touching a ghost cell) between start() and
//                  wait() = communication/computation overlap, and the boundary shell after.
// (An RMA variant, MPI_Put + MPI_Win_fence, is added later.)
#pragma once
#include "common/kernels.hpp"
#include "common/split.hpp"
#include "decomp.hpp"
#include <vector>

template <int DIM>
class Halo {
public:
    explicit Halo(const Decomp<DIM>& D) : D_(D) {
        const Block<DIM>& B = D.B;
        for (int a = 0; a < DIM; ++a) {
            // layer positions along axis a in padded coordinates: ghost-lo = 0, first interior = 1,
            // last interior = n, ghost-hi = n+1
            make_face(B, a, 1, send_[a][0]);            // first interior layer -> lower neighbour
            make_face(B, a, B.n[a], send_[a][1]);       // last interior layer  -> upper neighbour
            make_face(B, a, 0, recv_[a][0]);            // lower ghost layer   <- lower neighbour
            make_face(B, a, B.n[a] + 1, recv_[a][1]);   // upper ghost layer   <- upper neighbour
        }
    }
    ~Halo() {
        for (int a = 0; a < DIM; ++a)
            for (int s = 0; s < 2; ++s) { MPI_Type_free(&send_[a][s]); MPI_Type_free(&recv_[a][s]); }
    }
    Halo(const Halo&) = delete;
    Halo& operator=(const Halo&) = delete;

    // ---- blocking variant ----
    void exchange_blocking(double* p) {
        for (int a = 0; a < DIM; ++a) {
            const int lo = D_.nbr[a][0], hi = D_.nbr[a][1];
            // upward: send last layer to hi, receive lower ghost from lo
            MPI_Sendrecv(p, 1, send_[a][1], hi, 2 * a, p, 1, recv_[a][0], lo, 2 * a, D_.cart,
                         MPI_STATUS_IGNORE);
            // downward: send first layer to lo, receive upper ghost from hi
            MPI_Sendrecv(p, 1, send_[a][0], lo, 2 * a + 1, p, 1, recv_[a][1], hi, 2 * a + 1, D_.cart,
                         MPI_STATUS_IGNORE);
        }
    }

    // ---- non-blocking variant: start() posts everything, wait() completes it ----
    void start(double* p) {
        nreq_ = 0;
        for (int a = 0; a < DIM; ++a)
            for (int s = 0; s < 2; ++s) {
                const int nb = D_.nbr[a][s];
                if (nb == MPI_PROC_NULL) continue;
                // ghost layer on side s is filled by neighbour s; tag identifies direction of travel:
                // data moving "up" (towards higher coordinate) uses 2a, "down" uses 2a+1
                MPI_Irecv(p, 1, recv_[a][s], nb, 2 * a + (s == 0 ? 0 : 1), D_.cart, &req_[nreq_++]);
            }
        for (int a = 0; a < DIM; ++a)
            for (int s = 0; s < 2; ++s) {
                const int nb = D_.nbr[a][s];
                if (nb == MPI_PROC_NULL) continue;
                // s==1: our last layer goes up (tag 2a); s==0: our first layer goes down (tag 2a+1)
                MPI_Isend(p, 1, send_[a][s], nb, 2 * a + (s == 1 ? 0 : 1), D_.cart, &req_[nreq_++]);
            }
    }
    void wait() { MPI_Waitall(nreq_, req_, MPI_STATUSES_IGNORE); nreq_ = 0; }

private:
    // Face of the padded local array: thickness 1 at padded position `pos` along axis `a`,
    // all interior points along the other axes. MPI arrays are C-ordered (last index fastest),
    // so our axis order (axis 0 fastest) is reversed.
    static void make_face(const Block<DIM>& B, int a, int pos, MPI_Datatype& t) {
        int sizes[3], sub[3], start[3];
        for (int e = 0; e < DIM; ++e) {
            const int m = DIM - 1 - e;                  // position in the C-ordered array
            sizes[m] = B.ext[e];
            sub[m] = (e == a) ? 1 : B.n[e];
            start[m] = (e == a) ? pos : 1;              // other axes: skip the ghost cell
        }
        MPI_Type_create_subarray(DIM, sizes, sub, start, MPI_ORDER_C, MPI_DOUBLE, &t);
        MPI_Type_commit(&t);
    }

    const Decomp<DIM>& D_;
    MPI_Datatype send_[3][2], recv_[3][2];   // [axis][0=lower side, 1=upper side]
    MPI_Request req_[4 * 3];
    int nreq_ = 0;
};
