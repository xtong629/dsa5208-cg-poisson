// ops_mpi.hpp -- distributed operators for the shared CG loop (cg.hpp).
//   matvec : halo exchange of p, then stencil (blocking | non-blocking, with/without overlap)
//   dot    : local partial sum + MPI_Allreduce (the two global reductions of every CG iteration)
// Time accounting (see timers.hpp):
//   SEC_HALO      : time inside MPI for the halo exchange (Sendrecv, or Isend/Irecv posting + Waitall)
//   SEC_STENCIL   : stencil arithmetic (inner box + boundary shell in the overlapped variant)
//   SEC_ALLREDUCE : time in MPI_Allreduce. It includes waiting for slower ranks (load imbalance),
//                   because a collective cannot finish before every rank has arrived.
//   SEC_VECTOR    : axpy/xpay and the local part of the dot products
#pragma once
#include "common/ops_local.hpp"
#include "halo.hpp"

enum class HaloMode { Blocking, NonBlocking };

template <int DIM>
struct MpiOps : LocalOps<DIM> {
    const Decomp<DIM>& D;
    Halo<DIM>& H;
    HaloMode mode;
    bool overlap;
    SplitBoxes<DIM> split;

    MpiOps(const Decomp<DIM>& D_, Halo<DIM>& H_, int N, HaloMode m, bool ov)
        : LocalOps<DIM>(D_.B, N), D(D_), H(H_), mode(m), overlap(ov), split(D_.B) {}

    void matvec(double* p, double* Ap) {
        const Block<DIM>& B = this->B;
        if (mode == HaloMode::Blocking) {
            { ScopedTimer t(this->T, SEC_HALO); H.exchange_blocking(p); }
            { ScopedTimer t(this->T, SEC_STENCIL); stencil_all<DIM>(B, p, Ap, this->h2inv); }
        } else if (!overlap) {
            { ScopedTimer t(this->T, SEC_HALO); H.start(p); H.wait(); }
            { ScopedTimer t(this->T, SEC_STENCIL); stencil_all<DIM>(B, p, Ap, this->h2inv); }
        } else {
            // Hybrid design (MPI_THREAD_FUNNELED): the calling (master) thread posts all Isend/Irecv
            // outside any parallel region; then ALL threads compute the inner box; the master waits
            // (Waitall, again outside a parallel region); then all threads compute the shell.
            // Only the master thread ever calls MPI.
            { ScopedTimer t(this->T, SEC_HALO); H.start(p); }
            {   // inner points need no ghost data: compute them while the messages are in flight
                ScopedTimer t(this->T, SEC_STENCIL);
                stencil_range<DIM>(B, p, Ap, split.inner.lo, split.inner.hi, this->h2inv);
            }
            { ScopedTimer t(this->T, SEC_HALO); H.wait(); }   // exposed (non-hidden) comm time
            {
                ScopedTimer t(this->T, SEC_STENCIL);
                stencil_boxes<DIM>(B, p, Ap, split.shell.data(), (int)split.shell.size(), this->h2inv);
            }
        }
    }

    double dot(const double* a, const double* b) {
        double loc;
        { ScopedTimer t(this->T, SEC_VECTOR); loc = dot_local<DIM>(this->B, a, b); }
        double glob;
        { ScopedTimer t(this->T, SEC_ALLREDUCE); MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, D.cart); }
        return glob;
    }
};
