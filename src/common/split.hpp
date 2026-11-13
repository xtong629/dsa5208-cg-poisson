// split.hpp -- inner box / boundary shell split of a local block (shared by CPU and GPU codes).
#pragma once
#include "grid.hpp"
#include "kernels.hpp"
#include <algorithm>
#include <vector>

// Split the interior into the "inner" box (stencil needs no ghost cell) and the boundary shell
// (points on a face where a ghost value is needed), as disjoint boxes [lo,hi) in local interior
// coordinates. Used for overlap: compute inner while the halo is in flight, shell afterwards.
template <int DIM>
struct SplitBoxes {
    Box inner;
    std::vector<Box> shell;
    explicit SplitBoxes(const Block<DIM>& B) {
        for (int e = 0; e < 3; ++e) {
            inner.lo[e] = (e < DIM) ? 1 : 0;
            inner.hi[e] = (e < DIM) ? B.n[e] - 1 : 1;
            if (inner.hi[e] < inner.lo[e]) inner.hi[e] = inner.lo[e];   // empty if n < 3
        }
        // axis a contributes its lower and upper plane; lower axes are restricted to their inner
        // range (their planes were already taken), higher axes use the full range
        for (int a = 0; a < DIM; ++a)
            for (int s = 0; s < 2; ++s) {
                if (s == 1 && B.n[a] == 1) continue;       // single layer: lower plane is the whole thing
                Box b;
                for (int e = 0; e < 3; ++e) {
                    if (e >= DIM) { b.lo[e] = 0; b.hi[e] = 1; }
                    else if (e < a) { b.lo[e] = 1; b.hi[e] = std::max(1, B.n[e] - 1); }
                    else if (e > a) { b.lo[e] = 0; b.hi[e] = B.n[e]; }
                    else { b.lo[e] = (s == 0) ? 0 : B.n[e] - 1; b.hi[e] = b.lo[e] + 1; }
                }
                bool empty = false;
                for (int e = 0; e < 3; ++e) if (b.hi[e] <= b.lo[e]) empty = true;
                if (!empty) shell.push_back(b);
            }
    }
    // number of points covered; must equal the local interior count (self-check)
    long covered() const {
        auto vol = [](const Box& b) { return (long)(b.hi[0] - b.lo[0]) * (b.hi[1] - b.lo[1]) * (b.hi[2] - b.lo[2]); };
        long v = vol(inner);
        for (const Box& b : shell) v += vol(b);
        return v;
    }
};
