#!/usr/bin/env python3
"""validate.py -- the small set of correctness tests (run locally and again on the cluster).

  python3 scripts/validate.py serial          test 1 (serial)
  python3 scripts/validate.py mpi             tests 1-4 for build/mpi   (needs mpirun)
  python3 scripts/validate.py hybrid          same tests for build/hybrid with OMP threads
  python3 scripts/validate.py all             serial + mpi (+ hybrid if built)

Tests
  1  Discretisation error decreases as O(h^2): observed order between consecutive grids in (1.9, 2.1),
     for serial and for a parallel run (hybrid: 2 ranks x 2 threads).  CG must have converged and the recurrence residual must
     match the explicitly recomputed ||b - A x||.
  2  Parallel == serial: relative difference ||x - x_serial|| / ||x_serial|| below TOL_SOL for
     several rank counts, both decompositions (slab / block) and all halo variants.
  3  Iteration counts equal to the serial count within +-ITER_SLACK (global sums are
     re-associated when the rank count changes, so rounding differs; see Lecture 8).
  4  All halo variants (blocking, non-blocking with/without overlap) agree with each other.
  5  Awkward sizes: N not divisible by the process grid, and subdomains thinner than 3 points
     (no inner region, shell covers everything).
Environment: MPIRUN (default "mpirun"), MPIRUN_FLAGS (e.g. "--oversubscribe"), VALIDATE_NP (rank
counts, default "1,2,4,8"), OMP_THREADS (hybrid).
"""
import csv, math, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPIRUN = os.environ.get("MPIRUN", "mpirun").split()
MPIFLAGS = os.environ.get("MPIRUN_FLAGS", "").split()
NPS = [int(x) for x in os.environ.get("VALIDATE_NP", "1,2,4,8").split(",")]
THREADS = [int(x) for x in os.environ.get("OMP_THREADS", "1,2,3,4").split(",")]
NCPU = os.cpu_count() or 1   # hybrid combos with ranks*threads > NCPU are skipped (OpenMP threads would spin)
TOL_SOL = 1e-9       # relative difference of the solutions (solver tol 1e-13 -> ~1e-11 expected)
ITER_SLACK = 2
failures = []
TMP = tempfile.mkdtemp()


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def run(cmd, env=None):
    e = dict(os.environ)
    if env: e.update(env)
    r = subprocess.run(cmd, cwd=ROOT, env=e, capture_output=True, text=True)
    if r.returncode != 0:
        print("COMMAND FAILED:", " ".join(cmd), "\n", r.stdout, r.stderr)
        failures.append("command failed: " + " ".join(cmd))
        return False
    return True


def last_row(path):
    return list(csv.DictReader(open(path)))[-1]


def serial_ref(dim, N, tol="1e-13"):
    """run the serial program; returns (row, dump file)"""
    c = os.path.join(TMP, f"s_{dim}_{N}.csv"); d = os.path.join(TMP, f"s_{dim}_{N}.bin")
    if os.path.exists(c): os.remove(c)
    run(["build/serial", "--dim", str(dim), "--N", str(N), "--tol", tol, "--csv", c, "--dump", d])
    return last_row(c), d


def par(binary, np_, dim, N, decomp, halo, overlap, dump, tol="1e-13", threads=1):
    c = os.path.join(TMP, "p.csv")
    if os.path.exists(c): os.remove(c)
    cmd = MPIRUN + MPIFLAGS + ["-np", str(np_), f"build/{binary}", "--dim", str(dim), "--N", str(N),
           "--decomp", decomp, "--halo", halo, "--overlap", str(overlap), "--tol", tol,
           "--compare", dump, "--csv", c]
    ok = run(cmd, {"OMP_NUM_THREADS": str(threads), "OMP_WAIT_POLICY": "passive"})
    return last_row(c) if ok else None


def order_test(rows, label):
    for a, b in zip(rows, rows[1:]):
        h1, h2 = 1 / (int(a["N"]) + 1), 1 / (int(b["N"]) + 1)
        for key in ("err_l2", "err_max"):
            p = math.log(float(a[key]) / float(b[key])) / math.log(h1 / h2)
            check(1.9 < p < 2.1, f"{label} N {a['N']}->{b['N']} {key}: observed order {p:.3f}")
    for r in rows:
        check(r["converged"] == "1" and float(r["true_rel_resid"]) < 1e-9,
              f"{label} N={r['N']}: converged, true residual {float(r['true_rel_resid']):.1e}")


def test1(binary):
    print("[test 1] O(h^2) discretisation error")
    sizes = {2: [15, 31, 63, 127, 255], 3: [7, 15, 31, 63]}
    for dim in (2, 3):
        c = os.path.join(TMP, f"o_{binary}_{dim}.csv")
        if os.path.exists(c): os.remove(c)
        for N in sizes[dim]:
            if binary == "serial":
                run(["build/serial", "--dim", str(dim), "--N", str(N), "--tol", "1e-12", "--csv", c])
            else:
                run(MPIRUN + MPIFLAGS + ["-np", "2" if binary == "hybrid" else "4", f"build/{binary}",
                     "--dim", str(dim), "--N", str(N), "--tol", "1e-12", "--csv", c],
                    {"OMP_NUM_THREADS": "2", "OMP_WAIT_POLICY": "passive"})
        order_test(list(csv.DictReader(open(c))), f"{binary} {dim}D")


def tests_parallel(binary):
    threads_list = THREADS if binary == "hybrid" else [1]
    variants = [("blocking", 0), ("nonblocking", 0), ("nonblocking", 1)]
    print(f"[tests 2-4] {binary}: parallel vs serial, decompositions x halo variants")
    for dim, N in ((2, 127), (3, 31)):
        sref, dump = serial_ref(dim, N)
        it_s = int(sref["iters"])
        for th in threads_list:
            for np_ in NPS:
                if np_ * th > NCPU: continue
                for decomp in ("slab", "block"):
                    errs = []
                    for halo, ov in variants:
                        r = par(binary, np_, dim, N, decomp, halo, ov, dump, threads=th)
                        if r is None: continue
                        d, it = float(r["diff_serial"]), int(r["iters"])
                        if binary == "hybrid":
                            check(r["thread_level"] == "funneled" and int(r["threads"]) == th,
                                  f"hybrid np={np_} th={th}: MPI thread level {r['thread_level']}, threads {r['threads']}")
                        errs.append(float(r["err_l2"]))
                        tag = f"{dim}D N={N} np={np_} th={th} {decomp:5s} {halo}{'+overlap' if ov else ''}"
                        check(d < TOL_SOL, f"{tag}: diff vs serial {d:.2e}")
                        check(abs(it - it_s) <= ITER_SLACK, f"{tag}: iters {it} (serial {it_s})")
                    if errs:
                        check(max(errs) - min(errs) <= 1e-10 * max(errs),
                              f"{dim}D np={np_} th={th} {decomp}: halo variants agree on err_L2")


def test5(binary):
    th = 2 if binary == "hybrid" else 1
    print(f"[test 5] {binary}: uneven / tiny subdomains (threads={th})")
    cases = [(2, 4, 8), (2, 10, 8), (2, 50, 6), (3, 4, 8), (3, 5, 8), (3, 5, 6), (3, 29, 8), (3, 7, 4)]
    for dim, N, np_ in cases:
        sref, dump = serial_ref(dim, N)
        for decomp in ("block", "slab"):
            if decomp == "slab" and N < np_: continue
            for halo, ov in (("blocking", 0), ("nonblocking", 1)):
                r = par(binary, np_, dim, N, decomp, halo, ov, dump, threads=th)
                if r is None: continue
                check(float(r["diff_serial"]) < TOL_SOL and abs(int(r["iters"]) - int(sref["iters"])) <= ITER_SLACK,
                      f"{dim}D N={N} np={np_} {decomp} {halo}{'+ov' if ov else ''}: diff {float(r['diff_serial']):.1e}, iters {r['iters']} (serial {sref['iters']})")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    bins = []
    if what in ("serial", "all"): bins.append("serial")
    if what in ("mpi", "all"): bins.append("mpi")
    if what == "hybrid" or (what == "all" and os.path.exists(os.path.join(ROOT, "build/hybrid"))): bins.append("hybrid")
    for b in bins:
        if not os.path.exists(os.path.join(ROOT, "build", b)):
            print(f"build/{b} not found - run `make {b}` first"); sys.exit(2)
        test1(b)
        if b != "serial":
            tests_parallel(b)
            test5(b)
    print("\nALL VALIDATION TESTS PASSED" if not failures else f"\nVALIDATION FAILED ({len(failures)} failures)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
