#!/usr/bin/env python3
"""plot.py -- figures and summary tables from the result CSVs (median over repetitions).

  python3 scripts/plot.py --results results/atlas [--results results/vanda] --out figures
  python3 scripts/plot.py --results results/local_smoke --out figures_localtest --label "LOCAL TEST DATA"

Needs only numpy + matplotlib (python >= 3.9). Time, problem-size and rank-count axes are
logarithmic; efficiency and time-fraction axes are linear in y (log in x).

Terminology (Lecture 7):  T_s = runtime of the SERIAL program, T_p(n) = runtime on n ranks,
  speedup = T_s / T_p(n),  efficiency = T_s / (n T_p(n)).
  Strong scaling: fixed N, increase n.  Weak scaling: N grows with n (fixed points per rank).
Definitions of the non-standard quantities used in the figures:
  * time per iteration      = total solve time / number of CG iterations (timing runs use a fixed count)
  * weak-scaling efficiency = [T_s,iter(base) / points(base)] / [T_p,iter(n) / (points(n)/n)], i.e. time per
                              point-per-rank relative to the serial program; the problem size N(n) is rounded
                              to an integer, so points per rank are only approximately constant and this
                              normalisation removes that rounding effect.
  * communication fraction  = (halo exchange + Allreduce time) / total time (mean over ranks).
  * Amdahl fit              = least-squares fit (in log space) of the serial fraction S in
                              speedup = 1 / (S + (1-S)/n) to the measured strong-scaling speedup.
"""
import argparse, csv, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 14, "axes.titlesize": 13, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10, "figure.titlesize": 14})
from matplotlib.ticker import FuncFormatter, ScalarFormatter

NUMERIC = {"dim", "N", "unknowns", "ranks", "threads", "overlap", "iters", "converged", "rel_resid",
           "true_rel_resid", "err_l2", "err_max", "diff_serial", "halo_doubles", "local_points", "t_total",
           "t_per_iter", "t_stencil", "t_halo", "t_allreduce", "t_vector", "t_other", "rep",
           "reps", "t_allreduce_us"}
# colour-blind-safe palette (Okabe-Ito)
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]
SECTIONS = [("t_stencil", "stencil"), ("t_halo", "halo exchange"), ("t_allreduce", "Allreduce (incl. wait)"),
            ("t_vector", "vector updates"), ("t_other", "other")]
LABEL = None   # watermark text for test data


# ----------------------------------------------------------------------------- data helpers
def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                for k in NUMERIC:
                    if k in r and r[k] != "":
                        try: r[k] = float(r[k])
                        except ValueError: pass
                rows.append(r)
    return rows


def sel(rows, **kw):
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            if callable(v):
                if not v(r.get(k)): ok = False; break
            elif r.get(k) != v: ok = False; break
        if ok: out.append(r)
    return out


def group_median(rows, keyf, fields):
    """{key: {field: median, 'n': count}} -- median over repetitions."""
    g = {}
    for r in rows: g.setdefault(keyf(r), []).append(r)
    out = {}
    for k, rs in g.items():
        d = {f: float(np.median([x[f] for x in rs if isinstance(x.get(f), float)])) for f in fields
             if any(isinstance(x.get(f), float) for x in rs)}
        d["n"] = len(rs)
        out[k] = d
    return out


def per_iter(d, field):
    return d[field] / d["iters"]


def serial_times(rows):
    """T_s(dim, N) from the serial program, fixed-iteration runs (median over repetitions)."""
    s = sel(rows, program="serial", tag="serial_fixed")
    return group_median(s, lambda r: (int(r["dim"]), int(r["N"])), ["t_total", "iters"])


def pow2_axis(ax, which="x"):
    getattr(ax, "set_%sscale" % which)("log", base=2)
    fmt = FuncFormatter(lambda v, _: ("%d" % v) if v >= 1 else "%g" % v)
    getattr(ax, "%saxis" % which).set_major_formatter(fmt)


def finish(fig, out, name, title=None, bottom=0.02):
    if title: fig.suptitle(title, fontsize=14)
    if LABEL:
        fig.text(0.5, 0.5, LABEL, fontsize=34, color="red", alpha=0.18, ha="center", va="center", rotation=25)
        fig.text(0.01, 0.005, LABEL + " -- not results for the report", fontsize=7, color="red")
    fig.tight_layout(rect=(0, bottom, 1, 0.96 if title else 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "%s.%s" % (name, ext)), dpi=160)
    plt.close(fig)
    print("wrote", os.path.join(out, name + ".png"))


def amdahl_fit(n, s):
    """serial fraction S minimising squared log error of speedup=1/(S+(1-S)/n)."""
    n, s = np.asarray(n, float), np.asarray(s, float)
    grid = np.concatenate(([0.0], np.logspace(-6, 0, 600)))
    err = [np.sum((np.log(s) - np.log(1.0 / (g + (1 - g) / n))) ** 2) for g in grid]
    return float(grid[int(np.argmin(err))])


def new_fig(ncols, w=3.8, h=3.3, nrows=1):
    return plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)


# ----------------------------------------------------------------------------- figures
def fig_serial(rows, out):
    s = sel(rows, program="serial")
    if not s: return
    fixed = group_median(sel(s, tag="serial_fixed"), lambda r: (int(r["dim"]), int(r["unknowns"])), ["t_total", "iters"])
    conv = group_median(sel(s, tag="serial_conv"), lambda r: (int(r["dim"]), int(r["unknowns"])), ["t_total", "iters"])
    fig, axs = new_fig(2)
    for i, dim in enumerate((2, 3)):
        pts = sorted(k for k in fixed if k[0] == dim)
        if pts:
            x = np.array([k[1] for k in pts], float); y = np.array([per_iter(fixed[k], "t_total") for k in pts])
            axs[0][0].loglog(x, y, "o-", color=C[i], label="%dD (%d-point stencil)" % (dim, 5 if dim == 2 else 7))
            axs[0][0].loglog(x, y[0] * x / x[0], "--", color=C[i], alpha=0.5, lw=1)
        pts = sorted(k for k in conv if k[0] == dim)
        if pts:
            x = np.array([k[1] for k in pts], float); y = np.array([conv[k]["t_total"] for k in pts])
            axs[0][1].loglog(x, y, "s-", color=C[i], label="%dD, to tolerance" % dim)
            axs[0][1].loglog(x, y[0] * (x / x[0]) ** (1 + 1.0 / dim), "--", color=C[i], alpha=0.5, lw=1)
    axs[0][0].set(xlabel="problem size (number of unknowns)", ylabel="time per CG iteration [s]",
                  title="(a) cost of one iteration\n(dashed: linear in unknowns)")
    axs[0][1].set(xlabel="problem size (number of unknowns)", ylabel="time to solution [s]",
                  title="(b) run to tolerance\n(dashed: unknowns^(1+1/d))")
    for a in axs[0]: a.grid(True, which="both", alpha=0.25); a.legend(fontsize=10)
    finish(fig, out, "fig_serial_scaling", "Serial runtime vs problem size")


def strong_table(rows, ts):
    st = sel(rows, program="mpi", tag="strong")
    g = group_median(st, lambda r: (int(r["dim"]), int(r["N"]), int(r["ranks"])), ["t_total", "iters"])
    table = []
    for (dim, N, n), d in sorted(g.items()):
        t_s = ts.get((dim, N), {}).get("t_total")
        row = {"dim": dim, "N": N, "ranks": n, "T_p": d["t_total"], "iters": d["iters"], "T_s": t_s}
        if t_s: row["speedup"] = t_s / d["t_total"]; row["efficiency"] = t_s / (n * d["t_total"])
        table.append(row)
    return table


def write_table(rows, path, cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows: w.writerow([("%.6g" % r[c]) if isinstance(r.get(c), float) else r.get(c, "") for c in cols])
    print("wrote", path)


def fig_strong(rows, out, tabdir):
    ts = serial_times(rows)
    table = strong_table(rows, ts)
    if not table: return
    write_table(table, os.path.join(tabdir, "strong_scaling.csv"),
                ["dim", "N", "ranks", "iters", "T_s", "T_p", "speedup", "efficiency"])
    # extra reference: MPI with 1 rank (not the T_s of the definition)
    for dim in (2, 3):
        sizes = sorted({r["N"] for r in table if r["dim"] == dim})
        if not sizes: continue
        fig, axs = new_fig(3, w=3.5, h=3.8)
        for i, N in enumerate(sizes):
            tr = [r for r in table if r["dim"] == dim and r["N"] == N]
            n = np.array([r["ranks"] for r in tr], float)
            tp = np.array([r["T_p"] for r in tr])
            lab = "N=%d  (%.3g unknowns)" % (N, float(N) ** dim)
            axs[0][0].loglog(n, tp, "o-", color=C[i], label=lab)
            if tr[0].get("T_s"):
                axs[0][0].axhline(tr[0]["T_s"], color=C[i], ls=":", lw=1)   # T_s of the serial program
                sp = np.array([r["speedup"] for r in tr]); ef = np.array([r["efficiency"] for r in tr])
                axs[0][1].loglog(n, sp, "o-", color=C[i], label=lab)
                axs[0][2].semilogx(n, ef, "o-", color=C[i], label=lab)
                if len(n) >= 3 and sp.max() > 1.0:
                    S = amdahl_fit(n, sp)
                    nn = np.logspace(0, np.log2(n.max()), 60, base=2)
                    axs[0][1].loglog(nn, 1 / (S + (1 - S) / nn), "--", color=C[i], alpha=0.6, lw=1,
                                     label="Amdahl fit S=%.2g" % S)
            nn1 = [r for r in rows if r.get("program") == "mpi" and r.get("tag") == "strong" and r["dim"] == dim
                   and r["N"] == N and r["ranks"] == 1]
            if nn1 and tr[0].get("T_s"):   # MPI with 1 rank: extra reference only
                t1 = float(np.median([r["t_total"] for r in nn1]))
                axs[0][1].plot([1], [tr[0]["T_s"] / t1], "x", color=C[i], ms=8)
        nmax = max(r["ranks"] for r in table if r["dim"] == dim)
        nn = np.array([1, nmax])
        axs[0][1].loglog(nn, nn, "k:", lw=1, label="ideal")
        for a in axs[0]: pow2_axis(a); a.set_xlabel("MPI ranks n"); a.grid(True, which="both", alpha=0.25)
        axs[0][0].set(ylabel="runtime T_p(n) [s]",
                      title="(a) runtime, fixed iteration count\n(dotted: serial T_s)")
        axs[0][1].set(ylabel="speedup T_s / T_p(n)", title="(b) speedup (x: MPI on 1 rank)")
        axs[0][2].set(ylabel="efficiency T_s / (n T_p(n))", title="(c) efficiency"); axs[0][2].axhline(1, color="k", ls=":", lw=1)
        axs[0][2].set_ylim(0, 1.3)
        axs[0][0].legend(fontsize=10); axs[0][1].legend(fontsize=10)
        finish(fig, out, "fig_strong_%dD" % dim, "Strong scaling, %dD Poisson, block decomposition, non-blocking + overlap" % dim)


def fig_weak(rows, out, tabdir):
    ts = serial_times(rows)
    w = [r for r in sel(rows, program="mpi") if str(r.get("tag", "")).startswith("weak_base")]
    if not w: return
    table = []
    for dim in (2, 3):
        bases = sorted({r["tag"] for r in w if r["dim"] == dim})
        if not bases: continue
        fig, axs = new_fig(2, w=4.6)
        for i, tag in enumerate(bases):
            g = group_median(sel(w, tag=tag, dim=dim), lambda r: (int(r["ranks"]), int(r["N"])),
                             ["t_total", "iters", "unknowns", "t_halo", "t_allreduce"])
            keys = sorted(g)
            n = np.array([k[0] for k in keys], float)
            ti = np.array([per_iter(g[k], "t_total") for k in keys])
            pts = np.array([g[k]["unknowns"] for k in keys])
            base = int(tag.replace("weak_base", ""))
            axs[0][0].loglog(n, ti, "o-", color=C[i], label="%d^%d points per rank" % (base, dim))
            t_s = ts.get((dim, base))
            ts_iter = pts_s = None
            note = ""
            if t_s:
                ts_iter = t_s["t_total"] / t_s["iters"]; pts_s = float(base) ** dim
            elif 1 in [k[0] for k in keys]:   # no serial run at this size: fall back to MPI on 1 rank
                k1 = [k for k in keys if k[0] == 1][0]
                ts_iter = per_iter(g[k1], "t_total"); pts_s = g[k1]["unknowns"]
                note = " (baseline: MPI 1 rank)"
                print("warning: no serial run for %dD N=%d; weak-scaling baseline is MPI with 1 rank" % (dim, base))
            if ts_iter:
                eff = (ts_iter / pts_s) / (ti / (pts / n))
                axs[0][0].axhline(ts_iter, color=C[i], ls=":", lw=1)
                axs[0][1].semilogx(n, eff, "o-", color=C[i], label="%d^%d points per rank%s" % (base, dim, note))
                for k, e, t in zip(keys, eff, ti):
                    table.append({"dim": dim, "base": base, "ranks": k[0], "N": k[1], "t_iter": float(t),
                                  "weak_efficiency": float(e), "scaled_speedup": float(e) * k[0]})
        for a in axs[0]:
            pow2_axis(a); a.set_xlabel("MPI ranks n"); a.grid(True, which="both", alpha=0.25)
            if a.get_legend_handles_labels()[0]: a.legend(fontsize=10)
        axs[0][0].set(ylabel="time per CG iteration [s]", title="(a) time per iteration")
        axs[0][1].set(ylabel="weak-scaling efficiency", title="(b) efficiency")
        axs[0][1].axhline(1, color="k", ls=":", lw=1); axs[0][1].set_ylim(0, 1.3)
        finish(fig, out, "fig_weak_%dD" % dim, "Weak scaling, %dD Poisson (fixed points per rank)" % dim)
    if table: write_table(table, os.path.join(tabdir, "weak_scaling.csv"),
                          ["dim", "base", "ranks", "N", "t_iter", "weak_efficiency", "scaled_speedup"])


def fig_comm(rows, out):
    st = sel(rows, program="mpi", tag="strong")
    if not st: return
    for dim in (2, 3):
        sizes = sorted({int(r["N"]) for r in st if r["dim"] == dim})
        if not sizes: continue
        fig, axs = new_fig(len(sizes), w=max(5.0, 8.0 / len(sizes)), h=3.4, nrows=2)
        for j, N in enumerate(sizes):
            g = group_median(sel(st, dim=dim, N=N), lambda r: int(r["ranks"]),
                             [f for f, _ in SECTIONS] + ["t_total", "iters"])
            ns = sorted(g)
            n = np.array(ns, float)
            ax = axs[0][j]
            for i, (f, lab) in enumerate(SECTIONS):
                y = np.array([g[k].get(f, 0.0) / g[k]["iters"] for k in ns])
                if np.all(y > 0): ax.loglog(n, y, "o-", color=C[i], label=lab)
            ax.loglog(n, [g[k]["t_total"] / g[k]["iters"] for k in ns], "k--", label="total")
            pow2_axis(ax); ax.set(xlabel="MPI ranks n", ylabel="time per CG iteration [s]", title="N=%d" % N)
            ax.grid(True, which="both", alpha=0.25)
            ax2 = axs[1][j]
            tot = np.array([g[k]["t_total"] for k in ns])
            parts = [("t_stencil", "stencil"), ("t_vector", "vector updates"), ("t_halo", "halo exchange"),
                     ("t_allreduce", "Allreduce (incl. wait)"), ("t_other", "other")]
            colors = {"t_stencil": C[0], "t_vector": C[3], "t_halo": C[1], "t_allreduce": C[2], "t_other": C[4]}
            bottom = np.zeros(len(ns)); xs = np.log2(n); width = 0.7 if len(ns) < 2 else 0.8 * min(np.diff(xs))
            for f, lab in parts:
                y = np.array([g[k].get(f, 0.0) for k in ns]) / tot
                ax2.bar(xs, y, width, bottom=bottom, color=colors[f], label=lab); bottom += y
            ax2.set_xticks(xs); ax2.set_xticklabels(["%d" % v for v in n])
            ax2.set(xlabel="MPI ranks n (log2 spacing)", ylabel="fraction of total runtime\n(mean over ranks)", ylim=(0, 1.02))
        hs, ls = axs[0][0].get_legend_handles_labels()   # one shared legend below the panels (not on top of the data)
        fig.legend(hs, ls, loc="lower center", ncol=6, fontsize=10, frameon=False)
        finish(fig, out, "fig_comm_breakdown_%dD" % dim,
               "Where the time goes, %dD (strong scaling)" % dim, bottom=0.07)


def fig_decomp(rows, out):
    d = sel(rows, program="mpi", tag="decomp_halo")
    if not d: return
    for dim in (2, 3):
        dd = sel(d, dim=dim)
        if not dd: continue
        fig, axs = new_fig(3, w=3.5, h=3.8)
        for i, dec in enumerate(("slab", "block")):
            # surface (values sent per halo exchange) and exposed halo time, blocking variant
            g = group_median(sel(dd, decomp=dec, halo="blocking"), lambda r: int(r["ranks"]),
                             ["halo_doubles", "t_halo", "t_total", "iters", "local_points"])
            if not g: continue
            ns = sorted(g); n = np.array(ns, float)
            axs[0][0].loglog(n, [g[k]["halo_doubles"] for k in ns], "o-", color=C[i], label=dec)
            axs[0][1].loglog(n, [g[k]["t_halo"] / g[k]["iters"] for k in ns], "o-", color=C[i], label=dec)
            axs[0][2].loglog(n, [g[k]["t_total"] / g[k]["iters"] for k in ns], "o-", color=C[i], label=dec)
        for a in axs[0]: pow2_axis(a); a.set_xlabel("MPI ranks n"); a.grid(True, which="both", alpha=0.25); a.legend(fontsize=10)
        axs[0][0].set(ylabel="doubles per halo exchange", title="(a) halo surface")
        axs[0][1].set(ylabel="halo time per iteration [s]", title="(b) communication time")
        axs[0][2].set(ylabel="time per iteration [s]", title="(c) total time")
        finish(fig, out, "fig_decomposition_%dD" % dim, "Slab (1D) vs block (%dD) decomposition, %dD, blocking halo exchange" % (dim, dim))
        # halo variants
        fig, axs = new_fig(2, w=5.0)
        names = [("blocking", 0, "blocking Sendrecv"), ("nonblocking", 0, "non-blocking, no overlap"),
                 ("nonblocking", 1, "non-blocking + overlap")]
        for j, dec in enumerate(("slab", "block")):
            ax = axs[0][j]
            nsall = sorted({int(r["ranks"]) for r in dd if r["decomp"] == dec})
            for v, (h, ov, lab) in enumerate(names):
                g = group_median(sel(dd, decomp=dec, halo=h, overlap=float(ov)), lambda r: int(r["ranks"]), ["t_total", "iters"])
                xs = np.arange(len(nsall)) + (v - 1) * 0.27
                y = [g[k]["t_total"] / g[k]["iters"] if k in g else np.nan for k in nsall]
                ax.bar(xs, y, 0.26, color=C[v], label=lab)
            ax.set_yscale("log"); ax.set_xticks(range(len(nsall))); ax.set_xticklabels(["%d" % k for k in nsall])
            ax.set(xlabel="MPI ranks n", ylabel="time per iteration [s]", title=dec)
            ax.grid(True, which="both", axis="y", alpha=0.25)
            if j == 0: ax.legend(fontsize=10)
        finish(fig, out, "fig_halo_variants_%dD" % dim, "Halo-exchange variants, %dD" % dim)


def fig_allreduce(rows, out, root_dirs):
    rr = []
    for d in root_dirs:
        p = os.path.join(d, "allreduce.csv")
        if os.path.exists(p): rr += load([p])
    st = sel(rows, program="mpi", tag="strong")
    if not rr: return
    g = group_median(rr, lambda r: int(r["ranks"]), ["t_allreduce_us"])
    ns = sorted(g); n = np.array(ns, float)
    fig, axs = new_fig(1, w=6.0, h=4.2)
    ax = axs[0][0]
    ax.loglog(n, [g[k]["t_allreduce_us"] for k in ns], "o-", color=C[0], label="micro-benchmark: MPI_Allreduce, 1 double")
    for i, dim in enumerate((2, 3)):
        s = group_median(sel(st, dim=dim), lambda r: (int(r["N"]), int(r["ranks"])), ["t_allreduce", "iters"])
        for j, Nn in enumerate(sorted({k[0] for k in s})):
            ks = sorted(k for k in s if k[0] == Nn)
            # 2 Allreduces per CG iteration (+1 at start, ignored)
            ax.loglog([k[1] for k in ks], [s[k]["t_allreduce"] / (2.0 * s[k]["iters"]) * 1e6 for k in ks], "s--",
                      color=C[1 + i], alpha=0.5 + 0.25 * j, ms=4, label="inside CG, %dD N=%d (incl. wait)" % (dim, Nn))
    pow2_axis(ax); ax.set(xlabel="MPI ranks n", ylabel="time per Allreduce [microseconds]", title="Allreduce latency vs rank count")
    ax.grid(True, which="both", alpha=0.25); ax.set_ylim(top=2e4); ax.legend(fontsize=9, loc="upper left")
    finish(fig, out, "fig_allreduce_latency")


def fig_hybrid(rows, out):
    h = [r for r in rows if str(r.get("tag", "")).startswith("hybrid_total")]
    if not h: return
    for dim in (2, 3):
        hd = [r for r in h if r["dim"] == dim]
        totals = sorted({int(str(r["tag"]).replace("hybrid_total", "").replace("_pure_mpi", "")) for r in hd})
        if not totals: continue
        fig, axs = new_fig(2, w=4.8)
        for i, tot in enumerate(totals):
            hy = sel(hd, tag="hybrid_total%d" % tot)
            g = group_median(hy, lambda r: int(r["threads"]), ["t_total", "iters", "t_halo", "t_allreduce"])
            ths = sorted(g); x = np.array(ths, float)
            axs[0][0].loglog(x, [g[k]["t_total"] / g[k]["iters"] for k in ths], "o-", color=C[i], label="%d cores (ranks x threads)" % tot)
            axs[0][1].semilogx(x, [(g[k]["t_halo"] + g[k]["t_allreduce"]) / g[k]["t_total"] for k in ths], "o-", color=C[i], label="%d cores" % tot)
            ref = sel(hd, tag="hybrid_total%d_pure_mpi" % tot)
            if ref:
                axs[0][0].axhline(float(np.median([r["t_total"] / r["iters"] for r in ref])), color=C[i], ls=":", lw=1)
        for a in axs[0]: pow2_axis(a); a.set_xlabel("OpenMP threads per rank"); a.grid(True, which="both", alpha=0.25); a.legend(fontsize=10)
        axs[0][0].set(ylabel="time per iteration [s]", title="(a) fixed total cores (dotted: pure MPI binary)")
        axs[0][1].set(ylabel="communication fraction", title="(b) (halo + Allreduce) / total")
        finish(fig, out, "fig_hybrid_%dD" % dim, "MPI+OpenMP configurations at fixed core count, %dD" % dim)


def fig_gpu(rows, out):
    gs = [r for r in rows if r.get("tag") in ("gpu_1gpu", "cpu_node")]
    if gs:
        fig, axs = new_fig(2, w=4.8)
        for i, dim in enumerate((2, 3)):
            gg = group_median(sel(gs, tag="gpu_1gpu", dim=dim), lambda r: int(r["unknowns"]), ["t_total", "iters"])
            cc = group_median(sel(gs, tag="cpu_node", dim=dim), lambda r: int(r["unknowns"]), ["t_total", "iters", "ranks"])
            if gg:
                ks = sorted(gg); axs[0][0].loglog(ks, [gg[k]["t_total"] / gg[k]["iters"] for k in ks], "o-", color=C[i], label="%dD, 1 GPU" % dim)
            if cc:
                ks = sorted(cc); axs[0][0].loglog(ks, [cc[k]["t_total"] / cc[k]["iters"] for k in ks], "s--", color=C[i], label="%dD, CPU node (%d ranks)" % (dim, int(cc[ks[0]]["ranks"])))
            both = sorted(set(gg) & set(cc))
            if both: axs[0][1].loglog(both, [cc[k]["t_total"] / gg[k]["t_total"] for k in both], "o-", color=C[i], label="%dD" % dim)
        axs[0][0].set(xlabel="problem size (number of unknowns)", ylabel="time per iteration [s]", title="(a) 1 GPU vs 1 CPU node")
        axs[0][1].set(xlabel="problem size (number of unknowns)", ylabel="GPU speedup over CPU node", title="(b) speedup")
        for a in axs[0]: a.grid(True, which="both", alpha=0.25); a.legend(fontsize=10)
        finish(fig, out, "fig_gpu_vs_cpu", "GPU (CUDA + MPI) vs CPU node")
    gm = [r for r in rows if r.get("tag") == "gpu_multi"]
    if gm:
        for dim in (2, 3):
            d = sel(gm, dim=dim)
            if not d: continue
            fig, axs = new_fig(2, w=4.8)
            variants = sorted({(r["halo"], r["overlap"], r["comm_path"]) for r in d})
            t1 = None
            for i, v in enumerate(variants):
                g = group_median(sel(d, halo=v[0], overlap=v[1], comm_path=v[2]), lambda r: int(r["ranks"]), ["t_total", "iters", "t_halo"])
                ks = sorted(g); n = np.array(ks, float)
                lab = "%s%s, %s" % (v[0], "+overlap" if v[1] else "", v[2])
                axs[0][0].loglog(n, [g[k]["t_total"] / g[k]["iters"] for k in ks], "o-", color=C[i % 8], label=lab)
                if 1 in g and t1 is None: t1 = g[1]["t_total"]
                if t1: axs[0][1].semilogx(n, [t1 / (k * g[k]["t_total"]) for k in ks], "o-", color=C[i % 8], label=lab)
            for a in axs[0]: pow2_axis(a); a.set_xlabel("GPUs (one rank per GPU)"); a.grid(True, which="both", alpha=0.25); a.legend(fontsize=10)
            axs[0][0].set(ylabel="time per iteration [s]", title="(a) strong scaling")
            axs[0][1].set(ylabel="efficiency vs 1 GPU", title="(b) efficiency"); axs[0][1].set_ylim(0, 1.3)
            finish(fig, out, "fig_gpu_multi_%dD" % dim, "Multi-GPU scaling, %dD, N=%d" % (dim, int(d[0]["N"])))


# ----------------------------------------------------------------------------- main
def main():
    global LABEL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", action="append", required=True, help="results directory (repeatable)")
    ap.add_argument("--out", default="figures", help="output directory for figures (tables go to <out>/tables)")
    ap.add_argument("--label", default=None, help="watermark text, e.g. 'LOCAL TEST DATA'")
    a = ap.parse_args()
    LABEL = a.label
    files = []
    for d in a.results:
        files += [f for f in sorted(glob.glob(os.path.join(d, "*.csv"))) if os.path.basename(f) != "allreduce.csv"]
    if not files: sys.exit("no CSV files found in %s" % a.results)
    rows = load(files)
    os.makedirs(a.out, exist_ok=True)
    tabdir = os.path.join(a.out, "tables"); os.makedirs(tabdir, exist_ok=True)
    print("%d rows from %d files" % (len(rows), len(files)))
    fig_serial(rows, a.out); fig_strong(rows, a.out, tabdir); fig_weak(rows, a.out, tabdir)
    fig_comm(rows, a.out); fig_decomp(rows, a.out); fig_allreduce(rows, a.out, a.results)
    fig_hybrid(rows, a.out); fig_gpu(rows, a.out)


if __name__ == "__main__":
    main()
