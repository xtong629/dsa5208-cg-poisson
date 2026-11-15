#!/usr/bin/env python3
"""Build report.pdf from report.md: pandoc + tectonic (LaTeX), A4, page numbers.
Figures named `fig_*` in the text are embedded (PNG) with a caption right after the block that first mentions them.
Usage: python3 scripts/build_pdf.py            (needs pandoc and tectonic on PATH)"""
import os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP = {
 "fig_serial_scaling": "Serial CG on one core: time per iteration (left) and time to reach the tolerance (right) against the number of unknowns, 2D and 3D, logarithmic axes.",
 "fig_strong_2D": "Strong scaling in 2D ($N=4096$, $8192$): runtime, speedup $T_s/T_p(n)$ and efficiency against the number of MPI ranks $n$; Amdahl fit dashed.",
 "fig_strong_3D": "Strong scaling in 3D ($N=256$, $512$): runtime, speedup and efficiency against the number of MPI ranks.",
 "fig_weak_2D": "Weak scaling in 2D ($1024^2$ and $2048^2$ points per rank): time per iteration and weak-scaling efficiency.",
 "fig_weak_3D": "Weak scaling in 3D ($64^3$ and $128^3$ points per rank): time per iteration and weak-scaling efficiency.",
 "fig_comm_breakdown_2D": "Share of one iteration spent in stencil, halo exchange, Allreduce and vector updates, 2D.",
 "fig_comm_breakdown_3D": "Share of one iteration spent in stencil, halo exchange, Allreduce and vector updates, 3D.",
 "fig_allreduce_latency": "Latency of an Allreduce of one double (micro-benchmark) and Allreduce time measured inside the solver, against the number of ranks.",
 "fig_decomposition_2D": "Slab against block decomposition, 2D $N=4096$.",
 "fig_decomposition_3D": "Slab against block decomposition, 3D $N=256$.",
 "fig_halo_variants_2D": "Blocking, non-blocking and overlapped halo exchange, 2D.",
 "fig_halo_variants_3D": "Blocking, non-blocking and overlapped halo exchange, 3D.",
 "fig_hybrid_2D": "MPI+OpenMP: time per iteration for different ranks $\\times$ threads splits, 2D $N=4096$ (64 and 256 cores).",
 "fig_hybrid_3D": "MPI+OpenMP: time per iteration for different ranks $\\times$ threads splits, 3D $N=256$ (64 and 256 cores).",
 "fig_gpu_vs_cpu": "One A40 GPU against 36 CPU cores: time per iteration against problem size, 2D and 3D.",
 "fig_gpu_multi_2D": "One and two GPUs, 2D $N=8192$, for the halo variants.",
 "fig_gpu_multi_3D": "One and two GPUs, 3D $N=384$, for the halo variants.",
}
src = open(os.path.join(ROOT, "report.md"), encoding="utf-8").read()
blocks = re.split(r"\n\s*\n", src)
seen, out, n = set(), [], 0
for b in blocks:
    out.append(b)
    if b.lstrip().startswith("|") or b.startswith("---"):   # keep tables intact; figures follow the next text block
        pass
    names = [m for m in re.findall(r"`(fig_[A-Za-z0-9_]+)`", b) if m in CAP and m not in seen]
    # only for the introducing paragraph (the one that lists the figure in bold or "Figures:")
    if names and re.search(r"\*\*Figure|\*\*Figures", b):
        for nm in names:
            seen.add(nm); n += 1
            out.append(f"![{CAP[nm]}](figures/{nm}.png){{width=100%}}")
text = "\n\n".join(out)
pre = os.path.join(ROOT, ".report_preamble.tex")
open(pre, "w").write(r"""\usepackage{float}
\floatplacement{figure}{H}
\usepackage{booktabs}
\setlength{\parskip}{4pt}
\usepackage{hyperref}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=black}
""")
tmp = os.path.join(ROOT, ".report_build.md")
open(tmp, "w", encoding="utf-8").write(text)
cmd = ["pandoc", tmp, "-o", os.path.join(ROOT, "report.pdf"), "--pdf-engine=tectonic", "--resource-path=" + ROOT,
       "-V", "papersize=a4", "-V", "geometry:margin=2.2cm", "-V", "fontsize=10pt", "-V", "colorlinks=true",
       "-f", "markdown+tex_math_dollars+pipe_tables", "--toc", "--toc-depth=2", "-H", pre, "-V", "CJKmainfont=Songti SC"]
r = subprocess.run(cmd, cwd=ROOT)
os.remove(tmp); os.remove(pre)
print("embedded figures:", n, "of", len(CAP), "| missing:", sorted(set(CAP) - seen))
sys.exit(r.returncode)
