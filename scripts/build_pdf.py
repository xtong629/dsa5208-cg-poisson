#!/usr/bin/env python3
"""Build report.pdf from report.md: pandoc + tectonic (LaTeX), A4, page numbers.
Figures are the ![Figure N: ...](figures/x.png) images in report.md (numbered by LaTeX in order).
Usage: python3 scripts/build_pdf.py            (needs pandoc and tectonic on PATH)"""
import os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(ROOT, "report.md"), encoding="utf-8").read()
# images are written in report.md as ![Figure N: caption](...); LaTeX numbers the captions itself, so drop our "Figure N: " prefix
# and check that the numbers in the markdown are 1, 2, 3, ... in order (so the text references match the captions).
nums = [int(m) for m in re.findall(r"!\[Figure (\d+): ", src)]
assert nums == list(range(1, len(nums) + 1)), "figure numbers in report.md are not consecutive: %s" % nums
n = len(nums)
text = re.sub(r"!\[Figure \d+: ", "![", src)
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
       "-f", "markdown+tex_math_dollars+pipe_tables", "--toc", "--toc-depth=2", "-H", pre]
r = subprocess.run(cmd, cwd=ROOT)
os.remove(tmp); os.remove(pre)
print("figures:", n)
sys.exit(r.returncode)
