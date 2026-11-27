#!/bin/bash
# Build paper/report_H_results.pdf from report_H_results.md (pandoc -> xelatex; lmodern is not installed, so it is removed).
set -e
R="$(cd "$(dirname "$0")/.." && pwd)"; T=$(mktemp -d); cd $T
pandoc "$R/paper/report_H_results.md" -s -o rep.tex
sed -i '/usepackage{lmodern}/d' rep.tex
sed -i "s#{../experiments/#{$R/experiments/#g; s#\\\\usepackage{amssymb,amsmath}#\\\\usepackage{amssymb,amsmath}\\\\usepackage{fontspec}\\\\setmainfont{DejaVu Sans}#" rep.tex
xelatex -interaction=nonstopmode rep.tex >/dev/null; xelatex -interaction=nonstopmode rep.tex > build.log
grep -o "Output written.*" build.log; cp rep.pdf "$R/paper/report_H_results.pdf"
