#!/bin/bash
# Build paper/<name>.pdf from paper/<name>.md (pandoc -> xelatex; lmodern is not installed, so it is removed).
# Usage: bash paper/build_report_H.sh [name ...]   (default: report_H_results_vi report_H_results_en)
set -e
R="$(cd "$(dirname "$0")/.." && pwd)"
NAMES=("$@"); [ ${#NAMES[@]} -gt 0 ] || NAMES=(report_H_results_vi report_H_results_en)
for N in "${NAMES[@]}"; do
  T=$(mktemp -d); cd $T
  pandoc "$R/paper/$N.md" -s -o rep.tex
  sed -i '/usepackage{lmodern}/d' rep.tex
  sed -i "s#{../experiments/#{$R/experiments/#g; s#\\\\usepackage{amssymb,amsmath}#\\\\usepackage{amssymb,amsmath}\\\\usepackage{fontspec}\\\\setmainfont{DejaVu Sans}#" rep.tex
  xelatex -interaction=nonstopmode rep.tex >/dev/null; xelatex -interaction=nonstopmode rep.tex > build.log || true
  echo "$N: $(grep -o 'Output written.*' build.log)"; grep -c "Missing character" build.log || true
  cp rep.pdf "$R/paper/$N.pdf"
done
