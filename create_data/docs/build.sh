#!/bin/bash
# Build the two pipeline documents and copy them over the PDFs in create_data/.
set -e
cd "$(dirname "$0")"
for f in dataset_pipeline_vi dataset_pipeline_en; do
  xelatex -interaction=nonstopmode $f.tex >/dev/null; xelatex -interaction=nonstopmode $f.tex >/dev/null
done
cp dataset_pipeline_vi.pdf "../Quy trình tạo dataset Component-Level Negation VQA Benchmark.pdf"
cp dataset_pipeline_en.pdf ../Dataset_Pipeline_Component-Level_Negation_VQA_EN.pdf
rm -f *.aux *.log *.out
echo "built"
