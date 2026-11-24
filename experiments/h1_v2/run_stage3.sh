#!/bin/bash
# Stage 3 (PLAN.md): H4 forms (--forms), H3-B comprehension (COMP_T/A/V), controls (C0_box, OCR_V) on pilot200.
# Usage: bash run_stage3.sh [model ...] (default: all 5). Smoke (--pilot 2) of the new conds for every model first.
# Stops at the first failure (exit 1, "TRIAL STOPPED"); answers are cached, so a rerun resumes.
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NEG=~/anaconda3/envs/negation/bin/python
MODELS_TO_RUN=("$@"); [ $# -eq 0 ] && MODELS_TO_RUN=(qwen2.5-omni-7b qwen2.5-omni-3b minicpm-o-2.6 phi-4-mm internomni)
fail() { echo "FAILED $1 $(date)"; echo "TRIAL STOPPED $(date)"; exit 1; }
genpy() { case $1 in qwen2.5-omni-*) echo "$NEG";; minicpm-o-2.6) echo ~/anaconda3/envs/omni_neg/bin/python;;
          phi-4-mm) echo ~/anaconda3/envs/omni_phi/bin/python;; internomni) echo ~/anaconda3/envs/omni_intern/bin/python;; esac; }
bs() { case $1 in qwen2.5-omni-*) echo 4;; *) echo 1;; esac; }
run() {  # model, then run_h1v2 args
  local m=$1; shift
  $(genpy $m) -u run_h1v2.py --model $m --batch_size $(bs $m) --generate_only "$@" || fail "$m gen $*"
  $NEG -u run_h1v2.py --model $m --score_only "$@" || fail "$m score $*"
}
for m in "${MODELS_TO_RUN[@]}"; do
  echo "=== smoke $m $(date)"
  run $m --pilot 2 --conds COMP_T COMP_A COMP_V C0_box OCR_V
done
echo "SMOKE DONE $(date)"
for m in "${MODELS_TO_RUN[@]}"; do
  echo "=== $m comprehension + controls $(date)"
  run $m --pilot 200 --conds COMP_T COMP_A COMP_V
  run $m --pilot 200 --conds C0_box OCR_V
  echo "=== $m forms (H4) $(date)"
  run $m --pilot 200 --forms
done
echo "TRIAL DONE $(date)"
