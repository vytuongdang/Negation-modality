#!/bin/bash
# H3-B comprehension v2 (Vy OK 2026-11-24 10:44): COMP2_T/A/V + verbatim REP_T/REP_A/OCR_V on pilot200, 5 models.
# Usage: bash run_comp2.sh [--wait] [model ...]. --wait: start only after job-stage3 has ended with "TRIAL DONE"
# in pico/logs/stage3.log (one GPU job at a time). Smoke --pilot 2 for all models first. Stops at the first failure.
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1
NEG=~/anaconda3/envs/negation/bin/python
LOG3=../../pico/logs/stage3.log
if [ "$1" = "--wait" ]; then
  shift
  echo "waiting for job-stage3 $(date)"
  while tmux has-session -t job-stage3 2>/dev/null; do sleep 120; done
  grep -q "TRIAL DONE" $LOG3 || { echo "stage3 did not finish with TRIAL DONE -> not starting"; echo "TRIAL STOPPED $(date)"; exit 1; }
fi
MODELS_TO_RUN=("$@"); [ $# -eq 0 ] && MODELS_TO_RUN=(qwen2.5-omni-7b qwen2.5-omni-3b minicpm-o-2.6 phi-4-mm internomni)
CONDS="COMP2_T COMP2_A COMP2_V REP_T REP_A OCR_V"
fail() { echo "FAILED $1 $(date)"; echo "TRIAL STOPPED $(date)"; exit 1; }
genpy() { case $1 in qwen2.5-omni-*) echo "$NEG";; minicpm-o-2.6) echo ~/anaconda3/envs/omni_neg/bin/python;;
          phi-4-mm) echo ~/anaconda3/envs/omni_phi/bin/python;; internomni) echo ~/anaconda3/envs/omni_intern/bin/python;; esac; }
bs() { case $1 in qwen2.5-omni-*) echo 4;; *) echo 1;; esac; }
run() {
  local m=$1; shift
  $(genpy $m) -u run_comp2.py --model $m --batch_size $(bs $m) --generate_only "$@" || fail "$m gen $*"
  $NEG -u run_comp2.py --model $m --score_only "$@" || fail "$m score $*"
}
for m in "${MODELS_TO_RUN[@]}"; do echo "=== smoke $m $(date)"; run $m --pilot 2 --conds $CONDS; done
echo "SMOKE DONE $(date)"
for m in "${MODELS_TO_RUN[@]}"; do echo "=== $m $(date)"; run $m --pilot 200 --conds $CONDS; done
~/anaconda3/bin/python analyze_comp.py --tag pilot200 --version v2
echo "TRIAL DONE $(date)"
