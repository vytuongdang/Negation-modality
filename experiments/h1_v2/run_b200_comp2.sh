#!/bin/bash
# B200 COMP v2 (Vy OK 2026-11-27, in parallel with the NEU job on GPU 2). Same steps as run_comp2.sh, but GPU 2 and
# the fixed pilot1000 list: run_comp2.register() + run_pilot.select_list composed on the command line (no code edit).
# Judge-free (COMP2 letter-parsed, REP/OCR string similarity). OCR_V is in the list only so that the output file names
# match analyze_comp.py --version v2; its pilot1000 answers are already cached (no new generation).
# Usage: bash run_b200_comp2.sh [model ...]. Smoke --pilot 2 for all models first; stops at the first failure.
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NEG=~/anaconda3/envs/negation/bin/python
MODELS_TO_RUN=("$@"); [ $# -eq 0 ] && MODELS_TO_RUN=(qwen2.5-omni-7b qwen2.5-omni-3b minicpm-o-2.6 phi-4-mm internomni)
CONDS="COMP2_T COMP2_A COMP2_V REP_T REP_A OCR_V"
fail() { echo "FAILED $1 $(date)"; python3 ../../pico/notify.py "B200 COMP v2 FAILED at $1 (GPU 2). Log: pico/logs/comp2.log" || true; exit 1; }
genpy() { case $1 in qwen2.5-omni-*) echo "$NEG";; minicpm-o-2.6) echo ~/anaconda3/envs/omni_neg/bin/python;;
          phi-4-mm) echo ~/anaconda3/envs/omni_phi/bin/python;; internomni) echo ~/anaconda3/envs/omni_intern/bin/python;; esac; }
bs() { case $1 in qwen2.5-omni-*) echo 4;; *) echo 1;; esac; }
LISTRUN='import sys, run_comp2, run_pilot; run_comp2.register(); run_pilot.R.select = run_pilot.select_list; sys.argv = ["run_pilot"] + sys.argv[1:]; sys.exit(run_pilot.R.main())'
run_smoke() { local m=$1; shift
  $(genpy $m) -u run_comp2.py --model $m --batch_size $(bs $m) --generate_only "$@" || fail "$m smoke gen"
  $NEG -u run_comp2.py --model $m --score_only "$@" || fail "$m smoke score"; }
run_list() { local m=$1; shift
  $(genpy $m) -u -c "$LISTRUN" --model $m --batch_size $(bs $m) --generate_only --pilot 1000 "$@" || fail "$m gen"
  $NEG -u -c "$LISTRUN" --model $m --score_only --pilot 1000 "$@" || fail "$m score"; }
free=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
[ "$free" -gt 40000 ] || fail "preflight: GPU 2 has only $free MiB free"
for m in "${MODELS_TO_RUN[@]}"; do echo "=== smoke $m $(date)"; run_smoke $m --pilot 2 --conds $CONDS; done
echo "SMOKE DONE $(date)"
for m in "${MODELS_TO_RUN[@]}"; do echo "=== $m $(date)"; run_list $m --conds $CONDS; done
$NEG analyze_comp.py --tag pilot1000 --version v2 || fail "analyze"
echo "COMP2 DONE $(date)"
python3 ../../pico/notify.py "B200 COMP v2 DONE (5 models, pilot1000). Next: analysis + report." || true
