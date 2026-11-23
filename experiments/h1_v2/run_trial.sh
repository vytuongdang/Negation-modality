#!/bin/bash
# Stage-1 pilot (PLAN.md): H1 core (C0, T, A, V) on the pilot200 selection (932 questions after exclusions), one model
# at a time. Usage: bash run_trial.sh [model ...]   (default: all 4). Stops at the first failure (exit 1, no
# "TRIAL DONE"), since 2026-11-23 (a crash in 3 models was hidden behind "TRIAL DONE").
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1
NEG=~/anaconda3/envs/negation/bin/python
MODELS_TO_RUN=("$@"); [ $# -eq 0 ] && MODELS_TO_RUN=(qwen2.5-omni-7b qwen2.5-omni-3b minicpm-o-2.6 phi-4-mm)
fail() { echo "FAILED $1 $(date)"; echo "TRIAL STOPPED $(date)"; exit 1; }
for m in "${MODELS_TO_RUN[@]}"; do
  echo "=== $m $(date)"
  case $m in
    qwen2.5-omni-*) $NEG -u run_h1v2.py --model $m --pilot 200 --batch_size 4 || fail $m ;;
    minicpm-o-2.6)  ~/anaconda3/envs/omni_neg/bin/python -u run_h1v2.py --model $m --pilot 200 --generate_only --batch_size 1 || fail "$m gen"
                    $NEG -u run_h1v2.py --model $m --pilot 200 --score_only || fail "$m score" ;;
    phi-4-mm)       ~/anaconda3/envs/omni_phi/bin/python -u run_h1v2.py --model $m --pilot 200 --generate_only --batch_size 1 || fail "$m gen"
                    $NEG -u run_h1v2.py --model $m --pilot 200 --score_only || fail "$m score" ;;
    internomni)     ~/anaconda3/envs/omni_intern/bin/python -u run_h1v2.py --model $m --pilot 200 --generate_only --batch_size 1 || fail "$m gen"
                    $NEG -u run_h1v2.py --model $m --pilot 200 --score_only || fail "$m score" ;;
    *) fail "unknown model $m" ;;
  esac
done
echo "TRIAL DONE $(date)"
