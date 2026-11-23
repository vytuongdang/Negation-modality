#!/bin/bash
# Stage 0 (PLAN.md, Vy 2026-11-23): --pilot 2 (10 questions) on every model, full pipeline incl. judge v3 + labels.
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1
NEG=~/anaconda3/envs/negation/bin/python
for m in qwen2.5-omni-7b qwen2.5-omni-3b; do
  echo "=== $m $(date)"; $NEG -u run_h1v2.py --model $m --pilot 2 --batch_size 4 || echo "FAILED $m"
done
echo "=== minicpm gen $(date)"; ~/anaconda3/envs/omni_neg/bin/python -u run_h1v2.py --model minicpm-o-2.6 --pilot 2 --generate_only --batch_size 1 || echo "FAILED minicpm gen"
echo "=== phi gen $(date)"; ~/anaconda3/envs/omni_phi/bin/python -u run_h1v2.py --model phi-4-mm --pilot 2 --generate_only --batch_size 1 || echo "FAILED phi gen"
echo "=== internomni gen $(date)"; ~/anaconda3/envs/omni_intern/bin/python -u run_h1v2.py --model internomni --pilot 2 --generate_only --batch_size 1 || echo "FAILED internomni gen"
for m in minicpm-o-2.6 phi-4-mm internomni; do
  echo "=== score $m $(date)"; $NEG -u run_h1v2.py --model $m --pilot 2 --score_only || echo "FAILED score $m"
done
echo "SMOKE DONE $(date)"
