#!/bin/bash
# Stage-1 trial (Vy, 2026-11-19): H1 core (C0, T, A, V) on 1,000 questions (200 per branch), 4 models, one at a time.
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1
NEG=~/anaconda3/envs/negation/bin/python
for m in qwen2.5-omni-7b qwen2.5-omni-3b; do
  echo "=== $m $(date)"; $NEG -u run_h1v2.py --model $m --pilot 200 --batch_size 4 || echo "FAILED $m"
done
echo "=== minicpm-o-2.6 $(date)"; ~/anaconda3/envs/omni_neg/bin/python -u run_h1v2.py --model minicpm-o-2.6 --pilot 200 --generate_only --batch_size 1 || echo "FAILED minicpm gen"
echo "=== phi-4-mm $(date)"; ~/anaconda3/envs/omni_phi/bin/python -u run_h1v2.py --model phi-4-mm --pilot 200 --generate_only --batch_size 1 || echo "FAILED phi gen"
for m in minicpm-o-2.6 phi-4-mm; do
  echo "=== score $m $(date)"; $NEG -u run_h1v2.py --model $m --pilot 200 --score_only || echo "FAILED score $m"
done
echo "TRIAL DONE $(date)"
