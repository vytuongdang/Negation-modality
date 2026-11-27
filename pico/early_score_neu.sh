#!/bin/bash
# Early NEU judge for batch-1 Qwen models while minicpm still generates (only judge writer at this time).
# watch-neu-score kills this session as soon as run_b200_neu.sh prints any "=== score" line (script resumes from judge cache).
ROOT=/home/jovyan/gpu-4-nodes-volume2/vy/cross_multimodal_negation
cd $ROOT/experiments/h1_v2 && export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=2
for m in qwen2.5-omni-7b qwen2.5-omni-3b; do echo "=== early score $m $(date)"; ~/anaconda3/envs/negation/bin/python -u run_neu.py --model $m --score_only || { echo "EXIT 1"; exit 1; }; done
echo "EXIT 0"
