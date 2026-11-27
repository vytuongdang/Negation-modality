#!/bin/bash
# early_gen.sh <model> <python>: NEU generation for one batch-2 model started before run_b200_neu.sh reaches it (same command,
# same answers cache). A watcher kills this early copy as soon as the NEU script prints "--- gen <model>" (then the script's
# own copy resumes from the cache), so the same model never generates twice at once.
m=$1; py=$2; ROOT=/home/jovyan/gpu-4-nodes-volume2/vy/cross_multimodal_negation
cd $ROOT/experiments/h1_v2 && PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=2 $py -u run_neu.py --model $m --generate_only --batch_size 1
