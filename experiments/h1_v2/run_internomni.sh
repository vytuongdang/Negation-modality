#!/bin/bash
# Stage 2: InternOmni pilot200 C0/T/A/V (generate in omni_intern, score in negation)
set -e
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1
date
~/anaconda3/envs/omni_intern/bin/python -u run_h1v2.py --model internomni --pilot 200 --generate_only --batch_size 1 || { echo "TRIAL STOPPED (generate)"; exit 1; }
~/anaconda3/envs/negation/bin/python -u run_h1v2.py --model internomni --pilot 200 --score_only || { echo "TRIAL STOPPED (score)"; exit 1; }
echo "TRIAL DONE $(date)"
