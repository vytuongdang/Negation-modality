#!/bin/bash
# Sensitivity: H1 full + H4 pilot analyses without units that have a think_error judge row (any model).
cd "$(dirname "$0")/.." && export PYTHONNOUSERSITE=1
P=~/anaconda3/envs/negation/bin/python
M="qwen2.5-omni-7b qwen2.5-omni-3b minicpm-o-2.6 phi-4-mm internomni"
$P analyze.py --tag all --models $M --asr_ok_only --remove audit/rm_think_error_all.txt && \
$P analyze.py --tag pilot1000_forms --models $M --asr_ok_only --remove audit/rm_think_error_pilot1000_forms.txt
echo "EXIT $?"
