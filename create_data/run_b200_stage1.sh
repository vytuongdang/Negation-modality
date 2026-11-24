#!/bin/bash
# B200 stage 1 (Vy 2026-11-24): build the full dataset (steps 3-7 + TTS + exclusions) on GPU 2 ONLY.
# Input: dataset/b1/units.jsonl (2,799 images / 4,675 units, from subset_images.py). Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/src"
ROOT="$(cd ../.. && pwd)"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=2
PY=~/anaconda3/envs/negation/bin/python
fail() { python3 "$ROOT/pico/notify.py" "B200 stage 1 (dataset) FAILED at: $1. Log: pico/logs/stage1.log" || true; exit 1; }
trap 'fail "line $LINENO"' ERR
free=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
[ "$free" -gt 40000 ] || { echo "GPU 2 has only $free MiB free"; fail "preflight (GPU 2 busy)"; }
echo "== $(date -u +%T) steps 3-7 (B2 LLM, B3, forms, export, visual) on GPU 2 ($free MiB free)"
$PY -u run_pipeline.py --from_step 3
echo "== $(date -u +%T) TTS (cosyvoice env) + Whisper check"
~/anaconda3/envs/cosyvoice/bin/python -u tts_negation.py --src samples_compact_all.jsonl --dst samples_compact_all_audio.jsonl
echo "== $(date -u +%T) excluded units"
$PY -u flag_excluded_units.py
echo "STAGE1 DONE $(date -u +%T): dataset/out/samples_compact_all_audio.jsonl"
python3 "$ROOT/pico/notify.py" "B200 stage 1 (dataset) DONE on GPU 2. Details in pico/logs/stage1.log" || true
