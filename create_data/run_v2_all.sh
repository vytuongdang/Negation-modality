#!/bin/bash
# Dataset v2.
#   bash run_v2_all.sh select   # steps 0-2: annotations, B1 candidates, image selection -> stop, open
#                               #   dataset/b1/review/review.html and check it
#   bash run_v2_all.sh rest     # steps 3-7 (B2, B3, forms, export, visual) + TTS for every form
set -e
cd "$(dirname "$0")/src"
export PYTHONNOUSERSITE=1
PY=~/anaconda3/envs/negation/bin/python
case "$1" in
  select) $PY -u run_pipeline.py --to_step 2 ;;
  rest)   $PY -u run_pipeline.py --from_step 3
          ~/anaconda3/envs/cosyvoice/bin/python -u tts_negation.py --src samples_compact_all.jsonl \
              --dst samples_compact_all_audio.jsonl
          $PY -u flag_excluded_units.py   # double labels / part-whole / attribute overlap -> out/excluded_units.jsonl
          echo "DATASET V2 DONE: dataset/out/samples_compact_all_audio.jsonl" ;;
  *) echo "usage: bash run_v2_all.sh select|rest"; exit 1 ;;
esac
