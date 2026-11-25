#!/bin/bash
# B200 pilot runs on the FIXED list pilot1000.json (Vy 2026-11-25 schedule, steps 1-3). GPU 2 ONLY, stops at the
# first failure. Uses run_pilot.py (= run_h1v2.py on the list; run_h1v2.py and pilot1000.json are not edited).
#   bash run_b200_pilot.sh smoke [model]   # run_h1v2.py --pilot 2 --forms on ONE model (default qwen2.5-omni-3b)
#   bash run_b200_pilot.sh forms           # H4: T/A/V x exceptive/affirmative (+ cached C0/T/A/V), tag pilot1000_forms
#   bash run_b200_pilot.sh comp            # H3-B: --conds COMP_T COMP_A COMP_V
#   bash run_b200_pilot.sh ctrl            # controls: --conds C0_box OCR_V
# Generation in two parallel batches (never all 5 at once): BATCH1 (default qwen 7b, qwen 3b, minicpm), then BATCH2
# (default phi, internomni) once batch 1 has finished generating. Scoring (judge, shared results/judge.jsonl) runs one
# model at a time, in env `negation`, and may overlap with batch 2's generation. Override the split with
# BATCH1="m1 m2 m3" BATCH2="m4 m5". Per-model generation logs: pico/logs/<stage>_gen_<model>.log.
# Answers are cached (results/<model>/answers.jsonl), so a crashed run resumes where it stopped.
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=2
STAGE=$1; shift
NEG=~/anaconda3/envs/negation/bin/python
case $STAGE in
  smoke) ARGS="--forms" ;;
  forms) ARGS="--forms" ;;
  comp)  ARGS="--conds COMP_T COMP_A COMP_V" ;;
  ctrl)  ARGS="--conds C0_box OCR_V" ;;
  *) echo "usage: $0 smoke [model] | forms | comp | ctrl"; exit 1 ;;
esac
PIDS=()
fail() {
  echo "FAILED $1 $(date)"
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done   # only this script's own generation jobs
  python3 "$ROOT/pico/notify.py" "B200 pilot $STAGE FAILED at $1 (GPU 2). Log: pico/logs/pilot_$STAGE.log" || true
  exit 1
}
gpu_check() {
  nvidia-smi -i 2 --query-gpu=memory.used,memory.total --format=csv,noheader
  free=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free" -gt "$1" ] || fail "preflight ($2): GPU 2 has only $free MiB free, need $1"
}
genpy() {  # python of the model's own env
  case $1 in
    qwen2.5-omni-*) echo $NEG ;;
    minicpm-o-2.6)  echo ~/anaconda3/envs/omni_neg/bin/python ;;
    phi-4-mm)       echo ~/anaconda3/envs/omni_phi/bin/python ;;
    internomni)     echo ~/anaconda3/envs/omni_intern/bin/python ;;
    *) fail "unknown model $1" ;;
  esac
}
bs() { case $1 in qwen2.5-omni-*) echo 4 ;; *) echo 1 ;; esac; }

if [ "$STAGE" = smoke ]; then
  m=${1:-qwen2.5-omni-3b}
  gpu_check 30000 "smoke $m"
  echo "=== smoke $m $(date)"
  $(genpy $m) -u run_h1v2.py --model $m --pilot 2 $ARGS --generate_only --batch_size $(bs $m) || fail "smoke $m gen"
  $NEG -u run_h1v2.py --model $m --pilot 2 $ARGS --score_only || fail "smoke $m score"
  echo "PILOT smoke DONE $(date)"
  exit 0
fi

read -ra B1 <<< "${BATCH1:-qwen2.5-omni-7b qwen2.5-omni-3b minicpm-o-2.6}"
read -ra B2 <<< "${BATCH2:-phi-4-mm internomni}"

start_gen() {  # start generation of a batch in the background; fills PIDS / NAMES
  PIDS=(); NAMES=()
  for m in "$@"; do
    log="$ROOT/pico/logs/${STAGE}_gen_$m.log"
    echo "--- gen $m -> $log $(date)"
    $(genpy $m) -u run_pilot.py --model $m $ARGS --generate_only --batch_size $(bs $m) > "$log" 2>&1 &
    PIDS+=($!); NAMES+=($m)
    sleep 60   # let each model load before the next one (memory peaks)
  done
}
wait_gen() {
  for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { unset 'PIDS[i]'; fail "${NAMES[$i]} gen (see pico/logs/${STAGE}_gen_${NAMES[$i]}.log)"; }
    echo "--- gen ${NAMES[$i]} done $(date)"
  done
  PIDS=()
}
score() {
  for m in "$@"; do
    echo "=== score $m $(date)"
    $NEG -u run_pilot.py --model $m $ARGS --score_only || fail "$m score"
  done
}

echo "=== PILOT $STAGE: batch1 ${B1[*]} | batch2 ${B2[*]} $(date)"
gpu_check 60000 "batch1"
start_gen "${B1[@]}"; wait_gen
gpu_check 40000 "batch2"
start_gen "${B2[@]}"
score "${B1[@]}"            # judge one model at a time, overlaps batch 2 generation
wait_gen
score "${B2[@]}"
echo "PILOT $STAGE DONE $(date)"
python3 "$ROOT/pico/notify.py" "B200 pilot $STAGE DONE on GPU 2 (${B1[*]} ${B2[*]}). Next: analysis." || true
