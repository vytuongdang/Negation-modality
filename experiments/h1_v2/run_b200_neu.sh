#!/bin/bash
# B200 NEU control (Vy 2026-11-27) on the fixed 1,000-question pilot, GPU 2 only, stops at the first failure.
#   bash run_b200_neu.sh            # all 5 models: gen batch 1 (qwen7b, qwen3b, minicpm in parallel), then gen batch 2
#                                   # (phi, intern in parallel) while batch 1 is scored ONE model at a time (shared judge.jsonl)
# Assets first (done by hand 2026-11-27): create_data/src/neu_assets.py words / render / tts.
# Logs: pico/logs/neu_gen_<model>.log (generation), this script's own log for scoring.
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=2
NEG=~/anaconda3/envs/negation/bin/python
LOGS=$ROOT/pico/logs
fail() { echo "FAILED $1 $(date)"; python3 "$ROOT/pico/notify.py" "B200 NEU FAILED at $1 (GPU 2). Log: pico/logs/neu_run.log" || true; exit 1; }
genpy() { case $1 in qwen2.5-omni-*) echo $NEG ;; minicpm-o-2.6) echo ~/anaconda3/envs/omni_neg/bin/python ;;
  phi-4-mm) echo ~/anaconda3/envs/omni_phi/bin/python ;; internomni) echo ~/anaconda3/envs/omni_intern/bin/python ;; esac; }
bs() { case $1 in qwen2.5-omni-*) echo 4 ;; *) echo 1 ;; esac; }
gen_batch() {   # start generation of the given models in parallel, return their pids in PIDS
  PIDS=()
  for m in "$@"; do
    echo "--- gen $m $(date)"
    $(genpy $m) -u run_neu.py --model $m --generate_only --batch_size $(bs $m) > $LOGS/neu_gen_$m.log 2>&1 &
    PIDS+=($!)
  done
}
wait_batch() { for i in "${!PIDS[@]}"; do wait ${PIDS[$i]} || fail "gen ${BATCH[$i]}"; done; }
score() { echo "=== score $1 $(date)"; $NEG -u run_neu.py --model $1 --score_only || fail "score $1"; }

free=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
[ "$free" -gt 60000 ] || fail "preflight: GPU 2 has only $free MiB free"
$NEG -u run_neu.py --dry_run | head -1 || fail "dry run"

BATCH=(qwen2.5-omni-7b qwen2.5-omni-3b minicpm-o-2.6); gen_batch "${BATCH[@]}"; wait_batch
echo "batch 1 generated $(date)"
BATCH=(phi-4-mm internomni); gen_batch "${BATCH[@]}"; B2=("${PIDS[@]}")
for m in qwen2.5-omni-7b qwen2.5-omni-3b minicpm-o-2.6; do score $m; done
PIDS=("${B2[@]}"); wait_batch
echo "batch 2 generated $(date)"
for m in phi-4-mm internomni; do score $m; done
echo "NEU run DONE $(date)"
python3 "$ROOT/pico/notify.py" "B200 NEU run DONE (5 models, pilot1000). Next: analyze_neu.py + report." || true
