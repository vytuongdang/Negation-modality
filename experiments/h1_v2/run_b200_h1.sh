#!/bin/bash
# B200 stage 2 (Vy 2026-11-24): H1 core (C0, T, A, V) on GPU 2 ONLY. Stops at the first failure.
#   bash run_b200_h1.sh smoke [models...]   # --pilot 2 (10 questions), measures s/answer, tag pilot2
#   bash run_b200_h1.sh full  [models...]   # --all (every unit of dataset/out), tag all
# Default models: qwen2.5-omni-7b qwen2.5-omni-3b minicpm-o-2.6 phi-4-mm internomni. Answers are cached per model
# (results/<model>/answers.jsonl), so a crashed run resumes where it stopped. Generation runs in the model's own
# env, scoring (judge Qwen3-8B + labels) in the `negation` env; batch sizes as on server 1 (4 for Qwen, 1 for the rest).
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=2
MODE=$1; shift
case $MODE in smoke) SEL="--pilot 2" ;; full) SEL="--all" ;; *) echo "usage: $0 smoke|full [models]"; exit 1 ;; esac
NEG=~/anaconda3/envs/negation/bin/python
MODELS_TO_RUN=("$@"); [ $# -eq 0 ] && MODELS_TO_RUN=(qwen2.5-omni-7b qwen2.5-omni-3b minicpm-o-2.6 phi-4-mm internomni)
fail() { echo "FAILED $1 $(date)"; python3 "$ROOT/pico/notify.py" "B200 H1 $MODE FAILED at $1 (GPU 2). Log: pico/logs/h1_$MODE.log" || true; exit 1; }
free=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
[ "$free" -gt 30000 ] || fail "preflight: GPU 2 has only $free MiB free"
for m in "${MODELS_TO_RUN[@]}"; do
  echo "=== $m $MODE $(date)"
  case $m in
    qwen2.5-omni-*) $NEG -u run_h1v2.py --model $m $SEL --batch_size 4 || fail $m ;;
    minicpm-o-2.6)  ~/anaconda3/envs/omni_neg/bin/python -u run_h1v2.py --model $m $SEL --generate_only --batch_size 1 || fail "$m gen"
                    $NEG -u run_h1v2.py --model $m $SEL --score_only || fail "$m score" ;;
    phi-4-mm)       ~/anaconda3/envs/omni_phi/bin/python -u run_h1v2.py --model $m $SEL --generate_only --batch_size 1 || fail "$m gen"
                    $NEG -u run_h1v2.py --model $m $SEL --score_only || fail "$m score" ;;
    internomni)     ~/anaconda3/envs/omni_intern/bin/python -u run_h1v2.py --model $m $SEL --generate_only --batch_size 1 || fail "$m gen"
                    $NEG -u run_h1v2.py --model $m $SEL --score_only || fail "$m score" ;;
    *) fail "unknown model $m" ;;
  esac
done
echo "H1 $MODE DONE $(date)"
python3 "$ROOT/pico/notify.py" "B200 H1 $MODE DONE on GPU 2 (${MODELS_TO_RUN[*]}). Next: analysis + report." || true
