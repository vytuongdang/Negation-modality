#!/bin/bash
# B200 H3-C (Vy OK 2026-11-27): T_first / A_first vs T / A on pilot1000, GPU 2, stops at the first failure.
#   bash run_b200_h3c.sh smoke  [models]     # --pilot 2 (10 units): gen + score each model
#   bash run_b200_h3c.sh gen N  [models]     # N = 200 | 1000, generation only (no judge)
#   bash run_b200_h3c.sh score N [models]    # scoring (judge); waits while any NEU scoring process runs (shared judge.jsonl)
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=2
NEG=~/anaconda3/envs/negation/bin/python
MODE=$1; shift
case $MODE in smoke) SEL="--pilot 2" ;; gen|score) SEL="--n $1"; shift ;; *) echo "usage"; exit 1 ;; esac
MODELS_TO_RUN=("$@"); [ $# -eq 0 ] && MODELS_TO_RUN=(qwen2.5-omni-7b qwen2.5-omni-3b minicpm-o-2.6 phi-4-mm internomni)
fail() { echo "FAILED $1 $(date)"; python3 "$ROOT/pico/notify.py" "B200 H3-C $MODE FAILED at $1 (GPU 2). Log: pico/logs/h3c.log" || true; exit 1; }
genpy() { case $1 in qwen2.5-omni-*) echo "$NEG";; minicpm-o-2.6) echo ~/anaconda3/envs/omni_neg/bin/python;;
          phi-4-mm) echo ~/anaconda3/envs/omni_phi/bin/python;; internomni) echo ~/anaconda3/envs/omni_intern/bin/python;; esac; }
bs() { case $1 in qwen2.5-omni-*) echo 4;; *) echo 1;; esac; }
judge_free() { while pgrep -f "run_neu.py --model .* --score_only" >/dev/null || pgrep -f "run_h2.py .*--score_only" >/dev/null; do
  echo "waiting: another judge writer is running $(date)"; sleep 120; done; }
gen() { free=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free" -gt 40000 ] || fail "preflight $1: GPU 2 has only $free MiB free"
  $(genpy $1) -u run_h3c.py --model $1 $SEL --generate_only --batch_size $(bs $1) || fail "$1 gen"; }
sc() { judge_free; $NEG -u run_h3c.py --model $1 $SEL --score_only || fail "$1 score"; }
for m in "${MODELS_TO_RUN[@]}"; do
  echo "=== $MODE $m $SEL $(date)"
  case $MODE in smoke) gen $m; sc $m ;; gen) gen $m ;; score) sc $m ;; esac
done
echo "H3C $MODE DONE $(date)"
