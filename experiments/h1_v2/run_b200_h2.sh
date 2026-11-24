#!/bin/bash
# B200 H2 + H3-A (written 2026-11-24, NOT started: Vy decides when). GPU 2 only, stops at the first failure.
#   bash run_b200_h2.sh assets <pilot N | all>          # question images (CPU) + question audio (GPU 2, cosyvoice env)
#   bash run_b200_h2.sh run    <pilot N | all> [models] # generate + score H2 and H3-A, default all 5 models
# Options through environment variables: FAMILIES="H2 H3A", LAYOUTS="TA AT TV VT AV VA", H3A_FORM=affirmative|exceptive.
# Subset or full is the only choice you make: "pilot 40" (40 units per branch = 200 questions) or "all".
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=2
STEP=$1; SEL_KIND=$2
case $SEL_KIND in pilot) SEL="--pilot $3"; shift 3 ;; all) SEL="--all"; shift 2 ;; *) echo "usage: $0 assets|run pilot N|all [models]"; exit 1 ;; esac
NEG=~/anaconda3/envs/negation/bin/python
SRC=$ROOT/create_data/src
FAMILIES=${FAMILIES:-"H2 H3A"}; LAYOUTS=${LAYOUTS:-"TA AT TV VT AV VA"}; H3A_FORM=${H3A_FORM:-affirmative}
OPTS="--families $FAMILIES --layouts $LAYOUTS --h3a_form $H3A_FORM"
fail() { echo "FAILED $1 $(date)"; python3 "$ROOT/pico/notify.py" "B200 H2 $STEP FAILED at $1 (GPU 2). Log: pico/logs/h2_$STEP.log" || true; exit 1; }
free=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
[ "$free" -gt 30000 ] || fail "preflight: GPU 2 has only $free MiB free"
case $STEP in
  assets)
    (cd $SRC && $NEG -u h2_assets.py render $SEL) || fail "render"
    (cd $SRC && ~/anaconda3/envs/cosyvoice/bin/python -u h2_assets.py tts) || fail "tts"
    (cd $SRC && $NEG -u h2_assets.py status $SEL)
    python3 "$ROOT/pico/notify.py" "B200 H2 question assets DONE ($SEL)." || true ;;
  run)
    MODELS_TO_RUN=("$@"); [ $# -eq 0 ] && MODELS_TO_RUN=(qwen2.5-omni-7b qwen2.5-omni-3b minicpm-o-2.6 phi-4-mm internomni)
    for m in "${MODELS_TO_RUN[@]}"; do
      echo "=== $m H2/H3-A $SEL $(date)"
      case $m in
        qwen2.5-omni-*) $NEG -u run_h2.py --model $m $SEL $OPTS --batch_size 4 || fail $m ;;
        minicpm-o-2.6)  ~/anaconda3/envs/omni_neg/bin/python -u run_h2.py --model $m $SEL $OPTS --generate_only --batch_size 1 || fail "$m gen"
                        $NEG -u run_h2.py --model $m $SEL $OPTS --score_only || fail "$m score" ;;
        phi-4-mm)       ~/anaconda3/envs/omni_phi/bin/python -u run_h2.py --model $m $SEL $OPTS --generate_only --batch_size 1 || fail "$m gen"
                        $NEG -u run_h2.py --model $m $SEL $OPTS --score_only || fail "$m score" ;;
        internomni)     ~/anaconda3/envs/omni_intern/bin/python -u run_h2.py --model $m $SEL $OPTS --generate_only --batch_size 1 || fail "$m gen"
                        $NEG -u run_h2.py --model $m $SEL $OPTS --score_only || fail "$m score" ;;
        *) fail "unknown model $m" ;;
      esac
    done
    echo "H2 run DONE $(date)"
    python3 "$ROOT/pico/notify.py" "B200 H2/H3-A run DONE ($SEL; ${MODELS_TO_RUN[*]}). Next: analyze_h2.py + report." || true ;;
  *) echo "usage: $0 assets|run ..."; exit 1 ;;
esac
