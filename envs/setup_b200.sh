#!/bin/bash
# Build the experiment envs on B200 from server 1's package lists (envs/<env>.freeze.txt).
#
#   bash envs/setup_b200.sh                 # build all 4 envs (skips steps already done), then --check
#   bash envs/setup_b200.sh omni_neg        # only these envs (cosyvoice = TTS, built only when named)
#   bash envs/setup_b200.sh --check         # versions + a tiny CUDA test per env
#
# Same env names as server 1 (negation, omni_neg, omni_phi, omni_intern), created in /home/vy/conda-envs, and
# ~/anaconda3/envs is linked there so the run scripts (~/anaconda3/envs/<env>/bin/python) work unchanged.
# B200 = Blackwell (sm_100): server 1's torch 2.6+cu124 cannot run there, so every env gets torch 2.7.1+cu128;
# every other package keeps server 1's exact version (installed with --no-deps from the freeze list).
# vLLM / flashinfer / xgrammar are left out (not used by the experiments). flash_attn: prebuilt wheel for
# torch 2.7 if it installs, otherwise the env works without it (attention falls back to sdpa where the
# model code allows it; MiniCPM-o's remote code imports flash_attn -> see the --check output).
set -u
cd "$(dirname "$0")" || exit 1
CONDA=/opt/conda/bin/conda
ENVROOT=/home/vy/conda-envs
TORCH="torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1"
TORCH_INDEX=https://download.pytorch.org/whl/cu128
FA_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
SKIP='^(torch|torchvision|torchaudio|onnxruntime-gpu|triton|tokenspeed-triton|torch_c_dlpack_ext|torchcodec|vllm|flashinfer-python|flash_attn|flash-attn|xgrammar|nccl4py|cuda-[A-Za-z-]+|nvidia-(?!ml-py)[A-Za-z0-9-]+)( @|==)'
ALL="negation omni_neg omni_phi omni_intern"
CHECK="$ALL cosyvoice"
export PYTHONNOUSERSITE=1   # never mix in ~/.local packages

pyver() { [ "$1" = negation ] && echo 3.11 || echo 3.10; }

check() {
  for e in $CHECK; do
    py=$ENVROOT/$e/bin/python
    [ -x "$py" ] || { echo "== $e: MISSING"; continue; }
    echo "== $e"
    "$py" - <<'PY'
import importlib, torch
x = torch.ones(4, device="cuda")
print(f"  torch {torch.__version__}  cuda ok: {(x * 2).sum().item() == 8}  gpu: {torch.cuda.get_device_name(0)}")
for m in ("transformers", "accelerate", "peft", "soundfile", "librosa", "decord", "flash_attn", "onnxruntime", "whisper"):
    try:
        print(f"  {m} {getattr(importlib.import_module(m), '__version__', '?')}")
    except Exception as err:
        print(f"  {m}: NOT importable ({type(err).__name__}: {str(err)[:80]})")
PY
  done
}

link() {
  if [ -e ~/anaconda3/envs ] && [ ! -L ~/anaconda3/envs ]; then
    echo "~/anaconda3/envs exists and is not a link: not touching it"; return
  fi
  mkdir -p ~/anaconda3/bin
  ln -sfn "$ENVROOT" ~/anaconda3/envs
  ln -sf "$ENVROOT/negation/bin/python" ~/anaconda3/bin/python   # analysis scripts use ~/anaconda3/bin/python
}

build() {
  local e=$1 py=$ENVROOT/$1/bin/python req
  echo "===== $e $(date)"
  [ -x "$py" ] || "$CONDA" create -y -q -p "$ENVROOT/$e" "python=$(pyver "$e")" || { echo "FAILED create $e"; return 1; }
  "$py" -m pip install -q --upgrade pip
  "$py" -c "import torch, sys; sys.exit(torch.__version__ != '2.7.1+cu128')" 2>/dev/null ||
    "$py" -m pip install -q $TORCH --index-url $TORCH_INDEX || { echo "FAILED torch $e"; return 1; }
  req=$(mktemp); grep -vP "$SKIP" "$e.freeze.txt" > "$req"
  # old source packages (openai-whisper) import pkg_resources while building: setuptools < 81 + no build isolation
  "$py" -m pip install -q "setuptools<81" wheel
  # numpy first, at server 1's version: sdists built without isolation (pyworld) compile against it
  grep '^numpy==' "$e.freeze.txt" | xargs -r "$py" -m pip install -q --no-deps
  "$py" -m pip install -q --no-deps --no-build-isolation -r "$req" || { echo "FAILED packages $e"; rm -f "$req"; return 1; }
  rm -f "$req"
  # InternOmni tokenizer.model fails to load with sentencepiece 0.2.2 ("piece must not include null character"); 0.2.0 loads it
  if [ "$e" = omni_intern ]; then "$py" -m pip install -q --no-deps sentencepiece==0.2.0 || echo "WARNING $e: sentencepiece 0.2.0 pin failed"; fi
  if [ "$e" = cosyvoice ]; then   # server 1's onnxruntime-gpu 1.18 needs cuDNN 8; torch 2.7 ships cuDNN 9
    "$py" -m pip install -q --no-deps onnxruntime-gpu==1.22.0 || echo "WARNING $e: onnxruntime-gpu upgrade failed"
  fi
  if grep -q '^flash_attn' "$e.freeze.txt"; then
    "$py" -m pip install -q --no-deps "$FA_WHEEL" || echo "WARNING $e: flash_attn wheel did not install"
  fi
  echo "DONE $e $(date)"
}

if [ "${1:-}" = "--check" ]; then check; exit; fi
mkdir -p "$ENVROOT"
envs=${*:-$ALL}
for e in $envs; do build "$e"; done
link
check
echo "SETUP FINISHED $(date)"
