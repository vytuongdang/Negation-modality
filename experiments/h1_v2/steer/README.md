# Method part: reducing audio / visual negation violations (Qwen2.5-Omni 3B / 7B, pilot1000)

Written 2026-11-26 on B200 (Vy's instruction). **Code only so far: no GPU run until H2 / H3-A are finished and Vy says OK.**
New files only; run_h1v2.py, run_pilot.py, run_h2.py, omnilib.py and judge.py are imported read-only, never edited.

## Design

- **Questions:** the fixed pilot1000 list (`../pilot1000.json`), one negative sample per question.
  **Split by image:** TRAIN = source split train (673 questions), VAL = val/val2017/test (327). Each image is used
  once in the list, and `load_pilot()` asserts that no image is in both parts. Directions are computed on TRAIN only;
  every method is evaluated on VAL.
- **Prohibition direction** (`direction.py`): for every TRAIN question, two T-like inputs (same image and question,
  instruction as its own text item): "Do not mention X." vs "Mention X.", where X is the slot of the question's own
  instruction. The direction is d_L = mean hidden state over the instruction tokens (output of decoder layer L) for
  neg, minus the same for pos, at every layer. `--check_val` reports, per layer, how well the projection separates
  neg from pos on the VAL texts. Use it to choose layers without looking at violations.
- **Steering** (`run_steer.py gen --method steer`): during the prompt pass, h_L += alpha * d_L at the AUDIO
  instruction tokens (A) or at the IMAGE tokens (V: `--v_pos box` = only the tokens under the written instruction,
  `image` = all of them). The KV cache carries the change into generation. The sweep is over `--layers` ("16" or
  "12-16") and `--alphas`.
- **Controls / baselines:**
  - `--method rand`: a random direction with the same norm.
  - `--method cfg`: contrastive / CFG decoding, log p(with) + w (log p(with) - log p(without)), with a plausibility
    cut beta. "Without" = C0 for A and C0_box for V.
  - `--method transcribe`: transcribe-then-obey. The model first copies the instruction (audio: a transcription
    request; image: H1's OCR_V input, cached), then answers image + question + its own copy as a text item.
- **Same generation settings as H1:** greedy, fp16, 100 new tokens, and answers cut near 100 tokens are generated
  again with 256.
- **Caching:** cache `results/<model>/answers.jsonl`, keyed by omnilib.input_key(model|config_id, ...), resumable.
  The shared H1 cache is only read (for unsteered C0 / T / A / V / C0_box and OCR_V). Every row carries image_id,
  unit_id and sample_id.
- **Scoring:** the H1 judge path unchanged (`score_rows` = the scoring block of run_h1v2.main), with the judge
  cache `../results/judge.jsonl`. **Only one judge writer at a time:** do not score while another job judges.
- **Analysis** (`analyze_steer.py`): per config x channel, violation among C0-mentioned AND required recall
  (C0-mentioned and all questions) side by side. Each comes with a 95% image-bootstrap CI and a paired difference
  to the unsteered answer.

## Run (GPU 2 only, after Vy's OK)

```bash
cd experiments/h1_v2/steer; export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=2; PY=~/anaconda3/envs/negation/bin/python
CUDA_VISIBLE_DEVICES= $PY smoke_cpu.py                                  # CPU checks (must print ALL PASS)
$PY direction.py --model qwen2.5-omni-3b --check_val                     # ~673x2 forward passes, minutes
$PY run_steer.py gen --model qwen2.5-omni-3b --method steer --layers 8 12 16 20 24 --alphas 1 2 4 --limit 40   # small sweep
$PY run_steer.py gen --model qwen2.5-omni-3b --method rand  --layers 16 --alphas 4 --limit 40
$PY run_steer.py gen --model qwen2.5-omni-3b --method cfg --ws 1 2 --limit 40
$PY run_steer.py gen --model qwen2.5-omni-3b --method transcribe --limit 40
$PY run_steer.py score   --model qwen2.5-omni-3b --limit 40
$PY run_steer.py analyze --model qwen2.5-omni-3b --limit 40           # -> analysis_<model>_val_lim40.md
```
Then drop `--limit` for the chosen configs (327 VAL questions x 2 channels = 654 answers per config, about 6 min of
generation for 3B, plus the judge). Choose layer / alpha on TRAIN (`--split train`) if the sweep is wide, so the
VAL numbers are not tuned on VAL.

## Caveats to keep in mind

- The text direction is applied to audio / image token positions, whose hidden states may live in another part of
  the space. Whether it transfers is the question itself; `rand` shows whether any push of the same size does the same.
- A drop in violation together with a drop in required recall is over-suppression, not obedience. Always read both
  columns.
- CFG doubles the batch (pairs) and is batched as pairs of rows that are forced to the same token.
- `transcribe` for V reuses H1's OCR_V answers (94-100% exact copies on pilot1000).
