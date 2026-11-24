# PLAN approved by Vy (2026-11-23)

Written by Claude in Vy's terminal from Vy's approval sheet. Vy approves it by sending in Telegram:
"Duyệt, làm theo pico/PLAN.md". Follow it in order. Decisions and file paths behind it are in pico/NOTES.md
(section ">>> CURRENT STATE 2026-11-23 19:15" and the 2026-11-23 bullets); do not re-open decided points.

## Ground rules
- RUN only the stages marked RUN. Stages marked CODE ONLY: write + smoke-test the code on CPU or with
  `--pilot 2`, commit, report, then WAIT for Vy's OK in Telegram before any real run.
- One GPU job at a time, as a tmux session named `job-<name>`, output to `pico/logs/<name>.log`. Sleep until
  about 1 h before the job's ETA (write `pico/sleep_seconds`); never check more often than every 3 h.
- Smoke-test before every real run (`--pilot 2` = 10 questions). If something fails, fix it, note it, and say so
  in the next report; do not silently change the design.
- Reports: short, neutral English, numbers with 95% CIs, no direction claims ("audio is better") from pilots.
- Do NOT drop Phi-4 or the visual channel because of their results. Template N3 stays in all main numbers.
- Keep pico/NOTES.md current after each stage. Commit code with clear messages.
- Claude (Vy's terminal) does not edit the repo while you work, unless Vy asks; if you find edits you did not
  make, read them before changing anything.

## Design in force (all already in code, nothing to redo)
- Conditions: C0, T (instruction as its own text item), A (audio item), V (instruction written inside the
  image). Items after the image are joined with `omnilib.ITEM_SEP = "\n"` in every adapter.
- Primary metric: violation = any mention of the forbidden thing, broken down into 4 kinds:
  STATE (says it) / META (names it to exclude it) / DENY (says it is absent or false) / WRONG (gives a different
  value). Secondary: leak = STATE. Fields in scored rows: `violation`, `leak`, `target_label`, `count_rule`.
- Judge v3 reads the question (create_data/src/judge.py PROMPT_V3); thinking retry with 4096 tokens, else
  `judge_source = think_error`; label pass label-v4 on every judge-compliant row; count rule for bare-number
  answers (count branch; AMBIG kept with the judge and flagged).
- Answers that hit 100 tokens are regenerated with 256 (`omnilib.LONG_NEW_TOKENS`, `AnswerCache.final`).
- Excluded units: `create_data/dataset/out/excluded_units.jsonl` (79 of 1150: double label, part-whole,
  attribute overlap). Pilot = old pilot200 selection minus excluded = 932 questions.
- Bad images (Vy 2026-11-24): never rebuild to drop them; list them in `dataset/out/excluded_images.jsonl` and run
  `create_data/src/exclude_images.py` (adds their units to excluded_units.jsonl, reason bad_image; re-run after flag_excluded_units.py).
- Old results ("pilot v1") are kept as `results/<model>/scored_pilot200_sepv1.jsonl`: history only.

## Stage 0 - smoke test (RUN)
```
cd experiments/h1_v2
NEG=~/anaconda3/envs/negation/bin/python
$NEG -u run_h1v2.py --model qwen2.5-omni-7b --pilot 2 --batch_size 4
~/anaconda3/envs/omni_neg/bin/python -u run_h1v2.py --model minicpm-o-2.6 --pilot 2 --generate_only --batch_size 1
~/anaconda3/envs/omni_phi/bin/python -u run_h1v2.py --model phi-4-mm --pilot 2 --generate_only --batch_size 1
$NEG -u run_h1v2.py --model minicpm-o-2.6 --pilot 2 --score_only
$NEG -u run_h1v2.py --model phi-4-mm --pilot 2 --score_only
```
(qwen2.5-omni-3b: same as 7b.) Check in `results/<model>/scored_pilot2.jsonl`: every row has `target_label`,
`violation`, `leak`; `count_rule` is set on bare-number answers (Phi-4); no crash in the label pass; the
"N answers hit 100 tokens -> regenerate with 256" line appears. Read ~10 rows by eye.

## Stage 1 - H1 pilot v2, 4 models (RUN, ~6-7 GPU h)
```
cd experiments/h1_v2 && tmux new -d -s job-pilot2v2 "bash run_trial.sh > ../../pico/logs/pilot2v2.log 2>&1"
```
run_trial.sh = qwen-7b, qwen-3b (generate+score), minicpm-o and phi-4 (generate in their envs, score in
`negation`). Cached C0/V answers are reused; T/A are generated with the new setup; cut answers regenerated;
everything re-judged. Then:
```
$NEG analyze.py --tag pilot200
$NEG human_check.py score --tag pilot200_sepv1 --judge_tag pilot200 human_check/sheet_pilot200_sepv1_Vy67.csv
```
(Vy's 67 labels of the old sheet; items 10 and 45 would be WRONG under the 5-label scheme: both still count
as violations for the primary kappa.) Report to Vy: per model C0-mentioned n, violation T/A/V, T-A and T-V,
A-V with CIs; violation by kind; leak; N3 table (with / without N3); pilot v1 vs v2 side by side; judge v3 vs
Vy's labels (kappa primary and leak, and which disagreements remain).

## Stage 2 - InternOmni (RUN, ~2.5 GPU h)
- Weights are downloaded: `OpenGVLab/InternOmni` (17 GB, HF cache). README: transformers >= 4.37.2, remote
  code (InternViT-300M + InternLM2.5-7B + Whisper-large-v3 encoder).
- Env: try `omni_neg` (transformers 4.44.2) first. `negation` has transformers 5.17 and will likely break the
  remote code. If a new env is needed, clone one (`conda create --clone`) - never `pip install --user`.
- Add an adapter class to `experiments/common/omnilib.py` (same interface as the others: `tokenizer()`,
  `answer_batch(jobs, max_new_tokens)`, `close()`; image first, then the items joined with ITEM_SEP; audio and
  second-image items) and the key `internomni` in MODELS. Greedy decoding, max_new_tokens as the others.
- Smoke test `--pilot 2`, read the answers (does it actually use the audio? does it follow the chat format?),
  then pilot C0/T/A/V on the 932 questions, score in `negation`, add it to the report.
- Vy may later run InternOmni + Qwen3-Omni on a B200; keep the adapter portable.

## Stage 3 - H4, H3-B, controls on all 5 models (RUN after its code is ready, ~12-14 GPU h)
1. H4: `--forms` (affirmative + exceptive, T/A/V) on the pilot. Report Delta_m = F_aff,m - F_neg,m per channel
   (F = 1 - violation) with CIs, and the exceptive form next to it (Idea doc H4).
2. H3-B (NEW CODE): multiple-choice comprehension per channel, as in the Idea doc. Conditions COMP_T,
   COMP_A, COMP_V: the same image and the same instruction delivery as T / A / V, but the text question is
   replaced by: "According to the instruction [you were given / in the audio / written in the image], which
   one must not be mentioned? (A) ... (B) ... (C) ... Answer with the letter only." Options = the forbidden
   item + 2 other items of the same unit (other components; for count / attribute questions use the slot,
   e.g. "the number of spheres"), order shuffled with a seed per sample. Score by parsing the letter (no
   judge) into `results/<model>/comp_<tag>.jsonl`. Report NC per channel and the violation rate among
   questions the model understood vs did not understand.
3. Controls: `--conds C0_box OCR_V` (code ready; OCR is scored by string match into `ocr_<tag>.jsonl`).

## Stage 4 - H2 and H3-A (CODE ONLY, wait for Vy)
H2 = the main instruction (the question, a positive request) and the negation arrive in DIFFERENT channels,
for all 3 channel pairs and both directions:
| pair | direction 1 | direction 2 |
|---|---|---|
| text-audio | question text + negation audio (= A, exists) | question audio + negation text |
| text-visual | question text + negation in image (= V, exists) | question written in the image + negation text |
| audio-visual | question audio + negation in image | question written in the image + negation audio |
H3-A = the same 6 layouts, but the second instruction is a conflicting POSITIVE one instead of the negation
(e.g. "Describe the cat instead." / "Only tell me about the <other component>."), so the two directions show
general channel dominance without negation. Needs: TTS of the questions and of the H3-A instructions
(CosyVoice2 as in create_data/src/tts_negation.py, same 2 voices, 50/50, ASR check) and images with the
question / H3-A instruction written in them (create_data/src/render_visual.py style, not covering the
subject). Propose the exact H3-A wording to Vy in Telegram before generating audio. Pilot size when approved:
200 questions.

## Stage 5 - H3-C and H3-D (CODE ONLY, wait for Vy)
- H3-C: the negation BEFORE the image vs after it (Idea doc: "negation first" vs "target first"). Needs the
  adapters to accept the main image at a position other than first. Conditions T_first / A_first (text or
  audio instruction, then image, then question) vs T / A; for the visual channel use V_card placed before
  the image. T_pre / A_pre (instruction between image and question) are NOT H3-C. Pilot size when approved:
  200 questions, 5 models.
- H3-D: analysis only: violation vs the forbidden component's `area_ratio` (tertiles; also normalised by the
  C0 mention rate). Note the dataset filter keeps components >= 3% of the image, so "weak evidence" is limited.

## Later (CODE ONLY / wait)
Human check on pilot v2 (Vy 300, Huy 100 overlap; `human_check.py sample` + `html`), voice robustness (real
voices; a child voice only from a dataset with a clear licence), V_card / OCR_card, full dataset build
(~2.5k images), Qwen3-Omni on a B200.
