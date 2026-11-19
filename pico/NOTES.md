# Pico notes (memory across restarts)

## Now
- RUNNING (started 2026-11-19 19:35): tmux `job-select` = `run_v2_all.sh select` (steps 0-2), log pico/logs/select.log
  (ends with 'EXIT <code>'). Expected ~2 h. Next: review images (M1 step 2).
- DONE 19:47: Qwen2.5-Omni-7B and 3B downloaded to HF cache.
- WRITTEN (untested, needs dataset): experiments/common/omnilib.py (loader, batched Omni adapter, answer cache keyed
  by input hash incl. max_new_tokens=100, s/answer + ETA in log) and experiments/h1_v2/run_h1v2.py = H1 CORE
  (Vy 20:20): C0, T ("Q\n\nI"), A, V; one negative sample per unit (seeded draw for sweep units); --pilot 40 per
  branch (=200 questions, 800 answers); --conds T_join C0_box and --forms only LATER, after H1 core is reported.
  M2 order: pilot 200 on 7B (check by eye, measure s/answer, tell Vy ETA) -> full 7B -> report -> 3B.
- WRITTEN + tested on fake data: experiments/h1_v2/analyze.py (tables, McNemar, image bootstrap, H4 Delta, plots).
  Run it with ~/anaconda3/bin/python (base env has matplotlib; `negation` env has none; no installs needed).
- WRITTEN (untested): experiments/common/tts_sentences.py: TTS of a sentence list into its OWN folder (for H2),
  never deletes anything (tts_negation.py deletes per-sample wavs not in --src: do not use it for H2).

## State at hand-over (2026-11-19, set up by Claude with Vy)
- Dataset v2 code is new and only smoke-tested on ~10 images per source (B1 + image selection + visual render
  worked). B2 / baseline / B3 / forms / TTS with the new code have NOT run yet: expect small bugs.
- create_data/dataset/ holds only raw/ (annotations for every split). Everything else was deleted on purpose.
- Targets (config.py): COCO 950, VAW 950, CLEVR 600, GQA all that pass (relation is scarce, ~450-600 expected).
  Val first, train only to fill. B1 candidates seen in a test run: COCO 89 val + 2035 train; CLEVR 5751;
  VAW 239 val + 719 test + 4098 train; GQA 141 val + 977 train. COCO accepted ~70% of candidates.
- HF cache keeps Qwen3-8B, Qwen3-VL-8B, Qwen2.5-VL-3B, whisper-large-v3-turbo. Qwen-Omni must be downloaded.
- Vy's decisions: strict filters kept even though most COCO images come from train2017 (contamination is
  handled later by reporting val vs train separately). Pico reviews the selected images itself.

## Done
(nothing yet)

## Vy's stage plan (Telegram 20:07) — FOLLOW THIS
- After dataset v2: smoke test (run_h1v2.py --pilot 2 = 10 questions), then H1 core on a 1,000-question subset
  (--pilot 200 = 200/branch) for 4 models: Qwen2.5-Omni-7B, -3B, MiniCPM-o 2.6, Phi-4-MM. STOP after the subset,
  report neutral results with CIs on Telegram. NO full-dataset run until Vy decides models/branches.
- Gemma-3n-E4B offered as optional 5th (gated). Told Vy 20:0x.
- MiniCPM-o / Phi-4 need old transformers (4.44.2): env omni_old is broken (no `requests`). Cloning it into a NEW
  env `omni_neg`: DONE 20:22 (clone + pip-installed the missing deps inside it: requests, tqdm, safetensors, einops,
  psutil, filelock, regex, numba, cffi, jinja2, mpmath<1.4...). transformers 4.44.2, CUDA ok.
  Their adapters must be ported into omnilib (items API); run with --generate_only there, score in `negation`.
- MiniCPM-o 2.6 + Phi-4-MM downloaded (20:20).

## Reporting format (Vy 20:20/20:21): Telegram reports AND pico/status.txt in ENGLISH by default (Vietnamese
  only if Vy explicitly asks), short "-" items, one point each,
  blank line between items, no tables, no bold.

## Paper
- ACL 2027 ARR deadline: 2027-01-04 (web search 2026-11-19; verify on 2027.aclweb.org). ~6.5 weeks.
- paper/claims.md, paper/outline.md, paper/related_work.md, paper/ideas.md (I1 neutral audio, I2 comprehension
  probe, I3 target log-prob, I4 position) created 2026-11-19. Read pico/RESEARCH.md each session (§7 WHY, §10 ideas:
  pilot ≤300 samples, report to Vy, scale only after Vy agrees). Paper follows Idea-doc structure (H1,H2,H3-A..D,H4).

## Vy's changes (19:42)
- Reference-model baseline step REMOVED from the pipeline. Steps now: 0 ann, 1 B1, 2 select, 3 B2, 4 B3 (reads
  b2/units.jsonl, target drawn among ALL components), 5 forms, 6 export, 7 visual (+TTS). Experiments condition on
  the tested model's own C0. Dataset only for now; H1 after.
- Keep pico/status.txt current every step (2-4 lines Vietnamese: step+progress, remaining time step/milestone, next).
  Hourly heartbeat posts it. Telegram report after every finished step (pico/notify.py).

## Decisions
- T_sep = "Q\n\nI" (one text string with a blank line). Qwen2.5-Omni's chat template concatenates consecutive
  text blocks with NOTHING in between, so two content blocks would give "Q.Do not..." (worse than T_join).
- C0 answers are cached by input hash, so C0 is shared by template-sweep copies and by the forms of the same unit.

## Open problems / ideas
- Old pilot design confound (not a finding): text glued into the question vs audio as its own segment -> T_sep.
- Vy 20:05: NO v2 results yet; keep every report/note/paper text neutral (no direction) until v2 results with CIs.
- Telegram works since 20:10; catch-up report sent 19:56. Report after every finished step.
