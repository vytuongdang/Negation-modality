# Pico notes (memory across restarts)

## Now
- WAITING FOR VY (since 2026-11-23). H1 trial done + analysed + reported (Telegram 11-23). Do nothing until a
  "Note from Vy": each step only write 3600 into pico/sleep_seconds. Proposed next (needs Vy OK): comprehension
  probe per channel (ideas I2) + C0_box control on ~200 questions.
- H1 trial result (subset, analyze.py fixed so all 1,000 primary samples count; was 913 due to sweep-unit bug):
  violation among C0-mentioned (C0 ~80%, n~800/model), T / A / V:
  Qwen-7B 9.7 / 6.0 / 82.2 (T-A +3.7 [+1.9,+5.7]); Qwen-3B 54.0 / 44.8 / 86.2 (+9.1 [+6.3,+12.2]);
  MiniCPM-o 46.8 / 20.4 / 93.2 (+26.4 [+22.7,+30.2]); Phi-4 22.8 / 44.6 / 86.5 (-21.7 [-25.1,-18.2]).
  Phi-4 count: required recall 80% -> 22% (T), 34% (A) = over-suppression. Val/train confounded with source.
  Files: experiments/h1_v2/analysis_pilot200.md, experiments/REPORT.md, experiments/figures/h1v2_violation_pilot200.png.
- Trial finished 2026-11-20 09:06 (log pico/logs/h1_trial.log). Runner was stopped 11-20 01:03 to 11-23 16:40.

- Subset build steps 3-7 DONE (log pico/logs/rest_subset.log): B2 kept 1150/1150 (1486 s, human-verify 114);
  B3 1840 negative samples (1035 units + 115 sweep units x7; 22 target collisions skipped); by source/branch COCO
  ex 386, ex_closed 356, count 203; GQA rel 332; VAW attr 203; CLEVR count 195, attr 165. Forms 2070 (exc 1150,
  aff 920). Visual 3910 rendered, occludes_subject 210 (5.4%), font shrunk 445.
- TTS DONE 22:28: 1330 sentences (665 female); 33 (2.5%) ASR mismatches flagged asr_ok=false: 14 riding->"writing"
  (ambiguous for Whisper, maybe also for models), 8 trailing junk syllables ("bus paw", "car sap", "remote Kassel"),
  6 spelling variants (hair drier/dryer, onto/on to), 2 dropped "Answer", 3 other. All 3910 samples have audio,
  visual and box image (checked). Pilot/trial selection uses asr_ok units only.
- Smoke test (--pilot 2, 40 answers) OK for all 4 models; answers and judge scores checked by eye: correct.
  Speeds: Qwen-7B 0.69 s/answer (batch 4), MiniCPM-o 1.64, Phi-4 0.89 (batch 1). Phi-4 needs transformers 4.48.2
  (tokenizer fails in 4.44) -> NEW env `omni_phi` (clone of omni_neg + transformers 4.48.2). MiniCPM-o in omni_neg.
  Smoke-level observation only (n=8-9): V violations high in all models; MiniCPM-o A once listed absent options.
- DONE: tmux `job-h1trial` = experiments/h1_v2/run_trial.sh (Qwen-7B, Qwen-3B full; MiniCPM-o gen in
  omni_neg, Phi-4 gen in omni_phi, then their scoring in negation). Log pico/logs/h1_trial.log ('TRIAL DONE').
  ~3960 answers per model. Judge ~1 h/model (fast 45 min + thinking pass on ~16% disagreements). ETA all ~07:00 11-20.
  Then: analyze.py --tag pilot200 (base anaconda python), Telegram report with CIs, STOP for Vy.
- Experiment code (smoke-tested): experiments/common/omnilib.py (Omni batched; MiniCPM-o, Phi-4 single, env omni_neg
  with --generate_only, score in `negation` with --score_only), experiments/h1_v2/run_h1v2.py (H1 core C0/T/A/V,
  --pilot N per branch, one negative per unit), experiments/h1_v2/analyze.py (run with ~/anaconda3/bin/python),
  experiments/common/tts_sentences.py (H2 TTS into own folder).
- T = "Q\n\nI" (Qwen template joins text blocks with nothing between). max_new_tokens 100. Cache key = input hash.

## Done
- Selection (steps 0-2) finished 21:21 (5981 s). Images: COCO 950 (val2017 60, train 890), CLEVR 600 (val),
  VAW 950 (val 135, test 426, train 389), GQA 654 (val 90, train 564). Units 5,335: COCO existence 924,
  existence_closed 831, count 526; CLEVR attribute 850, count 600; VAW attribute 950; GQA relation 654.
  Rejections (COCO/VAW/GQA): blurry 186/244/130, no_unit_left (VLM check) 135/275/137, large text 64/115/58,
  b&w 33/49/64, clipped, dark/bright, near-dup, cross-source dup 0/12/52.
- Own review (contact sheets of 60 random accepted, 30 rejected; sheets in scratchpad, not kept):
  accepted: 6/60 (10%) doubtful ground truth that passed the VLM check (VAW tail "brown" on a white horse; COCO
  "banana" in a heavily filtered pear photo; dog only on a TV screen; VAW towel "blue"; multicolour umbrella
  "yellow"; VAW "branch" target in a motorcycle photo). ~5 more odd but usable components (GQA redundant
  relations eating/grazing on plant(s), "people holding onto boy", VAW "snow is snowy", "people are wearing black",
  generic name "animal"). Image quality fine: no blurry/dark accepted; some small text (labels, hats "Krispy
  Kreme"), a few filtered/vintage looks. CLEVR all correct (counts checked by eye).
  rejected: sensible; strict (~4/30 look like false rejects: sharp-looking "blurry", small watermark "large text",
  white-background "too bright"). Decision: NO filter change (problems are annotation errors, not image quality;
  borderline 10%; experiments condition on the model's own C0, so a wrong target mostly drops out). Report as
  limitation; B2 human-verify sample can estimate the GT error rate.
- Subset (21:28): subset_units.py --per_branch 230 -> 1,150 units, 1,142 distinct images; by source COCO 579,
  GQA 230, CLEVR 216, VAW 125; val-ish splits small (COCO val 36, GQA val 35, VAW val 15 + test 65, CLEVR val 216)
  -> val vs train check will be weak on the subset.

## Vy's plan (Telegram 20:49, OVERRIDES 20:48) — FOLLOW THIS
- When selection (steps 0-2) finishes: review images, then `cd create_data/src && python subset_units.py
  --per_branch 230` (keeps full list in b1/units_full.jsonl, writes subset to b1/units.jsonl), then
  `bash run_v2_all.sh rest` builds B2/B3/forms/export/visual/TTS for the subset only.
- Then smoke test run_h1v2.py (--pilot 2), then H1 core on the subset (--pilot 200 = 1,000 questions) for ALL 4
  models: Qwen2.5-Omni-7B, -3B, MiniCPM-o 2.6, Phi-4-MM (Vy 20:51, 2-model change reverted; next report: list the 4
  models + which need downloading = none, all 4 are in the HF cache). Report neutral results with CIs
  (English), then STOP. No other model, no full build, until Vy says continue.
- Do NOT build the full dataset until Vy has seen the subset results. (`subset_units.py --restore` later.)
- Caveat: sample_ids of the subset build will differ from a later full build (B3 numbers per build).
- Gemma-3n-E4B offered as optional 5th (gated). Told Vy 20:0x.
- MiniCPM-o / Phi-4 need old transformers (4.44.2): env omni_old is broken (no `requests`). Cloning it into a NEW
  env `omni_neg`: DONE 20:22 (clone + pip-installed the missing deps inside it: requests, tqdm, safetensors, einops,
  psutil, filelock, regex, numba, cffi, jinja2, mpmath<1.4...). transformers 4.44.2, CUDA ok.
  Their adapters must be ported into omnilib (items API); run with --generate_only there, score in `negation`.
- MiniCPM-o 2.6 + Phi-4-MM downloaded (20:20).

## Reporting format (Vy 20:20/20:21): Telegram reports AND pico/status.txt in ENGLISH by default (Vietnamese
  only if Vy explicitly asks), short "-" items, one point each,
  blank line between items, no tables, no bold.

- Vy 00:48: work token-efficiently (long sleeps, tail only, short reports); if usage limit near: short Telegram
  report with state + exact next step, then pause.

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

## State 2026-11-20 01:35 (written by Claude in Vy's terminal session, not by the runner)
- Runner stopped 01:03 (Vy asked STOP at 00:58; it was in usage-limit backoff, so it did not update NOTES or stop the job).
- job-h1trial was NOT stopped and keeps running overnight: Qwen-7B done; Qwen-3B at 3644/4000 answers at 01:30; then judge, MiniCPM-o, Phi-4. Check `tail pico/logs/h1_trial.log`.
- Runner auto-restarts 10:00 (systemd timer pico-restart-10h -> pico/restart_at_10.sh). First step then: check job, update NOTES, one short Telegram report, continue H1 trial plan.
- New quota rules (Vy): runner = Opus 5.5 --effort medium; 10 steps per session; while a GPU job runs, sleep until ~1 h before its ETA, never check more often than every 3 h (sleep_seconds max 43200). Heartbeat reports every 3 h. Telegram bot + email = Sonnet 5.5 medium.
