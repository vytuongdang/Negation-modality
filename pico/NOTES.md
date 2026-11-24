# Pico notes (memory across restarts)

## Now
- VY RULE (2026-11-23): code may be prepared, but NOTHING runs on the GPU until Vy explicitly says OK.
  Prepared, NOT run (commit a5da851): extra conds T_pre, A_pre, V_card (instruction on a white card = 2nd image),
  OCR_V, OCR_card (string match -> results/<model>/ocr_<tag>.jsonl). --conds runs write scored_<tag>_<conds>.jsonl
  and add cached C0. No GPU smoke test yet (Phi-4 2-image prompt untested). Default conds = core, input keys
  unchanged, so the running sepv2 trial is unaffected (checked diff).
- 2026-11-23 17:1x: RUNNING tmux job-h1sepv2 = experiments/h1_v2/run_trial.sh (Vy-approved T/A same setup,
  ITEM_SEP='\n', see pico/note.txt 17:05 + commit 0326287). Only T and A regenerate (2000 answers/model) + judge.
  Log pico/logs/h1_trial_sepv2.log ('TRIAL DONE'). ETA ~4-5 h (~22:00). Old numbers kept: results/*/scored_pilot200_sepv1.jsonl,
  review_*_sepv1.csv, analysis_pilot200_sepv1.md, figures/*_sepv1.png. When done: analyze.py --tag pilot200
  (~/anaconda3/bin/python), short neutral Telegram report new T vs A per model with CIs next to old ones, then STOP for Vy.
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

## State 2026-11-23 17:35 (Claude in Vy's terminal)
- Vy: STOP everything; discuss first what to run, run only after Vy approves.
- job-h1sepv2 (rerun of pilot200 T/A with the ITEM_SEP / T-own-item setup) was killed during qwen2.5-omni-3b generation (~550/2000). qwen2.5-omni-7b finished and is scored: T 11% / A 7% / V 82% (old setup 9.7 / 6.0 / 82.2). Old results kept as results/<model>/scored_pilot200_sepv1.jsonl.
  Resume command (answers are cached, finished ones are skipped): cd experiments/h1_v2 && tmux new -d -s job-h1sepv2 "bash run_trial.sh > ../../pico/logs/h1_trial_sepv2.log 2>&1"
  NOTE: results/qwen2.5-omni-3b|minicpm-o-2.6|phi-4-mm/scored_pilot200.jsonl may still hold the OLD-setup scores until the rerun finishes.
- Runner stopped (pico/stop, tmux pico killed). Extra conditions (T_pre, A_pre, V_card, OCR_V, OCR_card) are prepared, not run.
- 2026-11-23 judge fixes PREPARED (commit above, not run; Vy approves runs): judge.py retries unparseable thinking replies with 4096 tokens, else judge_source=think_error (was silently "fast"; 152/2269 thinking replies had no JSON). run_h1v2 adds target_label (STATE/META/DENY/NONE, LLM label pass only for rows where regex names the target but judge says compliant) + strict_violation; analyze.py prints a strict robustness table. human_check.py: sample 300 blind (model x C0/T/A/V) -> annotate -> kappa. Re-scoring needs only `--score_only` per model (judge cache: only retries + labels are new).
- 2026-11-23 count rule PREPARED (not run): run_h1v2.count_rule - bare-number answers in the count branch are scored by rule (a total that includes the forbidden objects = STATE; both readings = AMBIG, judge kept). On the old trial it changes only Phi-4: 73 rows compliant -> violation, 51 AMBIG. Vy's human check (67 items): agreement 92.5%, kappa 0.83; judge misses bare answers ("Cart") because it never sees the question -> judge v3 with the question is proposed, not done.
- 2026-11-23 DONE (Vy: drop them): COCO double labels (same object, two categories, IoU>0.7), e.g. pizza+sandwich on image 391106; ~121/945 COCO samples involve such a pair (hot dog/sandwich 33, backpack/handbag 19, chair/couch 14, car/truck 12, bowl/cup 11). Proposed filter: drop when the boxes overlap >0.7 AND share a supercategory.
  -> 66/579 COCO units in create_data/dataset/out/excluded_double_label.jsonl; load_dataset() skips them, run_h1v2 keeps the old selection minus them (pilot200: 944 questions), analyze.py drops them from old scored files; B1 filter DOUBLE_LABEL_IOU for rebuilds.
- 2026-11-23 PREPARED: answers that hit 100 tokens are regenerated with 256 (omnilib.LONG_NEW_TOKENS, AnswerCache.final); old trial: ~307 MiniCPM, 33 Qwen-7B.
- 2026-11-23 PREPARED: judge v3 = judge reads the question (create_data/src/judge.py PROMPT_V3, key v3; experiments pass (components, answer, question)); needs a full re-judge (~1 h GPU per model). Check against Vy's 67 labels: human_check.py score --tag pilot200_sepv1 --judge_tag <new tag> <Vy sheet>.
- 2026-11-23 Vy DECIDED: primary metric = violation = ANY mention of the forbidden thing (STATE / META / DENY: stated, named to exclude, denied or a different value), because we measure negation compliance; leak (= correct fact) is secondary. Code ready (run_h1v2 fields violation/leak/target_label; analyze uses violation; label pass runs on every judge-compliant row).
- 2026-11-23 Vy DECIDED: drop unavoidable overlaps. create_data/src/flag_excluded_units.py -> dataset/out/excluded_units.jsonl (replaces excluded_double_label.jsonl): 79/1150 units = double_label 66, part_whole 9 (keyboard in laptop, microwave in oven), attr_overlap 4 (pattern texture with color). Now 1071 questions; pilot200 = 932. Runs at the end of run_v2_all.sh rest.
- 2026-11-23 Vy DECIDED: keep template N3 ('Never refer to') in all main numbers and report it separately (analyze.py table with / without N3); worth a paragraph in the paper (wording effect).
- 2026-11-23 Vy: primary violation broken down by kind for the analysis: STATE / META / DENY (absent/false) / WRONG (different value); analyze.py tables 'Violation by kind' (overall, by branch, paired diffs by kind). Label prompt label-v4.

## >>> STATE 2026-11-24 07:20 (Pico) <<<
- Stage 1 DONE (TRIAL DONE 06:09). analyze.py -> experiments/h1_v2/analysis_pilot200.md, REPORT.md section "H1 pilot v2",
  figure h1v2_violation_pilot200.png. Violation among C0-mentioned (n 772-799), T/A/V: Qwen-7B 17.2/13.0/88.9
  (T-A +4.2 [+1.6,+6.6]); Qwen-3B 63.0/51.1/91.7 (+11.9 [+8.5,+15.5]); MiniCPM 51.1/31.2/95.2 (+20.0 [+16.2,+23.8]);
  Phi-4 26.0/40.9/91.3 (-14.9 [-19.2,-10.8]). V mostly STATE. Phi-4 count recall 74 -> 7% T. Same T-A signs as v1.
- human_check.py --judge_tag FIXED: drops labelled items whose answer was regenerated (18 of 67; T/A were redone).
  Judge v3 vs Vy on 49 identical answers: kappa 0.83, 93.9% (3 disagreements: WRONG count "2 spheres" missed,
  "table" vs "dining table" in C0, one FP). Old judge on all 67 with new metric: kappa 0.76.
- Stage 2 RUNNING since 07:07: tmux job-internomni = experiments/h1_v2/run_internomni.sh (gen omni_intern 3692 answers
  ~1.4 s -> ~1.5 h, then judge in negation ~1.5-2.5 h), log pico/logs/internomni_pilot.log ('TRIAL DONE' / 'TRIAL STOPPED').
  ETA ~10:30-11:00. Then analyze.py (adds internomni), short Stage 2 report, then Stage 3 per PLAN.md.
- Stage 2 DONE 09:41: InternOmni T/A/V 50.1/79.8/95.8 (n 779), T-A -29.8 [-34.0,-25.3], A-V -15.9; recall C0 70, T 74,
  A 27, V 73 -> A answers often short, only about the target (audio taken as question?). COMP_A will check. Reported.
- Stage 3 RUNNING since 10:10: tmux job-stage3, log pico/logs/stage3.log ('SMOKE DONE', 'TRIAL DONE'/'TRIAL STOPPED'). ETA ~23:00-24:00.
- 10:45 COMP v1 INVALID as comprehension (instruction after the MCQ is applied to the answer; Qwen-7B COMP_T 18.7%,
  A 45.1%, V 99.0%). Asked Vy (Telegram) to approve COMP v2 (quoted instruction first + verbatim repeat), see
  paper/ideas.md. Do NOT edit run_h1v2.py/comp.py while job-stage3 runs (it reloads them per model).
- 10:44 Vy APPROVED COMP v2. CODE READY 11:0x: experiments/h1_v2/run_comp2.py = wrapper around run_h1v2 (no edit to
  run_h1v2.py/comp.py): COMP2_T/A/V (instruction quoted first: "Here is an [spoken] instruction:" + item, then MCQ,
  same options as v1) + REP_T/REP_A (repeat word for word) + OCR_V; CPU-checked prompts. run_comp2.sh = smoke --pilot 2
  all 5, then pilot200 all 5, then analyze_comp.py --version v2 (-> analysis_comp2_pilot200.md, NC + verbatim, CIs).
  tmux job-comp2 = run_comp2.sh --wait: WAITS until job-stage3 ends with TRIAL DONE, then starts (log pico/logs/comp2.log).
  If stage3 STOPS, comp2 does not start: fix stage3, then start comp2 by hand. When done: brief neutral Telegram (NC per channel, CIs).
- 11:50 Stage 3 progress: Qwen-7B COMP + C0_box/OCR_V done, forms (H4) running since 11:09. Early peek (1 model, not a
  finding): C0_box target mention 84.5% vs C0 84.8% (n 932 pairs) -> the empty box itself does not change mentioning;
  OCR_V exact 99.8% -> Qwen-7B reads the written instruction yet V violation is ~89%. Full table after all 5 models.
- Stage 3 runner (was prepared 08:15): experiments/h1_v2/run_stage3.sh = smoke --pilot 2 of COMP_T/A/V C0_box OCR_V
  for all 5 models, then per model: COMP (comp_pilot200.jsonl), C0_box+OCR_V, --forms (H4). Launch when Stage 2 is done:
  cd experiments/h1_v2 && tmux new -d -s job-stage3 "bash run_stage3.sh > ../../pico/logs/stage3.log 2>&1" ; ~12-14 h.
  Analysis then: analyze.py --tag pilot200 (H4 section) + analyze_comp.py --tag pilot200 + OCR summary.

## >>> CURRENT STATE 2026-11-23 19:15 (read this first) <<<
- Runner restarted by Claude at Vy's request. NOTHING is running on the GPU. Do NOT start any GPU job until Vy approves it in Telegram (Vy's rule: code may be prepared, runs only after Vy says OK).
- All fixes below are committed and NOT run yet: T as its own item (ITEM_SEP), 256-token regeneration of cut answers, judge v3 (reads the question), thinking retry + think_error flag, count rule for bare numbers, primary metric = violation = any mention split into 4 kinds STATE/META/DENY/WRONG (leak = STATE, secondary), N3 kept + reported separately, 79 units excluded (excluded_units.jsonl; pilot = 932 questions), new conditions T_pre/A_pre/V_card/OCR_V/OCR_card ready, human_check.py + HTML page (Vy labelled 67 items of the old trial: 92.5% agree, kappa 0.83).
- Models: keep all 4 AND add InternOmni (OpenGVLab/InternOmni, InternLM2.5-7B + Whisper encoder, ~8B, fits 24 GB). Vy decided 2026-11-23. Download running in tmux dl-internomni (log pico/logs/dl_internomni.log); adapter not written yet.
- Old trial results = "pilot v1" (results/<model>/scored_pilot200_sepv1.jsonl): history only, NOT main numbers.
- PLAN awaiting Vy's OK, in order:
  1. Pilot v2: finish T/A for qwen-3b / minicpm / phi with the new setup (qwen-7b done), regenerate cut answers (256), re-judge all 4 models (judge v3 + count rule + kinds), analyze.py, check judge v3 against Vy's 67 labels (human_check.py score --judge_tag). ~4-5 GPU h. Run as ONE tmux job-* script; sleep until ~1 h before its ETA.
  2. InternOmni: adapter + 10-question smoke test, then pilot C0/T/A/V.
     ADAPTER PREPARED 2026-11-23 20:xx (omnilib.InternOmni, key `internomni`), NOT run on GPU. Env `omni_intern`
     (clone of omni_neg + sentencepiece, timm; transformers 4.44.2). fp16, eager attention (Turing; repo lacks
     flash_attention.py -> omnilib.internomni_local_dir builds experiments/common/models/internomni with weight
     symlinks + stub, and pre-copies code into the HF modules cache because tf 4.44 skips nested relative imports).
     Prompt = model's internlm2-chat template with its default (Chinese) system prompt, "<image>\n" + items joined
     by ITEM_SEP; 6 tiles + thumbnail at 448 (config max_dynamic_patch); one audio per prompt. Weights downloaded (17 GB). CPU check OK: C0/T/A/
     V_card prompts correct, 1792 img tokens, ~67 audio tokens. Smoke (after Vy OK): cd experiments/h1_v2 &&
     PYTHONNOUSERSITE=1 ~/anaconda3/envs/omni_intern/bin/python -u run_h1v2.py --model internomni --pilot 2
     --generate_only --batch_size 1 ; then score with `negation` env --score_only. Watch fp16 overflow (garbage
     text) and VRAM (17.5 GB weights).
  3. Controls pilot (5 models): OCR_V, OCR_card, C0_box, V_card, T_pre, A_pre.
  4. Human check (Vy 300, Huy 100 overlap) on pilot v2.
  5. Later: full dataset build (~2.5k images); Qwen3-Omni on a second server / rented GPU.
- 2026-11-23 Vy: Idea doc check -> plan must follow H1-H4. H2 and H3-A are REQUIRED and extended to all 3 channel pairs (text-audio, text-visual, audio-visual, both directions; 12 conditions; pilot 200 first). H3-B = multiple-choice comprehension per channel (Idea format). H3-C = negation BEFORE the image vs after. H3-D = analysis by target area (no new run). H4 forms are in the dataset already. Visual = typographic text (Vy's decision vs the Idea's prohibition sign) - state it in the paper. Do NOT drop the visual channel or Phi-4 because of their results.
- 2026-11-23 Vy: voices stay 2 (female/male CosyVoice2 clones) for main results. LATER (stage 6, not now): voice robustness subset - real voices (Vy + Huy ~100 sentences) and maybe a child voice (~200, only from a dataset with a clear licence; never record children).
- 2026-11-23 Vy: InternOmni is NOT urgent ("cứ từ từ"); Vy may run new models on a B200. Plan: this server runs the 4 current models for stages 1,3,4,5; InternOmni + Qwen3-Omni (bf16, fits on B200) run together on the B200 later. Prepare adapters + env spec + one run script so the B200 run is copy-and-run (code/data via git + rsync). Keep the InternOmni download (dl-internomni) for local adapter tests.
- 2026-11-23 Detailed approved plan in pico/PLAN.md (follow it when Vy says 'Duyệt, làm theo pico/PLAN.md').

## >>> STATE 2026-11-23 20:0x (Pico): PLAN.md APPROVED by Vy 19:42 -> follow pico/PLAN.md stage by stage <<<
- Stage 0 DONE 19:50 (experiments/h1_v2/run_smoke.sh, log pico/logs/smoke_v2.log): 5 models incl. InternOmni,
  no crash; all rows have violation/leak/target_label; count rule fires (Phi 3, InternOmni 4); MiniCPM 3 cut ->
  regenerated 256; InternOmni 1.5 s/answer, coherent; its A answers often short + only about the target (n=9:
  maybe it treats the audio as the question; check on pilot). Reported in Telegram.
- Stage 1 RUNNING since 19:52: tmux job-pilot2v2 (run_trial.sh), log pico/logs/pilot2v2.log ('TRIAL DONE').
  ETA ~02:30. Then: analyze.py --tag pilot200 ; human_check.py score --tag pilot200_sepv1 --judge_tag pilot200
  human_check/sheet_pilot200_sepv1_Vy67.csv ; report (PLAN Stage 1 list) ; then Stage 2 InternOmni pilot:
  ~/anaconda3/envs/omni_intern/bin/python -u run_h1v2.py --model internomni --pilot 200 --generate_only
  --batch_size 1, then `negation` --score_only (~932*4*1.5 s = 1.6 h + judge ~1 h).
- Stage 3 code: H3-B READY (CPU-tested, not GPU-smoked): conds COMP_T/COMP_A/COMP_V in run_h1v2 (comp.py;
  options = target + 2 others in the template {X} wording, fillers from the same source, seeded order, gold
  letters balanced A320/B306/C306 on the pilot), scored by letter -> results/<m>/comp_<tag>_COMP_T-COMP_A-COMP_V.jsonl;
  analyze_comp.py --tag pilot200 -> analysis_comp_pilot200.md. Run: --pilot 200 --conds COMP_T COMP_A COMP_V
  (smoke with --pilot 2 first). H4 = --forms (exists). Controls = --conds C0_box OCR_V.
- Vy RULE 2026-11-23 20:01 (REPLACES the 19:45 rule): NO event-by-event Telegram. Progress only via hourly
  heartbeat + pico/status.txt. Telegram report only: one short report per finished stage (incl. CODE ONLY stages
  when code is ready), plus urgent only (crash / job stopped, quota nearly out, decision needed from Vy).
- 2026-11-23 20:10 heartbeat now alerts Vy at once on new error lines in pico/logs/*.log (FAILED/Traceback/OOM/Killed); Claude edited pico/heartbeat.py only.
- 2026-11-23 22:29 CRASH (Vy woke me 22:57): Stage 1 qwen-3b / minicpm / phi failed with UnboundLocalError 'comp'
  (my H3-B edit at ~20:00 while the job ran: module `comp` vs local lambda `comp` in main). qwen-7b finished
  (scored_pilot200.jsonl 22:29, log pico/logs/pilot2v2_run1_qwen7b.log; took 2h40, mostly judge). FIX: lambda
  renamed target_comp; run_trial.sh takes a model list and exits 1 ("TRIAL STOPPED") on the first failure.
  Smoke --pilot 2 OK in negation / omni_neg / omni_phi + score_only. RESTARTED 22:59: job-pilot2v2 =
  run_trial.sh qwen2.5-omni-3b minicpm-o-2.6 phi-4-mm, log pico/logs/pilot2v2.log. ETA ~08:30 11-24 (3 x ~2.5 h
  judge + gen). Vy: check often enough that a crash idles the GPU at most 1 h -> sleep 3600 per check while running.

## >>> 2026-11-24 13:45 Note from Vy: PAPER DRAFT task (CPU only, between GPU checks, do not disturb jobs) <<<
- Step 1: paper/style_notes.md = writing-style notes from B-score (arXiv 2505.18545), VLMs are Biased (2505.23941),
  Deep and shallow biases in LMs (2609.09901): abstract, Fig 1, intro + numbered contributions, related work, method,
  results, limitations; never copy text. Telegram (English, short) when ready.
- Step 2: paper/draft_v1.md (plain markdown) from claims/outline/ideas/related_work/PLAN + Stage 1-3 results;
  every number from result files with 95% CI, marked pilot; [TBD: ...] for unfinished; neutral; Methods = real design
  (C0/T/A/V, violation kinds, judge v3, COMP v1/v2); "What is still missing" list at end. Telegram when ready.
- 14:2x paper/draft_v1.md written + committed (868e3c8; numbers from analysis_pilot200 + new experiments/h1_v2/marginal_cis.py
  -> paper/marginal_cis_pilot200.md; Qwen-7B C0_box-C0 -0.2 [-1.9,+1.6], OCR 930/932). style_notes.md being written by a
  background subagent (if it is missing at the next step: write it yourself from the 3 papers). DONE 14:5x: style_notes.md (subagent, 15-rule checklist) + draft_v1.md rewritten to it; both Telegram reports sent.
  Next for the paper: fill [TBD]s from Stage 3 (H4, OCR_V/C0_box all models) and COMP v2; Figure 1 real outputs.
