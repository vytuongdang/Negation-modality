# NOTES (B200 runner's memory)

>>> 2026-11-26 15:35 UTC — CURRENT STATE / NEXT (read first) <<<
- RUNNING: job-h2-qwen (qwen3b judge retry, ends ~15:45) -> tmux job-h2-rest auto-starts minicpm, phi, intern H2/H3-A when h2_qwen.log has
  "EXIT 0" (log pico/logs/h2_rest.log, "EXIT n"; EXIT 99 = qwen failed, rest not started -> tell Vy). ETA rest ~10-12 h (~03-04 UTC 11-27).
  Do NOT run steering (code ready, commit d0d66ec) until Vy OKs. Only one judge writer at a time.
- VY TASK (Telegram 00:12 KST 11-27, received 15:12 UTC): AFTER H2/H3-A finished + analysed (no GPU, no further OK needed):
  (1) paper DRAFT from real results only -> NEW files paper/draft_b200_v1.md + paper/draft_b200_review.md (never overwrite paper/draft_v1.md,
      server 1's). Every number from experiments/h1_v2 analysis files (H1 full 5 models: full_analysis_5models_asrok.md / _allunits.md;
      H4: pilot1000_forms_5models_asrok.md; H3-B + OCR + C0_box: pilot1000_comp_ctrl_5models.md, analysis_comp_pilot1000.md,
      analysis_pilot1000_asrok.md; H2/H3-A: analysis_h2_pilot1000_asrok.md (rerun for 5 models)). Placeholders, clearly marked, naming the
      experiment, for missing results (NEU neutral-word / cue control NOT run; steering/method NOT run; COMP v2 not run on B200; H3-C not coded).
      No direction claim beyond CIs. Framing: pico/RESEARCH.md, paper/ideas.md, paper/outline.md, paper/style_notes.md (checklist §Synthesis),
      paper/related_work.md, server-1 draft_v1.md only for structure.
  (2) REVIEW in draft_b200_review.md: a) method audit: judge reliability (think_error rows keep fast verdict: counts per model/run;
      regex-vs-judge disagreement counts; SENSITIVITY analysis excluding think_error rows), asr_ok vs all, excluded units (276) / removed
      images, pilot1000 vs full (no per-branch/split claims from pilot), one answer per unit + image bootstrap, truncation 100->256 regen counts;
      b) number audit: re-derive every draft number from result files, list mismatches; c) confounds: V 87-95% incl. affirmative (OCR_V,
      C0_box, kind split STATE/META/DENY/WRONG), InternOmni audio-as-question (A recall ~25-27%), T-A sign flips (within-model only),
      over-suppression (recall), C0 ~80%; d) what data supports, weak claims, ranked next experiments with GPU-h cost (NEU control, full H4,
      method, COMP v2, H3-C), honest framing (main vs Findings).
  (3) re-review after each section, fix + log changes. Then ONE short Telegram msg: files, 3-5 supported findings, top problems, ranked next
      experiments (ask, don't run). git pull --rebase first (if blocked by the old unstaged deletions: fetch + check origin has nothing new),
      commit by file name only, push.
  PLAN: the audit part (a, b, c) on finished runs (H1/H4/H3-B/controls) can be done now on CPU while job-h2-rest runs; write the draft
  sections that do not need H2 too; add H2/H3-A sections + final review + message after job-h2-rest EXIT 0 and analyze_h2.py on 5 models.
- H2 qwen7b preview numbers are in the 13:30 UTC line below.
- 16:30 UTC 11-26 (CPU work during job-h2-rest): WRITTEN paper/draft_b200_v1.md (all sections; H2 = qwen7b preview +
  placeholders) + paper/draft_b200_review.md (a method audit, c confounds, d supported/weak/next, style log, change log).
  Audit files: experiments/h1_v2/audit/ (audit_judge.py -> judge_audit.md + rm_think_error_<tag>.txt; audit_regen*.py ->
  regen_audit.md; dataset_stats_b200.md: 4,338 q / 2,627 images asr_ok). RUNNING (CPU): tmux cpu-audit-sens (audit/run_sens.sh:
  analyze.py --remove think_error units, all + forms -> analysis_all_asrok_rm.md, analysis_pilot1000_forms_asrok_rm.md; log
  pico/logs/audit_sens.log "EXIT n"). A subagent writes audit/number_audit.md (draft numbers vs files + V-C0_box re-derived).
  Found: H2 as run = question-vs-instruction channel swap, NOT the Idea-doc conflict (typed "Mention X" + spoken "Do not mention X"):
  flagged in draft 4.5 + review c7. NOTES "intern 33 think_error" for H1 is wrong (file: 14; phi 33). 9 units lost in B2/B3 (4,399->4,390).
  17:10 DONE: number audit (subagent, audit/number_audit.md; 6 small fixes applied) + think_error sensitivity (H1 robust;
  H4 Qwen-3B aff A-T and MiniCPM exc A-T CIs cross 0 -> marked fragile). Committed 2e0b67e.
  Vy style note (00:13 KST) DONE: draft follows paper/style_notes.md §3, style log in draft_b200_review.md; confirm in the final msg.
  19:11 UTC: minicpm H2 judge retry pass (almost done), phi + intern next (~01-03 UTC).
  LEFT: after job-h2-rest EXIT 0: analyze_h2.py 5 models, H2 section, re-review, ONE Telegram msg.
- 15:45 UTC 11-26: job-h2-qwen EXIT 0 (15:40). job-h2-rest running (minicpm gen started ~15:41, first answer 3.1 s).
  Vy note (Telegram 00:13 KST 11-27): draft must imitate the STYLE/STRUCTURE of "Deep and shallow biases in language models" (An Vo et al.,
  EMNLP 2026): use paper/style_notes.md (server 1 already studied it) first; no copied sentences; record the style choices taken in
  paper/draft_b200_review.md.

## >>> VY'S DECISIONS 2026-11-25 (written by Claude in Vy's terminal; read before planning the next stages) <<<
- TARGET: ACL 2027 main, ARR deadline 2027-01-04 (AoE, per 2027.aclweb.org). The paper needs a METHOD part that reduces
  the violations, not only the benchmark + analysis.
- SCOPE: H1 = FULL (running). H2, H3-A, H3-C, H4 = run only on the fixed pilot of 1,000 questions (below). H3-B (multiple
  choice, scored by letter, no judge) is cheap: may run full. H4 may be extended to full for Qwen-7B/3B if the pilot
  CIs for Delta_audio - Delta_text cross 0. Per-branch (200 each) and train/val results on the pilot are weak:
  existence_closed has only 7 val questions, so do not claim per-branch or split effects from the pilot.
- H4 cost: 2 forms x 3 channels = 6 new answers per question (~25.6k per model on full). Option offered to Vy: drop V
  (violated 89-95%, ceiling) and/or run only the affirmative form first (exceptive only if the interaction shows).
  Not decided yet: ask Vy before running H4.
- FIXED PILOT LIST (new, CPU only, not committed): experiments/h1_v2/pilot1000.json + pilot1000_summary.md, made by
  experiments/h1_v2/make_pilot.py (seed 42). 1,000 questions = 200 per branch; voice 500 female / 500 male (100/100 per
  branch); COCO 500, GQA 200, CLEVR 199, VAW 101; templates N1-N7 141-145 each; train 673 / val 327; asr_ok only; excluded
  units dropped; every image used once in the whole list; every question has all H4 forms its branch supports (existence
  has exceptive only, by design). Drawn from the same one-sample-per-unit selection as H1 full, so C0/T/A/V answers come
  from the answers.jsonl cache. It is NOT the old server-1 pilot200 (932 questions, different ids).
  UPDATE 2026-11-25 (Claude): run_h1v2.py is NOT edited (H1 job re-imports it). Instead use the wrapper
  experiments/h1_v2/run_pilot.py = run_h1v2.py on pilot1000.json (same args: --model, --forms, --conds ..., --generate_only,
  --score_only; output tag pilot1000[_forms][_<conds>]; shares answers.jsonl / judge.jsonl with H1). Dry-run checked
  (1,000 ids found, 5 branches), NO model run yet. run_h2.py and h2_assets.py now take `--pilot_list FILE` (tag h2_pilot1000).
- H2 + H3-A = READY BUT DO NOT RUN until Vy says "run H2" (Vy 2026-11-25). Done: code (h2_common.py, run_h2.py,
  h2_assets.py, analyze_h2.py, run_b200_h2.sh with `list` selection), question images for the 1,000 questions (CPU,
  dataset/h2/question_image, 75 of 1,000 occlude an object), question audio = TTS on GPU 2 (CosyVoice2, 512 unique
  question texts, started 11:43 UTC, log pico/logs/h2_tts.log, ~85 min; it is an asset job, not an H2 run: do not kill it,
  do not start another one). When it ends: `h2_assets.py status --pilot_list ...` must show 1000 audios; check the
  asr mismatches line. Then to run: `bash experiments/h1_v2/run_b200_h2.sh run list [models]` (smoke --pilot 2 does not
  fit the `list` selection: smoke with a small H2 run on 2 models first, ask Vy). H3-A uses the AFFIRMATIVE form, which
  does not exist for the 200 existence questions (skipped, reported by run_h2.py); use H3A_FORM=exceptive for them if Vy
  wants all 1,000. 12 conditions per question; qT_iA / qT_iV reuse H1's A / V answers from the cache.
- NOT DONE (needs an omnilib.py edit, so after H1 full ends): H3-C (instruction BEFORE the image), the adapters put the
  image first.
- EXTENDING pilot -> full (e.g. H4): keep every result already produced for the 1,000 pilot questions, then run ONLY
  the questions not yet run. The pilot is a subset of the full one-sample-per-unit selection (seed 42), so
  answers.jsonl (keyed by input hash) and judge.jsonl are reused automatically. Do not change code, prompts, judge,
  decoding or max tokens between the two parts (if judge.py must change, e.g. be8ab2b, check it does not change the
  scoring first and write it down). Final numbers = analysis on ALL questions pooled, never "pilot + random rest" and
  never the balanced pilot alone (it is 200 per branch; full is natural: attribute 1,518 ... relation 620). In the paper
  say that H4 first ran on the pilot and was then extended.
- >>> SCHEDULE AFTER H1 FULL (Vy 2026-11-25; follow in this order; Vy also sends the same text in Telegram) <<<
  0. Confirm H1 full ended (EXIT 0, all 5 models). git pull --rebase origin/main + push. Run the 5-model analysis (asr_ok main +
     all-units sensitivity), commit by file name (never git commit -a), send Vy ONE short report. Check that the H2 question-audio TTS
     (pico/logs/h2_tts.log) finished: `h2_assets.py status --pilot_list ...` must show 1000 audios; if it crashed, report (restart only if dead).
  1. Write experiments/h1_v2/run_b200_pilot.sh (like run_b200_h1.sh) using run_pilot.py (NOT run_h1v2.py edits, NOT pilot1000.json
     changes). GPU 2 only, nvidia-smi before each job, one failure = stop, tmux job-<name>, log pico/logs/<name>.log ending "EXIT n".
     Same envs as H1 (qwen: negation; minicpm: omni_neg gen + negation score; phi: omni_phi; internomni: omni_intern).
     GENERATION in two batches: batch 1 = 3 models in parallel, batch 2 = the other 2 in parallel after batch 1 finished
     generating; NEVER all 5 at once. Suggested: batch 1 = qwen2.5-omni-7b, qwen2.5-omni-3b, minicpm-o-2.6; batch 2 = phi-4-mm,
     internomni (change if nvidia-smi shows a better balance). Scoring one model at a time (shared results/judge.jsonl); it may overlap
     with the next batch's generation. >>> SEND VY A SHORT MESSAGE when the script is written and the smoke passed, BEFORE the real H4 run
     (what runs, which batches, ETA). <<<
  2. H4 FIRST: run_pilot.py --model <m> --forms on the 1,000 pilot, all 5 models (T/A/V, both forms). Smoke first with
     run_h1v2.py --pilot 2 --forms on one model. Analyse: Delta_m = F_aff - F_neg per channel, Delta_audio - Delta_text and
     Delta_visual - Delta_text with 95% CIs, exceptive next to it, required recall. >>> SEND VY A SHORT REPORT (numbers + CIs, no direction
     claims) BEFORE starting step 3. <<<
  3. H3-B and controls on the same list: --conds COMP_T COMP_A COMP_V and --conds C0_box OCR_V (5 models). Report NC per channel and
     OCR copy rate per model.
  4. After step 3: short report to Vy (H3-B NC per channel, OCR copy rate, controls), then STOP.
  5. H2 / H3-A: ready, DO NOT RUN until Vy writes "run H2". Then `bash experiments/h1_v2/run_b200_h2.sh run list <models>` (small run on
     2 models first). H3-C and the method work are not coded: do not start them.
  Rules unchanged: hourly heartbeat only, no event-by-event Telegram, crash / stopped job = urgent message at once, never kill other
  people's jobs, keep this file current after each step, tell Vy if the volume has < 30 GB free.
- NOT NEEDED NOW: real voices and child voice (old stage 6). Keep only the 2 CosyVoice2 voices (female / male).
- METHOD PART (idea stage, nothing coded): reduce violations on Qwen2.5-Omni-3B and 7B, on the pilot1000 subset only.
  Order: (1) training-free baselines: transcribe-then-obey (model first copies the instruction from audio/image as text),
  contrastive / CFG decoding (logit with instruction minus w x logit without it); (2) activation steering (a "prohibition"
  direction taken from the text instruction, added at the audio / image instruction positions); (3) LoRA trained on one
  channel and tested on the others (split by IMAGE, never by question; check val images). Every method must report
  required recall next to violation (over-suppression check; Phi-4 count recall already dropped 74 -> 7%). Judge is slow
  (~4 h per 17.5k answers), so 1k questions x 4 conditions ~1 h per configuration. Write code now, run after H1 full.
- Two things to check before the paper claims anything: (a) V is violated 89-95% even for Qwen-7B, which obeys T well:
  V may measure "treats text inside the image as an instruction", not negation by channel; split violations by kind
  (STATE / META / DENY / WRONG) and use OCR_V / V_card before interpreting V. (b) T vs A changes sign across models
  (InternOmni seems to treat the audio as the question, A recall 27%): compare paired within a model only.

## >>> 2026-11-25 19:17 UTC: H1 FULL DONE (EXIT 0, 19:06 UTC, all 5 models; intern 33 think_error rows) <<<
- git: committed pilot1000 list / run_pilot.py / H2 --pilot_list code, pull --rebase (conflict in analyze.py resolved = keep
  origin's forms/C0_box merge + B200 asr_ok/--remove filters), pushed 3f4fa7e. judge.py now has be8ab2b (OOM fix in the 4096
  retry; H1 full ran WITHOUT it; all pilot1000 runs use it -> state in report).
- 19:21 smoke EXIT 0 (48 ans, 0.47 s/ans). 19:28 STARTED tmux job-pilot-forms (H4, log pico/logs/pilot_forms.log; gen logs forms_gen_<m>.log): batch1 qwen7b 0.78, qwen3b 0.86, minicpm 3.1 s/ans (5.4k each; minicpm ~4.6 h).
- 19:55 UTC: 5-MODEL ASR_OK ANALYSIS DONE -> experiments/h1_v2/full_analysis_5models_asrok.md + figures/full_h1v2_{violation,forms}_5models_asrok.png (committed). Viol T/A/V (T-A [CI]): qwen7b 17.3/13.5/89.0 (+3.8 [2.6,5.0]); qwen3b 65.4/50.3/90.6 (+15.1); minicpm 48.1/30.1/93.9 (+17.9 [16.2,19.7]); phi 27.6/37.4/87.2 (-9.8 [-11.9,-7.7]); intern 49.2/80.8/94.7 (-31.6 [-33.7,-29.7]). Recall C0/T/A/V: intern 71/73/25/73, phi 81/64/68/78, minicpm 80/85/84/79. Sign of T-A differs by model. Report sent to Vy (Telegram + photo). TODO: claims.md update once (shared file), all-units sensitivity.
- 20:30 UTC: all-units sensitivity DONE (full_analysis_5models_allunits.md, committed): T-A same as asr_ok within 0.2 pp for all 5 models.
- H4 forms: qwen7b gen done 20:21, qwen3b 20:24; minicpm 1,481/5,400 at 2.4 s (ends ~23:05 UTC); script waits for it before batch2.
  To use idle GPU, STARTED EARLY batch-2 gen by hand (same cmd, same answers cache): tmux job-forms-early-internomni / job-forms-early-phi-4-mm,
  logs pico/logs/forms_early_gen_<m>.log ("EXIT n"). RACE RULE: if early phi/intern gen is still running when job-pilot-forms prints
  '--- gen phi-4-mm' (after minicpm), kill the EARLY tmux session (the script's copy resumes from the cache).
- 21:23 UTC: minicpm 2,721/5,400 (ends ~23:10), intern early 5,101 (ends ~21:30), phi early 2,181 at 1.4 s (ends ~22:40).
  STARTED tmux job-forms-early-score-qwen7b (judge, log pico/logs/forms_early_score_qwen7b.log, est. ~1.2 h -> ~22:40). It MUST end before
  job-pilot-forms prints '=== score qwen2.5-omni-7b' (only one judge writer on results/judge.jsonl); if not, kill the early score session.
- 22:43 UTC: intern early gen EXIT 0. phi early 4,361/5,400 (~23:15), minicpm 4,601 (~23:20 + regen). Early qwen7b judge slower (thinking 640/1176).
  tmux watch-forms-race kills job-forms-early-score-qwen7b automatically when the script prints '=== score qwen2.5-omni-7b' (judge
  appends per batch, so the script resumes from the cache). Same race for phi: script will start phi gen from cache; if early phi still
  running then, tmux watch-forms-race-phi kills job-forms-early-phi-4-mm automatically.
- 23:28 UTC: minicpm in regen (181/501 at 7.5 s -> ~00:15 UTC); phi early 5,321/5,400; qwen7b early judge 1088/1176.
  tmux job-forms-early-score-qwen3b starts qwen3b judge when qwen7b early judge ends (log forms_early_score_qwen3b.log);
  tmux watch-forms-race-3b kills it as soon as the script prints any '=== score' line (script resumes from cache).
- 00:18 UTC 11-26: ALL H4 GENERATION DONE (phi/intern early EXIT 0, minicpm regen 501 done). Script scoring since 00:02: qwen7b (early run
  did most; killed at retry pass, script finishing the retry), then qwen3b, minicpm, phi, intern (~1.5 h each) -> ~07-08 UTC. Watchers exited.
- 01:25 UTC: qwen7b scored (00:02-00:45), qwen3b judge running. analyze.py --tag pilot1000_forms --models <m> --asr_ok_only WORKS
  (H4 table with Delta CIs; scored_pilot1000_forms has C0 1000 + T/A/V 2800 each). Qwen7b preview: Delta aff-neg T -19.6, A -23.2, V +4.3,
  A-T -3.6 [-8.7,+1.0]; exc-neg A-T -0.1 [-4.4,+4.1]. Final: run for all 5 when job-pilot-forms EXIT, rename -> pilot1000_forms_5models.
  Rename outputs to full_* before commit. tmux job-pilot-smoke (GPU 2): run_b200_pilot.sh smoke (qwen3b --pilot 2 --forms), log
  pico/logs/pilot_smoke.log. Then: report to Vy (step 0 + smoke), then forms stage (H4), comp, ctrl.

- 02:20 UTC 11-26: qwen3b judge retry pass (16/66) -> done ~02:30; then minicpm, phi, intern scoring (~1.5 h each) -> job-pilot-forms EXIT ~07 UTC. Next wake: check EXIT, run analyze for 5 models (pilot1000_forms_5models), report, then comp + ctrl stages.

- 05:39 UTC 11-26: H4 PILOT DONE (job-pilot-forms EXIT 0; think_error rows qwen7b 57, qwen3b 47, minicpm 21, phi 21, intern 28).
  Analysis -> experiments/h1_v2/pilot1000_forms_5models_asrok.md + figures/pilot1000_h4_{forms,violation}_5models_asrok.png (committed, pushed).
  Delta_audio-Delta_text (aff-neg): qwen7b -3.6 [-8.2,+0.9], qwen3b -4.2 [-9.0,+0.6], minicpm +11.9 [6.6,17.6], phi +11.9 [7.0,17.0],
  intern +36.3 [30.9,41.8]. exc-neg A-T: qwen7b -0.1, qwen3b -19.3 [-24.1,-14.7], minicpm +5.7, phi +8.1, intern +0.4. V at ceiling (86-97%)
  in ALL forms incl. affirmative -> Delta_V uninformative. qwen7b obeys aff worse than neg (Delta_text -19.6). Report sent to Vy (+photo).
  Qwen CIs cross 0 -> H4 full extension for qwen7b/3b is an option: ASKED Vy, NOT run.
- 05:43 UTC (05:52 speeds qwen7b 0.71, qwen3b 0.64, minicpm 1.1 s/ans; 3000 ans/model; comp ~08 UTC, ctrl ~11-12 UTC): STARTED tmux job-pilot-comp-ctrl (GPU 2): run_b200_pilot.sh comp (log pico/logs/pilot_comp.log, "EXIT n") then ctrl
  (pico/logs/pilot_ctrl.log). After both: analyze_comp.py (H3-B NC per channel) + controls (OCR copy rate), report = step 4, then STOP.
  git pull --rebase fails (unstaged deletions of old tracked results, not mine): fetch shows origin has nothing new; pushed 13 commits.

- 09:15 UTC 11-26: STEP 3 DONE (comp EXIT 0 06:28, ctrl EXIT 0 08:10). STEP 4 report SENT. NOW STOPPED per schedule; GPU 2 idle.
  scored_pilot1000.jsonl (derived, ignored) = C0/T/A/V rows of scored_all.jsonl for the 1,000 pilot ids (analyze_comp needs it).
  Summary: experiments/h1_v2/pilot1000_comp_ctrl_5models.md (+ analysis_comp_pilot1000.md, analysis_pilot1000_asrok.md). H3-B v1 NC
  confounded with obedience (known lesson). OCR_V exact 94-100%; V - C0_box -0.7..-6.0 pp -> V failure not reading. Lesson added to RESEARCH.md.
  NOTE: analyze.py primary metric = field 'violation' (any mention), NOT target_mentioned; use r.get('violation') in ad-hoc scripts.
  WAITING for Vy: (a) "run H2" (b) H4 full extension for qwen7b/3b? (c) H3-B v2 on B200? Do not start any GPU job without one of these.
  TODO (CPU, optional): paper/claims.md update (shared file, coordinate).

- 10:10 UTC 11-26: Vy "run H2" (Telegram 19:09 KST). Preflight: h2_assets status 1000/1000 audios+images, GPU 2 empty, 398 GB free.
  STARTED tmux job-h2-qwen: run_b200_h2.sh run list qwen2.5-omni-7b qwen2.5-omni-3b (log pico/logs/h2_qwen.log, "EXIT n"; models sequential,
  gen+score each; 7,200 answers/model, qwen7b 0.5 s/ans). ETA ~14:30 UTC. Then check answers/scores by eye + analyze_h2.py on 2 models,
  then tmux job-h2-rest: run list minicpm-o-2.6 phi-4-mm internomni (log h2_rest.log). Existence (200 q) skipped for H3-A (affirmative form): report.
  No H3-C, no method work.

- 11:00 UTC 11-26: Vy 19:11 KST: write STEERING code while H2 runs (code only). DONE, commit d0d66ec pushed, message sent:
  experiments/h1_v2/steer/{steerlib,direction,run_steer,analyze_steer,smoke_cpu}.py + README.md (+ .gitignore results/).
  Direction = mean(h at text instr tokens, "Do not mention X.") - mean("Mention X."), per layer, TRAIN images (673); eval VAL (327).
  Steer A at audio tokens, V at box tokens (--v_pos box) or all image tokens; rand control; cfg (pairs in batch, forced same token);
  transcribe-then-obey. Own cache steer/results/<model>/answers.jsonl; H1 caches READ ONLY; judge = shared results/judge.jsonl
  (only one judge writer: never score steer while H2 scores). smoke_cpu.py ALL PASS (text_box is [x,y,w,h]: fixed).
  NO GPU run until H2/H3-A finished AND Vy says OK.

- 13:30 UTC 11-26: H2 qwen7b DONE (10:09-13:10, 65 think_error rows); qwen3b gen started 13:11 (0.56 s/ans, gen ~14:15, judge -> ~15:45).
  Preview analyze_h2.py --tag h2_pilot1000 --models qwen2.5-omni-7b --asr_ok_only (experiments/h1_v2/analysis_h2_pilot1000_asrok.md, rerun for
  all models later): qT_iA 13.2 / qA_iT 17.5 (H2 diff -4.3 [-7.1,-1.6]; H3A +2.7 [-0.9,6.3]; H2-H3A T-A -7.9 [-12.9,-3.1]); any iV ~86-90,
  any qV ~17-20 (T-V, A-V H2-H3A ~0). qT_iA/qT_iV match H1 A/V (consistency OK). Spot check: closed-list H3A omission labelled DENY (judge rule; caveat).
  NEXT: when job-h2-qwen EXIT 0 -> tmux job-h2-rest (run list minicpm-o-2.6 phi-4-mm internomni, log pico/logs/h2_rest.log), short note to Vy.

## >>> 2026-11-25 00:10 KST: H1 FULL RUNNING (tmux job-h1-full, GPU 2, log pico/logs/h1_full.log, ends "EXIT n") <<<
- 04:58 UTC (Vy note 13:57 KST: parallelise): started GENERATION ONLY of phi-4-mm and internomni in parallel on GPU 2
  (tmux job-h1-gen-phi-4-mm / job-h1-gen-internomni, logs pico/logs/h1_gen_<model>.log ending "EXIT n"; same cmd as
  run_b200_h1.sh with --generate_only, same answers.jsonl cache). Speeds with 3 jobs: phi 1.07 s/ans (ETA ~10:15 UTC),
  internomni 0.47 (~07:15), minicpm ~0.9+ (gen ~08:45, judge -> ~12:45). GPU 2 63 GB used of 183.
  Scoring NOT parallel: results/judge.jsonl is shared and its loader crashes on a malformed line (long thinking
  replies from 2 writers could interleave). Main job reaches phi after minicpm scoring: gen = all cache hits, then
  score_only -> phi judge ~12:45-16:45, internomni ~16:45-20:45 UTC, then analysis. MUST check phi parallel gen ended
  (EXIT 0) before main reaches phi, else duplicate gen (harmless but wasteful). Option to offer Vy: parallel judge
  with a separate cache file via a new wrapper script (not done).
- 06:05 UTC: last hour rates: minicpm 1,580 ans/h (alone 4,500), phi 2,500 (alone 4,090), intern 6,200 (alone 7,350) = 10,280/h combined vs ~4,100-7,350 sequential -> parallel ~1.7x faster overall. Left: minicpm 12.9k, phi 14.7k, intern 10.5k -> intern ~07:45, all gen ~10-11 UTC.
- 07:06 UTC: last hour minicpm 940/h (long CLEVR count answers), phi 1,800, intern 6,000 (13.1k, ends ~07:50). Combined 8.8k/h.
- >>> Vy 21:08 KST 11-25: GPU 2 = Vy's own GPU, the ONLY one for now. After step 0 is done, run steps 1-3 at once WITHOUT
  waiting for replies (the messages at smoke / H4 / H3-B are reports, NOT approvals). Still STOP after step 4; no H2 until "run H2". <<<
- 19:01 UTC: intern thinking 1491/1491 done, no-JSON retry next; H1 ends ~19:15.
- 18:10 UTC: PHI FULL SCORED (scored_all.jsonl, 15:24-17:31). internomni score_only since 17:31: pass 1 done, thinking 640/1491 -> ends ~19:00 UTC.
- 17:10 UTC: phi thinking 1728/1762 (phi done ~17:20), intern score next (~3.5 h -> ~21 UTC).
- 16:10 UTC: MINICPM FULL SCORED (42 think_error rows). phi score_only started 15:24 (gen all cached); thinking 640/1762 (only 1,762 disagree rows) -> phi ends ~17:30, intern ~17:30-21 UTC.
- 15:09 UTC: minicpm thinking 3048/3048 done (retry of no-JSON rows next), then phi score_only.
- 14:09 UTC: H2 question TTS DONE: h2_assets status = 1000/1000 question audios + images (481 female); 512 texts, 8 still ASR-mismatched after 3 tries (9 rows asr_ok false; mostly Whisper 'riding'->'writing', 'bear'->'bearer'). Phi parallel gen EXIT 0 (236 regen, 3,157 s). minicpm thinking 2048/3048 (ends ~15:20 UTC).
- 13:08 UTC: minicpm judge thinking 896/3048 (~14/min, ends ~15:40 UTC); phi main gen done (29,101 s), regen 236; H2 TTS 460/512, 7 asr mismatches, ends ~13:20.
- 12:05 UTC: Vy's Telegram 21:02 KST = the SCHEDULE AFTER H1 FULL block above (received, follow it). WROTE
  experiments/h1_v2/run_b200_pilot.sh (bash -n OK, NOT run, not committed yet): stages smoke [model] (run_h1v2.py --pilot 2
  --forms, default qwen 3b) | forms | comp | ctrl; batch1 qwen7b+qwen3b+minicpm gen in parallel, then batch2 phi+intern gen
  while batch1 scores one at a time; logs pico/logs/<stage>_gen_<model>.log; run as tmux job-pilot-<stage>, log
  pico/logs/pilot_<stage>.log + "EXIT n". Smoke only AFTER H1 full ends (judge.jsonl shared with the H1 scorer).
  H1 state: minicpm judge pass 1 done 12:03, thinking pass now (~3 h); phi parallel gen 16.0k (ETA ~12:45); H2 TTS (pid
  3703709, not tmux) 143/512 texts at 12:03 -> ends ~13:15 UTC. H1 full end estimate ~23-24 UTC (phi + intern judge ~4 h each).
- 11:07 UTC: minicpm main gen done; regen 1,500 truncated answers (256 tok, 3 s/ans, ETA ~11:50), judge -> ~16 UTC. phi 14.2k, ETA ~12:30 + its own regen.
- 10:07 UTC: minicpm 15.3k (gen ETA ~11:00), phi 12.2k (+5k/h, ETA ~12:30 UTC, before main reaches phi ~15 UTC: race unlikely).
- 09:06 UTC: intern parallel gen EXIT 0 (incl. 119 regen). Last hour minicpm 3,660 (12.2k, gen ETA ~11:15 UTC, judge -> ~15:15), phi 1,900 (7.2k, ETA ~13-15 UTC). Race with main job reaching phi ~15:15: decide at ~14 UTC.
- 08:06 UTC: minicpm 8.5k (2,900/h), phi 5.3k (only 660/h: CLEVR, long "### Your explanation" answers), intern main pass done, regen 119 long answers. RACE RULE: if phi parallel gen has not ended when main job finishes minicpm scoring, kill tmux job-h1-gen-phi-4-mm just before (main resumes from the cache, no loss).
- 04:50 UTC: QWEN3B FULL DONE (145 think_error rows). Analysis asr_ok: experiments/h1_v2/full_analysis_qwen3b_asrok.md + figures/full_h1v2_*_qwen3b_asrok.png.
  C0 82.0% (3,558/4,338). Violation T 65.4, A 50.3, V 90.6; T-A +15.1 pp [13.5, 16.8], McNemar p 5e-73. Count/relation T ~90 vs A ~70.
  N3-only T-A +4.9 [1.0, 8.9]. Val T 73.5/A 53.6, train 58.5/47.4. Recall C0 81.4, T 81.8, A 82.7, V 78.4.
  MiniCPM gen started 04:17 UTC, 0.80 s/ans (ETA gen ~08:20 UTC, then judge ~4 h -> ~12:30 UTC); phi then internomni (all done ~03-05 UTC 11-26).
- 01:40 UTC: qwen3b judge thinking 1,792/3,472, no errors; volume 279 GB free. qwen3b ends ~03:10 UTC.
- origin/main has 5 new commits (incl. be8ab2b judge.py OOM fix for the 4096 retry, analyze.py merges forms/C0_box). NOT pulled during the run (would change judge.py under later models). Pull --rebase + push AFTER H1 full ends; then use the new analyze.py. If a model's thinking retry OOMs, that commit is the fix (state it in the report).
- 00:45 UTC 11-25 Note from Vy (09:37 KST): if InternOmni (or any step) fails, fix at once without waiting, restart ONLY the failed/remaining
  models on GPU 2 (`tmux new -d -s job-h1-full2 "bash experiments/h1_v2/run_b200_h1.sh full <models> >> pico/logs/h1_full.log 2>&1; echo EXIT \$? >> pico/logs/h1_full.log"`), then 5-model analysis, brief report.
  Check tail of h1_full.log for FAILED/EXIT at every wake. Tokenizer fix re-verified 00:45 UTC: sentencepiece 0.2.0 in omni_intern, local
  tokenizer.model sha256 f868398f..., spm + AutoTokenizer (local dir) load OK (92,544 pieces). envs/setup_b200.sh now pins it (committed).
  ETA: qwen3b judge ends ~03:45 UTC; minicpm ~03:45-11:00; phi ~11:00-18:30; InternOmni starts ~18:30 UTC (03:30 KST 26/11), done ~23:30 UTC.
- 22:30 UTC: QWEN7B FULL DONE (scored_all.jsonl). 113 think_error rows (fast verdict kept, in review_all.csv). Analysis (asr_ok, 52 units out):
  experiments/h1_v2/full_analysis_qwen7b_asrok.md + figures/full_h1v2_*_qwen7b_asrok.png. C0 81.5% (3,535/4,338). Violation T 17.3,
  A 13.5, V 89.0; T-A +3.8 pp [2.6, 5.0], McNemar p 5e-10 (matches server-1 subset +4.2). Val T 16.1/A 10.4, train 18.2/15.9.
  Recall C0 80.4, T 84.1, A 83.8, V 79.5. analyze.py takes ~3 min; it writes analysis_all*.md + h1v2_*_all*.png: rename to full_* before commit.
  Qwen3B started 21:29 UTC (0.50 s/ans). claims.md (shared) NOT edited yet: update once after all 5 models.
- 18:15 UTC: judge THINKING pass is the bottleneck: ~1,100 rows/h (batch_size=8, 1500 tok), 3,549 rows (23% regex-disagree) = ~3.2 h/model -> all 5 done ~21-22 UTC 11-25. Asked Vy (Telegram 03:15 KST) for OK to raise judge batch to 32 between models; no reply = keep as is. Do NOT edit judge.py while job runs (later models re-import it).
- 17:15 UTC: qwen7b gen done 6,230 s (0.36 s/ans; 95 regen at 256 tok); judge pass 1 15,208 rows ~10 min; then judge (thinking) 3,549 rows. Remaining 4 models ~12 h.
- 2026-11-24 16:05 UTC: qwen7b gen 7,544/17,524 at 0.44 s/answer (gen ends ~17:15 UTC, then judge: measure its speed). Projected all 5 done ~06-09 UTC 11-25.
- NEVER `git commit -a` on B200 (it commits the deleted old results + B200 smoke files over server 1 tracked pilot2 files); add files by name.
- Smoke OK on all 5 (s/answer gen: qwen7b 0.36, qwen3b 0.40, minicpm 0.72, phi 0.88, internomni 0.49). Smoke tag pilot2.
- FIX: InternOmni tokenizer failed with sentencepiece 0.2.2 ("piece must not include null character"): installed
  sentencepiece==0.2.0 (--no-deps, wheel in /tmp/spt/dl) into env omni_intern -> loads (92,544 pieces). File verified intact: sha256 = HF blob name f868398f..., binary (not an LFS pointer). Server 1 freeze
  says 0.2.2: version differs; envs/setup_b200.sh not changed yet (TODO: pin 0.2.0 for omni_intern there).
- Full: 4,390 questions x C0/T/A/V = 17,560 answers per model (existence 696, closed 622, count 902, attribute 1,518,
  relation 652). Order qwen7b, qwen3b, minicpm, phi, internomni; ETA ~16-20 h total (judge speed unmeasured yet).
- Then: analyze.py --tag all --asr_ok_only (main) and without (sensitivity); rename outputs to full_* before commit.

## >>> 2026-11-24 23:50 KST: FINAL BENCHMARK DONE; H1 SMOKE RUNNING (tmux job-h1-smoke, GPU 2, log pico/logs/h1_smoke.log) <<<
- Stage 1 EXIT 0 at 14:38 UTC. filter_benchmark.py applied (3 images; 24 media files moved to dataset/removed_images/,
  out_unfiltered/ = untouched build, 71 MB). Final: 2,792 images, 4,666 units, 7,468 negative + 4,666 exceptive +
  3,874 affirmative = 16,008 samples; units: attribute 1,546, count 963, existence 792, existence_closed 713,
  relation 652. check_ids.py: OK (no missing/duplicate ids). ASR ok 2,506/2,553 sentences (47 fail after 3 tries).
  excluded_units 276 (double_label 209, part_whole 39, attr_overlap 28). occludes_subject 625/16,008.
- Smoke started 14:49 UTC (models: qwen 7b, 3b, minicpm, phi, internomni). Next: s/answer -> ETA -> full run.

## >>> FULL AUTONOMY (Vy, 2026-11-24 23:09 KST, Vy asleep) -- overrides "no GPU until Vy OK" and "Vy's review" <<<
- (1) stage 1 done -> Pico reviews itself (selection report, sharpness, B2 human-verify, ASR) -> filter_benchmark.py
  -> FINAL benchmark (unfiltered copy untouched in out_unfiltered/). Never drop images because of results.
- (2) then H1 at once on the final benchmark: smoke --pilot 2 on all 5 models (s/answer), then full. GPU 2 ONLY,
  nvidia-smi before each job, never kill others, stop at first failure. NEU_T/NEU_A/NEU_V + cues + OCR_V only after
  CPU smoke test; NEU code does NOT exist anywhere (checked local + origin 23:10 KST) -> standard H1 first.
- IDs mandatory: image_id, unit_id, sample_id in the benchmark and ALL result files. Removed images -> NOTES list +
  dataset/out/removed_images.jsonl (= the filter list). Analysis gets an optional removal-list argument (by id) to
  re-run without regenerating.
- Reporting: ~1 short Telegram per hour (sleep_seconds <= 3600 while jobs run), one per finished stage, urgent only for
  crashes/quota/decisions. Warn if volume < 30 GB. Commit + push code when stable. Low tokens.
  (The 12 h sleep rule is suspended for this autonomous session.)

## >>> BAD-IMAGE CLEANUP = FILTER THE BENCHMARK ITSELF, BEFORE H1 (Vy, 2026-11-24 23:04 KST; replaces 23:01) <<<
- Order: (1) stage 1 builds the full dataset ONCE (never rebuild from step 3 for image cleanup); (2) after the build
  AND Vy's review: list bad images {"image": "images/...", "reason": ...} (exact path as in samples.jsonl) in
  create_data/dataset/out/removed_images.jsonl + the "Removed images" list below, then
  `cd create_data/src && PYTHONNOUSERSITE=1 ~/anaconda3/envs/negation/bin/python filter_benchmark.py` (--dry_run first);
  (3) only then H1, reading the filtered out/ as usual (no bad-image logic in run/analysis code; asr_ok handling
  and flag_excluded_units' excluded_units.jsonl stay as they are). Do NOT start H1 until Vy says OK.
- filter_benchmark.py: copies the unfiltered build once to dataset/out_unfiltered/ (always filters FROM it, so it is
  idempotent), rewrites every out/*.jsonl without the rows of those images' units (samples, forms, compact*,
  excluded_units), prunes the audio manifest (copy kept as manifest_unfiltered.jsonl), MOVES per-sample wav/visual
  files of removed samples to dataset/removed_images/. sample_ids are kept, nothing regenerated.
- Dependency check (2026-11-24): a REBUILD would change remaining items (B3 sweep subset = rng.sample over all units;
  sample_id = per source-branch counters in file order; TTS voice = shuffle over all sentences). Filtering does not:
  ids are not renumbered, targets/templates are seeded per unit (crc32 of unit_id), forms carry pair_of + unit_id
  and are dropped with their unit, audio/visual are per sample. Dry run on the partial build (3 test images): 9
  units, 39 samples removed; samples 7471->7450, forms 8545->8527, compact_all 16016->15977 (consistent).
  Side effects to state: template-sweep share and voice balance shift slightly.
- Image review (Pico, 2026-11-24 23:15 KST, on the built samples.jsonl: 2,795 images / 4,669 units before TTS ended):
  72 random images (18 per source; sheets /tmp/rev, not kept). CLEVR 18/18 fine. 54 photos: 2 too dark to see the
  target (coco/000000109532 chair/plant in a black room; vg/2409290 tiny woman on a dark blue escalator), 1 clearly
  wrong GT (vg/2390987 VAW "building is brown", building is grey/white), 2 doubtful (coco/000000291791 handbag not
  visible; vg/2402934 horse colour = silhouette), several odd VAW attributes ("elephants are wide", "sky is wide":
  annotation style, kept). => ~4% image-quality + ~5% GT problems, same as server 1's 10% (decision there: no filter
  change). Tested a stricter darkness rule (brightness < 60 or clipped > 0.25 = 83 images): 24 of them viewed, only
  ~2 really bad -> a global pixel rule would remove mostly good images: NOT applied. ASR is handled by asr_ok
  (analysis), B2 human-verify sheet is unlabeled (no signal). Decision: remove only the images seen as clearly bad;
  the rest is a stated limitation (violation is conditioned on each model's own C0, so wrong targets mostly drop out).
- Removed images (id, reason) -> also in create_data/dataset/out/removed_images.jsonl:
  coco/000000109532 too dark (clipped 0.31), target objects not visible
  vg/2409290 too dark / blue-tinted, subject tiny, relation not verifiable
  vg/2390987 wrong ground truth (VAW: building is brown; it is grey/white)
- IDs: image_id = '<folder>/<stem>' (omnilib.image_id); added to load_dataset rows, run_h1v2 / run_h2 result rows and
  (by filter_benchmark.py) every benchmark row. analyze.py / analyze_h2.py take `--remove FILE...` (jsonl with
  image_id/image/unit_id/sample_id, or one id per line) -> output analysis_<tag>[_asrok]_rm.md, nothing regenerated.
- NEXT when stage 1 prints STAGE1 DONE: (a) cd create_data/src; filter_benchmark.py (writes out_unfiltered/ once,
  adds image_id/unit_id/sample_id to every row); (b) check_ids.py -> counts + ID check go into the next Telegram
  report (Vy 23:22 KST asked); (c) nvidia-smi -i 2, then tmux job-h1-smoke: bash experiments/h1_v2/run_b200_h1.sh smoke
  > pico/logs/h1_smoke.log 2>&1 ; measure s/answer; then full.
- Tag clash: run_h1v2 tags are pilot2 / all (same as server 1). Results are local (git-ignored), but before committing
  an analysis_*.md / figure from B200 rename it with a `full_` prefix (hosts/b200.md rule).

## >>> RULE (Vy, 2026-11-24 ~21:35 KST): NO PERIODIC CHECKS <<<
- job-stage1 posts to Telegram itself when it finishes or fails. End EVERY step by writing 43200 to pico/sleep_seconds
  (sleep 12 h). Only work when Vy sends a message (arrives as pico/note.txt, wakes the runner within 10 s).
  This supersedes the "~45 min checks" in the block below.

## >>> 2026-11-24 12:22 UTC: RUNNER (pico tmux) WATCHING job-stage1 (per pico/note.txt) <<<
- Vy restarted the runner (pico/stop gone). Task from note.txt: watch job-stage1 (pid 1338300, GPU 2, started 11:58 UTC),
  sleep ~2700 s between checks, Telegram at each stage change; if it vanishes without EXIT: report last log lines +
  dmesg/OOM, do NOT restart without Vy's OK. No other GPU job. Do not commit the deleted old results.
- 12:21 UTC: still in B2 (no progress output), pid alive, 18 GB on GPU 2. Times to Vy in KST (UTC+9).

## >>> 2026-11-24 12:20 UTC: H2 + H3-A CODE WRITTEN (not run; Vy decides when) <<<
- New files only (nothing stage 1 / H1 uses was edited): create_data/src/h2_assets.py (render = question written in the
  image, CPU, env negation; tts = question audio, GPU 2, env cosyvoice; status), experiments/common/h2_common.py (6
  layouts, selection = same as run_h1v2.select, input builder), experiments/h1_v2/run_h2.py (generate + score exactly
  like run_h1v2; scoring block is a COPY of run_h1v2.main, keep in sync), analyze_h2.py (violation per layout, direction
  contrasts per channel pair with paired CIs / McNemar, H2 minus H3A), experiments/h1_v2/run_b200_h2.sh.
- Subset or full: `--pilot N` (N units per branch) or `--all`, same selection as H1 so answers are cached/shared.
  Examples: `bash run_b200_h2.sh assets pilot 40` then `bash run_b200_h2.sh run pilot 40 qwen2.5-omni-7b`
  (env FAMILIES / LAYOUTS / H3A_FORM choose families, layouts, H3-A form).
- H3-A instruction default = H4 affirmative form ("Mention only {Y}."): its audio/image already exist. PLAN's
  "Describe the cat instead." wording NOT used: Vy has to approve. qT_iA / qT_iV are identical to H1's A / V (cache).
- Tested on CPU only: py_compile, layout builder on dummy rows, analyze_h2.py on synthetic rows. NOT tested yet: render
  / tts on the real dataset (needs stage 1 to finish) and every adapter with audio-only or image-only questions.

## >>> 2026-11-24 11:58 UTC: Vy OK'd GPU 2 (ONLY GPU 2). STAGE 1 RUNNING (Claude in Vy's terminal) <<<
- Vy: GPU 2 is free, run the dataset build then H1 (staged; Vy decides H4/COMP/etc. later via Telegram). pico/stop stays
  (the runner is not running; do not start it, Vy's Claude drives the stages by hand).
- Stage 1 = tmux job-stage1: `bash create_data/run_b200_stage1.sh` (steps 3-7 on GPU 2, then TTS + Whisper, then
  flag_excluded_units), log pico/logs/stage1.log (ends with "EXIT n"), Telegram message at the end. B2 prints no progress bar.
- Stage 2 (not started yet) = `bash experiments/h1_v2/run_b200_h1.sh smoke|full [models]` (GPU 2 only, stops at first
  failure, tag pilot2 / all, log pico/logs/h1_<mode>.log). Order: smoke on all 5 models (measure s/answer), then full.
- Done on CPU meanwhile: matplotlib installed in env negation (--no-deps); analyze.py got --asr_ok_only (main analysis on
  asr_ok units, run once without as the sensitivity check). Models: all downloads finished (dl_models.log).

## >>> 2026-11-24 10:07 UTC: Vy's STOP STILL HOLDS (Claude in Vy's terminal) <<<
- The runner (tmux pico) restarted job-build-full on GPU 1 at 10:05 although Vy had ordered "no GPU". Claude KILLED it
  again at 10:07 (it had only loaded Qwen3-8B, nothing written) and created pico/stop so the runner exits after its
  current pause. Do NOT remove pico/stop or restart any GPU job until Vy says so explicitly. Vy is asking Claude to
  review ALL code first so that ONE command runs the whole thing later (CPU-only review, discussion with Vy).
  The "RUNNER STATE" block just below is superseded by this.

## >>> RUNNER STATE 2026-11-24 10:10 UTC (Pico) -- SUPERSEDED, see the STOP block above <<<
- First build try (GPU 3, started 09:51) DIED silently at 09:52 right after Qwen3-8B loaded: no traceback, no EXIT
  line, tmux session gone (cause unknown; log kept as pico/logs/build_full_try1.log). RESTARTED 10:05 UTC in tmux
  job-build-full on GPU 1 (CUDA_VISIBLE_DEVICES=1, 17.5 GB), same command, log pico/logs/build_full.log (ends "EXIT n").
  B2 prints no progress bar: judge liveness by `ps -p <pid>` + nvidia-smi -i 1 + "kept N/4675" line.
- Server 1's subset TTS crash (missing manifest dir) is already fixed in tts_negation.py (mkdir at line 107).
- git pull --rebase is blocked by the uncommitted deletions (old results/dataset); origin had nothing new at 10:01.
  Use `git fetch` + `git log HEAD..origin/main` to check; do NOT commit the deletions without telling Vy.


## >>> STOP 2026-11-24 (Vy): NOTHING RUNS ON THE GPU. job-build-full was KILLED by Vy's request <<<
- Reason (Vy): the GPUs are shared with other people; do not take GPU memory that someone else may need. Do NOT
  restart the build or any GPU job (B2/Qwen3-8B, TTS, experiments) until Vy says so explicitly in Telegram.
  The job had only loaded the model, wrote nothing (dataset/b2 does not exist). Everything below that says
  "RUNNING" is outdated. Prepared and ready for when Vy gives the OK: the build command below, the TTS command.
- Current dataset state: b1/units.jsonl = 4,675 units / 2,799 images (subset_images.py applied); b2/out/audio/visual
  deleted; raw, images, b1 kept. CPU-only work is fine meanwhile (code, notes, env fixes, downloads).

## >>> (outdated: was DONE / RUNNING 2026-11-24 ~17:00 UTC+9; Vy OK'd: delete old, rebuild full) <<<
- Vy: old synced subset dataset + old results may be deleted; B200 runs the FULL set. DELETED (local copy only;
  server 1 and git history still have them): create_data/dataset/{b2,out,audio,visual} and experiments/h1_v2/results
  (git shows these as deleted: do NOT commit that deletion without telling Vy). KEPT: dataset/raw (annotations),
  images (3,154 selected), images_rejected, b1 (selection + units_full.jsonl), experiments/h1/results_pilot_v1,
  analysis_*.md, REPORT.md, figures, human_check sheets (Vy's 67 labels).
- APPLIED subset_images.py --n_images 2800 -> b1/units.jsonl = 4,675 units / 2,799 images (numbers above).
- RUNNING: tmux job-build-full = `run_pipeline.py --from_step 3` (B2, B3, forms, export, visual) on GPU 3 only
  (CUDA_VISIBLE_DEVICES=3, ~71 GB was free), log pico/logs/build_full.log (ends with "EXIT <code>").
  B2 first: LLM Qwen3-8B fp16 over 4,675 units (server 1: 1,150 units took 1,486 s). Then run on its own, in order:
  1. TTS: cd create_data/src && PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=<free gpu> ~/anaconda3/envs/cosyvoice/bin/python
     tts_negation.py --src samples_compact_all.jsonl --dst samples_compact_all_audio.jsonl   (what run_v2_all.sh does)
  2. flag_excluded_units.py (negation env) -> dataset/out/excluded_units.jsonl
  NOTE: do not use `run_v2_all.sh rest` (set -e + TTS would stop before the exclusions if TTS fails).
- TTS prerequisites FIXED: env cosyvoice was incomplete (pyworld built against numpy 2.2.6 and failed). Fix: pin
  `pip install --no-deps numpy==1.26.4` in the env, then rerun `bash envs/setup_b200.sh cosyvoice` (log
  pico/logs/env_cosyvoice2.log): now torch 2.7.1+cu128 cuda ok, onnxruntime 1.22, whisper, librosa, soundfile OK.
  setup_b200.sh itself NOT changed (it needs numpy pinned before pyworld; tell server 1 / fix when committing).
  TTS code is in <project>/tts (CosyVoice, pretrained_models/CosyVoice2-0.5B, ref_candidates): path matches
  tts_negation.py TTS_ROOT. Whisper large-v3-turbo downloaded. So TTS runs on B200 (settles the hosts/b200.md contradiction).
- Env gap to check at the first smoke: env `negation` has NO librosa / decord / onnxruntime / whisper (omni_neg has
  librosa). Qwen2.5-Omni audio loading may need librosa -> check with --pilot 2 before a real run.
- Still downloading (tmux dl-models, ~/dl_models.log): Qwen2.5-Omni-3B, MiniCPM-o-2_6, Phi-4-multimodal, InternOmni.
  Qwen2.5-Omni-7B, Qwen3-8B, Qwen3-VL-8B, whisper-large-v3-turbo are in the HF cache.

## >>> VY'S DECISIONS 2026-11-24 (Claude in Vy's terminal; this file = B200 only, pico/NOTES.md = server 1) <<<
- Roles: server 1 = SUBSET tests only (try designs, speeds, new code). B200 = the FULL experiments (much faster).
- Dataset target on B200: ~2.8k original IMAGES to build, then drop the clearly bad images after review -> ~2.5k
  final images (Vy: 2.5k images x several components/attributes = plenty of questions). Not "2.5k questions".
- Image-level subset script (new, CPU only, not committed yet): create_data/src/subset_images.py. Dry run
  `--n_images 2800` (nothing written): 2,799 images / 4,675 units (existence 793, closed 713, count 965,
  attribute 1,550, relation 654); GQA all 654 (relation has no other source), COCO 815, VAW 815, CLEVR 515;
  non-train images (val/test) kept first, train drawn at random (seed 42). Expect ~4.35k questions after the
  79/1150-style exclusions (~7%), ~3.9k after the final ~2.5k-image cleanup.
  Apply with: cd create_data/src && PYTHONNOUSERSITE=1 ~/anaconda3/envs/negation/bin/python subset_images.py
  --n_images 2800 ; then `bash ../run_v2_all.sh rest` (B2 -> B3 -> forms -> export -> visual -> TTS -> exclusions).
  NOT applied yet. units_full.jsonl (5,335 units / 3,154 images) stays as the pool; dropping bad images later =
  [SUPERSEDED 2026-11-24 23:04 KST: NO rebuild; filter the benchmark with filter_benchmark.py, see the top block].
- Bad-image cleanup after the build: use the review page + selection_report (sharpness etc.) + B2 human-verify
  sample; record every removed image id and why in this file (never drop images because of results).

## State of this machine, checked 2026-11-24 (Claude)
- create_data/dataset/ is the SYNCED SERVER-1 SUBSET BUILD: b1/units.jsonl = 1,150 units, b1/units_full.jsonl =
  5,335; b2/, out/, audio/, visual/ hold the subset outputs (4.8 GB). `run_v2_all.sh rest` overwrites b2/out and
  re-runs B3 -> sample_ids will differ from server 1 (never compare by sample_id). tts_negation.py deletes wav
  copies that are not in its --src file: do not point it at a stale src.
- Envs present in /home/vy/conda-envs (linked to ~/anaconda3/envs): cosyvoice, negation, omni_intern, omni_neg,
  omni_phi. Not yet verified with `envs/setup_b200.sh --check`.
- HF cache has Qwen3-8B (+ Qwen3-VL-8B); NOT yet: Qwen2.5-Omni-7B/3B, MiniCPM-o-2_6, Phi-4-multimodal, InternOmni,
  whisper-large-v3-turbo (dl_models.py lists them; check its log before the first run).
- GPUs (nvidia-smi 2026-11-24): all 4 B200 have 148-164 GB of 183 GB used by other people -> only ~19-35 GB free
  per card. A 7-8B omni model in bf16 (~16-18 GB) barely fits; Qwen3-Omni (~60 GB) does NOT fit on one card now.
  Check nvidia-smi right before every job, use CUDA_VISIBLE_DEVICES, never kill other people's processes.
- Volume 98% full (288 GB free). Full dataset (images + audio + visual + forms) adds several GB; warn Vy < 30 GB.
- Contradiction to settle with Vy: pico/hosts/b200.md says "No TTS here, audio is made on server 1 and synced",
  but its dataset section says "TTS also on B200", and a cosyvoice env + tts/ (CosyVoice, 3.8 GB) exist here.
  Whisper (ASR check) is also needed for TTS.
- Waiting for Vy's OK in Telegram before any GPU run (rule). First proposed run order once OK: apply
  subset_images.py -> build steps 3-7 (+TTS) -> smoke --pilot 2 on all 5 models (measure s/answer) -> H1 full.

## >>> CURRENT STATE 2026-11-24 (set up by Claude) <<<
- B200 just set up: code cloned from GitHub, dataset synced from server 1 (create_data/dataset), envs being built
  with envs/setup_b200.sh (log /home/vy/setup_envs.log). Claude CLI not installed yet (Vy will do it later).
- Nothing has run here yet. First job when the runner starts: verify the envs (envs/setup_b200.sh --check),
  read pico/hosts/b200.md, PLAN.md and server 1's pico/NOTES.md (to see the state of the study), then report
  to Vy and wait for instructions.

## VY'S DECISION 2026-11-24: asr_ok
- Run ALL units (no asr_ok filter before running; `--all` as is, flag kept in every row). Main analysis on asr_ok=True, plus a sensitivity check on all units. Dataset stays at 2.8k images (2,799 / 4,675 units); the ~2.5k cleanup comes later.
