# Pico notes (memory across restarts)

## Now
- RUNNING (started 2026-11-19 19:35): tmux `job-select` = `run_v2_all.sh select` (steps 0-2), log pico/logs/select.log
  (ends with 'EXIT <code>'). Expected ~2 h. Next: review images (M1 step 2).
- DONE 19:47: Qwen2.5-Omni-7B and 3B downloaded to HF cache.
- WRITTEN (untested, needs dataset): experiments/common/omnilib.py (data loader, batched Omni adapter, answer cache
  keyed by input hash, judge scoring) and experiments/h1_v2/run_h1v2.py (conditions C0, C0_box, T_join, T_sep, A, V;
  forms T_sep/A/V; --pilot N per branch / --all).
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
- H1 confound from the pilot: text glued into the question vs audio as its own segment -> T_sep condition.
- Telegram (pico/notify.py): 2026-11-19 19:40 not sent (topic not set up yet: Vy must /setup); report kept in
  pico/telegram_sent.log. Keep posting after every step anyway.
