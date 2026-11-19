# Pico notes (memory across restarts)

## Now
- Nothing running yet. Next: Milestone 1 step 1 (`cd create_data && bash run_v2_all.sh select`).

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

## Open problems / ideas
- H1 confound from the pilot: text glued into the question vs audio as its own segment -> T_sep condition.
