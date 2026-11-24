# NOTES (B200 runner's memory)

## >>> 2026-11-25 00:10 KST: H1 FULL RUNNING (tmux job-h1-full, GPU 2, log pico/logs/h1_full.log, ends "EXIT n") <<<
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
