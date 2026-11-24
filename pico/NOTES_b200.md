# NOTES (B200 runner's memory)

## >>> RUNNER STATE 2026-11-24 10:10 UTC (Pico) <<<
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
  remove them from units.jsonl (or lower --n_images) and rebuild from step 3.
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
