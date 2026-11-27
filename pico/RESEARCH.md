# Research brief: cross-multimodal negation (Vy + Huy) — read at the start of every session

Goal: a paper (target venue: ACL 2027; check the ARR deadline with a web search and keep it in NOTES.md).
This file is the shared understanding of the study. Pico keeps the "Lessons" section at the end growing.

## 1. The question and why it is new
Omni-models (image + audio + text in, e.g. Qwen2.5-Omni, MiniCPM-o, Phi-4-MM, Gemma-3n) are told what NOT to do
in more and more ways: typed, spoken, or shown. Negation in text-only and image-text models is studied; whether a
**negative constraint is followed equally when it arrives through different channels** is not.
RQ: *Is negation faithfulness the same when the same negative instruction is delivered by text, by speech, or
written in the image — and if not, is the gap about negation or about the channel?*

## 2. Identification logic (the heart of the paper — never lose this)
- A raw channel gap (violation_audio ≠ violation_text) has two readings: the model handles *negation* worse in that
  channel, or it follows *any instruction* worse there. Only the **polarity × channel interaction** separates them:
  Δ_m = F_aff,m − F_neg,m (negation penalty per channel), signal = Δ_audio − Δ_text (and Δ_visual − Δ_text).
- The **exceptive** form ("Mention everything except X": names X, no "not") separates "the word not" from "hearing or
  reading the target's name" (lexical priming).
- Everything is **paired within an image**: same image, same question, only the instruction changes. Use paired
  tests (McNemar per pair, bootstrap over images for CIs; template-sweep copies are not independent).
- Violation is measured only where the tested model states the target **without** any instruction (its own C0):
  otherwise "not mentioning" is not compliance. Report the C0 rate too (ceiling / salience).

## 3. Hypotheses on dataset v2
- **H1 (channel gap):** violation under T_join, T_sep, A, V vs C0. T_sep (instruction as its own text block) is the
  fair text control for audio (audio is always a separate segment); C0_box (empty box) is the fair control for V.
- **H2 (direct conflict):** text "Mention X." + audio "Do not mention X." and the swap, both orders. If negation were
  equally strong in both channels, swapping should not change P(X mentioned).
- **H3-A (modality dominance):** conflict without negation ("Mention only X" vs "Mention only Y", swapped). If the same
  channel wins with and without negation → dominance, not negation.
- **H3-B (comprehension):** can the model *report* which item was forbidden (multiple choice, question in text) per
  channel? Understanding ≠ obeying.
- **H3-C (activation order):** instruction before the image vs after (all v2 runs put the image first = "target
  first").
- **H3-D (evidence strength):** violation vs target size (area_ratio is observational and confounded with category;
  control for category or manipulate).
- **H4 (negation specificity):** Δ_m with affirmative forms, exceptive as the priming control. Affirmatives exist only
  where the question already names every option/slot (existence_closed, count, attribute, relation).
All hypotheses are **two-sided**: no direction is established yet. Nothing is concluded before dataset v2 results.

## 4. What we know so far: nothing conclusive yet
- No result on dataset v2 exists. Do not state or assume any direction (text better / audio better) anywhere — in
  notes, reports, the paper draft, or Telegram — until v2 results with confidence intervals support it.
- The old pilot (a different, smaller dataset of 435 samples, no longer used) is only a hint about *what to
  control*, not a finding: there the text instruction was glued into the question while audio came as a separate
  segment, so channel and segmentation were confounded (hence T_sep). Its numbers are in
  experiments/h1/results_pilot_v1 if a sanity comparison is ever needed; do not cite them as results.

## 5. Known confounds and how we handle them
- Affirmative leaks the answer in open existence → no affirmative there (closed-set existence instead).
- Keyword traps in scoring (hot dog/dog, teddy bear/bear) → LLM judge primary, regex second opinion, alias checks.
- Question always in text favours the text channel slightly → keep the question in text for every condition and
  say so; T_sep makes text and audio both "separate instruction" conditions.
- ASR: does the model *hear* "don't"? → TTS is Whisper-checked (asr_ok); H3-B tests the model's own comprehension.
- Visual instruction must not hide the target → text placed off the question's objects; flagged occludes_subject.
- Ceiling risk (Qwen-Omni-7B ~7-13%) → report per model and branch; harder branches (attribute/relation) matter.
- Train-split images (COCO/GQA train) may be in instruction-tuning data → report val vs train separately.
- Model-based filtering (VLM check of ground truth) may favour easy images → report it as a limitation.

## 6. Do / don't
DO: pilot before full runs; paired comparisons; CIs; report what is NOT supported; keep every run versioned and
reproducible (script + config + git commit in the results folder); write results into paper/ as they arrive.
DON'T: change the dataset, judge, prompts or exclusion rules after seeing results to make an effect appear;
compare channels without the matching control; claim causality from H3-D observational sizes; drop a model or a
branch because it "looks bad" (report it); spend GPU on full runs before the pilot looks right.

## 7. Going deeper: WHY is a channel better or worse? (the analysis that makes it a paper)
A gap alone is a finding; an explanation is a paper. For every gap found, try to explain it with cheap, decisive
tests (subset first), for example:
- Segmentation / position: T_join vs T_sep vs A; instruction before vs after the question or the image (H3-C).
- "Is it the sound or the words?": feed the Whisper transcript of the audio as text (same words, text channel);
  vary the voice (female/male, speed, a synthetic vs natural voice); add a neutral-audio control (speech that says
  something irrelevant) to see whether any audio segment changes behaviour.
- Comprehension vs compliance (H3-B): can the model repeat/identify the instruction per channel? Does compliance
  fail even when comprehension succeeds?
- Cue type: explicit negators (N1-N5) vs implicit (N6 avoid, N7 leave out) vs exceptive, per channel.
- Error anatomy: classify violations (states the target plainly, names it while saying it won't, synonym/alias,
  partial), and over-suppression (drops required components) per channel.
- Inside the model (Qwen-Omni weights are open): log-probability of the target token(s) at the answer position per
  condition; attention mass from answer tokens to the instruction tokens/segment; how this changes with channel.
  Keep these small and well-controlled; they explain, they do not replace the behavioural results.
- Salience / evidence strength (H3-D): target size and category; does the channel gap grow with salience?

## 8. Paper plan (keep paper/ updated)
- paper/claims.md: each candidate claim → the exact evidence (table/figure, numbers, test) → status
  (supported / not / pending). The story follows the evidence; likely shapes:
  (a) negation is channel-specific (Δ_audio ≠ Δ_text) — the headline we would like but must earn;
  (b) audio is a stronger instruction channel in general (salience), negation-specific effects small;
  (c) no channel gap: robustness result + benchmark contribution.
- paper/outline.md **follows the Idea document's structure** (create_data/Idea Vy + Huy.pdf): motivation and the
  NF vs IF table → H1 (does the channel matter) → H2 (direct conflict) → H3-A..D (why: dominance, comprehension,
  activation order, evidence strength) → H4 (negation-specific, Δ_m, exceptive priming control). The Idea document
  is a rough draft: where results or better designs suggest a change, propose it (see §10), do not silently rewrite.
- paper/outline.md also: intro, related work (search the web for negation in VLMs/omni-models, instruction following
  across modalities, audio instruction following; real papers with links only), benchmark, experiments, analysis,
  limitations.
- experiments/REPORT.md + experiments/figures/: numbers and plots the paper will use.

## 9. Lessons (Pico adds one line each time something is learned; date | lesson)
- 2026-11-19 | Gluing the instruction into the question string vs a separate audio segment confounds channel with
  segmentation; always pair audio with T_sep.
- 2026-11-19 | Baseline filter removed (Vy): targets now include components the tested model never states, so every
  violation number MUST be conditioned on that model's own C0; expect C0 rates well below 100% and report them.
- 2026-11-19 | Prior work (MCR-Bench, ALME; verify the links before citing) reports TEXT dominating audio in content
  conflicts. Whether negation behaves the same is open: we have no v2 result yet (the old pilot is not evidence).
  H2 vs H3-A is where this is decided — high value for the paper.
- 2026-11-19 | (Vy) Never pre-conclude a direction before v2 results with CIs; keep notes, reports and the draft
  neutral.
- 2026-11-24: comparing human labels with a re-scored run must match on the answer text, not on (sample_id, condition): regenerated answers silently pair a human label with a different answer (18/67 items).
- 2026-11-24: a probe question placed BEFORE an active instruction gets the instruction applied to the probe answer (COMP v1: model avoided the forbidden option, 18.7% < chance). Quote the instruction as content, first, when testing comprehension.

## 10. New ideas: propose, pilot on a subset, discuss, then scale
The Idea document is a starting point, not a cage. When Pico sees a better design, a missing control, or a
surprising pattern worth chasing:
1. Write it in paper/ideas.md: the idea, why (which result or gap motivates it), what outcome would mean what,
   cost (GPU hours), risk.
2. Run a **small subset pilot** right away if it is cheap (≤ ~300 samples, ≤ ~2 GPU hours, a new versioned folder
   under experiments/, never touching finished runs or the dataset).
3. Report it to Vy on Telegram (and in the next email): the idea, the pilot numbers with CIs, and a clear
   recommendation (drop / refine / scale up). **Scale up only after Vy agrees** (a "Note from Vy").
4. Record the outcome in ideas.md and a Lesson in §9, whatever it was.
- 2026-11-23 | H1 trial: the T vs A gap changes sign across models (Phi-4 opposite to Qwen/MiniCPM), so any
  channel claim must be per model; V violation 82-93% everywhere needs a reading-vs-obeying test before any claim.
- 2026-11-23 | Check the n in every analysis table against the design (913 vs 1,000 revealed a selection bug).
- 2026-11-23 | Never edit a script that a running job will start again later (run_trial.sh re-imports
  run_h1v2.py per model): a new module name `comp` clashed with a local lambda and crashed 3 of 4 models, and the
  runner hid it behind "TRIAL DONE". Runner scripts must stop on the first failure; new code goes in only between jobs.
- 2026-11-26 (B200, pilot1000, 5 models): models copy the in-image instruction almost verbatim (OCR_V exact 94-100%) yet V lowers mention vs the empty-box control C0_box by only 0.7-6 pp: V failure is not a reading failure; the written instruction is read but mostly not treated as an instruction.
- 2026-11-27 (B200, NEU pilot1000, Qwen-7B/3B only): a neutral in-image instruction ("Include the word W") is ignored as much as the negative one (follow 7-10%), and a cue sentence does not help -> V is "image text is not an instruction", not negation. The typed/spoken gap flips sign between NEG and NEU (D_A - D_T -7.3 / -20.5, CIs exclude 0), while vs the affirmative form (H4) the Qwen CIs include 0: the conclusion depends on the control; report both controls side by side.
