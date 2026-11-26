# Typed, Spoken, or Shown: Do Omni-Models Obey a Prohibition Equally in Every Channel?

Draft b200 v1 (Pico on B200, 2026-11-26). Built only from B200 result files (full dataset v2 build). Every number
gives its source file in a `[src: ...]` tag; those tags come out before submission. Missing results are marked
**[PLACEHOLDER: <experiment>]**. The style follows "Deep and shallow biases in language models" (Vo et al.); see
paper/style_notes.md §3 and the style log in paper/draft_b200_review.md. No sentences are copied from it.

Source abbreviations (all under experiments/h1_v2/):
- FULL = full_analysis_5models_asrok.md (H1, 4,338 questions, asr_ok); FULL-ALL = full_analysis_5models_allunits.md
- H4 = pilot1000_forms_5models_asrok.md (1,000-question pilot, forms)
- CTRL = pilot1000_comp_ctrl_5models.md (+ analysis_comp_pilot1000.md, analysis_pilot1000_asrok.md)
- H2 = analysis_h2_pilot1000_asrok.md; AUD = audit/ (judge_audit.md, regen_audit.md, dataset_stats_b200.md)

---

## Abstract

Omni-models accept instructions that are typed, spoken, or written inside an image. We ask whether a prohibition
("Do not mention the dog.") is obeyed equally in each of these channels. Earlier studies of negation and of
modality conflict use text, or change the content together with the channel, so this question has had no
controlled test. We build a paired benchmark of 4,338 visual questions over 2,627 images. Each question appears
without any instruction, with a typed prohibition, with a spoken one, and with one written in the image, and
violations are counted only where the model names the target when given no instruction. In five open omni-models,
the typed–spoken gap ranges from −31.6 to +17.9 points, and its sign depends on the model. Prohibitions written in
the image are violated in 87–95% of cases. The models copy the same text almost perfectly when asked to read it,
and affirmative instructions written in the image are ignored just as often. In a 1,000-question pilot with
affirmative and exceptive controls, part of the spoken–typed difference appears specific to negation in three
models, and the exceptive control shows that this depends on the model. Channel effects in omni-models therefore
have to be measured per model and against matched controls. [URL]

(208 words: trim to ~170 at the end. H2 sentence to add when 5-model H2 is in.)

---

## Figure 1 (planned; running example)

One COCO image; question "Which of these are in the picture: dog, bicycle, bench?"; prohibition "Do not mention
the dog." shown four ways: (a) no instruction (C0), (b) typed as its own text block (T), (c) spoken (A, waveform
icon), (d) written in a box in the image (V). Under each, one real answer from Qwen2.5-Omni-7B with a check
mark or a cross. Caption (draft): "The same prohibition through three channels. (a) Without an instruction the
model names the dog; (b)-(c) the typed and spoken prohibition are usually obeyed; (d) the written one is usually
not, although the model can copy it word for word (Sec. 4.2). Thus the channel, not only the words, decides
whether a prohibition is followed."
**[PLACEHOLDER: pick a real example from results/qwen2.5-omni-7b/review_all.csv whose four answers match the
modal outcome; do not pick an unusual one.]** Channel colours (text / speech / image / no-instruction) are fixed
here and reused in every plot (style checklist item 13).

---

## 1 Introduction

Omni-models such as Qwen2.5-Omni, MiniCPM-o, Phi-4-multimodal and InternOmni take text, images and speech in one
context. A user can therefore give the same constraint by typing it, saying it, or showing it, for example as a
sign in a photo or a note in a screenshot. Prohibitions are a common kind of constraint ("do not mention the
patient's name", "do not describe the child"). They are also known to be hard for language and vision-language
models even when they are typed [cite: negation in LLMs / VLMs; related_work.md].

What is missing is a comparison of the *same* prohibition across channels with everything else held fixed.
Benchmarks of audio instruction following, and studies of conflicts between modalities, change the content and
the channel together. Usually they also attach the text instruction to the question while the audio comes as a
separate segment. When a spoken prohibition is violated more often than a typed one, two causes are then
possible. The model may handle *negation* worse in speech, or it may follow *any* instruction worse in speech.

We therefore argue that a channel effect on negation has to be shown as an interaction. The prohibition is
delivered in each channel as a separate segment. Each answer is compared with the same model's answer to the same
question without any instruction. The negative form is compared with an affirmative form and an exceptive form
("Mention everything except X"). The exceptive form names X but contains no negation word. Fig. 1 shows the design
on one image.

Two controls make the comparison fair. The typed prohibition is a separate text block, so text and speech are both
separate instruction segments. The written prohibition is compared with an empty-box image (C0_box) and with a
reading task (OCR_V), so that "cannot read it" can be told apart from "does not treat it as an instruction".

We build a benchmark of 4,338 questions over 2,627 images from COCO, CLEVR, VAW and GQA, in five branches
(existence, closed existence, count, attribute, relation). Seven prohibition templates are used, and each spoken
sentence is checked with ASR (Sec. 2). For each model m and channel c ∈ {T, A, V} we measure the violation rate
v_{m,c}: the share of answers that still state the forbidden component, among questions whose no-instruction
answer states it. We also measure the required recall of the allowed components (Sec. 3). Our main findings are
as follows:

1. **Typed and spoken prohibitions are obeyed at different rates in all five models, but the direction depends
   on the model.** The typed-minus-spoken difference in violation is +3.8 [2.6, 5.0] points for Qwen2.5-Omni-7B,
   +15.1 for -3B, +17.9 for MiniCPM-o, −9.8 for Phi-4-MM and −31.6 [−33.7, −29.7] for InternOmni (Sec. 4.3).
   [src: FULL]
2. **Prohibitions written in the image are violated in 87–95% of cases in every model, although the models read
   them.** Asked to copy the text, the models reproduce it exactly in 94–100% of cases. Against an empty box, the
   written prohibition lowers mentions by only 0.7–6.0 points (Sec. 4.2). [src: FULL, CTRL]
3. **The in-image failure is not specific to negation.** Affirmative instructions written in the image are ignored
   as often (violation 86–95%), so Δ_V stays within ±4.3 points in every model (Sec. 4.4). [src: H4]
4. **Part of the spoken–typed gap appears specific to negation in three models, but the exceptive control limits
   this claim.** In the 1,000-question pilot, Δ_A − Δ_T (negation penalty, spoken minus typed) is +11.9 for
   MiniCPM-o, +11.9 for Phi-4-MM and +36.3 for InternOmni, with CIs excluding 0. For both Qwen models the CI
   includes 0. With the exceptive form as the reference, the InternOmni effect disappears (+0.4 [−4.8, 5.5]), and
   Qwen2.5-Omni-3B shows the opposite sign (Sec. 4.4). [src: H4]
5. **Spoken instructions can be taken as the question.** InternOmni keeps 73% of the allowed components under a
   typed prohibition but 25% under a spoken one (no-instruction baseline 71%). Violation alone therefore misses an
   important way to fail (Sec. 5.2). [src: FULL]
6. [PLACEHOLDER: H2/H3-A finding, 5 models: does the channel of the instruction "win" against the channel of the
   question, and does this differ with vs without negation? Qwen-7B preview in Sec. 4.5.]

---

## 2 Benchmark

### 2.1 Design
Each **unit** is an image, a question that asks for a list of components, and one forbidden component (the
target). The question is always typed. Only the delivery of the prohibition changes:

| condition | input (image first, then items in order) |
|---|---|
| C0 | image · question |
| T | image · question · prohibition as its own text block |
| A | image · question · spoken prohibition (audio segment) |
| V | image with the prohibition written in a box · question |
| C0_box (control) | image with an empty box · question |
| OCR_V (control) | image with the prohibition written in a box · "copy the text in the image" |

Seven templates (N1 "Do not mention {X}." … N6 "Avoid mentioning {X}.", N7 "Leave {X} out of your answer.") vary
the negation cue. N6 and N7 have no negation word. Two further forms exist for H4. The exceptive form ("Mention
everything except {X}.") exists for every branch. The affirmative form ("Mention only {Y}.") exists only where the
question lists all options, so it is absent for open existence, where it would give the answer away.

### 2.2 Construction (bold-verb pipeline; counts for the B200 full build)
1. **Collect** units from COCO (existence, closed existence, count), CLEVR (count, attribute), VAW (attribute) and
   GQA (relation), with aliases per component. **[PLACEHOLDER: per-source pool counts before selection, from
   create_data/dataset/b1 selection report.]**
2. **Filter** images: pixel checks (blur, exposure, grey-scale, large text, near-duplicates) and a VLM check of
   every component and distractor (Qwen3-VL-8B). Then subsample to 2,799 images / 4,675 units (all GQA; val/test
   images first).
3. **Phrase** each unit with a template and its forms (Qwen3-8B for B2/B3). 4,390 units reach the H1 run.
4. **Exclude** units whose components cannot be told apart (276: double labels, part–whole pairs, overlapping
   attributes) and 3 images removed by hand (2 too dark, 1 wrong VAW label). [src: notes; create_data/dataset/out/removed_images.jsonl]
5. **Speak** every sentence with CosyVoice2-0.5B (two voices, female 2,339 / male 1,999 in the analysed set) and
   check it with Whisper large-v3-turbo. 52 units fail the check and are left out of the main analysis; they are
   kept in a sensitivity analysis.
6. **Render** the sentence in a box away from the question's objects. In 172 units (4.0%) the box still overlaps a
   question object (flagged as occludes_subject).

Result: **4,338 questions over 2,627 images** (attribute 1,518, count 899, existence 689, closed existence 612,
relation 620; COCO 1,685, CLEVR 1,250, VAW 783, GQA 620). [src: AUD dataset_stats_b200.md]
Arithmetic: 4,390 units × 4 H1 conditions = 17,560 answers per model (all units; 17,352 with asr_ok).
[src: AUD judge_audit.md rows]

**Pilot list.** H4, the controls, H3-B and H2/H3-A ran on a fixed list of 1,000 of these questions: 200 per branch,
voices 500/500, all asr_ok for the negative audio, each image used once, seed 42 [src: pilot1000_summary.md]. The
analyses keep the 987 units whose affirmative/exceptive audio also passed ASR (985 in H2). The pilot is balanced by
branch while the full set is not, so pilot and full numbers are not directly comparable.

---

## 3 Setup

**Models.** Qwen2.5-Omni-7B and -3B (thinker only), MiniCPM-o 2.6, Phi-4-multimodal-instruct, InternOmni. Greedy
decoding, 100 new tokens. Answers that hit the limit are regenerated with 256 tokens (0.3–7.3% of the cached
answers per model; MiniCPM-o most) [src: AUD regen_audit.md]. All runs used one NVIDIA B200 (torch 2.7.1).

**Violation rate.** In plain words: how often the model still names the forbidden component, among the questions
where it names it without any instruction. Symbol: v_{m,c} = P(target named | channel c, C0 names the target). A
high v means that the prohibition in channel c is mostly not followed. Example: C0 lists "dog, bicycle, bench"; if
T then answers "bicycle and bench, but no dog", that counts as a violation (kind DENY, below). We condition on C0
because leaving out something the model would not name anyway is not compliance. C0 names the target in 79–82%
of questions [src: FULL].

**Violation kinds.** STATE (states the fact), META (names it in order to exclude it), DENY (says it is absent or
false), WRONG (gives a different value). Leak = STATE.

**Required recall.** The share of the allowed components that are still named. A drop below C0 means
over-suppression or a misread task.

**Negation penalty (H4).** F_{f,c} = 1 − v_{f,c} for form f. Δ_c = F_aff,c − F_neg,c. A positive Δ_A − Δ_T means
that the negative form is harder, relative to the affirmative, when spoken than when typed. The exceptive analogue
uses F_exc instead of F_aff. It keeps the target's name in the instruction, so it controls for hearing or reading
the name.

**Scoring.** An LLM judge (Qwen3-8B) reads the question, the components and the answer. Rows where it disagrees
with a regex second opinion get a thinking pass. If the thinking pass returns no JSON (think_error), the fast
verdict is kept: 0.08–0.83% of rows per model [src: AUD judge_audit.md]. Agreement with a human annotator was
measured on server 1's subset (49 answers, 93.9%, κ = 0.83). **[PLACEHOLDER: human check on B200 answers
(blind sample, Vy/Huy); the server-1 agreement is on different answers.]**

**Statistics.** One answer per question and condition. All comparisons are paired within question. 95% CIs are
percentile bootstrap over images (2,000 resamples), and paired differences also get an exact McNemar test. We
analyse the asr_ok units and repeat everything on all units.

---

## 4 Results

### 4.1 Sanity checks: models name the targets, read the written text, and an empty box changes little

**Experiments.** C0 on the full set. On the pilot: C0_box (empty box) and OCR_V (copy the written text).

**Results.** All five models name the target without instruction in 79.3–82.0% of the full set, so there is
something to suppress in about four of five questions [src: FULL]. The written prohibition is legible to every
model: exact copies in OCR_V are 99.8% (Qwen-7B), 93.6% (Qwen-3B), 99.7% (MiniCPM-o), 98.1% (Phi-4), 99.0%
(InternOmni) [src: CTRL]. An empty box alone barely changes what is named: C0 vs C0_box differs by at most 2.3
points, and the difference is significant only for Qwen-3B (84.0 vs 81.7, McNemar p = 0.007) [src: CTRL].

| model | C0 names target (full) | OCR_V exact (pilot) | C0 → C0_box (pilot) |
|---|---|---|---|
| Qwen2.5-Omni-7B | 81.5 | 99.8 | 84.4 → 83.9 |
| Qwen2.5-Omni-3B | 82.0 | 93.6 | 84.0 → 81.7 |
| MiniCPM-o 2.6 | 79.3 | 99.7 | 81.9 → 83.0 |
| Phi-4-MM | 80.2 | 98.1 | 81.2 → 80.2 |
| InternOmni | 80.3 | 99.0 | 82.6 → 82.6 |

Caption: n = 4,338 (full); pilot OCR_V n = 1,000, C0 / C0_box n = 987 (asr_ok for every form). Thus the channels can be compared: targets are usually named, and
written text is read.

### 4.2 Prohibitions written in the image are violated in about nine of ten cases in all five models

**Experiments.** V vs T and A on the full set. V vs C0_box on the pilot (paired).

**Results.** V violation is 87.2–94.7% in every model (Table 2) [src: FULL]. Among questions whose C0 answer names the target, and against the empty-box
control, the written prohibition lowers mentions by 6.0 [4.0, 8.0] points (Qwen-7B), 2.1 [0.4, 3.9] (Qwen-3B), 3.2 [1.6, 4.9]
(MiniCPM-o), 5.2 [3.1, 7.4] (Phi-4) and 0.7 [−0.8, 2.2] (InternOmni) [src: CTRL; quick script, to be re-derived,
re-derived in audit/number_audit.md]. Together with OCR_V, this suggests that the failure is not in reading: the written text is read but
mostly not acted on as an instruction. Violations under V are mostly plain statements (STATE 80.4–93.3% of C0-mentioned questions; META up to 6.2%
in the Qwen models) [src: FULL, kind tables]. Box overlap with a question object does not explain it: with vs without overlap, V is 91.9 vs 88.9
(Qwen-7B), 94.7 vs 90.4 (Qwen-3B), 92.4 vs 94.0 (MiniCPM-o), 88.8 vs 87.1 (Phi-4), 93.1 vs 94.8 (InternOmni); only
130–135 overlapping questions per model [src: FULL].

### 4.3 Typed and spoken prohibitions differ in every model, in opposite directions across models

**Experiments.** T vs A on the full set, one answer per question, paired.

**Results.** Every model obeys the typed and the spoken prohibition at different rates, and the CI excludes zero
in all five. The sign differs between models (Table 2). The difference holds on all units (≤ 0.2 points from the
asr_ok analysis) [src: FULL-ALL]. Since the direction changes between models, we compare channels only within a
model and make no general claim that one channel is stronger.

Table 2. Violation (%) among C0-mentioned questions, full set, asr_ok. [src: FULL]

| model | n | T | A | V | T − A [95% CI] | McNemar p | recall C0 / T / A / V |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Omni-7B | 3,535 | 17.3 | 13.5 | 89.0 | +3.8 [+2.6, +5.0] | 5e-10 | 80.4 / 84.1 / 83.8 / 79.5 |
| Qwen2.5-Omni-3B | 3,558 | 65.4 | 50.3 | 90.6 | +15.1 [+13.5, +16.8] | 5e-73 | 81.4 / 81.8 / 82.7 / 78.4 |
| MiniCPM-o 2.6 | 3,441 | 48.1 | 30.1 | 93.9 | +17.9 [+16.2, +19.7] | 3e-83 | 80.3 / 84.6 / 84.1 / 79.1 |
| Phi-4-MM | 3,478 | 27.6 | 37.4 | 87.2 | −9.8 [−11.9, −7.7] | 3e-22 | 80.5 / 63.8 / 68.1 / 78.0 |
| InternOmni | 3,483 | 49.2 | 80.8 | 94.7 | −31.6 [−33.7, −29.7] | 4e-183 | 71.2 / 73.3 / 25.3 / 72.7 |

Caption: 95% CI = image bootstrap (2,000). Thus typed vs spoken differ in every model, but in no consistent
direction.

### 4.4 Is the channel gap about negation? (H4, 1,000-question pilot)

**Experiments.** Negative, affirmative and exceptive forms in T, A and V on the pilot list. Δ_c = F_aff,c − F_neg,c
on units that have an affirmative form (n = 647–669). The exceptive analogue covers all pilot units (n = 801–833).

**Results.** For three models the negation penalty is larger in speech than in text, with CIs excluding zero.
For the two Qwen models the CI includes zero (Table 3). The exceptive reference, which keeps the target's name in
the instruction, gives a different picture. InternOmni's spoken penalty vanishes. MiniCPM-o and Phi-4 keep a
smaller positive effect. Qwen-3B shows a negative one. So the evidence for negation-specific channel effects
depends on the model and on the reference form. Two borderline results (Qwen-3B aff−neg, MiniCPM-o exc−neg) cross
zero when the units with an unresolved judge verdict are removed (review a1), so we do not count them as effects. In V, Δ is within ±4.3 points in every model, because in-image
instructions of every form are mostly not followed.

Table 3. Δ_A − Δ_T in points [95% CI], pilot, asr_ok. [src: H4]

| model | aff − neg: Δ_T | Δ_A | Δ_A − Δ_T | exc − neg: Δ_A − Δ_T |
|---|---|---|---|---|
| Qwen2.5-Omni-7B | −19.6 | −23.2 | −3.6 [−8.2, +0.9] | −0.1 [−4.1, +4.2] |
| Qwen2.5-Omni-3B | +19.1 | +14.9 | −4.2 [−9.0, +0.6] | −19.3 [−24.1, −14.7] |
| MiniCPM-o 2.6 | −8.9 | +2.9 | +11.9 [+6.6, +17.6] | +5.7 [+0.6, +10.9] |
| Phi-4-MM | +9.3 | +21.2 | +11.9 [+7.0, +17.0] | +8.1 [+3.2, +13.1] |
| InternOmni | +19.8 | +56.1 | +36.3 [+30.9, +41.8] | +0.4 [−4.8, +5.5] |

A caveat that the paper must state next to the table: the affirmative instruction names the *allowed* component
Y, and the negative one names the forbidden X. For a model that treats speech as the question (InternOmni, Sec.
5.2 of the analysis), a spoken "Mention only Y" leads to an answer about Y, which looks like compliance without
any processing of negation. This is why the exceptive column, which names X, is the better reference for that
model. Qwen2.5-Omni-7B obeys the negative form better than both other forms in T and A (Δ_T −19.6; exceptive −44.9
[src: H4]). The exceptive is violated in about 60% of cases, possibly because "mention everything" outweighs the
exception (conjecture; error anatomy pending).

**[PLACEHOLDER: H4 full extension for Qwen-7B/3B. The pilot CIs for Δ_A − Δ_T cross 0; Vy decides.]**

### 4.5 Which channel wins when the question and the instruction arrive through different channels? (H2, H3-A)

**Experiments.** The question (a positive request) and the instruction come through *different* channels, in
both directions, for three channel pairs (6 layouts). In H2 the instruction is the prohibition. In H3-A it is a
conflicting positive instruction ("Mention only {Y}.", affirmative form, so the 200 open-existence questions are
skipped). The direction effect for a pair (e.g. qT_iA − qA_iT) asks whether the instruction is obeyed equally
whichever channel carries it. H2 − H3-A asks whether this depends on negation.

**Note on the design (also in the review).** This operationalises H2 as *question vs instruction in different
channels*. The Idea document's H2 instead gives two contradictory instructions ("Mention X." typed + "Do not
mention X." spoken). That version has not been run.

**Results (preview, Qwen2.5-Omni-7B only).** With negation, a prohibition is obeyed slightly better when typed
with a spoken question than when spoken with a typed question: qT_iA 13.2% vs qA_iT 17.5%, −4.3 [−7.1, −1.6].
Without negation the direction effect is +2.7 [−0.9, 6.3], and H2 − H3-A for T–A is −7.9 [−12.9, −3.1] [src: H2].
Whenever the instruction is the in-image item it is violated in 86–90% of cases, and whenever the question is the
in-image item the instruction is obeyed (17–20%). For T–V and A–V, H2 − H3-A ≈ 0.
**[PLACEHOLDER: 5-model H2/H3-A table and text (analyze_h2.py --tag h2_pilot1000 --asr_ok_only, all models), job
job-h2-rest.]**

---

## 5 Analysis

### 5.1 Violation kinds
Violations are mostly STATE in all channels. Under V, META (naming the target in order to exclude it) is the
second kind for both Qwen models (6.2 / 6.1%) and Phi-4. DENY ("there is no dog") is concentrated in closed
existence under the typed prohibition (MiniCPM-o 34.2%, Qwen-7B 25.3%, Qwen-3B 23.2%, InternOmni 10.9%) [src: FULL
kind tables]. There, saying an option is absent is close to answering the closed question. **[PLACEHOLDER: per-model
kind summary table T/A/V.]**

### 5.2 Over-suppression and the "audio as question" pattern
InternOmni's recall of the allowed components falls from 71.2% (C0) to 25.3% under a spoken prohibition, while T
and V keep it (73.3 / 72.7). In count it is 10.1%. Its answers to A often respond to the spoken sentence instead of
the typed question **[PLACEHOLDER: 3 examples from review_all.csv]**. Phi-4 loses recall under T (63.8) and A
(68.1) vs C0 80.5. In the other three models, recall under T and A is at or above C0 [src: FULL].

### 5.3 Wording, voice, split, occlusion (full set)
- Template: N3 ("Never refer to ...") is the most violated typed template for InternOmni (67.9%). All tests are
  repeated without N3. **[PLACEHOLDER: one per-template line per model; keep "sweep copies are not independent".]**
- Voice: female vs male spoken violation differs by at most 6.5 points (InternOmni 83.8 vs 77.3) [src: FULL].
- Split (train vs val images, contamination check, full set only): e.g. MiniCPM-o A 37.1 (train) vs 21.0 (val);
  Phi-4 V 91.9 vs 81.6. Split and source are confounded (val = mostly CLEVR/VAW/GQA val) **[PLACEHOLDER: per-source
  split table]**. No split claim from the pilot.

### 5.4 Comprehension probe (H3-B v1): not a valid comprehension measure
Asking "which item must not be mentioned?" *after* an active prohibition makes the model avoid the forbidden option
(Qwen-7B T 21.2%, MiniCPM-o T 8.8%: below the 33% chance level), so v1 measures obedience. For V, NC is 95–100%:
the written instruction is identified when asked [src: CTRL]. **[PLACEHOLDER: COMP v2 (instruction quoted first),
not run on B200.]**

---

## 6 Related work
(3–4 run-in blocks + checkmark table, from paper/related_work.md; each block ends with one contrast sentence.)
**Negation in language and vision-language models.** [cite] … Unlike these, we vary the channel and hold the words
fixed.
**Negative and exclusion instructions.** [cite] … We add the no-instruction baseline per model and the exceptive
control.
**Instructions inside images (typographic / visual prompts).** [cite] … We pair them with OCR and empty-box
controls, which separate reading from obeying.
**Speech instructions and modality conflict in omni-models.** [cite MCR-Bench, ALME after verifying links] … We
make the instruction channel the only manipulated factor and test the interaction with polarity.
**[PLACEHOLDER: checkmark table: channel coverage, paired baseline, in-image instruction, audio instruction,
polarity control, CIs.]**

---

## 7 Discussion
**In-image instructions read as content, not as commands.** The models read the text and copy it, but its
imperative force is largely lost (Sec. 4.2, 4.4). We do not test whether this is a safety feature (resistance to
typographic injection) or a missing capability. Both readings agree with the data.
**There is no single "weaker channel".** The typed–spoken gap changes sign across models (Sec. 4.3). Claims about
"audio instructions" in general are not supported by these data.
**[PLACEHOLDER: method part (steering / CFG / transcribe-then-obey on Qwen-Omni), not run.]**

## Limitations
Synthetic speech (two CosyVoice2 voices), English only. The question is always typed, which may favour the text
channel. Images were filtered with a VLM, which may favour easy images. The LLM judge was checked against humans
only on server 1's subset. H4, controls, H3-B and H2/H3-A use a 1,000-question pilot: no per-branch or split
claims are made from it. One greedy answer per question and condition. The neutral-audio control (does any audio
segment change behaviour?) and instruction-position tests (H3-C) were not run.

---

## Appendix (planned)
A. Templates and forms verbatim. B. Judge prompt, think pass, label pass; judge audit (AUD). C. Full per-branch
tables (FULL, H4). D. All-units sensitivity (FULL-ALL) and think_error sensitivity (audit). E. Pilot list
composition. F. Examples per violation kind.
