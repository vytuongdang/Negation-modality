# Typed, Spoken, or Shown: Do Omni-Models Obey a Prohibition Equally in Every Channel?

Draft v1 (Pico, 2026-11-24) for Vy + Huy. Target: ACL 2027 (ARR). Plain markdown; style follows paper/style_notes.md.
All numbers are PILOT numbers (dataset v2 subset, 932 questions) from the files named in brackets. Unfinished work
is marked [TBD: ...]. No claim about a "better" channel is made beyond what the per-model CIs show.

Channel coding used everywhere (text, figures, tables): C0 = no instruction (grey), T = typed (blue),
A = spoken (orange), V = written in the image (green).

Running example (pilot item coco-existence-000199): a photo of a stir-fry; question "What foods are present in the
image?"; instruction "Leave the carrot out of your answer." (template N7).

---

## Abstract

Omni-models take text, speech and images in one prompt, so a user can type, say, or show what a model must not do.
Work on negation has studied the content of queries, not whether a prohibition survives the channel it arrives in.
We build a paired benchmark in which each image-question pair has one forbidden answer component and the same
instruction ("Leave the carrot out of your answer.") is given as a separate text item, as synthetic speech, or as
text written in the image; violations are counted only where the model names the component without any
instruction. In a pilot with five open omni-models (932 questions), in-image instructions are violated in 88.9%
[86.7, 91.0] to 95.8% [94.2, 97.0] of cases, although the one model checked so far transcribes them almost
perfectly (930 of 932). Typed and spoken instructions differ by 4 to 30 points, but the sign of the difference
changes across models. [TBD: one sentence on H4, whether the gap is specific to negation.] [TBD: one sentence on
the conflict tests.] The design separates reading, understanding and obeying an instruction per channel.
[TBD: URL.]

---

## Figure 1 (planned)

(a) The running example: one image, the question, and "Leave the carrot out of your answer." delivered three ways
(typed block, waveform, text on the image), plus the no-instruction answer C0 that names the carrot.
(b) [TBD: the real outputs of 2-3 models for the three channels, each marked obeyed / violated.]
(c) Violation rate per model and channel, dots with 95% CIs (from experiments/figures/h1v2_violation_pilot200.png).
Caption idea: "The same prohibition in three channels. (a) Input. (b) Outputs. (c) Violation among questions where
the model names the target without instruction (n = 772-799 per model, image bootstrap CIs). Written in the image,
the prohibition is almost always violated; typed versus spoken depends on the model."

---

## 1 Introduction

Omni-models such as Qwen2.5-Omni, MiniCPM-o and Phi-4-multimodal accept images, audio and text together. In use,
constraints reach them through all three: a user types "don't mention the price", says it to a voice assistant, or
the rule sits on a slide in the picture. If a model obeys "do not mention X" when typed but not when spoken or
shown, a user's constraint depends on how it was delivered.

Negation is a known weak spot. Vision-language models often read negated queries like affirmative ones (NegBench,
NegVQA), and naming a forbidden concept can make it more likely to appear (arXiv 2404.15154). Separately, audio-text
studies find that models follow text when text and audio disagree (MCR-Bench, ALME). But no study holds the
instruction fixed and changes only the channel, so it is unknown whether a prohibition is channel-robust.

We ask: is a negative instruction obeyed equally when the same words arrive typed, spoken, or written in the image,
and if not, is the gap about negation or about the channel? Figure 1 shows the setup on one example.

A fair comparison needs four controls. (i) A model can only comply with "do not mention the carrot" if it would
otherwise mention it, so violation is conditioned on its own no-instruction answer C0. (ii) Audio is always a
separate segment, so the typed instruction is also a separate item (not glued into the question). (iii) Text on the
image covers pixels, so an empty box of the same size is a control (C0_box). (iv) A model may disobey because it did
not read or hear the instruction, or because it follows any instruction worse in that channel; we therefore probe
transcription and comprehension per channel, and compare negative with affirmative and exceptive forms (H4).

We build a component-level negation benchmark (Sec. 3), define a paired protocol (Sec. 4), and test five open
omni-models on a 932-question pilot (Sec. 5-6). Our pilot findings:

1. The five models name the target without instruction in 82.8% [80.6, 85.3] to 85.7% [83.5, 87.9] of questions,
   so roughly 780 questions per model can test compliance (Sec. 5.1).
2. Written in the image, the prohibition is violated in 88.9% to 95.8% of cases in every model (Sec. 5.2). For
   Qwen2.5-Omni-7B this is not a reading failure: it transcribes the written instruction exactly in 930/932 images,
   and the empty box alone changes nothing (-0.2 pp [-1.9, +1.6]) (Sec. 5.1). [TBD: other four models.]
3. Typed and spoken prohibitions differ in every model (T - A from +20.0 [+16.2, +23.8] to -29.8 [-34.0, -25.3]
   pp), but three models violate more when typed and two more when spoken: the pilot shows no single channel
   ordering (Sec. 5.3).
4. A comprehension probe placed before the instruction is itself obeyed: Qwen2.5-Omni-7B picks the forbidden item
   in only 18.7% [16.1, 21.0] of typed cases, below the 33.3% chance level. Comprehension must be tested with the
   instruction quoted as content (Sec. 6.1).
5. [TBD: H4: whether the channel gaps remain when the instruction is affirmative or exceptive (Sec. 5.4).]

---

## 2 Related work

**Negation in vision-language models.** NegBench (Alhamoud et al., CVPR 2025, arXiv 2501.09425) and NegVQA
(arXiv 2505.22946) show that VLMs misread negated captions and questions. They test understanding of negated
content; we test compliance with a prohibition on the answer, delivered through different channels.

**Negative instructions and priming.** Negative prompts in text-to-image models can push generation toward the
forbidden concept (arXiv 2404.15154). Negation remains hard for LLMs across sizes and languages (Vrabcová et al., arXiv 2503.22395). Our exceptive form ("Mention
everything except the carrot.") names the target without a negation word and serves as the priming control.

**Instructions inside the image.** Typographic and visual prompt injection show that drawn-in text can steer LVLMs
(arXiv 2408.03554, 2503.11519, 2604.12371). These are adversarial; our V channel is a legitimate instruction from the
user, with an OCR check and an empty-box control.

**Speech versus text instructions.** Spoken input is followed less reliably than the same text in many
audio-language models (arXiv 2603.01502; VoiceBench [TBD: citation]); under audio-text conflict models tend to
follow text (MCR-Bench, arXiv 2508.15407; ALME, arXiv 2602.11488); XModBench (arXiv 2510.15148) and OmniBench report
cross-modal inconsistency. None tests a negative instruction across channels with a paired baseline; our H2/H3-A
conflict tests ask whether dominance changes with polarity.

| work | negation | text instr. | audio instr. | in-image instr. | paired no-instr. baseline | CIs |
|---|---|---|---|---|---|---|
| NegBench / NegVQA | yes | yes | no | no | no | [TBD] |
| visual prompt injection | no | yes | no | yes | partly | [TBD] |
| MCR-Bench / ALME | no | yes | yes | no | no | [TBD] |
| ours | yes | yes | yes | yes | yes | yes |

[TBD: verify every row and link before submission.]

---

## 3 Benchmark (dataset v2)

The pipeline has six steps (counts for the pilot subset; full build [TBD]):

1. **Collect** image-question units from COCO (existence, closed existence, count), CLEVR (count, attribute), VAW
   (attribute) and GQA (relation). Each question has several answer components with aliases (e.g. "dining table" /
   "table"); one is the target.
2. **Filter** images with pixel checks (blur, brightness, black-and-white, large text, near-duplicates) and a VLM
   check (Qwen3-VL-8B, yes/no on every component and distractor): COCO 950, CLEVR 600, VAW 950, GQA 654 images,
   5,335 units.
3. **Subset and clean** for the pilot: 1,150 units (230 per branch); drop 79 where two components cannot be told
   apart (same object with two labels, part-whole such as keyboard / laptop, overlapping attributes): 1,071 units.
   One negative sample per unit, ASR-checked audio only: 932 questions.
4. **Phrase** the prohibition with one of seven templates (Table 2); a 10% subset repeats all seven. Forms for H4:
   exceptive "Mention everything except {X}." (all branches) and affirmative "Mention only {Y}." (closed existence,
   count, attribute, relation only; in open existence it would give away the answer).
5. **Speak** every sentence with CosyVoice2-0.5B (two cloned voices, female / male, 50/50) and transcribe it with
   Whisper: 1,330 sentences, 33 (2.5%) mismatches flagged and left out of the pilot.
6. **Render** the sentence onto the image away from the question's objects (5.4% still overlap a component,
   flagged), plus an empty-box twin for C0_box.

Arithmetic: 932 questions x 4 H1 conditions (C0, T, A, V) = 3,728 answers per model in the H1 pilot.

Table 2 (templates): N1 "Do not mention {X}." · N2 "Don't say anything about {X}." · N3 "Never refer to {X}." ·
N4 "Make no mention of {X}." · N5 "Answer without mentioning {X}." · N6 "Avoid mentioning {X}." · N7 "Leave {X}
out of your answer." N6 and N7 carry no negation word.

Quality: in our own review of 60 random accepted units, 6 (10%) had doubtful ground truth; conditioning on C0
removes most of them from the violation counts. [TBD: Table 1, statistics of the full build per source, branch,
split, form.]

---

## 4 Experimental setup

**Models.** Qwen2.5-Omni-7B and -3B (thinker only), MiniCPM-o 2.6, Phi-4-multimodal-instruct, InternOmni. Greedy
decoding, 100 new tokens; answers that hit the limit are regenerated with 256.

**Conditions.** Input order: image, question, instruction; items separated by a newline.
C0: image + question. T: question, then the instruction as its own text item. A: question, then the spoken
instruction. V: the image with the instruction written in it, then the question. Controls: C0_box (empty box, no
instruction) and OCR_V (transcribe the text in the image).

**Violation rate.** In plain words: how often the model still names the forbidden component, among questions where
it names that component without any instruction. Symbol: v_m = P(target named | channel m, C0 names target).
A high v_m means the prohibition in channel m is mostly ignored. Example: if C0 lists "broccoli, tofu, carrot" and
the T answer lists "broccoli, tofu, and no carrot", that is a violation. Conditioning on C0 is needed because not
naming something the model would never name is not compliance.

**Violation kinds.** STATE (states the fact: "there is a carrot"), META (names it to exclude it: "apart from the
carrot..."), DENY (says it is absent or false: "there is no carrot"), WRONG (gives a different value, e.g. a wrong
count). Leak = STATE is a secondary metric.

**Negation penalty (H4).** F_m = 1 - v_m is compliance. Delta_m = F_aff,m - F_neg,m: how much harder the negative
form is than the affirmative one in channel m. A high Delta_audio - Delta_text would mean negation, not instruction
following in general, is harder by speech.

**Required recall.** Share of the other (allowed) components still named; a drop means over-suppression.

**Scoring.** Judge v3: Qwen3-8B reads the question, the components and the answer; disagreements with a regex
second opinion get a thinking pass; bare-number answers in the count branch are scored by rule; a label pass
assigns the violation kind. Against one human annotator (Vy) on 49 unchanged answers: 93.9% agreement, kappa 0.83.
[TBD: blind 300-item check (Vy) and 100-item overlap (Huy).]

**Statistics.** All comparisons are paired within question. 95% CIs are percentile bootstrap over images
(1,000-2,000 resamples); paired differences also get an exact McNemar test.

**Comprehension (H3-B).** COMP v1: three-option multiple choice ("which item were you asked not to mention?") with
the instruction in its usual slot after the question. COMP v2: the instruction is quoted first as content ("Here
is an [spoken] instruction:"), then the same multiple choice, plus a verbatim repeat task per channel (REP_T, REP_A).

---

## 5 Results (pilot, 932 questions)

### 5.1 Sanity checks: the models name the targets, and Qwen2.5-Omni-7B reads the written instruction

**Experiments.** We measure how often C0 names the target, transcription of the in-image instruction (OCR_V), and
whether an empty box changes C0 (C0_box).

**Results.** All five models name the target without instruction in 83-86% of questions (Table 3). For
Qwen2.5-Omni-7B, OCR_V is exact in 930/932 images (99.8%), and C0_box - C0 = -0.2 pp [-1.9, +1.6] (n = 932).
Speech: all pilot audio passed the Whisper check (Sec. 3). [TBD: OCR_V, C0_box for the other four models.]

Table 3. Share of questions where C0 names the target (95% CI, image bootstrap; n = 932 per model) and the number
kept for compliance tests. Takeaway: about 780 testable questions per model, with similar rates across models.
[paper/marginal_cis_pilot200.md]

| model | C0 names target % [95% CI] | n C0-mentioned |
|---|---|---|
| Qwen2.5-Omni-7B | 84.8 [82.5, 87.1] | 790 |
| Qwen2.5-Omni-3B | 85.7 [83.5, 87.9] | 799 |
| MiniCPM-o 2.6 | 84.3 [82.0, 86.7] | 786 |
| Phi-4-MM | 82.8 [80.6, 85.3] | 772 |
| InternOmni | 83.6 [81.3, 85.9] | 779 |

### 5.2 Prohibitions written in the image are violated in about nine of ten cases in all five models

**Experiments.** Violation rate v_m for T, A, V among C0-mentioned questions (Table 4).

**Results.** V is violated in 88.9-95.8% of cases, above T and A in every model (paired A - V from -15.9
[-19.0, -12.8] pp for InternOmni to -75.8 [-78.9, -72.8] pp for Qwen2.5-Omni-7B, McNemar p < 1e-20 for every T vs V and A vs V pair
[experiments/REPORT.md, analysis_pilot200.md]). Most V violations are STATE (82-94% of C0-mentioned questions): the
model answers as if the written instruction were absent. For Qwen2.5-Omni-7B, which reads it (Sec. 5.1), this
points to obeying, not reading, as the failure; whether the model understands the text as an instruction to it is
open [TBD: COMP v2].

Table 4. Violation rate (%) among C0-mentioned questions, 95% image-bootstrap CIs (n = 772-799 per model).
Takeaway: V is high everywhere; T versus A depends on the model. [paper/marginal_cis_pilot200.md]

| model | T | A | V |
|---|---|---|---|
| Qwen2.5-Omni-7B | 17.2 [14.7, 20.0] | 13.0 [10.8, 15.3] | 88.9 [86.7, 91.0] |
| Qwen2.5-Omni-3B | 63.0 [59.6, 66.2] | 51.1 [47.4, 54.6] | 91.7 [89.7, 93.6] |
| MiniCPM-o 2.6 | 51.1 [47.7, 54.6] | 31.2 [28.0, 34.5] | 95.2 [93.6, 96.6] |
| Phi-4-MM | 26.0 [23.1, 29.1] | 40.9 [37.4, 44.4] | 91.3 [89.2, 93.1] |
| InternOmni | 50.1 [46.5, 53.5] | 79.8 [77.0, 82.5] | 95.8 [94.2, 97.0] |

### 5.3 Typed and spoken prohibitions differ in every model, in opposite directions across models

**Experiments.** Paired difference T - A per model (Table 5).

**Results.** Every CI excludes zero. Qwen2.5-Omni-7B, Qwen2.5-Omni-3B and MiniCPM-o violate more when typed;
Phi-4-MM and InternOmni violate more when spoken. The same signs appeared for the four models of an earlier run
with a different scoring rule (pilot v1), so the pattern is not a product of the judge change. Two model-specific
behaviours qualify the numbers:
- InternOmni under A: required recall falls from 70.1% (C0) to 26.8% (T 74.3%, V 73.4%); many answers are one
  sentence about the forbidden item only, possibly because the spoken instruction is taken as the question
  (conjecture; [TBD: COMP v2 A]).
- Phi-4-MM over-suppresses in the count branch: required recall 74% at C0, 7% under T, 17% under A.

Table 5. Paired T - A (pp), 95% image-bootstrap CI, C0-mentioned questions. Takeaway: large gaps, no common
direction. [experiments/REPORT.md]

| model | T - A (pp) [95% CI] |
|---|---|
| Qwen2.5-Omni-7B | +4.2 [+1.6, +6.6] |
| Qwen2.5-Omni-3B | +11.9 [+8.5, +15.5] |
| MiniCPM-o 2.6 | +20.0 [+16.2, +23.8] |
| Phi-4-MM | -14.9 [-19.2, -10.8] |
| InternOmni | -29.8 [-34.0, -25.3] |

### 5.4 Is the channel gap specific to negation? (H4)

[TBD: Delta_m per channel with CIs; Delta_audio - Delta_text and Delta_visual - Delta_text; exceptive versus
negative per channel. Source: Stage 3 forms run (running), analyze.py H4 section.]

### 5.5 Which channel wins when channels conflict? (H2, H3-A)

[TBD: H2 = positive request and negation in different channels (3 channel pairs x 2 directions); H3-A = same
layouts with a conflicting positive instruction. Code only; waiting for Vy.]

---

## 6 Analysis

### 6.1 A comprehension probe before the instruction is obeyed, not answered

In COMP v1 the multiple-choice question came first and the instruction after it. Qwen2.5-Omni-7B picked the
forbidden item in 18.7% [16.1, 21.0] of typed cases (below 33.3% chance), 45.1% [42.1, 48.2] spoken, 99.0%
[98.4, 99.6] written in the image (n = 932 each) [experiments/h1_v2/analysis_comp_pilot200.md]. Its answers avoid the
forbidden option, i.e. the instruction is applied to the probe. COMP v1 therefore mixes reading, understanding and
obeying, and we do not interpret its channel ordering; COMP v2 quotes the instruction as content.
[TBD: COMP v2 for all five models.]

### 6.2 Violation kinds

For Qwen2.5-Omni-7B, META and DENY stay at 2-4% in every channel; DENY is concentrated in closed existence (21.5%
under T), where "there is no carrot" still names the carrot. [TBD: all models, with CIs.]

### 6.3 Wording, split, voice, occlusion

Template N3 ("Never refer to") is kept in all main numbers and reported separately: for Qwen2.5-Omni-7B, T - A is
+4.8 [+2.3, +7.5] without N3 and -1.1 [-10.3, +8.0] with N3 only (n = 87). Train versus val images: T 17.3 vs 17.0,
A 14.2 vs 10.7, V 87.9 vs 90.9 (split confounded with source; CLEVR is all val). Voice: A 13.4% female (n = 409)
vs 12.6% male (n = 381). Occlusion: V 88.9% vs 88.1% (n = 42 occluded). All for Qwen2.5-Omni-7B.
[TBD: CIs and all models; explicit (N1-N5) versus implicit (N6-N7) cues per channel; H3-C instruction before the
image; H3-D violation by target area.]

---

## 7 Limitations

- Pilot size: 932 questions from a 1,150-unit subset; the full build is not done.
- Synthetic speech: two cloned voices, English only; real and diverse voices are not yet tested.
- The question is always typed, which may favour the text channel; T is a separate item to match audio.
- The visual channel is typographic text, not a prohibition sign (a design decision).
- Images were filtered with a VLM check, which may favour easy images; about 10% of reviewed units had doubtful
  ground truth.
- The judge is an LLM (Qwen3-8B); agreement with one annotator is kappa 0.83 on 49 answers; a larger blind check is
  pending.
- Five open models up to about 8B parameters; no larger or closed omni-models.
- Not measured yet: comprehension per channel (COMP v2), negation specificity (H4), conflict (H2, H3-A), instruction
  position (H3-C).

---

## What is still missing (for draft v2)

1. Stage 3 for the remaining four models: OCR_V, C0_box, COMP v1 (running on the GPU).
2. COMP v2 (understanding per channel + verbatim repeat), five models (queued after Stage 3).
3. H4 affirmative / exceptive forms, Delta_m per channel with CIs (running in Stage 3).
4. H2 and H3-A conflict experiments (code only; needs Vy's OK, new TTS and rendered images).
5. H3-C (instruction before the image) and H3-D (target area).
6. Human check of pilot v2 (Vy 300 items, Huy 100 overlap).
7. Full dataset build (~2.5k images) and Table 1 statistics.
8. Appendix tables per model (kinds, templates, split, voice, occlusion) with CIs.
9. Figure 1 with real model outputs; verified citations and related-work table.
