# Does a prohibition survive the channel? Negative instructions in text, speech, and image for omni-models

Draft v1 (Pico, 2026-11-24) for Vy + Huy. Target: ACL 2027 (ARR). Plain markdown.
All numbers are PILOT numbers (dataset v2 subset, 932 questions) unless marked otherwise. Every number comes from
the result files named in brackets. Placeholders look like [TBD: ...]. No claim about which channel is "better"
is made beyond what the confidence intervals show, per model.

---

## Abstract

Omni-models accept text, speech and images in a single prompt, so a user can type, say, or show what the model must
not do. We ask whether a negative instruction ("Do not mention the dog.") is obeyed equally when it arrives through
these three channels. We build a component-level negation VQA benchmark on COCO, VAW, GQA and CLEVR in which every
question has a forbidden answer component, and the same instruction is delivered as a separate text segment, as
synthetic speech, or as text written inside the image. Violations are counted only where the model states the
component without any instruction, so every comparison is paired within an image. In a pilot with five open
omni-models (932 questions), instructions written in the image are violated in 89-96% of cases, although the one
model checked so far transcribes them almost perfectly (Qwen2.5-Omni-7B: 930/932 exact), while the text-versus-speech gap is large but
changes sign across models (from 20.0 points more violations under text to 29.8 points more under speech).
[TBD: one sentence on H4, whether the gap is specific to negation (affirmative and exceptive forms).]
[TBD: one sentence on H2/H3-A conflict results.] We release the benchmark, audio, rendered images and scoring code.

---

## Figure 1 (planned)

Left: one image (e.g. a kitchen), the question "What foods are present in the image?", and the instruction
"Leave the carrot out of your answer." shown three ways: as a text block, as a waveform, written on the image.
Right: violation rate among C0-mentioned questions per channel for the five models (dots with 95% CIs,
experiments/figures/h1v2_violation_pilot200.png restyled).
Caption idea: "The same prohibition, three channels. Written in the image it is read but almost never obeyed; text
versus speech depends on the model."

---

## 1 Introduction

Omni-models such as Qwen2.5-Omni, MiniCPM-o and Phi-4-multimodal take images, audio and text in one prompt. In use,
constraints reach them through all three: a user types "don't mention the price", says it aloud in a voice
assistant, or the constraint sits on a slide or a sign in the picture. Negative constraints are a known weak spot
of language and vision-language models: negated queries are often treated like affirmative ones, and naming a
forbidden concept can make it more likely to appear. Whether a prohibition is followed *equally well* when the same
words come through a different channel has not been measured.

A fair comparison is harder than it looks. (i) A model can only "comply" with "do not mention X" if it would have
mentioned X otherwise, so violation must be conditioned on the model's own unconstrained answer. (ii) Channels
differ in more than modality: an instruction glued into the question string is not the same input as a separate
audio segment, and text drawn on an image also covers pixels. (iii) A channel gap may reflect instruction following
in general rather than negation; separating the two needs affirmative and exceptive versions of the same
constraint. (iv) A model may fail to obey because it did not hear or read the instruction; comprehension must be
probed separately.

We design the benchmark and the experiments around these four points and test five open omni-models.

Contributions:
1. A component-level negation VQA benchmark (dataset v2): images from four sources, five question branches
   (existence, closed existence, count, attribute, relation), a forbidden component per question with aliases,
   seven negative templates, affirmative and exceptive forms, ASR-checked synthetic speech in two voices, and
   rendered in-image instructions with an empty-box control. [TBD: final size after the full build.]
2. A paired evaluation protocol: violation among questions where the model's own no-instruction answer (C0) states
   the target, four violation kinds (STATE, META, DENY, WRONG), an LLM judge that reads the question
   (agreement with a human annotator kappa = 0.83 on 49 answers, pilot), and image-level bootstrap CIs.
3. A pilot study on five omni-models showing that (a) in-image instructions are violated in 89-96% of cases (and, for the one model
   checked so far, read correctly), and (b) the text-versus-speech gap is model-specific in sign. [TBD: H4 result.]
4. Diagnostics that separate reading, understanding and obeying: OCR of the in-image instruction, an empty-box
   control, multiple-choice comprehension per channel, and conflict tests between channels (H2, H3-A). [TBD: results.]

---

## 2 Related work

Negation in vision-language models. NegBench (Alhamoud et al., CVPR 2025, arXiv 2501.09425) and NegVQA
(arXiv 2505.22946) show that VLMs often misread negated queries and captions. These works test understanding of
negated *content*; we test compliance with a prohibition on what the *answer* may say, and how the channel of the
prohibition matters.

Negative instructions and priming. Naming a forbidden concept can increase its appearance in generation (negative
prompts in text-to-image models, arXiv 2404.15154) [TBD: verify Truhn et al., arXiv 2503.22395, before citing].
Our exceptive form ("Mention everything except X") names X without a negation word and serves as the priming control.

Instructions inside the image. Typographic and visual prompt injection show that text drawn into an image can steer
LVLMs (arXiv 2408.03554, 2503.11519, 2604.12371). Our visual channel is the benign case: a legitimate instruction
written in the image, with an empty-box control for the covered pixels.

Speech versus text and modality conflict. Spoken instructions are followed less reliably than the same text in many
audio-language models (arXiv 2603.01502; VoiceBench [TBD: citation]), and when audio and text disagree, models tend
to follow text (MCR-Bench, arXiv 2508.15407; ALME, arXiv 2602.11488); XModBench (arXiv 2510.15148) and OmniBench
report cross-modal inconsistency in omni-models. Whether *negative* instructions show the same dominance pattern is
the question our H2 / H3-A conflict design answers. [TBD: verify every link before submission.]

---

## 3 Benchmark (dataset v2)

Sources and branches. Images come from COCO (existence, closed existence, count), CLEVR (count, attribute), VAW
(attribute) and GQA (relation). For each image a question has several answer components (e.g. the foods in the
image); one component is the target that the instruction forbids. Components carry aliases (e.g. "dining table" /
"table") for scoring.

Image selection. Pixel filters (blur, brightness, black-and-white, large text, near-duplicates) and a VLM check
(Qwen3-VL-8B yes/no on every component, distractor, and large text) select images. Full selection: COCO 950, CLEVR
600, VAW 950, GQA 654 images, 5,335 units. The pilot uses a subset of 1,150 units (230 per branch); 79 units are
excluded where two components cannot be separated (same object with two labels, part-whole such as keyboard /
laptop, attribute overlap), leaving 1,071. Own review of 60 random accepted images: 6 (10%) had doubtful ground
truth; this mostly drops out because violation is conditioned on the model's own C0.

Instructions. Seven negative templates: N1 "Do not mention {X}.", N2 "Don't say anything about {X}.", N3 "Never
refer to {X}.", N4 "Make no mention of {X}.", N5 "Answer without mentioning {X}.", N6 "Avoid mentioning {X}.",
N7 "Leave {X} out of your answer." (N6 and N7 have no negation word.) One template per question; a 10% subset
repeats all seven (template sweep). Forms for H4: exceptive "Mention everything except {X}." (all branches) and
affirmative "Mention only {Y}." (only where the question already lists every option: closed existence, count,
attribute, relation; in open existence it would leak the answer).

Speech. Every sentence is synthesised with CosyVoice2-0.5B (zero-shot cloning, one female and one male voice, 50/50)
and transcribed with Whisper; 33 of 1,330 sentences (2.5%) do not match and are flagged (asr_ok = false); the pilot
uses asr_ok units only.

In-image text. The instruction is rendered on the image away from the question's objects; 5.4% of rendered images
still overlap a component (occludes_subject, reported separately). For each, an empty-box image with the same box and
no text is the visual control (C0_box).

[TBD: Table 1, statistics of the final build per source / branch / split / form.]

---

## 4 Experimental setup

Models. Qwen2.5-Omni-7B and -3B (thinker only), MiniCPM-o 2.6, Phi-4-multimodal-instruct, InternOmni. Greedy
decoding, 100 new tokens (answers that hit the limit are regenerated with 256).

Conditions (H1). Input order is image, then question, then instruction; items are separated by a newline.
- C0: image + question, no instruction.
- T: question, then the instruction as its own text item (the fair text match for audio, which is always a
  separate segment).
- A: question, then the spoken instruction.
- V: the image with the instruction written in it, then the question.
Controls: C0_box (empty box, no text, no instruction), OCR_V (the model transcribes the written instruction).

Comprehension (H3-B). COMP v1 asked a three-option multiple-choice question ("which item were you asked not to
mention?") with the instruction in its usual slot after the question. This turned out to measure obedience, not
comprehension (Section 6). COMP v2 quotes the instruction first as content ("Here is an [spoken] instruction:"),
then asks the same multiple-choice question, and adds a verbatim repeat task per channel (REP_T, REP_A).

Scoring. Primary metric: violation = any mention of the forbidden component among questions whose C0 answer states
it. Each violation is labelled STATE (states the fact), META (names it to exclude it), DENY (says it is absent or
false) or WRONG (gives a different value); leak = STATE is secondary. Judge v3: Qwen3-8B reads the question, the
components and the answer; disagreements with a regex second opinion get a thinking pass; bare-number answers in
the count branch are scored by rule. Required recall (share of the other components still stated) measures
over-suppression.

Statistics. Paired comparisons within question; 95% CIs from 1,000-2,000 bootstrap resamples over images
(template-sweep copies are not independent); exact McNemar tests.

Human check. Vy labelled 67 answers of an earlier run; on the 49 whose answer is unchanged, judge v3 agrees on
93.9% (kappa 0.83). [TBD: blind 300-item check (Vy) + 100 overlap (Huy) on pilot v2.]

---

## 5 Results (pilot, 932 questions)

### 5.1 The model states the target without instruction in 83-86% of questions

| model | C0 states target % [95% CI] | n C0-mentioned |
|---|---|---|
| Qwen2.5-Omni-7B | 84.8 [82.5, 87.1] | 790 |
| Qwen2.5-Omni-3B | 85.7 [83.5, 87.9] | 799 |
| MiniCPM-o 2.6 | 84.3 [82.0, 86.7] | 786 |
| Phi-4-MM | 82.8 [80.6, 85.3] | 772 |
| InternOmni | 83.6 [81.3, 85.9] | 779 |

[paper/marginal_cis_pilot200.md]

### 5.2 Finding 1: instructions written in the image are almost never obeyed

Violation rate (%) among C0-mentioned questions, 95% image-bootstrap CIs [paper/marginal_cis_pilot200.md]:

| model | T | A | V |
|---|---|---|---|
| Qwen2.5-Omni-7B | 17.2 [14.7, 20.0] | 13.0 [10.8, 15.3] | 88.9 [86.7, 91.0] |
| Qwen2.5-Omni-3B | 63.0 [59.6, 66.2] | 51.1 [47.4, 54.6] | 91.7 [89.7, 93.6] |
| MiniCPM-o 2.6 | 51.1 [47.7, 54.6] | 31.2 [28.0, 34.5] | 95.2 [93.6, 96.6] |
| Phi-4-MM | 26.0 [23.1, 29.1] | 40.9 [37.4, 44.4] | 91.3 [89.2, 93.1] |
| InternOmni | 50.1 [46.5, 53.5] | 79.8 [77.0, 82.5] | 95.8 [94.2, 97.0] |

V exceeds both T and A in every model (paired A-V from -15.9 [-19.0, -12.8] for InternOmni to -75.8
[-78.9, -72.8] for Qwen2.5-Omni-7B; experiments/REPORT.md). The violations under V are mostly plain statements
(STATE 82-94% of C0-mentioned questions): the model answers the question as if the written instruction were not there.

Is the instruction read? For Qwen2.5-Omni-7B, asked to transcribe it, the model reproduces it exactly in 930 of 932
images (99.8%), and the empty box alone does not change whether the target is mentioned (C0_box - C0 = -0.2 pp
[-1.9, +1.6], n = 932). So, for this model, the in-image instruction is read and the covered pixels do not matter,
yet it is violated in 88.9% of cases. [TBD: OCR_V and C0_box for the other four models (Stage 3 running).]
[TBD: COMP v2: does the model *understand* the written text as an instruction?]

### 5.3 Finding 2: the text-versus-speech gap is large but its sign depends on the model

| model | T - A (pp) [95% CI] |
|---|---|
| Qwen2.5-Omni-7B | +4.2 [+1.6, +6.6] |
| Qwen2.5-Omni-3B | +11.9 [+8.5, +15.5] |
| MiniCPM-o 2.6 | +20.0 [+16.2, +23.8] |
| Phi-4-MM | -14.9 [-19.2, -10.8] |
| InternOmni | -29.8 [-34.0, -25.3] |

[experiments/REPORT.md] Every CI excludes zero, but three models violate more under text and two more under speech.
There is therefore no single channel ordering across models in this pilot. The same signs appeared in an earlier
run with a different scoring rule (pilot v1, four models), which suggests the pattern is not an artefact of the
judge change.

Two model-specific behaviours qualify these numbers:
- InternOmni under A: required recall falls from 70.1% (C0) to 26.8% (A) (T 74.3%, V 73.4%). Many answers are one
  sentence about the forbidden item only, as if the spoken instruction were taken as the question. Its high A
  violation may reflect the role it assigns to audio rather than negation. [TBD: COMP v2 A for InternOmni.]
- Phi-4-MM over-suppresses in the count branch (required recall 74% at C0 to 7% under T, 17% under A): it drops
  the other counts as well.

### 5.4 Is the gap about negation? (H4)

[TBD: Δ_m = F_aff,m - F_neg,m per channel with CIs; Δ_audio - Δ_text and Δ_visual - Δ_text; exceptive versus
negative per channel (priming). Source: Stage 3 forms run, analyze.py H4 section.]

### 5.5 Which channel wins in a conflict? (H2, H3-A)

[TBD: H2 (positive request and negation in different channels, 3 channel pairs x 2 directions) and H3-A (same
layouts with a conflicting positive instruction). Code only; waiting for Vy.]

---

## 6 Analysis

Comprehension probes can themselves be obeyed. In COMP v1 the multiple-choice question came first and the
instruction after it. Qwen2.5-Omni-7B picked the forbidden item in only 18.7% [16.1, 21.0] of text cases, below the
33.3% chance level, versus 45.1% [42.1, 48.2] for audio and 99.0% [98.4, 99.6] for visual
[experiments/h1_v2/analysis_comp_pilot200.md]. The answers show the model applying the instruction to its
multiple-choice answer (it avoids the forbidden option). COMP v1 is therefore a measure of obedience; COMP v2 quotes
the instruction as content first. We do not interpret the v1 channel ordering further: it mixes reading,
understanding and obeying. [TBD: COMP v2 per channel and model.]

Violation kinds. For Qwen2.5-Omni-7B, META and DENY are small (2-4%) in every channel; DENY is concentrated in
closed existence (21.5% under T), where saying "there is no X" still names X. [TBD: table for all models.]

Template wording. N3 ("Never refer to") is kept in all main numbers and reported separately; for
Qwen2.5-Omni-7B, T - A without N3 is +4.8 [+2.3, +7.5] and with N3 only -1.1 [-10.3, +8.0] (n = 87).
[TBD: template effects for all models; explicit (N1-N5) versus implicit (N6-N7) cues per channel.]

Split. For Qwen2.5-Omni-7B, violation on train-split images and val images is similar (T 17.3 vs 17.0, A 14.2 vs
10.7, V 87.9 vs 90.9); split is confounded with source (CLEVR is all val). [TBD: CIs; all models.]

Voice and occlusion. Qwen2.5-Omni-7B: A violation 13.4% (female voice, n = 409) vs 12.6% (male, n = 381); V 88.9%
without and 88.1% with occlusion (n = 42). [TBD: CIs; all models.]

[TBD: H3-C instruction before vs after the image; H3-D violation vs target area.]

---

## 7 Limitations

- Pilot size: 932 questions from a 1,150-unit subset; the full build is not done.
- Synthetic speech (two cloned voices, English only); real and diverse voices are planned.
- The question is always in text, which may favour the text channel; T is a separate text item to match audio.
- Image selection uses a VLM check, which may favour easy images; about 10% of reviewed accepted units had doubtful
  ground truth.
- The judge is an LLM (Qwen3-8B); agreement with one human annotator is kappa 0.83 on 49 answers. A larger blind
  check is pending.
- The visual channel is typographic text, not a symbolic prohibition sign (Vy's design decision).
- Five open models up to ~8B parameters; no closed or larger omni-models yet.

---

## What is still missing (for v2 of this draft)

1. Stage 3 for the remaining four models: OCR_V, C0_box, COMP v1 (running).
2. COMP v2 (understanding per channel, verbatim repeat) for all five models (queued after Stage 3).
3. H4: affirmative and exceptive forms, Δ_m per channel with CIs (running in Stage 3).
4. H2 and H3-A conflict experiments (code only; needs Vy's OK and new TTS / rendered images).
5. H3-C (instruction before the image) and H3-D (target area) analysis.
6. Human check of pilot v2 (Vy 300 items, Huy 100 overlap).
7. Full dataset build (~2.5k images) and the final benchmark statistics table.
8. Per-model tables for violation kinds, templates, split, voice, occlusion with CIs (Appendix).
9. Verified citations for every related-work entry; Figure 1.
