# Ideas (Pico): idea → why → what each outcome would mean → cost → status
Pilot on ≤300 samples in a new versioned folder; scale only after Vy agrees.

## I1. Neutral-audio control (is it audio presence, or the spoken words?)  [proposed 2026-11-19]
- Conditions: T_sep + neutral spoken sentence ("Please look at the picture carefully."), and A_neutral_only
  (question + neutral audio, no instruction: does any audio change what is said?).
- Why: if A and T_sep differ in H1 v2, one reading is that ANY audio segment changes behaviour (salience /
  attention), not the spoken words. Adding an irrelevant audio segment next to the text instruction tests this.
  Only worth running if H1 v2 shows an A vs T_sep difference (either direction).
- Outcomes: T_sep+neutral ≈ T_sep → audio presence irrelevant, the words matter; T_sep+neutral ≠ T_sep → presence
  effect (a confound to report). A_neutral_only vs C0 shows if audio alone suppresses mentions.
- Cost: 2 extra conditions × ~200 samples × 2 models ≈ 0.5 GPU h + 2 TTS sentences.

## I2. Comprehension probe per channel (H3-B, cheap form)  [proposed 2026-11-19]
- After the same input (T_sep / A / V), ask instead: "Which thing were you asked not to mention? Answer in a few
  words." Score with the judge against the target. Gives P(understood) and P(obeyed | understood) per channel.
- Why: separates "did not hear/read it" (ASR/OCR failure, esp. V and A) from "understood but did not obey".
- Cost: 3 conditions × ~200 samples × 2 models ≈ 0.5 GPU h.

## I3. Target log-probability at the answer start (Qwen-Omni, open weights)  [proposed 2026-11-19]
- Teacher-force the model's own C0 answer prefix up to the first mention of the target; compare log p(target
  token) under C0 vs T_sep vs A vs V. A continuous, low-variance measure of suppression strength per channel.
- Why: binary violation may be near floor/ceiling for some models; log-prob shows graded suppression.
- Cost: forward passes only, ~200 samples × 4 conds, < 0.5 GPU h.

## I4. Instruction position (H3-C)  [from Idea doc, refine after H1]
- Instruction before the image vs after the question, text and audio, same samples.

## 2026-11-24: H3-B comprehension v1 is not valid as a comprehension test (Qwen-7B, 932 q)
- COMP_T / COMP_A put the multiple-choice question FIRST and the instruction AFTER it (same slot as T / A).
  Result: COMP_T 18.7% correct (below 33% chance), COMP_A 45.1%, COMP_V 99.0%. Answers are "B. So, what's next on
  your mind?": the model treats the trailing instruction ("Leave the carrot out of your answer.") as a command on
  its MCQ answer and avoids the forbidden option (existence: 11/175 correct, it picks an absent object instead).
- So v1 measures obedience, not comprehension. Keep v1 as data (it shows the instruction is applied), do not use it as H3-B.
- Proposal COMP v2 (needs Vy OK; code after Stage 3 ends, ~1-2 GPU h for 5 models): the instruction is quoted as
  content, delivered first, and the question asks about it:
  T: [image] "Here is an instruction:" + instruction text item + "Which item does this instruction say must not be mentioned? (A)..(B)..(C).. Answer with the letter only."
  A: same with the audio item; V: the instruction image + "Which item does the instruction written in the image say must not be mentioned? ...".
  Plus a verbatim check per channel: repeat the instruction word for word (T trivial copy, A = ASR_A, V = OCR_V exists).
