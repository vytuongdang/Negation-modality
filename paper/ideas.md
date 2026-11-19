# Ideas (Pico): idea → why → what each outcome would mean → cost → status
Pilot on ≤300 samples in a new versioned folder; scale only after Vy agrees.

## I1. Neutral-audio control (is it audio presence, or the spoken words?)  [proposed 2026-11-19]
- Conditions: T_sep + neutral spoken sentence ("Please look at the picture carefully."), and A_neutral_only
  (question + neutral audio, no instruction: does any audio change what is said?).
- Why: pilot v1 had audio negation obeyed better than text. If adding an irrelevant audio segment ALSO lowers
  violations of the text instruction, the "audio advantage" is a generic salience/attention effect of an audio
  segment, not better understanding of spoken negation.
- Outcomes: T_sep+neutral ≈ T_sep → audio presence irrelevant, the words matter; T_sep+neutral < T_sep → presence
  effect (big confound to report). A_neutral_only vs C0 shows if audio alone suppresses mentions.
- Cost: 2 extra conditions × ~200 samples × 2 models ≈ 0.5 GPU h + 2 TTS sentences.

## I2. Comprehension probe per channel (H3-B, cheap form)  [proposed 2026-11-19]
- After the same input (T_sep / A / V), ask instead: "Which thing were you asked not to mention? Answer in a few
  words." Score with the judge against the target. Gives P(understood) and P(obeyed | understood) per channel.
- Why: separates "did not hear/read it" (ASR/OCR failure, esp. V and A) from "understood but did not obey".
- Cost: 3 conditions × ~200 samples × 2 models ≈ 0.5 GPU h.

## I3. Target log-probability at the answer start (Qwen-Omni, open weights)  [proposed 2026-11-19]
- Teacher-force the model's own C0 answer prefix up to the first mention of the target; compare log p(target
  token) under C0 vs T_sep vs A vs V. A continuous, low-variance measure of suppression strength per channel.
- Why: binary violation is noisy and at ceiling for 7B (pilot 7-13%); log-prob shows graded suppression.
- Cost: forward passes only, ~200 samples × 4 conds, < 0.5 GPU h.

## I4. Instruction position (H3-C)  [from Idea doc, refine after H1]
- Instruction before the image vs after the question, text and audio, same samples.
