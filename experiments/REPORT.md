# Experiments report (Pico)

## H1 trial on dataset v2, stage-1 subset (2026-11-20; analysed 2026-11-23)

Setup: 1,000 questions (200 per branch: existence, existence_closed, count, attribute, relation), one negative
sample per question, audio with asr_ok only. Conditions: C0 (no instruction), T (instruction as its own text block
after the question), A (spoken instruction after the question), V (instruction written in the image). Greedy,
max_new_tokens 100. Scoring: Qwen3-8B judge (closed set), regex second opinion, thinking pass on disagreements.
Violation = target stated, among questions where the model's own C0 answer states it (n ≈ 800 per model).
CIs: 95% percentile bootstrap over images (2,000 resamples); McNemar exact on paired outcomes.
Full tables: h1_v2/analysis_pilot200.md; figures: figures/h1v2_violation_pilot200.png.

| model | C0 states target | viol. T | viol. A | viol. V | T−A (pp) [95% CI] | McNemar T vs A |
|---|---|---|---|---|---|---|
| Qwen2.5-Omni-7B | 80.3% | 9.7 | 6.0 | 82.2 | +3.7 [+1.9, +5.7] | p=3e-4 |
| Qwen2.5-Omni-3B | 81.0% | 54.0 | 44.8 | 86.2 | +9.1 [+6.3, +12.2] | p=5e-9 |
| MiniCPM-o 2.6 | 81.4% | 46.8 | 20.4 | 93.2 | +26.4 [+22.7, +30.2] | p=4e-39 |
| Phi-4-MM | 80.1% | 22.8 | 44.6 | 86.5 | −21.7 [−25.1, −18.2] | p=6e-30 |

What the trial supports (subset only, 4 models):
- The violation rate differs by channel within every model (paired tests, CIs exclude 0).
- T vs A: the sign is model-dependent (A lower for both Qwen models and MiniCPM-o; T lower for Phi-4-MM), so
  there is no single channel ordering across models.
- V: 82-93% violation in all models, far above T and A.

What it does NOT show yet:
- Whether channel gaps are negation-specific (needs affirmative/exceptive forms, H4).
- Why V is so high: models may not read the written instruction at all (OCR/attention) rather than read and
  ignore it. Needs a comprehension probe (ideas.md I2) and the C0_box control (empty box, same pixels).
- val vs train: differences exist but are confounded with source (CLEVR is all val), not interpretable here.
- Required recall: mostly unchanged; exception Phi-4-MM count branch (recall 80% → 22% under T, 34% under A):
  over-suppression, it drops the other counts too.
- Occlusion (≈40 flagged questions/model) and voice show no large differences (small n).
