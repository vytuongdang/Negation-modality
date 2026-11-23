# Experiments report (Pico)

## H1 pilot v2 on dataset v2 subset (2026-11-24, MAIN pilot numbers)

Setup: 932 questions (79 overlap units excluded), C0/T/A/V, T = instruction as its own text item (ITEM_SEP),
cut answers regenerated with 256 tokens, judge v3 (reads the question) + count rule + label pass.
Violation = any mention of the forbidden thing (STATE/META/DENY/WRONG), among questions where C0 states the target.
Full tables: experiments/h1_v2/analysis_pilot200.md; figure experiments/figures/h1v2_violation_pilot200.png.

| model | n C0-mentioned | T | A | V | T-A [95% CI] | A-V [95% CI] |
|---|---|---|---|---|---|---|
| Qwen2.5-Omni-7B | 790 | 17.2 | 13.0 | 88.9 | +4.2 [+1.6, +6.6] | -75.8 [-78.9, -72.8] |
| Qwen2.5-Omni-3B | 799 | 63.0 | 51.1 | 91.7 | +11.9 [+8.5, +15.5] | -40.7 [-44.4, -36.8] |
| MiniCPM-o 2.6 | 786 | 51.1 | 31.2 | 95.2 | +20.0 [+16.2, +23.8] | -64.0 [-67.6, -60.3] |
| Phi-4-MM | 772 | 26.0 | 40.9 | 91.3 | -14.9 [-19.2, -10.8] | -50.4 [-54.4, -46.6] |

What this supports (pilot, one subset, one dataset): the visual (typographic) instruction is violated far more
than text or audio in all 4 models (V mostly STATE: the model answers the question and ignores the written
instruction; whether it reads the text at all is NOT yet known -> OCR_V / COMP_V controls). The T-vs-A difference
is model-dependent in sign: 3 models violate more in T, Phi-4 more in A. So there is no general "audio is worse"
or "text is worse" result. Direction of T-A and the sign flip for Phi-4 match pilot v1 (old setup/metric).
Not supported yet: any negation-specific claim (needs H4 affirmative/exceptive forms and comprehension controls).
Phi-4 over-suppresses in count (required recall 74% at C0 -> 7% T, 17% A).
Judge v3 vs Vy's labels on the 49 labelled answers that were not regenerated: kappa 0.83, 93.9% agree
(3 disagreements: a WRONG count "2 spheres" scored compliant, "table" for "dining table" in C0, one false positive).

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
