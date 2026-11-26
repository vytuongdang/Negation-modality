# pilot1000 (B200): H3-B comprehension v1 + visual controls, 5 models (2026-11-26)

Runs: run_b200_pilot.sh comp (COMP_T/A/V) and ctrl (C0_box, OCR_V), GPU 2, both EXIT 0 (comp 05:43-06:28, ctrl -08:10 UTC).
C0/T/A/V rows for the same 1,000 sample_ids are taken from H1 full (results/<m>/scored_all.jsonl ->
scored_pilot1000.jsonl, derived, git-ignored). Full tables: analysis_comp_pilot1000.md, analysis_pilot1000_asrok.md.

## H3-B v1: NC = forbidden item picked (3 options, chance 33%), n=1000, 95% image-bootstrap CI

| model | T | A | V |
|---|---|---|---|
| qwen2.5-omni-7b | 21.2 [18.7, 23.8] | 48.9 [45.9, 52.0] | 98.7 [98.0, 99.4] |
| qwen2.5-omni-3b | 61.0 [58.0, 64.0] | 75.4 [72.8, 78.0] | 99.4 [98.9, 99.8] |
| minicpm-o-2.6 | 8.8 [7.1, 10.6] | 42.5 [39.4, 45.6] | 97.4 [96.4, 98.3] |
| phi-4-mm | 54.5 [51.3, 57.6] | 68.9 [65.9, 71.7] | 99.9 [99.7, 100.0] |
| internomni | 54.6 [51.7, 57.8] | 33.8 [30.8, 36.7] | 95.1 [93.7, 96.4] |

NOT a clean comprehension measure: the MCQ ("which one must not be mentioned?") comes after an active "do not mention X";
picking X = mentioning it, so a model that obeys avoids the right option (Qwen-7B T 21%, MiniCPM T 9%: below chance).
Low T/A NC can mean obedience, not misunderstanding (RESEARCH.md Lessons 2026-11-24). v2 design (run_comp2.py: instruction
quoted first, then MCQ) is the valid one; not run on B200. V NC 95-100%: the image instruction is read and identified.

## Controls

OCR_V (model asked to copy the text written in the image), n=1000: exact copy qwen7b 99.8%, qwen3b 93.6%, minicpm 99.7%,
phi 98.1%, intern 99.0%.

C0 vs C0_box (empty box, no text; target stated without instruction): qwen7b 84.4 vs 83.9 (McNemar p=0.60), qwen3b 84.0 vs
81.7 (p=0.007), minicpm 81.9 vs 83.0 (p=0.13), phi 81.2 vs 80.2 (p=0.28), intern 82.6 vs 82.6 (p=1). The box alone hardly
changes what is said.

Violation (any mention) among C0-mentioned, V (instruction written in the box) minus C0_box (empty box), paired, asr_ok, 95%
image-bootstrap CI (quick script; n slightly larger than analyze.py's because of how primaries are chosen):
qwen7b -6.0 [-8.0, -4.0] (n=846); qwen3b -2.1 [-3.9, -0.4]; minicpm -3.2 [-4.9, -1.6]; phi -5.2 [-7.4, -3.1];
intern -0.7 [-2.2, +0.8].

Reading: every model reads the written instruction (OCR 94-100%, V NC 95-100%), but writing it in the image lowers mention of
the target by only 0-6 pp relative to an empty box. V's ~90% violation is therefore not a reading failure: the written text
is mostly not acted on as an instruction. This supports treating V as "in-image text as instruction", separate from the T/A
negation comparison.
