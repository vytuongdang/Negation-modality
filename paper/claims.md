# Candidate claims → evidence → status (updated by Pico; story follows the evidence)

| # | claim | evidence needed | status |
|---|---|---|---|
| C1 | Omni-models' compliance with "do not mention X" differs by channel (text / audio / in-image). | H1 v2: violation among C0-mentioned, T_sep vs A vs V, McNemar + image bootstrap, per model | partly (pilot v2 2026-11-24, 932 q, 4 models, judge v3): V violated 89-95% in all models vs T 17-63%, A 13-51%; T-A sign model-dependent (+4.2/+11.9/+20.0 Qwen7B/Qwen3B/MiniCPM, -14.9 Phi-4, -29.8 InternOmni, all CIs exclude 0; InternOmni A recall 27% = audio may be taken as the question). V needs OCR/comprehension control before any claim beyond "typographic instruction ignored". |
| C2 | How the text instruction is segmented (glued to the question vs its own block) changes compliance. | T_join vs T_sep paired; A vs T_sep | pending |
| C3 | The channel gap is negation-specific (polarity × channel interaction). | H4: Δ_audio − Δ_text (and visual) with CIs, affirmative units | pending |
| C4 | Naming the target (priming) rather than the word "not" drives violations. | exceptive vs negative per channel | pending |
| C5 | Under direct text/audio conflict, the channel that wins depends on polarity (not plain dominance). | H2 vs H3-A, both orders | pending |
| C6 | Effects are not over-suppression. | required recall per condition | partly: pilot v2 recall stable except Phi-4 count (74 -> 7% T, 17% A) |
| C7 | Results hold on val (non-train) images. | split table | pending |
| C8 | Benchmark contribution: component-level negation VQA with paired text/audio/visual instructions. | dataset v2 stats | pending (M1 running) |
