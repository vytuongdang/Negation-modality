# Number audit: paper/draft_b200_v1.md (2026-11-26)

Read-only audit. Sources = files named in the draft's `[src: ...]` tags (FULL = full_analysis_5models_asrok.md,
FULL-ALL = full_analysis_5models_allunits.md, H4 = pilot1000_forms_5models_asrok.md, CTRL = pilot1000_comp_ctrl_5models.md
+ analysis_comp_pilot1000.md + analysis_pilot1000_asrok.md, H2 = analysis_h2_pilot1000_asrok.md, AUD = audit/*.md).
Model-to-row mapping was checked per model section (`## <model>`) in every file. Rounding: a draft value is OK if it is
the source value rounded (e.g. McNemar 4.65e-10 -> "5e-10").

Status: OK / MISMATCH / NOT FOUND / NOTE (number right but wording or n slightly off).

## 1. Claim table

| # | draft location | claim | source value | status |
|---|---|---|---|---|
| 1 | Abstract | 4,338 visual questions | dataset_stats: asr_ok units 4338; FULL: 4338 primary per model | OK |
| 2 | Abstract | 2,627 images | dataset_stats: asr_ok images 2627 | OK |
| 3 | Abstract | four conditions (C0/T/A/V) | FULL condition set | OK |
| 4 | Abstract | five models | 5 sections in FULL | OK |
| 5 | Abstract | T−A gap −31.6 to +17.9 | FULL: min InternOmni −31.6, max MiniCPM +17.9 | OK |
| 6 | Abstract | V violated 87–95% | FULL V ALL: 89.0/90.6/93.9/87.2/94.7 (87.2–94.7) | OK |
| 7 | Abstract | 1,000-question pilot | pilot1000_summary: 1000 | OK (analyses use 987 units, see NOTE 33) |
| 8 | Abstract | "three models" negation-specific | H4 aff−neg A−T CI excl. 0: MiniCPM, Phi, Intern | OK |
| 9 | Abstract note | ~185 words | abstract body = 208 words (wc) | MISMATCH (editorial) |
| 10 | Fig.1 | three channels / four ways | design | OK |
| 11 | Intro ¶ | 4,338 questions / 2,627 images | dataset_stats | OK |
| 12 | Intro ¶ | 4 sources (COCO, CLEVR, VAW, GQA), 5 branches | dataset_stats | OK |
| 13 | Intro ¶ | seven templates | dataset_stats template_id N1–N7 | OK |
| 14 | Finding 1 | Qwen-7B +3.8 [2.6, 5.0] | FULL qwen7b T vs A +3.8 [+2.6, +5.0] | OK |
| 15 | Finding 1 | -3B +15.1 | FULL qwen3b +15.1 | OK |
| 16 | Finding 1 | MiniCPM-o +17.9 | FULL minicpm +17.9 | OK |
| 17 | Finding 1 | Phi-4 −9.8 | FULL phi −9.8 | OK |
| 18 | Finding 1 | InternOmni −31.6 [−33.7, −29.7] | FULL intern −31.6 [−33.7, −29.7] | OK |
| 19 | Finding 2 | V violated 87–95% every model | FULL 87.2–94.7 | OK |
| 20 | Finding 2 | OCR exact copy 94–100% | CTRL 93.6–99.8 | OK (rounded) |
| 21 | Finding 2 | empty box: lowers mentions 0.7–6.0 points | CTRL V−C0_box −0.7 … −6.0 | OK (re-derived, Task 2) |
| 22 | Finding 3 | affirmative in image ignored 86–95% | H4 affirmative visual 86.4/92.5/93.1/91.0/95.0 | OK |
| 23 | Finding 3 | Δ_V near zero in every model | H4 aff−neg visual +4.3 [+1.3,+7.3] (Qwen-7B), others within ±1.9 | NOTE (Qwen-7B CI excludes 0) |
| 24 | Finding 4 | MiniCPM-o +11.9 | H4 minicpm aff−neg A−T +11.9 [+6.6,+17.6] | OK |
| 25 | Finding 4 | Phi-4 +11.9 | H4 phi +11.9 [+7.0,+17.0] | OK |
| 26 | Finding 4 | InternOmni +36.3 | H4 intern +36.3 [+30.9,+41.8] | OK |
| 27 | Finding 4 | Qwen CIs include 0 | H4 qwen7b [−8.2,+0.9], qwen3b [−9.0,+0.6] | OK |
| 28 | Finding 4 | InternOmni exc +0.4 [−4.8, 5.5] | H4 intern exc−neg A−T +0.4 [−4.8,+5.5] | OK |
| 29 | Finding 4 | Qwen-3B opposite sign (exc) | H4 qwen3b exc −19.3 | OK |
| 30 | Finding 5 | InternOmni recall T 73%, A 25%, C0 71% | FULL intern recall ALL 71.2/73.3/25.3 | OK |
| 31 | §2.1 | seven templates; N6, N7 without negation word | design (templates in FULL template tables N1–N7) | OK |
| 32 | §2.2 step 2 | 2,799 images / 4,675 units | pico/NOTES_b200.md l.355/364/388 (draft has no src tag here) | OK (src: notes) |
| 33 | §2.2 step 3 | 4,390 units reach H1 | dataset_stats 4390; FULL-ALL 4390 | OK |
| 34 | §2.2 step 4 | 276 excluded units | FULL header 276; NOTES_b200 l.241 | OK |
| 35 | §2.2 step 4 | 3 images removed by hand (too dark) | removed_images.jsonl: 3 rows; 2 "too dark", 1 "wrong ground truth (VAW colour)" | MISMATCH (reason) |
| 36 | §2.2 step 5 | female 2,339 / male 1,999 | dataset_stats voice | OK (sum 4338) |
| 37 | §2.2 step 5 | 52 units fail ASR check | FULL "52 units left out"; 4390−4338 | OK |
| 38 | §2.2 step 6 | 172 units (4.0%) occludes_subject | dataset_stats True 172; 172/4338 = 3.96% | OK |
| 39 | §2.2 Result | 4,338 / 2,627 | dataset_stats | OK |
| 40 | §2.2 Result | attribute 1,518 | dataset_stats | OK |
| 41 | §2.2 Result | count 899 | dataset_stats | OK |
| 42 | §2.2 Result | existence 689 | dataset_stats | OK |
| 43 | §2.2 Result | closed existence 612 | dataset_stats | OK |
| 44 | §2.2 Result | relation 620 | dataset_stats | OK (branch sum 4338) |
| 45 | §2.2 Result | COCO 1,685 | dataset_stats | OK |
| 46 | §2.2 Result | CLEVR 1,250 | dataset_stats | OK |
| 47 | §2.2 Result | VAW 783 | dataset_stats | OK |
| 48 | §2.2 Result | GQA 620 | dataset_stats | OK (source sum 4338) |
| 49 | §2.2 Arithmetic | 4,390 × 4 = 17,560 per model | judge_audit "all" rows 17560 each model | OK |
| 50 | §2.2 Arithmetic | 17,352 with asr_ok | 4,338 × 4 = 17,352 (derived; not printed in judge_audit) | OK (derived) |
| 51 | §2.2 Pilot | 1,000 questions, 200 per branch | pilot1000_summary | OK |
| 52 | §2.2 Pilot | voices 500/500 | pilot1000_summary | OK |
| 53 | §2.2 Pilot | all asr_ok | pilot1000_summary says asr_ok; but H4/CTRL analyses drop 13 units by unit-level asr_ok (forms audio) -> 987 | NOTE |
| 54 | §2.2 Pilot | seed 42, each image once | pilot1000_summary | OK |
| 55 | §3 Models | 100 / 256 new tokens | regen_audit header | OK |
| 56 | §3 Models | regenerated 0.3–7.3%, MiniCPM-o most | regen_audit 0.4/0.3/7.3/1.1/0.7 | OK |
| 57 | §3 Violation | C0 names target 79–82% | FULL 79.3–82.0 | OK |
| 58 | §3 Scoring | think_error 0.08–0.83% per model | judge_audit run "all": 0.64/0.83/0.24/0.19/0.08 | OK |
| 59 | §3 Scoring | human agreement 49 answers, 93.9%, κ = 0.83 | pico/NOTES.md l.144 (no src tag in draft) | OK (src: notes) |
| 60 | §3 Statistics | 2,000 bootstrap resamples | FULL "image bootstrap"; Table 2 caption | OK |
| 61 | §4.1 | C0 names target 79.3–82.0% | FULL | OK |
| 62 | §4.1 | OCR exact 99.8 / 93.6 / 99.7 / 98.1 / 99.0 | CTRL qwen7b/qwen3b/minicpm/phi/intern same | OK |
| 63 | §4.1 | C0 vs C0_box differ by at most 2.3 | CTRL diffs 0.5/2.3/1.1/1.0/0.0 | OK |
| 64 | §4.1 | Qwen-3B 84.0 vs 81.7, p = 0.007 | CTRL p=0.007 (analysis_pilot1000_asrok p=0.00674) | OK |
| 65 | §4.1 | only Qwen-3B significant | CTRL p 0.60/0.007/0.13/0.28/1 | OK |
| 66 | §4.1 table | C0 full 81.5/82.0/79.3/80.2/80.3 | FULL per model | OK |
| 67 | §4.1 table | OCR column | CTRL | OK |
| 68 | §4.1 table | Qwen-7B 84.4→83.9 | CTRL | OK |
| 69 | §4.1 table | Qwen-3B 84.0→81.7 | CTRL | OK |
| 70 | §4.1 table | MiniCPM 81.9→83.0 | CTRL | OK |
| 71 | §4.1 table | Phi 81.2→80.2 | CTRL | OK |
| 72 | §4.1 table | Intern 82.6→82.6 | CTRL | OK |
| 73 | §4.1 caption | n = 1,000 (pilot) | OCR n=1000, but C0/C0_box rates are over n=987 (analysis_pilot1000_asrok) | MISMATCH (minor) |
| 74 | §4.2 | V 87.2–94.7% | FULL | OK |
| 75 | §4.2 | Qwen-7B 6.0 [4.0, 8.0] | CTRL −6.0 [−8.0, −4.0] (sign flipped for "lowers") | OK |
| 76 | §4.2 | Qwen-3B 2.1 [0.4, 3.9] | CTRL −2.1 [−3.9, −0.4] | OK |
| 77 | §4.2 | MiniCPM 3.2 [1.6, 4.9] | CTRL −3.2 [−4.9, −1.6] | OK |
| 78 | §4.2 | Phi 5.2 [3.1, 7.4] | CTRL −5.2 [−7.4, −3.1] | OK |
| 79 | §4.2 | Intern 0.7 [−0.8, 2.2] | CTRL −0.7 [−2.2, +0.8] | OK |
| 80 | §4.2 | V STATE 80.4–93.3% | FULL V STATE 80.4/82.3/91.1/85.6/93.3 | OK |
| 81 | §4.2 | META up to 6.2% in Qwen | FULL V META qwen7b 6.2, qwen3b 6.1 (others ≤1.1) | OK |
| 82 | §4.2 | Qwen-7B overlap 91.9 vs 88.9 | FULL qwen7b True 91.9 / False 88.9 | OK |
| 83 | §4.2 | Qwen-3B 94.7 vs 90.4 | FULL qwen3b | OK |
| 84 | §4.2 | MiniCPM 92.4 vs 94.0 | FULL minicpm | OK |
| 85 | §4.2 | Phi 88.8 vs 87.1 | FULL phi | OK |
| 86 | §4.2 | Intern 93.1 vs 94.8 | FULL intern | OK |
| 87 | §4.2 | 130–135 overlapping per model | FULL occludes True n 135/132/132/134/130 | OK |
| 88 | §4.3 | CI excludes 0 in all five | FULL T vs A CIs | OK |
| 89 | §4.3 | all units ≤ 0.2 pts from asr_ok | T−A all-units vs asr_ok: qwen7b +3.8/+3.8, qwen3b +15.0/+15.1, minicpm +18.0/+17.9, phi −9.6/−9.8, intern −31.6/−31.6 (max 0.2) | OK |
| 90 | Table 2 | Qwen-7B n 3,535, 17.3/13.5/89.0, +3.8 [+2.6,+5.0], 5e-10, recall 80.4/84.1/83.8/79.5 | FULL qwen7b: same; p 4.65e-10 | OK |
| 91 | Table 2 | Qwen-3B n 3,558, 65.4/50.3/90.6, +15.1 [+13.5,+16.8], 5e-73, 81.4/81.8/82.7/78.4 | FULL qwen3b: same; p 4.66e-73 | OK |
| 92 | Table 2 | MiniCPM n 3,441, 48.1/30.1/93.9, +17.9 [+16.2,+19.7], 3e-83, 80.3/84.6/84.1/79.1 | FULL minicpm: same; p 2.85e-83 | OK |
| 93 | Table 2 | Phi n 3,478, 27.6/37.4/87.2, −9.8 [−11.9,−7.7], 3e-22, 80.5/63.8/68.1/78.0 | FULL phi: same; p 3.14e-22 | OK |
| 94 | Table 2 | Intern n 3,483, 49.2/80.8/94.7, −31.6 [−33.7,−29.7], 4e-183, 71.2/73.3/25.3/72.7 | FULL intern: same; p 4.01e-183 | OK |
| 95 | Table 2 caption | bootstrap 2,000 | analyze.py convention | OK |
| 96 | §4.4 | aff n = 647–669 | H4 n 669/664/649/647/663 | OK |
| 97 | §4.4 | exc n = 801–833 | H4 n 833/829/808/801/815 | OK |
| 98 | §4.4 | three models CI excludes 0; Qwen include 0 | H4 | OK |
| 99 | §4.4 | in V, Δ within ±4.3 | H4 visual Δ: aff +4.3/−1.1/+0.6/−1.9/+0.5; exc +2.8/−0.7/−1.1/+0.9/−0.7 | OK |
| 100 | Table 3 | Qwen-7B −19.6 / −23.2 / −3.6 [−8.2,+0.9] / −0.1 [−4.1,+4.2] | H4 qwen7b same | OK |
| 101 | Table 3 | Qwen-3B +19.1 / +14.9 / −4.2 [−9.0,+0.6] / −19.3 [−24.1,−14.7] | H4 qwen3b same | OK |
| 102 | Table 3 | MiniCPM −8.9 / +2.9 / +11.9 [+6.6,+17.6] / +5.7 [+0.6,+10.9] | H4 minicpm same | OK |
| 103 | Table 3 | Phi +9.3 / +21.2 / +11.9 [+7.0,+17.0] / +8.1 [+3.2,+13.1] | H4 phi same | OK |
| 104 | Table 3 | Intern +19.8 / +56.1 / +36.3 [+30.9,+41.8] / +0.4 [−4.8,+5.5] | H4 intern same | OK |
| 105 | §4.4 caveat | Qwen-7B Δ_T −19.6; exceptive −44.9 | H4 qwen7b aff−neg T −19.6; exc−neg T −44.9 (A −45.0) | OK |
| 106 | §4.4 caveat | exceptive violated ~60% (Qwen-7B) | H4 qwen7b exceptive T 62.4, A 58.3 | OK |
| 107 | §4.5 | 200 open-existence questions skipped in H3-A | pilot1000_summary existence 200 (no affirmative) | OK |
| 108 | §4.5 | 6 layouts, 3 pairs | H2 file: 6 conditions per family | OK |
| 109 | §4.5 | qT_iA 13.2 vs qA_iT 17.5, −4.3 [−7.1, −1.6] | H2 qwen7b same | OK |
| 110 | §4.5 | H3-A direction +2.7 [−0.9, 6.3] | H2 same | OK |
| 111 | §4.5 | H2 − H3-A T–A −7.9 [−12.9, −3.1] | H2 same | OK |
| 112 | §4.5 | in-image instruction violated 86–90% | H2 qT_iV 90.1, qA_iV 90.0; H3A 86.4, 88.3 | OK |
| 113 | §4.5 | in-image question: instruction obeyed (17–20%) | H2 qV_iT 19.7, qV_iA 17.4; H3A 19.5, 18.7 | OK |
| 114 | §4.5 | T–V, A–V H2 − H3-A ≈ 0 | H2 +0.7 [−3.9,5.8], +1.2 [−3.7,6.1] | OK |
| 115 | §5.1 | DENY next most common kind | FULL: true for most cells, but META > DENY in V for Qwen-7B (6.2 vs 2.4), Qwen-3B (6.1 vs 2.1), Phi (1.1 vs 0.6), and Phi T (0.2 vs 0.1) | MISMATCH (wording) |
| 116 | §5.1 | DENY largest for T in closed existence (InternOmni 10.9%) | FULL intern exist_closed T DENY 10.9 (= InternOmni's max). But other models are higher: MiniCPM 34.2, Qwen-7B 25.3, Qwen-3B 23.2 | NOTE (value OK; "largest" only within InternOmni) |
| 117 | §5.2 | Intern recall 71.2 → 25.3; T/V 73.3 / 72.7 | FULL intern | OK |
| 118 | §5.2 | count 10.1% | FULL intern count A recall 10.1 | OK |
| 119 | §5.2 | Phi T 63.8, A 68.1, C0 80.5 | FULL phi | OK |
| 120 | §5.2 | other three: T, A ≥ C0 | FULL qwen7b 84.1/83.8 ≥ 80.4; qwen3b 81.8/82.7 ≥ 81.4; minicpm 84.6/84.1 ≥ 80.3 | OK |
| 121 | §5.3 | N3 most violated typed template, InternOmni 67.9% | FULL intern template T: N3 67.9 (max) | OK |
| 122 | §5.3 | voice diff ≤ 6.5 (Intern 83.8 vs 77.3) | FULL voice A: diffs 1.9/4.7/0.3/0.2/6.5 | OK |
| 123 | §5.3 | MiniCPM A 37.1 (train) vs 21.0 (val) | FULL minicpm split A | OK |
| 124 | §5.3 | Phi V 91.9 vs 81.6 | FULL phi split V | OK |
| 125 | §5.4 | Qwen-7B T NC 21.2% | CTRL / analysis_comp qwen7b T 21.2 | OK |
| 126 | §5.4 | MiniCPM T NC 8.8% | CTRL minicpm T 8.8 | OK |
| 127 | §5.4 | chance 33% | CTRL | OK |
| 128 | §5.4 | V NC 95–100% | CTRL V 98.7/99.4/97.4/99.9/95.1 | OK |
| 129 | Limitations | two voices, 1,000-question pilot | dataset_stats / pilot summary | OK |

Claims checked: 129 (rows; several rows bundle a full table row of 4–8 numbers).

## 2. Mismatches / notes with suggested text

1. **Abstract word note (#9).** Abstract body is 208 words, not ~185. Suggest: "(~210 words: trim to 170 at the end.)"
2. **§2.2 step 4 (#35).** removed_images.jsonl has 3 images: 2 too dark, 1 wrong ground truth. Suggest: "and 3 images
   removed by hand (two too dark to verify, one with a wrong VAW label)."
3. **§4.1 caption (#73).** C0 / C0_box rates are on the 987 pilot units that pass unit-level asr_ok (analysis_pilot1000_asrok.md
   "n=987"); OCR_V is on 1,000. Suggest: "n = 4,338 (full); pilot: 1,000 for OCR_V, 987 for C0 vs C0_box."
4. **§2.2 pilot (#53), related.** "all asr_ok" (per pilot1000_summary) holds for the negative-form audio, but analyze.py drops 13
   pilot units whose form (affirmative/exceptive) audio failed ASR, so H4/CTRL use 987 units (H2: 985). Suggest adding
   "(987 after also requiring the affirmative/exceptive audio to pass the ASR check)".
5. **Finding 3 (#23).** "Δ_V is near zero in every model": Qwen-7B Δ_V = +4.3 [+1.3, +7.3] (CI excludes 0). Suggest
   "Δ_V is within ±4.3 points in every model".
6. **§5.1 (#115).** META, not DENY, is the second kind under V for both Qwen models (6.2 / 6.1 vs DENY 2.4 / 2.1) and Phi.
   Suggest: "DENY or META is the next most common kind (META under V for the Qwen models)".
7. **§5.1 (#116).** InternOmni 10.9% is not the largest DENY value; MiniCPM-o reaches 34.2% (T, closed existence), Qwen-7B
   25.3%, Qwen-3B 23.2%. Suggest: "and is largest for the typed prohibition in closed existence (MiniCPM-o 34.2%,
   Qwen-7B 25.3%)".
8. **§4.2 V − C0_box CIs (#75–79).** Re-derived (Task 2): point estimates and n reproduce exactly; CIs differ by ≤ 0.2 from
   the quick script (bootstrap RNG). Draft CIs are acceptable; if the draft cites the re-derived script, use: Qwen-7B 6.0
   [4.0, 8.0], Qwen-3B 2.1 [0.4, 3.9], MiniCPM-o 3.2 [1.7, 4.8], Phi-4 5.2 [3.0, 7.5], InternOmni 0.7 [−0.7, 2.2].
   The "[src: CTRL; quick script, to be re-derived]" tag can point to audit/rederive_v_minus_c0box.py.

No numeric mismatch was found in Table 2, Table 3, the §4.1 table, the findings list, or the H2 preview.

## 3. Task 2: V minus C0_box re-derivation

Script: experiments/h1_v2/audit/rederive_v_minus_c0box.py (pure python3, `python3 -I`). Inputs:
results/<model>/scored_pilot1000.jsonl (C0, V) and scored_pilot1000_C0_box-OCR_V.jsonl (C0_box). Violation fields are JSON
booleans in these files (script accepts bool or string). Paired by C0_box `pair_of` (= its sample_id in every row).
Duplicates: each file has exactly one row per unit and condition (1,000 each; assert on V), so no template-sweep
handling was needed. C0 rows in the C0_box file are identical (verdict and answer) to the C0 rows in scored_pilot1000.jsonl.
Every pilot row has asr_ok = True, so "asr_ok" filtering by row keeps all 1,000 units ("row"). The analyze.py unit set
additionally drops 13 units whose affirmative/exceptive audio failed ASR (from scored_pilot1000_forms.jsonl) -> "987".
Bootstrap: images resampled with replacement (here 1 unit per image), 2,000 resamples, `random.Random(0)`, 2.5/97.5 percentiles.

Difference = mean(V violation) − mean(C0_box violation), pp, among C0-mentioned units.

| model | reported (CTRL) | re-derived, C0-mentioned, 1,000-unit set | n | same, 987-unit set | n | C0_box-mentioned, 1,000 set | n | C0_box-mentioned, 987 set | n |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-omni-7b | −6.0 [−8.0, −4.0] n=846 | −6.0 [−8.0, −4.0] | 846 | −6.0 [−8.2, −4.1] | 833 | −8.9 [−10.8, −7.0] | 839 | −8.8 [−10.7, −7.0] | 828 |
| qwen2.5-omni-3b | −2.1 [−3.9, −0.4] | −2.1 [−3.9, −0.4] | 842 | −2.2 [−4.0, −0.4] | 829 | −5.9 [−7.6, −4.3] | 816 | −6.0 [−7.7, −4.3] | 806 |
| minicpm-o-2.6 | −3.2 [−4.9, −1.6] | −3.2 [−4.8, −1.7] | 819 | −3.2 [−4.8, −1.7] | 808 | −4.8 [−6.3, −3.5] | 830 | −4.9 [−6.5, −3.5] | 819 |
| phi-4-mm | −5.2 [−7.4, −3.1] | −5.2 [−7.5, −3.0] | 812 | −5.4 [−7.6, −3.2] | 801 | −9.1 [−11.0, −7.1] | 802 | −9.2 [−11.4, −7.2] | 792 |
| internomni | −0.7 [−2.2, +0.8] | −0.7 [−2.2, +0.7] | 827 | −0.7 [−2.2, +0.9] | 815 | −3.6 [−5.0, −2.3] | 828 | −3.7 [−4.9, −2.5] | 815 |

Reading:
- The reported values reproduce: same point estimates and n=846 for Qwen-7B with the 1,000-unit set (which explains
  "n slightly larger than analyze.py's": analyze.py uses the 987-unit set, whose n equals its C0-mentioned n
  833/829/808/801/815). CI endpoints differ by ≤ 0.2 pp (bootstrap RNG). Using the 987 set changes estimates by ≤ 0.2 pp.
- Conditioning on C0_box instead gives larger effects (−3.6 to −9.2, all CIs exclude 0, InternOmni included). This version
  selects on the subtracted variable (C0_box violation = 1 for every kept unit), so it is biased toward a negative
  difference by regression to the mean; the C0-conditioned version (draft's definition) is the fair one. The draft's
  "0.7–6.0 points" and "InternOmni CI includes 0" depend on this choice and should name the conditioning explicitly.
