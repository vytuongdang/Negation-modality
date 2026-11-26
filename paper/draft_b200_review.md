# Review of paper/draft_b200_v1.md (Pico on B200, 2026-11-26)

Written as a critical co-author. Each point says what was checked, with which file, and what it means for the
paper. Change log at the end.

## (a) Method audit

### a1. Judge reliability [src: experiments/h1_v2/audit/judge_audit.md, audit_judge.py]
- think_error rows (the thinking judge returned no JSON, so the fast verdict is kept) are rare. H1 full: Qwen-7B 113
  (0.64%), Qwen-3B 145 (0.83%), MiniCPM-o 42 (0.24%), Phi-4 33 (0.19%), InternOmni 14 (0.08%). Forms pilot:
  21–57 per model (0.22–0.61%). Controls: 3–5. H2: Qwen-7B 65, Qwen-3B 38 (other models pending).
  NOTE: NOTES_b200 (2026-11-25) says "intern 33 think_error rows" for H1 full, but the scored file has 14. Phi has
  33. The note probably swapped the two models. The file counts are the ones to use.
- Judge vs regex disagreement on the target (any condition): 5.6–10.3% of rows in H1 full (InternOmni highest,
  10.3%), 5.3–8.6% in forms, 2.5–12.7% in controls. These rows are the ones that go to the thinking pass, so
  disagreement is expected. It shows how much rests on the judge rather than on string matching. InternOmni's
  higher rate matches its longer, less list-like answers to spoken instructions (conjecture).
- **Sensitivity without think_error units** (union over models: 290 units in H1 full, 129 in forms; analyze.py
  --remove): [PENDING: audit/run_sens.sh -> analysis_all_asrok_rm.md, analysis_pilot1000_forms_asrok_rm.md].
- Human agreement exists only for server 1's subset (49 answers, 93.9%, κ 0.83). **No human check on B200
  answers.** This is the largest method gap for a judge-scored paper. Recommendation: a blind 300-item sample
  stratified by model × channel (Vy), plus 100 overlapping items (Huy), before submission.
- Judge code: H1 full ran without be8ab2b (OOM fix in the 4096 retry); all pilot1000 runs have it. The fix touches
  the retry path only. To verify: re-score a sample of 200 H1 rows with the current judge and compare. Cheap; needs
  the judge on the GPU (~10 min).

### a2. asr_ok vs all units
Main analysis = asr_ok units (4,338). All units (4,390): T − A changes by ≤ 0.2 points for every model
[src: full_analysis_5models_allunits.md vs _asrok.md; number audit to confirm]. Only 52 units fail ASR, so the
filter cannot drive the result. The pilot list is asr_ok only by construction.

### a3. Excluded units and removed images
276 units are excluded by rule (double label, part–whole, attribute overlap; flag_excluded_units.py), and 3 images
by hand (too dark; removed_images.jsonl). All of this was decided **before** any result (NOTES 11-24) and applied
in every analysis. Also 4,675 units after image subsetting → 4,399 after exclusions → 4,390 reach the H1 run.
**Open:** 9 units are lost in B2/B3 with no recorded reason. Check the B3 log before quoting pipeline counts.

### a4. Pilot vs full
H4, controls, H3-B and H2/H3-A use the 1,000-question pilot (200 per branch). Per-branch n is 200 (existence_closed
has 7 val questions), so the draft makes **no per-branch or split claims from the pilot**. Checked: Sec. 5.3 split
numbers come from the full set only. The pilot is balanced, so its pooled numbers weight branches differently from
the full set. Example: Qwen-7B V is 89.0 (full) vs 90.2 (pilot negative); T is 17.3 vs 17.5. Never mix them in one
table.

### a5. One answer per unit + image bootstrap
One greedy answer per question and condition. CIs resample images (2,627 in full, 1,000 in pilot), which is right
because units share images in the full set. Template-sweep copies (425 units) are used only in template tables,
not in tests. Greedy decoding means the CIs cover item sampling, not sampling noise in the model. State this.

### a6. Truncation and regeneration [src: audit/regen_audit.md]
Answers regenerated at 256 tokens after hitting 100: Qwen-7B 156 (0.4%), Qwen-3B 100 (0.3%), MiniCPM-o 2,110
(7.3%), Phi-4 305 (1.1%), InternOmni 184 (0.7%) of all cached answers (all runs share the cache). Longer answers
can mention more, and MiniCPM-o's regenerated share is not small. By condition (audit/audit_regen_cond.py; tags pool
H1 and forms answers): MiniCPM-o C0 12.9%, T 6.7%, A 4.1%, V 12.4%, C0_box 10.6%. In the other models it is ≤ 2.8%
everywhere, with C0/V/C0_box highest. Reading: greedy decoding means the regenerated answer extends the cut one, so
regeneration only completes answers and never changes the first 100 tokens. It is a fix, not a bias. But answers
under C0 and V are longer, so they have more room to name the target. For MiniCPM-o T vs A (6.7 vs 4.1%) the
difference is too small to drive a 17.9-point gap. Report the per-condition rates in the appendix.

## (b) Number audit
Done independently (subagent, read-only), see experiments/h1_v2/audit/number_audit.md. 129 claim rows were checked. Tables
2, 3, §4.1, the findings list, the H2 preview and the dataset counts all match. Found and FIXED in the draft: abstract
word count (208, not ~185); removed images = 2 too dark + 1 wrong VAW label; pilot C0/C0_box n = 987, not 1,000
(13 units fail ASR on their affirmative/exceptive audio; H2 uses 985); "Δ_V near zero" → "within ±4.3" (Qwen-7B
Δ_V +4.3 [1.3, 7.3] excludes 0); §5.1 kind ranking was wrong (under V, META > DENY for the Qwen models and Phi-4;
DENY is large in closed existence under T: MiniCPM-o 34.2%, Qwen-7B 25.3%, Qwen-3B 23.2%).
V − C0_box re-derived from raw rows (audit/rederive_v_minus_c0box.py): every point estimate is identical, and the
CI ends differ by ≤ 0.2. The original script used 1,000 units, analyze.py 987: same values within 0.2. Conditioning
on C0_box instead of C0 makes every effect larger (−3.6 to −9.1; InternOmni's CI then excludes 0). The C0-conditioned
version is the fair one (it does not select on the value being subtracted), and the draft now says which
conditioning it uses.

New confound found by the audit (c11 below): DENY counts as a violation.

## (c) Confounds and how the draft handles them

1. **V at 87–95%, including the affirmative.** V is high in every form (affirmative 86–95%), OCR_V exact copy is
   94–100%, and V − C0_box is only 0.7–6 points. So V measures "is in-image text treated as an instruction",
   not negation by channel. The draft therefore keeps V out of the negation claims (Δ_V ≈ 0 is reported as
   uninformative). Kinds under V are mostly STATE (80–93%). The Qwen models show 6% META, i.e. they sometimes
   acknowledge the written instruction and still name the target. Good material for Fig./appendix examples.
2. **InternOmni treats audio as the question.** A recall is 25.3% vs C0 71.2%. Its A violation (80.8%) therefore
   partly reflects "answers the spoken sentence", which names X, rather than ignoring a prohibition. Consequence
   for H4: the affirmative audio instruction names Y, not X, so its Δ_A (+56.1) is inflated by this behaviour.
   The exceptive reference (names X) gives +0.4 [−4.8, 5.5]. The draft states this caveat next to Table 3. It must
   not report InternOmni's +36.3 as a negation effect.
3. **T − A sign flips.** The draft compares channels within a model only and states that no general direction is
   supported. Good. Do not average T − A across models.
4. **Over-suppression.** Recall is reported next to violation (Table 2). Phi-4 drops under T/A (63.8/68.1 vs 80.5)
   and InternOmni under A. For any method section later, recall must be a co-primary metric.
5. **C0 ≈ 80%.** Violation is conditioned on C0, and n per model is given. Residual concern: conditioning on C0
   selects the questions where each model names the target, so the sets differ slightly between models (3,441–3,558).
   Between-model comparisons are on different subsets. Within-model channel comparisons are on the same subset.
6. **Question always typed.** Every condition has a typed question, which may favour T. This is stated as a
   limitation. H2's swapped layouts (question spoken or written) partly address it.
7. **H2 design differs from the Idea document.** The run uses question-vs-instruction channel swaps (PLAN stage 4),
   not "Mention X typed + Do not mention X spoken". The draft says so in 4.5. Before submission, decide with Vy
   whether to (i) rename this experiment ("instruction channel vs question channel") and keep "H2 = direct conflict"
   for future work, or (ii) also run the Idea-doc version (needs new TTS for "Mention {X}."; ~1–2 GPU h per model on
   the pilot).
8. **H3-B v1 is confounded with obedience** (known lesson 2026-11-24). The draft uses only its V result (the
   written instruction is identified) and calls T/A NC uninterpretable.
9. **Split / source confounding.** Val images are mostly CLEVR/VAW/GQA, and train images mostly COCO. A train–val
   difference is therefore also a source difference. No contamination claim without a per-source split table.
10. **Torch 2.7.1 on B200 vs 2.6 on server 1.** Small numerical differences are possible. All draft numbers are
    from B200 only, so this does not matter as long as the two machines' results are not mixed.

11. **DENY counted as violation.** In closed existence under T, a large part of the violations are "there is no X"
    (MiniCPM-o: 34.2 of its 85.9 violation points there; FULL kind table). For a closed question, saying an option is absent
    still names it, and it is often false (the VLM check says X is present), so counting it is defensible. But it
    should be shown as its own column, and the headline T − A should be repeated with leak (= STATE only). The leak
    column exists in FULL (paired kind table): T − A by leak is +2.9 (Qwen-7B), +14.8 (Qwen-3B), +18.1 (MiniCPM-o),
    −8.0 (Phi-4), −32.7 (InternOmni). Every sign and CI matches the any-mention metric, so S1 does not depend on
    counting DENY.

## (d) What the data support, what is weak, what to run next

**Supported (full set, CIs exclude 0, robust to all-units):**
- S1. Typed and spoken prohibitions are obeyed at different rates within every model tested, with a
  model-dependent sign.
- S2. Prohibitions written in the image are violated in ~9/10 cases although they are read (OCR, empty-box
  control); affirmative written instructions behave the same → not negation-specific.
- S3. InternOmni's recall drops under spoken instructions (audio-as-question), so violation must be read with
  recall.

**Weak / pilot only:**
- W1. Negation-specific spoken–typed penalty (H4): CIs exclude 0 for 3 models with the affirmative reference, but
  the exceptive reference shrinks or reverses it. Pilot n ≈ 650–830. Do not headline it.
- W2. H2/H3-A: one model so far. H2 − H3-A for T–A on Qwen-7B is −7.9 [−12.9, −3.1], i.e. the direction effect
  differs with vs without negation. This is interesting, but it is one model, the pilot, and an operationalisation
  that differs from the Idea doc.
- W3. Anything per branch, split, template or voice from the pilot.

**Ranked next experiments (ask Vy; costs on B200 GPU 2, generation + judge):**
1. Human check of the judge on B200 answers (300 blind + 100 overlap). No GPU, ~3–4 h of human time. Highest
   priority: every number depends on the judge.
2. H4 full extension for Qwen-7B/3B (Vy's rule: Qwen CIs cross 0). About 2 forms × 3 channels × ~3.3k extra
   questions ≈ 20k answers per model: ~3 h gen + ~4 h judge per model ≈ 14 GPU-h for both. Could drop V (ceiling)
   → ~9 GPU-h.
3. NEU control (neutral spoken sentence next to the typed prohibition, and neutral audio alone): tells "any audio
   segment changes behaviour" apart from "the spoken words". Pilot, 5 models: 2 conds × 1,000 ≈ 10k answers ≈ 4–5
   GPU-h + 2 TTS sentences. Directly needed to explain S1.
4. COMP v2 (instruction quoted first, then MCQ): the valid H3-B. 3 conds × 1,000 × 5 models, letter-scored (no
   judge) ≈ 1.5–2 GPU-h. Cheap, and it separates comprehension from compliance per channel.
5. Method (steering / CFG / transcribe-then-obey, code ready d0d66ec) on Qwen-7B/3B pilot: ~1 h per configuration
   (judge-bound). Needed for an ACL main paper (Vy 11-25).
6. H3-C (instruction before the image): needs an omnilib.py adapter change; ~3 GPU-h on the pilot for 5 models.

**Honest framing (main vs Findings).** As it stands (S1–S3 + controls), this is a solid benchmark-and-analysis
paper whose headline is partly negative: there is no general weaker channel, in-image instructions are read but
not obeyed, and negation specificity depends on the model. That fits Findings well. For the main track it needs
(i) the human-validated judge, (ii) an explanation of S1 (NEU control + COMP v2), and (iii) the method part with
recall reported.

## Style choices taken from "Deep and shallow biases in language models" (Vo et al., via paper/style_notes.md §3)
Structure and style only; no sentences copied.
- Abstract of one-job sentences: phenomenon → gap → what we build → headline numbers → controls → what it buys → URL.
- Figure 1 as a worked running example (one image, four deliveries), reused in the metric definitions.
- Intro of five paragraphs ending in "Our main findings are as follows" with a numbered list. Each item has a
  number and a section pointer, and one item is a limiting result (finding 4).
- Each metric defined as plain words → symbol → "a high X means" → worked example (Sec. 3).
- Benchmark construction as a numbered bold-verb pipeline with counts (Sec. 2.2).
- Claim-titled result subsections with run-in **Experiments** / **Results**; results open with the takeaway;
  captions end with "Thus, ...".
- Related work placed after the results, with run-in blocks each ending in a contrast sentence plus a checkmark
  table.
- Discussion with claim-like run-in headings, separating what was measured from what was not.
Deviation from the model paper (on purpose): we report CIs and paired tests on every comparison (it reports none),
because our effects are smaller and model-dependent.

## Change log
- 2026-11-26 16:xx: draft v1 written (all sections except H2 5-model); section pointers fixed (findings pointed
  to 5.x, results are 4.x); V kind range corrected to STATE 80.4–93.3 (first draft said 86–93, not in the file);
  occlusion placeholder filled from FULL.
- 2026-11-26 17:xx: number audit applied (6 fixes above); confound c11 (DENY) added with the leak-only check.
