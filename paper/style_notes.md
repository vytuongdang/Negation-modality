# Style notes: three reference papers (writing style only)

- [1] B-score (Vo, Taesiri, Kim, Nguyen; ICML 2025): https://arxiv.org/abs/2505.18545 (html: https://arxiv.org/html/2505.18545)
- [2] Vision Language Models are Biased (Vo, Nguyen, Taesiri, Dang, Nguyen, Kim; ICLR 2026): https://arxiv.org/abs/2505.23941 (html: https://arxiv.org/html/2505.23941)
- [3] Deep and shallow biases in language models (Vo, Dang, Nguyen, Villa-Cueva, Solorio, Nguyen, Kim; arXiv 9 Sep 2026): https://arxiv.org/abs/2609.09901 (html: https://arxiv.org/html/2609.09901)

All three IDs resolved (fetched 2026-11-24) and match the expected titles. Notes describe structure and
style only; no sentences are copied. Numbers below are quoted only to show *how* numbers are used.

---

## 1. B-score (2505.18545)

**Abstract** (~165 words, 7 sentences). Roles: (1) known phenomenon with two concrete, memorable examples
(gender, the number 7); (2) the research question as "we investigate whether..."; (3) the test set, with
its taxonomy enumerated inline as (1)/(2)/(3); (4) headline finding, flagged with "Interestingly";
(5) the proposed metric; (6) downstream benefit on named benchmarks, with a parenthetical definition of the
task; (7) code/data URL. Almost no numbers (only "9 topics"); the gains are stated qualitatively.

**Figure 1.** Three panels on one running example (GPT-4o asked for a random digit): (a) bar chart of
single-turn vs multi-turn probabilities with B-score, (b) three independent single-turn chats, (c) one
multi-turn chat. Caption is two sentences: the failure with its number (~70% for "7"), then the contrast
condition with its number (~10%), each pointing to a panel letter. The figure sits above the first
paragraph of the intro, so the reader sees the phenomenon before any prose.

**Introduction.** P1: the phenomenon and how bias is usually measured (single-turn repetition), with
Fig. 1b pointers. P2: an explicit italic-style question ("Would an LLM ... ?") followed by a one-word
answer ("Yes") and the Fig. 1 example. P3: a conjecture listing three kinds of bias; prior work covers
only one -> this is the gap. P4: what we propose (framework with 4 question types; the metric, defined in
one sentence with its formula in words). Then "main findings" as a **numbered list of 4 items**, each one
sentence, each a finding (not a contribution claim), each ending with a section pointer; only the last
item has numbers (+9.3, +2.9). The models tested are listed with small logo icons.

**Related work** (~700 words, sec. 2, before method). Four run-in bold headings (topic phrases). Each
block: a compressed survey with dense citation lists, then a "Our work differs ... in two aspects: (1)...
(2)..." or "In contrast, our approach..." sentence. Positioning is explicit and enumerated.

**Method.** Very short (sec. 3): the two conditions defined in one run-in paragraph each, a one-sentence
definition of bias, then the metric as a single displayed equation. The interpretation is given as a
three-case list (score > 0, ~0, < 0) with what each means, using the Fig. 1 example again. A closing
paragraph states practical properties (unsupervised, post-hoc). Benchmark section (sec. 4) opens with a
small table of 4 example questions (one per category) on a single topic, then counts, then the 4
categories as a numbered list with one-line definitions, then a control (option-order shuffling).

**Results.** Every subsection title is a full claim, e.g. "LLMs become less biased when viewing response
history in subjective & random questions". Each subsection has run-in "Experiment" and "Results"
paragraphs. Findings are given as before -> after numbers (0.77 -> 0.29) with figure/table pointers in
parentheses after every number. Captions restate the takeaway ("The B-score captures the trend ... while
the confidence score does not"). Category icons (subjective, random, easy, hard) are reused in text,
tables and captions as colour/icon coding. No CIs, no significance tests; runs are repeated 10x "to
mitigate run-to-run variability" and means are reported.

**Limitations.** A short run-in paragraph inside "Discussion and Conclusions" (3 sentences: scope of
evaluation, untested benchmarks, compute overhead). Conclusion opens "First, ... Second, ... Third, ...".

**Appendix.** Used heavily for full question sets, prompts, threshold tuning, and "additional results"
whose subsection titles are again findings (e.g. "On well-known BBQ bias benchmark, our conclusions
remain the same"), plus one with a "Recommendation" run-in heading.

---

## 2. Vision Language Models are Biased (2505.23941)

**Abstract** (~180 words, 6 sentences). (1) LLM prior knowledge helps but can sway answers (sets up
mechanism); (2) "In this work, we test..." with the task named (counting, identification); (3) headline
result with a concrete parenthetical example (4th stripe on a 3-stripe logo) and a hard number
(17.05% mean accuracy across 7 domains, domains enumerated); (4) an intervention result with number
(background removal, +21.09 points) and its interpretation; (5) a reasoning-length trend with number
(~40% then decline); (6) one-line contribution summary (failure mode + framework); then URL. Number-heavy.

**Figure 1.** A grid of 7 example counterfactual images (one per domain), each with the exact prompt and
each model's answer marked with a cross/tick against GT and the "bias" answer. Caption is one short
sentence; the figure itself carries the message. It acts as a teaser showing all tasks at once.

**Introduction.** P1: prior knowledge in LLMs, including memorised visual facts (concrete: flag stars,
chicken legs, pointing to Fig. 1). P2: why VLMs inherit it (early vs late fusion). P3: the gap: prior
evidence used Y/N prompts that contain the biased statement, so it is unclear (1) how much the image vs
the prompt drives errors and (2) whether bias affects neutral everyday tasks. P4: aim + one concrete test
(3-legged chicken + exact question). P5: the framework, human review, 7 domains listed with icons in
decreasing popularity, 5 named models. Then "Our key findings are:" with a **numbered list of 6 items**;
items carry many numbers (1.01%, 2.50%, +21.09), some have sub-sentences per domain, all end with section
pointers. Several items are written as mechanisms ("confirming...", "suggesting...").

**Related work** (~850 words incl. a comparison table, sec. 2). Three run-in headings. Includes Table 1:
a checkmark comparison against 4 prior benchmarks on design axes (biased prompt? biased image? generation
method, in-image text injection, top score, question type). The section ends with "existing benchmarks
have four limitations (1)-(4)" and a mirrored "We address these by (1)-(4)" list: an explicit 1:1 gap ->
fix mapping.

**Benchmark section.** Opens with a pipeline figure (sanity check -> counterfactual -> background
hypothesis). Justifies the task choice (why counting) in a few sentences. Run-in headings: "Taxonomy",
"Controls" (resolutions, two neutral prompt wordings), "Bias Definition" (bias rate defined in one
sentence with an example). The three questions Q1-Q3 are printed verbatim. Each task subsection has an
"Images" paragraph with a numbered generation procedure (Step 1/2/3) and an explicit count arithmetic
(91 animals x 3 resolutions = 273). Full details are pushed to per-task appendices.

**Results.** Starts with a **sanity check** subsection showing models know the unmodified subjects
(100%), which licenses the later causal claims. Titles are claims ("VLMs struggle to count the
signature elements in counterfactual images"; "Background contributes significantly to VLM counting
failures"; "Thinking longer reduces bias in VLMs, but overthinking harms accuracy"). Each has run-in
"Experiments"/"Results"; one subsection opens with a direct question ("What in the CF images made VLMs
count so poorly?") then "We hypothesize...". Domain-by-domain results use paired comparisons with a
one-clause hedged explanation ("likely because..."). Uncertainty: no CIs in the main text; robustness is
argued by 5 repeated runs varying < 1 point (appendix, mean +- std). Hedging verbs: "suggesting",
"likely", "perhaps".

**Limitations.** Very short run-in paragraph (image generators have their own biases; one example).
Followed by ICLR-required ethics, LLM-use and reproducibility statements.

**Appendix.** Huge (A.1-A.23 plus per-task appendices E-K), with a table of contents; every A.x title is
a finding. Main text refers out constantly ("More results are in Sec. A.x").

---

## 3. Deep and shallow biases in language models (2609.09901)

Resolved: the ID points to this exact title (submitted 9 Sep 2026). Has an ACL-style unnumbered
"Limitations" section and an "use of AI assistants" appendix, so it is the closest template for us.

**Abstract** (~156 words, 9 sentences). (1) phenomenon in plain words; (2) gap: prior work conflates two
things; (3) "We introduce" the score and what it measures (two parts); (4) scale + headline number in
words ("about a quarter", across 4,442 prompts and four models); (5) names the two categories; (6)-(7)
two further findings (origin in pretraining; harder to remove), no numbers; (8) a "therefore" sentence
stating what the measure buys; (9) URL. Sentences are short and each does one job.

**Figure 1.** A framework diagram that is also a worked example: two real prompts side by side (butterfly
= Deep, number = Shallow), showing direct samples, reframed samples and whether the top answer returns.
Caption: one sentence naming the figure ("Our bias evaluation framework"), one sentence on procedure,
then (a) and (b) each defined by its example. The same two examples, with the same numbers, recur in the
intro, in the metric definitions (sec. 2), and in the dataset steps: a true running example.

**Introduction** (5 paragraphs + list). P1: phenomenon with two concrete numbers from Fig. 1, then prior
work. P2: gap (one fixed phrasing cannot separate stable bias from prompt dependence) + why it matters
(prompt sensitivity citations). P3: "We therefore argue..." + the Fig. 1 contrast + naming of the two
categories. P4: second motivation (tracing origin, predicting removability). P5: the framework in
notation (DR, FR, pi = DR*FR) and the dataset size, with section pointers. Then "Our main findings are as
follows" with a **numbered list of 5 items**; most items carry ranges or deltas (35.8%-77.7%; -4.5 vs
-10.3 points) and every item ends with a section pointer. One item states a negative/limited result
honestly (clean attribution works only for a small fraction).

**Method (sec. 2) and Dataset (sec. 3).** Method comes first, dataset second, related work late (sec. 5).
Each quantity gets its own short subsection or run-in heading (Direct prompting, Scenario reframing,
Cluster judge, Bias depth score, Bias types, Total bias proportion). Pattern per quantity: plain-language
definition -> symbol -> "A high X means..." -> worked example with the running prompts. Thresholds are
justified (0.65 = 20 of 30 samples; uniform baseline 0.5 for two options). Categories given as a bullet
list with the formal condition first, then a one-line gloss. Dataset is a numbered 6-step pipeline, each
step a bold verb (Extract, Filter, Select, Rewrite, Deduplicate, Reframe) with the count after each step.
Design choices are defended with "this matters for three reasons. First/Second/Third".

**Results.** Subsection titles are claims ("Most LLM biases are shallow"; "Deep bias is mostly set during
pretraining"; "Deep biases are harder to remove than Shallow biases"). Each opens with a link back to the
previous subsection and a question in the form "We therefore ask: (i)... (ii)...", then bold run-in
"Experiments" and "Results". Results paragraphs start with the takeaway sentence, then numbers with
arrows (8.0% -> 25.0%) and figure pointers. Captions end with a "Thus, ..." sentence giving the
takeaway, and state n per bin. A qualitative table shows four real examples with their training-data
evidence. No CIs or significance tests reported.

**Related work** (sec. 5, three run-in headings) with a checkmark comparison table (Table 3) against
five prior methods. Each block ends with "We go further by..." / "Unlike prior work, ..." / "We address
these questions by..." plus section pointers.

**Discussion / Limitations.** Discussion uses bold run-in headings that are themselves claims (e.g. "Bias
depth also indicates which mitigation to use") and explicitly separates what was measured from what was
not ("We do not measure these harms directly"). Limitations is a single focused paragraph on the central
weakness (attribution rarely clean) and why, plus a future direction.

---

## Cross-paper patterns

- Same group, same house style: claim-titled result subsections; run-in "Experiments"/"Results";
  numbered findings list at the end of the intro with section pointers; colour/icon coding for
  categories and models; heavy appendix whose titles are also findings.
- Abstracts 155-180 words, 6-9 sentences, one URL at the end. [2] and [3] put a headline number in.
- Figure 1 is always a concrete example of the phenomenon (not an architecture box), with a 1-3 sentence
  caption that points to panels by letter.
- Related work is positioned by an explicit contrast sentence per block and, in [2] and [3], a
  checkmark table. In [3] (ACL-style) it moves after the results.
- Statistics: none of the three report CIs or tests in the main text; they rely on repeated runs and
  large deltas. We must do better here (pilot sizes, CIs), but keep their "number + pointer" rhythm.

---

## Synthesis: style checklist for our paper (ACL 2027, cross-modal negation)

1. **Abstract, 150-180 words, 7-9 one-job sentences**: phenomenon -> gap (no controlled comparison of the
   same prohibition across text / speech / in-image) -> what we build (paired benchmark + no-instruction
   baseline) -> 1-2 headline numbers with CIs, worded neutrally -> one control finding -> what the
   design buys -> URL. Say "pilot" in the abstract if the numbers are pilot numbers.
2. **Figure 1 = one running example**, not a pipeline box: the same image + "Do not mention the dog."
   delivered three ways (typed, spoken, written in the image) plus the no-instruction baseline, with each
   model's output and a violated/obeyed mark. Caption: 2-3 sentences, panel letters, takeaway last.
3. **Reuse that running example everywhere** (intro, metric definitions, template table, error analysis),
   the way [3] reuses its two prompts.
4. **Intro in five paragraphs**: setting (omni-models take constraints in any channel) -> gap (prior
   negation/instruction work is text-only or confounds channel with content) -> our argument + Fig. 1 ->
   why the paired design and controls are needed -> design in notation with section pointers.
5. **Numbered findings list (4-6 items)** closing the intro, each one sentence, each with a number + CI
   or delta and a "(Sec. X)" pointer. State findings, not "we propose" claims; include at least one
   null or limiting result.
6. **Define every metric with the [3] pattern**: plain words -> symbol (e.g. violation rate given the
   baseline mentions the object, Delta_m per channel m) -> "A high X means..." -> worked example. Justify
   any threshold or conditioning choice in one sentence.
7. **Benchmark construction as a numbered, bold-verb pipeline** with the item count after each step
   (sources -> selection -> VLM check -> templates -> TTS + ASR check -> image rendering), arithmetic
   shown once (items x channels x forms = N). Full prompts and templates verbatim in the appendix.
8. **Put a sanity-check / control subsection first in Results** (as in [2]): the model mentions the
   object without instruction (baseline), it can read the in-image text (OCR control), it transcribes the
   speech (ASR). Only then compare channels.
9. **Result subsection titles are neutral claims**, sized to the evidence ("Spoken prohibitions are
   violated more often than typed ones in two of three models"), each with run-in **Experiments** and
   **Results** paragraphs; first sentence of Results is the takeaway.
10. **Every number gets an uncertainty and a pointer**: "x% [95% CI a, b]; Tab. N"; paired differences
    reported with the paired test (McNemar) and bootstrap CIs over images; say n in every caption.
    Never write "significantly" without the test.
11. **Captions carry the takeaway**: end figure/table captions with one sentence stating what to read
    from them, plus n and the CI method.
12. **Neutral, calibrated verbs**: "is associated with", "suggests", "in this pilot"; avoid "fails",
    "collapses", "blind" unless the CI excludes the null and the effect is large. Hedge explanations with
    one clause ("possibly because..."), as [2] does, and label them as conjecture.
13. **Consistent colour/icon coding** for the three channels (text / speech / image) and the baseline,
    used identically in Fig. 1, plots, tables and inline mentions.
14. **Related work after results or before benchmark, 3-4 run-in headed blocks** (negation in VLMs;
    negative instructions; typographic/visual prompts; speech vs text instructions), each ending in one
    explicit contrast sentence, plus a small checkmark table (channel coverage, paired baseline,
    in-image instruction, audio instruction, CIs).
15. **Limitations as its own focused section** (ACL requires it): synthetic TTS speech, English only,
    question always in text, VLM-filtered images, LLM-judge error, pilot sample size; say what was not
    measured. Appendix subsection titles should also be findings, with a contents list if long.
