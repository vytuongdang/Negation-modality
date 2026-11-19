# Paper outline (draft, ACL 2027 via ARR; ARR deadline found by web search: 2027-01-04, verify on 2027.aclweb.org)

1. Introduction — omni-models receive constraints by typing, speaking, showing; does a prohibition survive the
   channel? A fair comparison needs controls (T_sep, C0_box, affirmative/exceptive, conflict).
   Contributions: benchmark (C8), controlled channel comparison (C1-C2), negation-specificity test (C3-C4),
   conflict/dominance analysis (C5).
2. Related work — negation in VLMs; negative instructions / pink elephant; typographic/visual instructions;
   speech vs text instructions and modality conflict (paper/related_work.md).
3. Benchmark — sources (COCO, VAW, GQA, CLEVR), branches, components + aliases, image selection + VLM check,
   templates N1-N7, forms, TTS (+ASR check), visual rendering (+box control). Stats table.
4. Setup — models, conditions, scoring (LLM judge + regex), metrics (violation | C0-mentioned, required recall,
   Δ_m), statistics (McNemar, image bootstrap).
5. Results — H1 channel gap; H4 negation specificity; H2/H3-A conflict.
6. Analysis — templates, branch, voice, ASR, occlusion, split; error examples.
7. Limitations — synthetic speech, English only, question always in text, VLM-filtered images, judge errors.
