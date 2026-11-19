# Related work (collected by Pico; links from web search 2026-11-19, verify each before citing)

## Negation in vision-language models
- Alhamoud et al., *Vision-Language Models Do Not Understand Negation* (NegBench), CVPR 2025.
  https://arxiv.org/abs/2501.09425 — negated retrieval / MCQ; CLIP-style models near chance; fine-tuning on
  synthetic negated captions helps. Negation *in the query*, not a negative *instruction*.
- *NegVQA: Can Vision Language Models Understand Negation?* https://arxiv.org/abs/2505.22946 — 7,379 two-choice
  negated VQA questions, 20 VLMs struggle. Again comprehension of negated content, not prohibitions on output.
- Gap we fill: prohibitions on what the answer may say ("do not mention X"), and the channel carrying them.

## Negative instructions / "pink elephant" effect
- Hwang et al., arXiv 2404.15154 — negative prompts in text-to-image models push toward the forbidden concept
  ("white bear" effect). https://arxiv.org/abs/2404.15154 — motivates the exceptive form (naming X primes X).
- Truhn et al., arXiv 2503.22395 (2025), negation handling in LLMs across sizes/languages — NOT yet opened, check.

## Instructions inside the image (typographic / visual prompt injection)
- Empirical analysis of LVLMs against goal hijacking via visual prompt injection (2024).
  https://arxiv.org/abs/2408.03554 — drawn-in instructions make LVLMs abandon the task; depends on OCR + instruction following.
- Typographic visual prompt injection in cross-modality generation models, arXiv 2503.11519.
- *Reading Between the Pixels* (typographic attack success vs text-image embedding alignment), arXiv 2604.12371:
  text-only vs image-rendered attacks differ by model. Our V condition is the benign analogue (a legitimate
  instruction in the image); C0_box controls for the occluded pixels.

## Speech vs text instructions; modality conflict in audio / omni models
- Qwen2.5-Omni technical report, arXiv 2503.20215 (speech-converted text benchmarks).
- VoiceBench (spoken instruction following, synthetic + real speech) — find exact citation.
- Speech-text modality gap, layer-wise analysis, arXiv 2603.01502: S2T accuracy below matched T2T for all models.
- *When Audio and Text Disagree* (MCR-Bench), arXiv 2508.15407: LALMs prefer text when text and audio conflict.
- ALME (Billa), arXiv 2602.11488: text dominance in audio-text conflicts, 8 languages; tuning the LLM reduces it.
- XModBench (ICLR 2026), arXiv 2510.15148: tri-modal consistency; largest gaps audio vs text.
- OmniBench (NeurIPS 2025): open omni-models weak at tri-modal instruction following.
- Relevance: prior work says TEXT dominates audio in content conflicts; our pilot found audio NEGATION obeyed
  better than text. H2/H3-A test whether that is dominance or negation-specific. Strong framing point.
