# Component-Level Negation VQA: dataset pipeline

Implements "Quy trình tạo dataset Component-Level Negation VQA Benchmark" (B1 → B2 → B3 → baseline filter).
Pilot size: `N_IMAGES = 50` per source (COCO, CLEVR, VAW, GQA). All parameters live in `config.py`.

One command runs everything (from this folder):

    python3 run_pipeline.py                  # all steps 0-5
    python3 run_pipeline.py --step3 --step5  # only selected steps
    python3 run_pipeline.py --from_step 2    # step 2 and everything after
    python3 run_pipeline.py --n_images 20    # override images per source

Steps (each script can also be run on its own):

| Step | Script | Output |
|---|---|---|
| 0 | `python3 download_annotations.py` | `dataset/raw/` (annotations only; CLEVR/GQA members are read from the remote zip via HTTP Range) |
| 1 (B1) | `python3 b1_ground_truth.py` | `dataset/b1/units.jsonl`, `dataset/images/{coco,clevr,vg}/` |
| 2 (B2) | `python3 b2_components.py` | `dataset/b2/units.jsonl`, `dataset/b2/human_verify/{human_verify.csv,human_verify.html}` |
| 3 (B3) | `python3 b3_negation.py` | `dataset/out/samples_text_negative.jsonl` |
| 4 (doc §7) | `python3 baseline.py` | `dataset/out/samples.jsonl` (final, full record), `dataset/out/alias_report.csv`, `dataset/out/baseline_cache/` |
| 5 | `python3 export_compact.py` | `dataset/out/samples_compact.jsonl` (slim file for running models; `id` = `sample_id` in samples.jsonl) |

`mention.py` is the string matcher (aliases, plurals, "hot dog" vs "dog", count = number next to the noun);
the baseline filter uses it and later scoring should reuse it.

## Choices made where the doc leaves room

- **COCO count**: a (image, supercategory) pair gets a count question only if every category in it has ≤ 5
  instances and no `iscrowd` region (a crowd region makes the exact count unknown).
- **VAW**: slots are VAW parent types (color, material, shape, size, texture, state, action; `other` dropped),
  and a slot is used only when the object has exactly one positive label in it. "Unique name in the image" is
  checked against the GQA scene graph of the same VG image, so VAW images outside GQA are skipped.
- **CLEVR attribute**: the object must be the only one of its shape; the question asks color, size, material.
  **CLEVR count** asks only about shapes present in the scene.
- **GQA relation**: only verb-like relations (`-ing` plus "covered in/with", "filled with", "full of"); spatial
  ones (left of, near, behind...) are dropped. Each relation must point to one object, and the objects must have
  different names. The first GQA balanced `rel` question about the subject is kept as `extra.gqa_original_qa`.
  Pronoun: he/she for man/woman words, they for person words, otherwise it.
- **B2**: components come from annotation for every branch (GQA included, from the scene graph), so the LLM only
  writes aliases and reviews atomicity/question. Count aliases (number words) are rule-based. Generic words and
  aliases shared by two components are removed. LLM: `Qwen/Qwen3-8B` (fp16, transformers) because vLLM fails on
  this machine (flashinfer vs Python 3.10).
- **B3**: the target is the first component, in seeded random order, whose aliases do not collide with another
  component's (e.g. sandwich vs hot dog); units with no such component are dropped. 10% of units are expanded to
  all 7 templates (`negation.template_sweep = true`).
  `prompt` is the question only and `negation_text` the instruction only; evaluation joins them. N7's cue is `leave_out`, not "exceptive", to keep the name
  free for the later exceptive form.
- **Baseline**: reference models `Qwen2.5-VL-3B-Instruct` and `Qwen3-VL-8B-Instruct` (fp16, images downscaled to
  1024px on the long side). A sample is kept if at least one model mentions the target at baseline.
