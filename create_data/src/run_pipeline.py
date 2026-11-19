"""Run the dataset pipeline (v2), or selected steps.

    python3 run_pipeline.py --to_step 2        # annotations, B1 candidates, image selection -> stop for review
    python3 run_pipeline.py --from_step 3      # B2 ... visual (after the review page is approved)
    python3 run_pipeline.py --step5 --step6    # only selected steps

Steps:
    0  download annotations (every split in config.SPLITS)
    1  B1   candidates from the annotations (annotation-level filters)
    2  image selection: download val first, train only to fill; pixel checks, text space, VLM checks
       -> dataset/b1/review/review.html (human look before going on)
    3  B2   components, aliases, 10% human-verify sample (local LLM)
    4  baseline: reference models answer each question without negation; eligible targets
    5  B3   negative instruction (7 templates) -> dataset/out/samples.jsonl
    6  affirmative + exceptive forms (H4) -> dataset/out/samples_forms.jsonl
    7  export compact negative file
    8  visual channel: instruction written inside the image -> dataset/out/samples_compact_all.jsonl
then, in the cosyvoice env, TTS for every form (run_v2_all.sh does both).
"""
import argparse
import time

import config

STEPS = {
    0: ("download annotations", "download_annotations"),
    1: ("B1 candidates", "b1_ground_truth"),
    2: ("image selection", "select_images"),
    3: ("B2 components", "b2_components"),
    4: ("baseline (reference models)", "baseline"),
    5: ("B3 negative instructions", "b3_negation"),
    6: ("affirmative + exceptive forms", "b3_forms"),
    7: ("export compact", "export_compact"),
    8: ("visual channel", "render_visual"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, (name, _) in STEPS.items():
        ap.add_argument(f"--step{k}", action="store_true", help=f"run step {k}: {name}")
    ap.add_argument("--from_step", type=int, help="run this step and every step after it")
    ap.add_argument("--to_step", type=int, help="stop after this step")
    ap.add_argument("--seed", type=int, help=f"random seed (default {config.SEED})")
    args = ap.parse_args()
    if args.seed is not None:
        config.SEED = args.seed

    chosen = [k for k in STEPS if getattr(args, f"step{k}")]
    if args.from_step is not None or args.to_step is not None:
        lo = args.from_step if args.from_step is not None else 0
        hi = args.to_step if args.to_step is not None else max(STEPS)
        chosen = sorted(set(chosen) | {k for k in STEPS if lo <= k <= hi})
    if not chosen:
        chosen = list(STEPS)

    for k in chosen:
        name, module = STEPS[k]
        print(f"\n===== step {k}: {name} =====", flush=True)
        t = time.time()
        mod = __import__(module)
        if module == "render_visual":
            import sys
            sys.argv = [sys.argv[0]]
        mod.main()
        print(f"----- step {k} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
