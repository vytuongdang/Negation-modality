"""Image-level subset of the selected units (B200 full build, Vy 2026-11-24: build ~2.8k images, drop the worst later).

Step 2 leaves every selected unit in dataset/b1/units_full.jsonl (3,154 images, 5,335 units). This script picks
`--n_images` images and keeps ALL units of the picked images, writing dataset/b1/units.jsonl (units_full.jsonl is
never touched), so steps 3-7 + TTS (`bash run_v2_all.sh rest`) build exactly these images.

    python subset_images.py --n_images 2800 --dry_run   # print what would be picked, write nothing
    python subset_images.py --n_images 2800             # write units.jsonl + subset_info.json

Quota per source: --keep_all sources (default GQA: the relation branch has no other source) keep every image; the
other sources share the remaining quota in proportion to their image count. Inside a source, images from non-train
splits (val / test) are kept first, so the val-vs-train contamination check stays as strong as it can be; the rest
is filled with train images drawn at random (seeded). Seeded and deterministic.
"""
import argparse
import json
import random
from collections import Counter, defaultdict

from config import B1_DIR, SEED
from utils import read_jsonl, write_jsonl


def pick(units, n_images, keep_all):
    by_image = defaultdict(list)
    for u in units:
        by_image[u["image"]].append(u)
    by_source = defaultdict(list)
    for im, us in by_image.items():
        by_source[us[0]["source"]["dataset"]].append(im)
    fixed = sum(len(v) for s, v in by_source.items() if s in keep_all)
    free = {s: len(v) for s, v in by_source.items() if s not in keep_all}
    room = max(n_images - fixed, 0)
    quota = {s: len(v) for s, v in by_source.items() if s in keep_all}
    for s, n in free.items():
        quota[s] = min(n, round(room * n / sum(free.values())))
    rng = random.Random(f"{SEED}-subset-images")
    picked = []
    for s in sorted(by_source):
        ims = sorted(by_source[s])
        nontrain = [i for i in ims if by_image[i][0]["source"]["split"] != "train"
                    and not by_image[i][0]["source"]["split"].startswith("train")]
        train = [i for i in ims if i not in set(nontrain)]
        rng.shuffle(nontrain)
        rng.shuffle(train)
        picked += (nontrain + train)[:quota[s]]
    keep = set(picked)
    return [u for u in units if u["image"] in keep], quota


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=2800)
    ap.add_argument("--keep_all", nargs="*", default=["GQA"], help="sources that keep every image")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    full_path, cur = B1_DIR / "units_full.jsonl", B1_DIR / "units.jsonl"
    units = read_jsonl(full_path)
    out, quota = pick(units, args.n_images, set(args.keep_all))
    info = {"n_images_requested": args.n_images, "quota_per_source": quota, "n_units": len(out),
            "n_full": len(units), "distinct_images": len({u["image"] for u in out}),
            "by_branch": dict(Counter(u["branch"] for u in out)),
            "by_source": dict(Counter(u["source"]["dataset"] for u in out)),
            "by_split": dict(Counter(f"{u['source']['dataset']}/{u['source']['split']}" for u in out))}
    print(json.dumps(info, indent=1))
    if args.dry_run:
        print("dry run: nothing written")
        return
    out.sort(key=lambda u: u["unit_id"])
    write_jsonl(cur, out)
    (B1_DIR / "subset_info.json").write_text(json.dumps(info, indent=1))
    print(f"wrote {cur} ({len(out)} units); full list stays in {full_path}")


if __name__ == "__main__":
    main()
