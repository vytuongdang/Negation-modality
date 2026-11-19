"""Stage 1 (Vy, 2026-11-19): build the dataset only for a subset of the selected units.

After image selection, dataset/b1/units.jsonl holds every selected unit. This script keeps the full list in
dataset/b1/units_full.jsonl (written once, never overwritten) and replaces units.jsonl with a seeded, branch-
stratified subset, so steps 3-7 + TTS (`bash run_v2_all.sh rest`) build only what the subset needs.

    python3 subset_units.py --per_branch 230     # 230 units per branch (some are dropped later by B2 / B3)
    python3 subset_units.py --restore            # put the full list back (before building the full dataset)

Inside a branch, units are drawn at random (seed) with distinct images where possible.
"""
import argparse
import json
import random
import shutil
from collections import Counter, defaultdict

from config import B1_DIR, SEED
from utils import read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_branch", type=int, default=230)
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()
    full_path, cur = B1_DIR / "units_full.jsonl", B1_DIR / "units.jsonl"
    if args.restore:
        shutil.copy(full_path, cur)
        print(f"restored {cur} from {full_path}")
        return
    if not full_path.exists():
        shutil.copy(cur, full_path)
    units = read_jsonl(full_path)
    rng = random.Random(f"{SEED}-subset")
    by_branch = defaultdict(list)
    for u in units:
        by_branch[u["branch"]].append(u)
    picked, used_images = [], set()
    for b in sorted(by_branch):
        pool = sorted(by_branch[b], key=lambda u: u["unit_id"])
        rng.shuffle(pool)
        fresh = [u for u in pool if u["image"] not in used_images]
        rest = [u for u in pool if u["image"] in used_images]
        take = (fresh + rest)[:args.per_branch]
        used_images |= {u["image"] for u in take}
        picked += take
    picked.sort(key=lambda u: u["unit_id"])
    write_jsonl(cur, picked)
    info = {"per_branch": args.per_branch, "n_units": len(picked), "n_full": len(units),
            "by_branch": dict(Counter(u["branch"] for u in picked)),
            "by_source": dict(Counter(u["source"]["dataset"] for u in picked)),
            "by_split": dict(Counter(f"{u['source']['dataset']}/{u['source']['split']}" for u in picked)),
            "distinct_images": len({u["image"] for u in picked})}
    (B1_DIR / "subset_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
