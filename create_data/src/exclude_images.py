"""Drop bad images at run/analysis time WITHOUT regenerating anything (Vy, 2026-11-24).

Input:  dataset/out/excluded_images.jsonl  (written by hand during the image review), one row per image:
          {"image": "<image path or file name as in samples.jsonl>", "reason": "<why>"}
Effect: every unit of those images is added to dataset/out/excluded_units.jsonl with reason "bad_image"
        (the file flag_excluded_units.py writes, which load_dataset / analyze.py / analyze_h2.py / marginal_cis.py
        already read). Rows from flag_excluded_units.py are kept; earlier "bad_image" rows are replaced, so the
        script is idempotent. Re-run it after every run of flag_excluded_units.py (that script rewrites the file).

Why filtering and not a rebuild: B3's template-sweep subset (rng.sample over all units), the sample_id numbering
(per source-branch counters) and the TTS voice assignment (shuffle over all sentences) depend on the whole set,
so a rebuild without some images would change the remaining items. Filtering leaves them untouched.
Usage: cd create_data/src && PYTHONNOUSERSITE=1 ~/anaconda3/envs/negation/bin/python exclude_images.py [--dry_run]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from config import OUT_DIR
from utils import read_jsonl

LIST = OUT_DIR / "excluded_images.jsonl"
OUT = OUT_DIR / "excluded_units.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    bad = {Path(str(r["image"])).name: r["reason"] for r in read_jsonl(LIST)}
    samples = read_jsonl(OUT_DIR / "samples.jsonl")
    if (OUT_DIR / "samples_forms.jsonl").exists():
        samples += read_jsonl(OUT_DIR / "samples_forms.jsonl")
    units_of = defaultdict(dict)   # image file name -> {unit_id: sample}
    for s in samples:
        units_of[Path(str(s["image"])).name].setdefault(s["validation"]["unit_id"], s)
    missing = [i for i in bad if i not in units_of]
    if missing:
        raise SystemExit(f"{len(missing)} listed images have no unit in samples.jsonl, e.g. {missing[:5]}")
    kept = [r for r in (read_jsonl(OUT) if OUT.exists() else []) if r["reason"] != "bad_image"]
    seen = {r["unit_id"] for r in kept}
    new = []
    for img, reason in bad.items():
        for uid, s in units_of[img].items():
            if uid not in seen:
                seen.add(uid)
                new.append({"unit_id": uid, "reason": "bad_image", "detail": reason, "branch": s["branch"],
                            "image": s["image"]})
    print(f"{len(bad)} bad images -> {len(new)} more units excluded (others already flagged); "
          f"total {len(kept) + len(new)}; by branch: {dict(Counter(r['branch'] for r in new))}")
    if not args.dry_run:
        with open(OUT, "w") as f:
            for r in kept + new:
                f.write(json.dumps(r) + "\n")
        print(f"-> {OUT}")


if __name__ == "__main__":
    main()
