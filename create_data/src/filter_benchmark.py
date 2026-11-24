"""Remove bad images FROM THE BENCHMARK after the one-time build, regenerating nothing (Vy, 2026-11-24 23:04 KST).

Input:  dataset/out/excluded_images.jsonl  (written by hand after the image review), one row per image:
          {"image": "<path exactly as in samples.jsonl, e.g. images/vg/2345.jpg>", "reason": "<why>"}
        (paths, not file names: VAW and GQA share Visual Genome images)
Effect: 1. the first time, the unfiltered build is copied to dataset/out_unfiltered/ (never overwritten after);
        2. every *.jsonl of out_unfiltered/ is written back to out/ without the rows of the removed images
           (samples, forms, compact, compact_all, compact_all_audio, excluded_units, ...); sample_ids are kept
           as they are (no renumbering), so B3 targets/templates, forms (pair_of) and audio/visual paths of the
           remaining items are unchanged;
        3. the audio manifest keeps only sentences still used (sample_ids pruned; unfiltered copy kept next to it);
        4. the per-sample wav / visual files of removed samples are MOVED to dataset/removed_images/ (not deleted).
The script always starts from out_unfiltered/, so it is idempotent: add images to the list and run it again.
Run/analysis scripts read out/ as before and need no bad-image logic.
Why not a rebuild: B3's template-sweep subset (rng.sample over all units), the sample_id numbering (per
source-branch counters in file order) and the TTS voice assignment (shuffle over all sentences) depend on the
whole set, so a rebuild would change the remaining items.
Usage: cd create_data/src && PYTHONNOUSERSITE=1 ~/anaconda3/envs/negation/bin/python filter_benchmark.py [--dry_run]
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from config import DATA, OUT_DIR
from utils import read_jsonl

SRC = DATA / "out_unfiltered"
LIST = OUT_DIR / "excluded_images.jsonl"
MANIFEST = DATA / "audio" / "negation_sentences" / "manifest.jsonl"
MOVED = DATA / "removed_images"
ID_KEYS = ("sample_id", "id", "pair_of")


def name(p):
    return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    bad = {name(r["image"]): r["reason"] for r in read_jsonl(LIST)}
    if not SRC.exists() and not args.dry_run:
        shutil.copytree(OUT_DIR, SRC)
        print(f"unfiltered build copied to {SRC}")
    src = SRC if SRC.exists() else OUT_DIR
    samples = read_jsonl(src / "samples.jsonl") + (read_jsonl(src / "samples_forms.jsonl")
                                                   if (src / "samples_forms.jsonl").exists() else [])
    images = {name(s["image"]) for s in samples}
    missing = [i for i in bad if i not in images]
    if missing:
        raise SystemExit(f"{len(missing)} listed images are not in the benchmark, e.g. {missing[:5]}")
    drop_units = {s["validation"]["unit_id"] for s in samples if name(s["image"]) in bad}
    drop_ids = {s["sample_id"] for s in samples if s["validation"]["unit_id"] in drop_units}

    def dropped(r):
        if "image" in r and name(r["image"]) in bad:
            return True
        if r.get("unit_id") in drop_units or r.get("validation", {}).get("unit_id") in drop_units:
            return True
        return any(r.get(k) in drop_ids for k in ID_KEYS)

    removed_rows = []
    for f in sorted(src.glob("*.jsonl")):
        if f.name == LIST.name:
            continue
        rows = read_jsonl(f)
        keep = [r for r in rows if not dropped(r)]
        if f.name.startswith("samples_compact_all"):
            removed_rows += [r for r in rows if dropped(r)]
        print(f"{f.name}: {len(rows)} -> {len(keep)}")
        if not args.dry_run:
            with open(OUT_DIR / f.name, "w") as g:
                for r in keep:
                    g.write(json.dumps(r, ensure_ascii=False) + "\n")
    br = Counter(s["branch"] for s in read_jsonl(src / "samples.jsonl") if s["sample_id"] in drop_ids)
    print(f"{len(bad)} images -> {len(drop_units)} units, {len(drop_ids)} samples (all forms) removed; "
          f"negative samples by branch: {dict(br)}")

    if MANIFEST.exists():
        full = MANIFEST.with_name("manifest_unfiltered.jsonl")
        if not full.exists() and not args.dry_run:
            shutil.copy(MANIFEST, full)
        man = read_jsonl(full if full.exists() else MANIFEST)
        out = []
        for m in man:
            ids = [i for i in m.get("sample_ids", []) if i not in drop_ids]
            if ids or not m.get("sample_ids"):
                out.append(dict(m, sample_ids=ids))
        print(f"audio manifest: {len(man)} -> {len(out)} sentences")
        if not args.dry_run:
            with open(MANIFEST, "w") as g:
                for m in out:
                    g.write(json.dumps(m, ensure_ascii=False) + "\n")

    moved = 0
    for r in removed_rows:   # per-sample media of removed samples (compact_all rows carry the paths)
        for k in ("negation_audio", "negation_image", "box_image"):
            p = r.get(k)
            if not p:
                continue
            p = Path(p) if Path(p).is_absolute() else DATA / p
            if p.exists() and not args.dry_run:
                dst = MOVED / p.relative_to(DATA)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dst))
                moved += 1
    print(f"media files moved to {MOVED}: {moved}" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
