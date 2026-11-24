"""ID check of the benchmark (Vy 2026-11-24): every row of out/*.jsonl has image_id, unit_id, sample_id; sample ids
unique per file; ids consistent across files; counts of images / units / samples.
Usage: cd create_data/src && PYTHONNOUSERSITE=1 ~/anaconda3/envs/negation/bin/python check_ids.py [out_dir]"""
import sys
from collections import Counter
from pathlib import Path

from config import OUT_DIR
from utils import read_jsonl

out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR
ok = True
unit_img = {}
for f in ["samples.jsonl", "samples_forms.jsonl", "samples_compact_all.jsonl", "samples_compact_all_audio.jsonl"]:
    rows = read_jsonl(out / f)
    miss = {k: sum(not r.get(k) for r in rows) for k in ("image_id", "unit_id", "sample_id")}
    dup = sum(c - 1 for c in Counter(r.get("sample_id") for r in rows).values() if c > 1)
    for r in rows:   # a unit always belongs to one image
        if unit_img.setdefault(r.get("unit_id"), r.get("image_id")) != r.get("image_id"):
            miss["unit_with_2_images"] = miss.get("unit_with_2_images", 0) + 1
    bad = any(miss.values()) or dup
    ok &= not bad
    print(f"{f}: {len(rows)} rows, {len({r.get('image_id') for r in rows})} images, "
          f"{len({r.get('unit_id') for r in rows})} units, missing {miss}, duplicate sample_ids {dup}"
          + ("  <-- PROBLEM" if bad else ""))
neg = read_jsonl(out / "samples_compact_all_audio.jsonl")
ids = {r["sample_id"] for r in neg}
full = {r["sample_id"] for r in read_jsonl(out / "samples.jsonl")} | {r["sample_id"] for r in read_jsonl(out / "samples_forms.jsonl")}
print(f"compact_all_audio ids == samples+forms ids: {ids == full} ({len(ids)} vs {len(full)})")
ok &= ids == full
print("ID CHECK", "OK" if ok else "FAILED")
