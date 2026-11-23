"""Mark COCO units of the CURRENT dataset whose question group has one object labelled twice (Vy, 2026-11-23).

B1 now drops such units when the dataset is rebuilt (b1_ground_truth.double_labelled). This script applies the
same rule to the dataset that already exists, without rebuilding: it writes the unit ids to
    dataset/out/excluded_double_label.jsonl      {unit_id, image, supercategory, pairs}
and experiments/common/omnilib.load_dataset() skips every sample of those units (all forms, all conditions).

    python3 flag_double_labels.py
"""
import json
from collections import Counter, defaultdict

from b1_ground_truth import double_labelled
from config import OUT_DIR, RAW
from utils import read_jsonl

OUT = OUT_DIR / "excluded_double_label.jsonl"


def main():
    samples = [s for s in read_jsonl(OUT_DIR / "samples.jsonl") if s["source"]["dataset"] == "COCO"]
    forms = OUT_DIR / "samples_forms.jsonl"
    if forms.exists():
        samples += [s for s in read_jsonl(forms) if s["source"]["dataset"] == "COCO"]
    want = {int(str(s["image"]).split("/")[-1].split(".")[0]) for s in samples}
    cat_of, sc_of, by_img = {}, {}, defaultdict(list)
    for split in ("train2017", "val2017"):
        data = json.load(open(RAW / f"coco/instances_{split}.json"))
        cats = {c["id"]: c for c in data["categories"]}
        for a in data["annotations"]:
            if a["image_id"] in want:
                by_img[a["image_id"]].append(a)
                cat_of[a["id"]] = cats[a["category_id"]]["name"]
                sc_of[cats[a["category_id"]]["name"]] = cats[a["category_id"]]["supercategory"]
    flagged = {}
    for s in samples:
        uid = s["validation"]["unit_id"]
        if uid in flagged:
            continue
        img = int(str(s["image"]).split("/")[-1].split(".")[0])
        names = {cat_of[int(i)] for c in s["components"] for i in c.get("annotation_ids", []) if int(i) in cat_of}
        scs = {sc_of[n] for n in names}
        if len(scs) != 1:
            continue
        sc = scs.pop()
        by_cat = defaultdict(list)
        for a in by_img[img]:
            if sc_of.get(cat_of[a["id"]]) == sc:
                by_cat[cat_of[a["id"]]].append(a)
        pairs = double_labelled(by_cat)
        if pairs:
            flagged[uid] = {"unit_id": uid, "image": s["image"], "supercategory": sc, "pairs": pairs,
                            "branch": s["branch"]}
    with open(OUT, "w") as f:
        for r in flagged.values():
            f.write(json.dumps(r) + "\n")
    units = {s["validation"]["unit_id"] for s in samples}
    print(f"{len(flagged)} of {len(units)} COCO units flagged -> {OUT}")
    print("by branch:", dict(Counter(r["branch"] for r in flagged.values())))
    print("pairs:", Counter(p for r in flagged.values() for p in map(tuple, r["pairs"])).most_common(10))


if __name__ == "__main__":
    main()
