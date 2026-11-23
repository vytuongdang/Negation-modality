"""Units to leave out of every experiment, with the reason (Vy, 2026-11-23). Writes
    dataset/out/excluded_units.jsonl      {unit_id, reason, detail, branch, image}
experiments/common/omnilib.load_dataset() skips every sample of these units (all forms, all conditions), and
experiments/h1_v2/analyze.py drops them from older scored files. Runs at the end of `run_v2_all.sh rest`, and
can be run by hand on an existing dataset (no rebuild needed).

Reasons
  double_label  COCO: one object labelled with two categories of the same supercategory (boxes IoU >
                DOUBLE_LABEL_IOU), e.g. pizza + sandwich on one flatbread. The ground truth is wrong.
                (B1 also drops these when the dataset is rebuilt.)
  part_whole    COCO: a component is a PART of another one (PART_OF pairs, box > PART_INSIDE inside it), e.g. the
                keyboard of a laptop. Naming the part reveals the whole, so no answer is both right and compliant.
  attr_overlap  attribute questions where one asked value gives away another: a pattern texture (striped,
                spotted...) asked together with color, or two values sharing a word ("red striped" / "red").

    python3 flag_excluded_units.py
"""
import json
from collections import Counter, defaultdict

from b1_ground_truth import double_labelled
from config import OUT_DIR, RAW
from utils import read_jsonl

OUT = OUT_DIR / "excluded_units.jsonl"
PART_OF = {("keyboard", "laptop"), ("microwave", "oven")}   # (part, whole) pairs seen in COCO units
PART_INSIDE = 0.8          # share of the part's box inside the whole's box
PATTERN = {"striped", "spotted", "checkered", "plaid", "dotted", "polka dotted", "lined", "patterned", "floral",
           "camouflage", "spotty", "zebra print", "leopard print"}


def inside(part, whole):
    px, py, pw, ph = part
    wx, wy, ww, wh = whole
    ix = max(0, min(px + pw, wx + ww) - max(px, wx))
    iy = max(0, min(py + ph, wy + wh) - max(py, wy))
    return ix * iy / (pw * ph) if pw * ph else 0.0


def attr_overlap(s):
    slots = {c["slot"]: c["value"] for c in s["components"] if c["type"] == "attribute" and c.get("slot")}
    pattern = [v for k, v in slots.items() if k != "color" and (v in PATTERN or any(p in v for p in PATTERN))]
    if pattern and "color" in slots:
        return f"pattern {pattern[0]!r} with color {slots['color']!r}"
    vals = list(slots.values())
    for i, a in enumerate(vals):
        for b in vals[i + 1:]:
            if set(a.split()) & set(b.split()):
                return f"values share a word: {a!r} / {b!r}"
    return None


def main():
    samples = read_jsonl(OUT_DIR / "samples.jsonl")
    forms = OUT_DIR / "samples_forms.jsonl"
    if forms.exists():
        samples += read_jsonl(forms)
    units = {}
    for s in samples:
        units.setdefault(s["validation"]["unit_id"], s)
    coco = [s for s in units.values() if s["source"]["dataset"] == "COCO"]
    want = {int(str(s["image"]).split("/")[-1].split(".")[0]) for s in coco}
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

    def flag(s, reason, detail):
        uid = s["validation"]["unit_id"]
        if uid not in flagged:
            flagged[uid] = {"unit_id": uid, "reason": reason, "detail": detail, "branch": s["branch"],
                            "image": s["image"]}

    for s in coco:
        img = int(str(s["image"]).split("/")[-1].split(".")[0])
        names = {cat_of[int(i)] for c in s["components"] for i in c.get("annotation_ids", []) if int(i) in cat_of}
        scs = {sc_of[n] for n in names}
        if len(scs) == 1:
            sc = next(iter(scs))
            by_cat = defaultdict(list)
            for a in by_img[img]:
                if sc_of.get(cat_of[a["id"]]) == sc:
                    by_cat[cat_of[a["id"]]].append(a)
            pairs = double_labelled(by_cat)
            if pairs:
                flag(s, "double_label", pairs)
                continue
        boxes = defaultdict(list)
        for a in by_img[img]:
            boxes[cat_of[a["id"]]].append(a["bbox"])
        for part, whole in PART_OF:
            if part in names and whole in names and any(inside(p, w) > PART_INSIDE
                                                        for p in boxes[part] for w in boxes[whole]):
                flag(s, "part_whole", [part, whole])
                break
    for s in units.values():
        if s["branch"] == "attribute":
            d = attr_overlap(s)
            if d:
                flag(s, "attr_overlap", d)

    with open(OUT, "w") as f:
        for r in flagged.values():
            f.write(json.dumps(r) + "\n")
    print(f"{len(flagged)} of {len(units)} units excluded -> {OUT}")
    print("by reason:", dict(Counter(r["reason"] for r in flagged.values())))
    print("by branch:", dict(Counter(r["branch"] for r in flagged.values())))
    for r in flagged.values():
        if r["reason"] != "double_label":
            print(f"  {r['reason']:12} {r['unit_id']:28} {r['detail']}")


if __name__ == "__main__":
    main()
