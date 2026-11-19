"""B1: candidate (image, question, raw components) units from the source annotations, with every
annotation-level filter applied. Images are NOT downloaded here; step 2 (select_images.py) downloads candidates
split by split (val first) and applies the pixel and VLM checks until each source reaches its target.

Output: dataset/b1/candidates.jsonl, one row per candidate image:
    {source, split, image_id, image (path under dataset/), url | zip_member, width, height, license, coco_id,
     n_objects, units: [{branch, question, question_origin, original_answer, raw_components, protect_boxes, ...}]}

Annotation-level filters (config.py):
    all      image short side >= MIN_SHORT_SIDE (not CLEVR, 480x320 by design); object count <= MAX_OBJECTS;
             every component's object >= MIN_AREA_RATIO of the image and not touching the border;
             2-5 components; queried object unique in the image
    COCO     no crowd region anywhere in the image; license not NoDerivs; count only if every counted instance
             >= MIN_COUNT_INSTANCE_RATIO and <= MAX_COUNT instances per kind
    VAW      slots with exactly one explicit positive label; object name unique in the GQA scene graph
    GQA      verb-like relations, one object per relation, distinct object names; the subject must pass the
             3% / border rule, the related object only needs MIN_RELATION_OBJECT_RATIO (1%)
    VG       MAX_OBJECTS["VG"] counts "things": body parts, clothing and background (NOT_CLUTTER) are not counted
    CLEVR    attribute: the object is the only one of its shape; count: shapes present
Every unit also carries protect_boxes [[x, y, w, h, weight]]: the objects the visual instruction must not cover
(weight 1 = objects of the question, OTHER_WEIGHT = other annotated objects).
"""
import json
import random
import re
import zlib
from collections import Counter, defaultdict

from config import (B1_DIR, BORDER_PX, COCO_EXCLUDED_LICENSES, MAX_COMPONENTS, MAX_COUNT, MAX_OBJECTS,
                    MIN_AREA_RATIO, MIN_COMPONENTS, MIN_COUNT_INSTANCE_RATIO, MIN_RELATION_OBJECT_RATIO, MIN_SHORT_SIDE,
                    N_DISTRACTORS, OTHER_WEIGHT, RAW, SEED, SPLITS, URLS)
from utils import write_jsonl

# ---------------------------------------------------------------- English helpers

PLURAL_EXCEPTIONS = {
    "person": "people", "sheep": "sheep", "skis": "pairs of skis", "scissors": "pairs of scissors",
    "knife": "knives", "mouse": "computer mice", "tv": "TVs", "broccoli": "pieces of broccoli",
    "sports ball": "balls", "remote": "remotes", "glass": "glasses", "shelf": "shelves",
    "leaf": "leaves", "child": "children", "man": "men", "woman": "women",
}


def plural(noun):
    if noun in PLURAL_EXCEPTIONS:
        return PLURAL_EXCEPTIONS[noun]
    head, _, last = noun.rpartition(" ")
    if last in PLURAL_EXCEPTIONS:
        p = PLURAL_EXCEPTIONS[last]
    elif re.search(r"(s|x|z|ch|sh)$", last):
        p = last + "es"
    elif re.search(r"[^aeiou]y$", last):
        p = last[:-1] + "ies"
    else:
        p = last + "s"
    return f"{head} {p}".strip()


def join_list(items):
    items = list(items)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# COCO supercategory -> noun used in "What {noun} are present in the image?"
SUPERCAT_NOUN = {
    "vehicle": "vehicles", "outdoor": "outdoor objects", "animal": "animals", "accessory": "accessories",
    "sports": "pieces of sports equipment", "kitchen": "kitchen items", "food": "foods",
    "furniture": "pieces of furniture", "electronic": "electronic devices", "appliance": "appliances",
    "indoor": "indoor objects",
}

MALE = {"man", "boy", "guy", "gentleman"}
FEMALE = {"woman", "girl", "lady"}
PEOPLE = {"person", "child", "kid", "player", "skier", "surfer", "skateboarder", "snowboarder", "people", "batter",
          "catcher", "tennis player", "rider", "biker", "umpire", "pitcher", "spectator", "chef"}
PLURAL_SUBJECTS = {"people", "children", "kids", "men", "women"}
SINGULAR_IN_S = {"bus", "glass", "grass", "dress", "class", "lens", "cactus", "bonus", "canvas", "gas", "series",
                 "species", "news", "tennis", "chess", "mattress", "address", "compass", "cross", "boss", "moss"}


def is_plural(subject):
    """Plural object names such as 'cows', 'players', 'legs' (VAW/GQA use them); 'bus', 'glass' are singular."""
    last = subject.split()[-1]
    return (subject in PLURAL_SUBJECTS or last in PLURAL_SUBJECTS
            or (last.endswith("s") and not re.search(r"(ss|us|is)$", last) and last not in SINGULAR_IN_S))


def be(subject):
    return "are" if is_plural(subject) else "is"


def pronoun(subject):
    """Subject pronoun, verb 'be', possessive."""
    if is_plural(subject):
        return "they", "are", "their"
    if subject in MALE:
        return "he", "is", "his"
    if subject in FEMALE:
        return "she", "is", "her"
    if subject in PEOPLE:
        return "they", "are", "their"
    return "it", "is", "its"


# ---------------------------------------------------------------- geometry helpers

def area_ratio(w, h, W, H):
    return w * h / (W * H)


def touches_border(x, y, w, h, W, H):
    return x <= BORDER_PX or y <= BORDER_PX or x + w >= W - BORDER_PX or y + h >= H - BORDER_PX


def visible_ok(x, y, w, h, W, H):
    """Large enough and not cut by the image border."""
    return area_ratio(w, h, W, H) >= MIN_AREA_RATIO and not touches_border(x, y, w, h, W, H)


def rbox(b):
    return [round(float(t), 1) for t in b]


# ---------------------------------------------------------------- VG metadata and scene graphs

def load_vg_meta():
    vg = json.load(open(RAW / "vg/image_data.json"))
    return {str(x["image_id"]): x for x in vg}


def load_scene_graphs():
    sg = {}
    for split in ["val", "train"]:
        for k, g in json.load(open(RAW / f"gqa/{split}_sceneGraphs.json")).items():
            sg.setdefault(k, g)
    return sg


# ---------------------------------------------------------------- COCO: existence, closed-set existence, count

def b1_coco(split):
    data = json.load(open(RAW / f"coco/instances_{split}.json"))
    cats = {c["id"]: c for c in data["categories"]}
    imgs = {i["id"]: i for i in data["images"]}
    by_img = defaultdict(list)
    for a in data["annotations"]:
        by_img[a["image_id"]].append(a)
    stats = Counter()
    out = []
    for img_id, anns in by_img.items():
        info = imgs[img_id]
        W, H = info["width"], info["height"]
        if info["license"] in COCO_EXCLUDED_LICENSES:
            stats["license_noderivs"] += 1
            continue
        if min(W, H) < MIN_SHORT_SIDE:
            stats["small_image"] += 1
            continue
        if any(a["iscrowd"] for a in anns):
            stats["crowd"] += 1
            continue
        if len(anns) > MAX_OBJECTS["COCO"]:
            stats["too_many_objects"] += 1
            continue
        present_all = {cats[a["category_id"]]["name"] for a in anns}
        groups = defaultdict(lambda: defaultdict(list))  # supercat -> cat name -> anns
        for a in anns:
            c = cats[a["category_id"]]
            groups[c["supercategory"]][c["name"]].append(a)
        units = []
        for sc, by_cat in groups.items():
            if sc not in SUPERCAT_NOUN or not (MIN_COMPONENTS <= len(by_cat) <= MAX_COMPONENTS):
                continue
            # every category must have a clearly visible (largest) instance
            largest = {n: max(v, key=lambda x: x["area"]) for n, v in by_cat.items()}
            if not all(visible_ok(*largest[n]["bbox"], W, H) for n in by_cat):
                stats["component_small_or_cut"] += 1
                continue
            names = sorted(by_cat)
            protect = [[*rbox(a["bbox"]), 1.0 if cats[a["category_id"]]["name"] in by_cat else OTHER_WEIGHT]
                       for a in anns]

            def comp(n, kind):
                best = largest[n]
                c = {"type": kind, "subject": n, "annotation_ids": [x["id"] for x in by_cat[n]],
                     "bbox": rbox(best["bbox"]), "area_ratio": round(best["area"] / (W * H), 4)}
                if kind == "existence":
                    c["text"] = f"{n} exists"
                else:
                    k = len(by_cat[n])
                    c.update(value=k, text=f"there {'is' if k == 1 else 'are'} {k} {n if k == 1 else plural(n)}")
                return c

            noun = SUPERCAT_NOUN[sc]
            base = {"supercategory": sc, "protect_boxes": protect}
            units.append({**base, "branch": "existence", "question": f"What {noun} are present in the image?",
                          "question_origin": "template:exist_v1", "original_answer": names,
                          "raw_components": [comp(n, "existence") for n in names]})
            # closed-set existence: present categories + absent ones of the same supercategory ("absent" = no
            # annotation of that category at all; COCO labels are exhaustive). Every instruction form sees the
            # same names, so an affirmative form does not leak the answer.
            absent = sorted(c["name"] for c in cats.values()
                            if c["supercategory"] == sc and c["name"] not in present_all)
            if len(absent) >= N_DISTRACTORS:
                urng = random.Random(zlib.crc32(f"{SEED}-closed-{img_id}-{sc}".encode()))
                distractors = urng.sample(absent, N_DISTRACTORS)
                options = names + distractors
                urng.shuffle(options)
                units.append({**base, "branch": "existence_closed",
                              "question": f"Which of the following {noun} are present in the image: "
                                          f"{join_list(options)}?",
                              "question_origin": "template:exist_closed_v1", "options": options,
                              "distractors": distractors, "original_answer": names,
                              "raw_components": [comp(n, "existence") for n in names]})
            # count: every kind <= MAX_COUNT instances and every instance clearly visible; the question names the
            # kinds (as CLEVR count does), so an affirmative form adds nothing the question does not say
            if all(len(by_cat[n]) <= MAX_COUNT and all(x["area"] / (W * H) >= MIN_COUNT_INSTANCE_RATIO
                                                        for x in by_cat[n]) for n in names):
                units.append({**base, "branch": "count",
                              "question": f"How many {join_list(plural(n) for n in names)} are there in the image?",
                              "question_origin": "template:count_v2",
                              "original_answer": [f"{len(by_cat[n])} {n if len(by_cat[n]) == 1 else plural(n)}"
                                                  for n in names],
                              "raw_components": [comp(n, "count") for n in names]})
        if not any(u["branch"] == "existence" for u in units):
            stats["no_unit"] += 1
            continue
        out.append({"source": "COCO", "split": split, "image_id": img_id,
                    "image": f"images/coco/{info['file_name']}",
                    "url": URLS["coco_img"].format(split=split, id=img_id), "width": W, "height": H,
                    "license": info["license"], "flickr_url": info.get("flickr_url"),
                    "n_objects": len(anns), "units": units})
        stats["kept"] += 1
    return out, stats


# ---------------------------------------------------------------- CLEVR: attribute + count

CLEVR_SHAPES = ["cube", "sphere", "cylinder"]


def clevr_box(o):
    x, y, _ = o["pixel_coords"]
    r = 42 if o["size"] == "large" else 24
    return [x - r, y - r, 2 * r, 2 * r]


def b1_clevr(split):
    scenes = json.load(open(RAW / f"clevr/CLEVR_{split}_scenes.json"))["scenes"]
    questions = json.load(open(RAW / f"clevr/CLEVR_{split}_questions.json"))["questions"]
    ref_q = defaultdict(list)
    for q in questions:
        if len(ref_q[q["image_index"]]) < 3:
            ref_q[q["image_index"]].append({"question": q["question"], "answer": q["answer"],
                                            "question_index": q["question_index"]})
    stats, out = Counter(), []
    for s in scenes:
        idx, objs = s["image_index"], s["objects"]
        if len(objs) > MAX_OBJECTS["CLEVR"]:
            stats["too_many_objects"] += 1
            continue
        shape_count = Counter(o["shape"] for o in objs)
        base = {"reference_qa": ref_q[idx], "protect_boxes": [[*clevr_box(o), 1.0] for o in objs]}
        units = []
        for oi, o in enumerate(objs):  # attribute: the object is identified by its shape alone
            if shape_count[o["shape"]] != 1:
                continue
            slots = ["color", "size", "material"]
            units.append({**base, "branch": "attribute",
                          "question": f"Describe the {o['shape']} by its {join_list(slots)}.",
                          "question_origin": "template:clevr_attr_v1", "original_answer": [o[k] for k in slots],
                          "subject": o["shape"],
                          "raw_components": [{"type": "attribute", "subject": o["shape"], "slot": k, "value": o[k],
                                              "text": f"{o['shape']} is {o[k]}",
                                              "annotation_ids": [f"{idx}:obj{oi}:{k}"], "bbox": clevr_box(o)}
                                             for k in slots]})
        present = [sh for sh in CLEVR_SHAPES if shape_count[sh] > 0]
        if len(present) >= MIN_COMPONENTS:
            units.append({**base, "branch": "count",
                          "question": f"How many {join_list(plural(sh) for sh in present)} are there?",
                          "question_origin": "template:clevr_count_v1",
                          "original_answer": [f"{shape_count[sh]} {sh if shape_count[sh] == 1 else plural(sh)}"
                                              for sh in present],
                          "raw_components": [{"type": "count", "subject": sh, "value": shape_count[sh],
                                              "text": f"there {'is' if shape_count[sh] == 1 else 'are'} "
                                                      f"{shape_count[sh]} {sh if shape_count[sh] == 1 else plural(sh)}",
                                              "annotation_ids": [f"{idx}:obj{i}" for i, o in enumerate(objs)
                                                                 if o["shape"] == sh]} for sh in present]})
        if not any(u["branch"] == "attribute" for u in units):
            stats["no_unit"] += 1
            continue
        out.append({"source": "CLEVR", "split": split, "image_id": idx,
                    "image": f"images/clevr/{s['image_filename']}",
                    "zip_member": f"CLEVR_v1.0/images/{split}/{s['image_filename']}", "width": 480, "height": 320,
                    "license": "CC BY 4.0", "n_objects": len(objs), "units": units})
        stats["kept"] += 1
    return out, stats


# ---------------------------------------------------------------- VAW: attribute on real images

VAW_FILES = {"val": ["val.json"], "test": ["test.json"], "train": ["train_part1.json", "train_part2.json"]}
VAW_SLOTS = ["color", "material", "shape", "size", "texture", "state", "action"]  # parent types; "other" dropped
SLOT_PHRASE = {"action": "pose or activity"}


# Scene-graph names that are body parts, clothing, or background, not separate "things" in the scene; they do
# not count toward MAX_OBJECTS["VG"] (a lone man on grass already has man, shirt, pants, head, hand, grass, sky...).
NOT_CLUTTER = {
    "sky", "ground", "grass", "wall", "floor", "road", "street", "sidewalk", "water", "snow", "cloud", "clouds",
    "tree", "trees", "leaves", "leaf", "building", "field", "dirt", "sand", "ocean", "background", "hill",
    "mountain", "mountains", "bush", "bushes", "branch", "branches", "window", "windows", "door", "roof", "head",
    "hand", "hands", "hair", "leg", "legs", "ear", "ears", "eye", "eyes", "nose", "arm", "arms", "face", "tail",
    "mouth", "neck", "finger", "fingers", "foot", "feet", "wheel", "wheels", "tire", "tires", "letter", "letters",
    "number", "numbers", "logo", "shadow", "reflection", "line", "lines", "spot", "spots", "stripe", "stripes",
    "shirt", "pants", "jacket", "hat", "shorts", "shoe", "shoes", "helmet", "jeans", "coat", "cap", "glasses",
    "sunglasses", "dress", "sleeve", "collar", "belt", "sock", "socks", "boot", "boots", "handle", "light", "pole",
    "sign",
}


def n_things(g):
    return sum(o["name"] not in NOT_CLUTTER for o in g["objects"].values())


def vg_protect(g, main_ids=(), main_names=()):
    return [[o["x"], o["y"], o["w"], o["h"], 1.0 if (oid in main_ids or o["name"] in main_names) else OTHER_WEIGHT]
            for oid, o in g["objects"].items()]


def vg_image_ok(img, vg_meta, sg, stats):
    if img not in vg_meta or img not in sg:
        stats["no_scene_graph"] += 1
        return False
    m = vg_meta[img]
    if min(m["width"], m["height"]) < MIN_SHORT_SIDE:
        stats["small_image"] += 1
        return False
    if n_things(sg[img]) > MAX_OBJECTS["VG"]:
        stats["too_many_objects"] += 1
        return False
    return True


def vg_record(source, split, img, vg_meta, sg, units):
    m = vg_meta[img]
    return {"source": source, "split": split, "image_id": int(img), "image": f"images/vg/{img}.jpg",
            "url": m["url"], "width": m["width"], "height": m["height"], "license": "CC BY 4.0 (Visual Genome)",
            "coco_id": m.get("coco_id"), "n_objects": n_things(sg[img]),
            "n_scene_graph_objects": len(sg[img]["objects"]), "units": units}


def b1_vaw(split, vg_meta, sg):
    leaf_types = json.load(open(RAW / "vaw/attribute_types.json"))
    parents = json.load(open(RAW / "vaw/attribute_parent_types.json"))
    leaf_to_parent = {leaf: p for p, leaves in parents.items() for leaf in leaves}
    attr_to_slot = {a: leaf_to_parent.get(leaf, "other") for leaf, attrs in leaf_types.items() for a in attrs}
    insts = [i for f in VAW_FILES[split] for i in json.load(open(RAW / "vaw" / f))]
    stats = Counter()
    units_by_img = defaultdict(list)
    for inst in insts:
        img = str(inst["image_id"])
        if not vg_image_ok(img, vg_meta, sg, stats):
            continue
        g, m = sg[img], vg_meta[img]
        name = inst["object_name"]
        if sum(o["name"] == name for o in g["objects"].values()) != 1:  # the only object with that name
            stats["name_not_unique"] += 1
            continue
        x, y, w, h = inst["instance_bbox"]
        if not visible_ok(x, y, w, h, m["width"], m["height"]):
            stats["object_small_or_cut"] += 1
            continue
        by_slot = defaultdict(list)
        for a in inst["positive_attributes"]:
            by_slot[attr_to_slot.get(a, "other")].append(a)
        slots = [s for s in VAW_SLOTS if len(by_slot.get(s, [])) == 1][:MAX_COMPONENTS]
        if len(slots) < MIN_COMPONENTS:
            stats["lt2_slots"] += 1
            continue
        _, _, poss = pronoun(name)
        ar = round(area_ratio(w, h, m["width"], m["height"]), 4)
        units_by_img[img].append({
            "branch": "attribute", "instance_id": inst["instance_id"],
            "question": f"Describe the {name} by {poss} {join_list(SLOT_PHRASE.get(s, s) for s in slots)}.",
            "question_origin": "template:vaw_attr_v1", "original_answer": [by_slot[s][0] for s in slots],
            "subject": name, "protect_boxes": vg_protect(g, main_names={name}) + [[*rbox(inst["instance_bbox"]), 1.0]],
            "raw_components": [{"type": "attribute", "subject": name, "slot": s, "value": by_slot[s][0],
                                "text": f"{name} {be(name)} {by_slot[s][0]}", "annotation_ids": [inst["instance_id"]],
                                "bbox": rbox(inst["instance_bbox"]), "area_ratio": ar} for s in slots]})
    out = []
    for img, units in units_by_img.items():  # one object per image
        u = random.Random(zlib.crc32(f"{SEED}-vaw-{img}".encode())).choice(units)
        out.append(vg_record("VAW", split, img, vg_meta, sg, [u]))
    stats["kept"] = len(out)
    return out, stats


# ---------------------------------------------------------------- GQA: relation

EXTRA_RELATIONS = {"covered in", "covered with", "covered by", "filled with", "full of"}


def is_verb_relation(rel):
    return rel.split()[0].endswith("ing") or rel in EXTRA_RELATIONS


def b1_gqa(split, vg_meta, sg):
    qs = json.load(open(RAW / f"gqa/{split}_balanced_questions.json"))
    subj_q = {}  # (image, subject id) -> first balanced relation question about it (kept as the original QA)
    for qid, q in qs.items():
        if q["types"]["semantic"] != "rel" or not q["semantic"]:
            continue
        m = re.match(r".* \((\d+)\)$", q["semantic"][0]["argument"])
        if m:
            subj_q.setdefault((q["imageId"], m.group(1)), {"question_id": qid, "question": q["question"],
                                                           "answer": q["answer"], "fullAnswer": q["fullAnswer"]})
    del qs
    stats = Counter()
    units_by_img = defaultdict(list)
    for (img, sid), gq in subj_q.items():
        if not vg_image_ok(img, vg_meta, sg, stats):
            continue
        g = sg[img]
        objs, W, H = g["objects"], g["width"], g["height"]
        if sid not in objs:
            continue
        subj = objs[sid]
        if Counter(o["name"] for o in objs.values())[subj["name"]] != 1:
            stats["name_not_unique"] += 1
            continue
        if not visible_ok(subj["x"], subj["y"], subj["w"], subj["h"], W, H):
            stats["subject_small_or_cut"] += 1
            continue
        rels = defaultdict(set)
        for r in subj["relations"]:
            if is_verb_relation(r["name"]) and r["object"] in objs:
                rels[r["name"]].add(r["object"])
        seen, uniq = set(), []
        for rn, o in sorted(rels.items()):  # one object per relation, distinct names, clearly visible
            if len(o) != 1:
                continue
            oid = next(iter(o))
            ob = objs[oid]
            if ob["name"] in seen or ob["name"] == subj["name"] or \
                    area_ratio(ob["w"], ob["h"], W, H) < MIN_RELATION_OBJECT_RATIO:
                continue
            seen.add(ob["name"])
            uniq.append((rn, oid))
        if len(uniq) < MIN_COMPONENTS:
            stats["lt2_relations"] += 1
            continue
        uniq = uniq[:MAX_COMPONENTS]
        name = subj["name"]
        pron, verb, _ = pronoun(name)
        units_by_img[img].append({
            "branch": "relation", "subject_id": sid,
            "question": f"For the {name}, say {join_list(f'what {pron} {verb} {rn}' for rn, _ in uniq)}.",
            "question_origin": "template:gqa_rel_v1", "gqa_original_qa": gq,
            "original_answer": [objs[oid]["name"] for _, oid in uniq], "subject": name, "pronoun": [pron, verb],
            "protect_boxes": vg_protect(g, main_ids={sid} | {oid for _, oid in uniq}),
            "raw_components": [{"type": "relation", "subject": name, "relation": rn, "value": objs[oid]["name"],
                                "text": f"{name} {be(name)} {rn} {objs[oid]['name']}", "annotation_ids": [sid, oid],
                                "bbox": [objs[oid][k] for k in "xywh"],
                                "area_ratio": round(area_ratio(objs[oid]["w"], objs[oid]["h"], W, H), 4)}
                               for rn, oid in uniq]})
    out = []
    for img, units in units_by_img.items():  # one subject per image
        u = random.Random(zlib.crc32(f"{SEED}-gqa-{img}".encode())).choice(units)
        out.append(vg_record("GQA", split, img, vg_meta, sg, [u]))
    stats["kept"] = len(out)
    return out, stats


# ---------------------------------------------------------------- main

def main():
    vg_meta = load_vg_meta()
    print("loading GQA scene graphs (train + val)", flush=True)
    sg = load_scene_graphs()
    rows, report = [], {}
    for source in ["COCO", "CLEVR", "VAW", "GQA"]:
        for split in SPLITS[source]:
            fn = {"COCO": lambda: b1_coco(split), "CLEVR": lambda: b1_clevr(split),
                  "VAW": lambda: b1_vaw(split, vg_meta, sg), "GQA": lambda: b1_gqa(split, vg_meta, sg)}[source]
            cand, stats = fn()
            report[f"{source}/{split}"] = dict(stats)
            print(f"{source}/{split}: {len(cand)} candidate images, "
                  f"{dict(Counter(u['branch'] for c in cand for u in c['units']))}; filtered: {dict(stats)}",
                  flush=True)
            rows += cand
    write_jsonl(B1_DIR / "candidates.jsonl", rows)
    (B1_DIR / "candidates_report.json").write_text(json.dumps(report, indent=1))
    print(f"{len(rows)} candidate images -> {B1_DIR / 'candidates.jsonl'}")


if __name__ == "__main__":
    main()
