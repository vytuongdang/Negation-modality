"""B2: turn B1 ground truth into verified atomic components.

Components themselves come straight from the annotation (COCO/VAW/CLEVR/GQA scene graph), so the LLM's job is
to (a) generate alternative names used later for scoring, and (b) review atomicity and the question.
Every component is cross-checked against the original annotation IDs; a stratified sample (config.HUMAN_VERIFY_RATE, 10%) is exported
for human verification.
"""
import csv
import html
import json
import random
from collections import Counter, defaultdict

from config import (B1_DIR, B2_DIR, B2_MODEL, B2_PROMPT_VERSION, HUMAN_VERIFY_RATE, MIN_COMPONENTS, RAW, ROOT,
                    SEED, SPLITS)
from utils import read_jsonl, write_jsonl

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

PROMPT = """You are helping build a visual question answering benchmark. Each item has a question about an image and a list of ground-truth components (atomic facts) taken from human annotation.

Question: {question}
Components:
{components}

For each component, give "aliases": other words or short phrases that a model's answer could use to refer to the component's VALUE (shown in square brackets), so that a string matcher can detect when the value is mentioned.
- existence and relation: other names for exactly that object (synonyms, plural form), e.g. couch -> sofa, couches.
- attribute: near-synonyms of exactly that attribute value, e.g. wooden -> wood, made of wood.
- count: give "aliases": [] (number words are added automatically) and give "subject_aliases": other names for the counted object.
Rules for every alias:
- It must be specific: a word someone would use only for this value. Never give broader categories (animal, furniture, clothing, object, item, shape, material, color, vehicle).
- Never give a different thing that is merely related (racket -> bat is wrong, shorts -> trousers is wrong, cloth -> wool is wrong).
- Never give a word that names another component's value. Do not repeat the value itself. At most 5 aliases; an empty list is fine.

Also judge:
- "atomic": true if the component is a single fact (one object, one attribute value, one relation).
- "question_ok": true if the question is natural English and asks for every component.
- "issues": short note of any problem, or "".

Reply with JSON only:
{{"components": [{{"id": "C1", "atomic": true, "aliases": ["..."], "subject_aliases": ["..."]}}], "question_ok": true, "issues": ""}}"""


def render_components(comps):
    lines = []
    for c in comps:
        if c["type"] == "existence":
            v = c["subject"]
        elif c["type"] == "count":
            v = str(c["value"])
        else:
            v = c["value"]
        lines.append(f'- {c["id"]} ({c["type"]}): {c["text"]}  [{v}]')
    return "\n".join(lines)


# ---------------------------------------------------------------- annotation cross-check

def load_annotation_index():
    """Annotation lookup for every split used (config.SPLITS)."""
    from b1_ground_truth import VAW_FILES
    coco_ids = {}
    for split in SPLITS["COCO"]:
        for a in json.load(open(RAW / f"coco/instances_{split}.json"))["annotations"]:
            coco_ids[a["id"]] = a["image_id"]
    vaw = {inst["instance_id"]: inst for split in SPLITS["VAW"] for f in VAW_FILES[split]
           for inst in json.load(open(RAW / "vaw" / f))}
    gqa = {}
    for split in ["val", "train"]:
        for k, g in json.load(open(RAW / f"gqa/{split}_sceneGraphs.json")).items():
            gqa.setdefault(k, g)
    clevr = {s["image_index"]: s for split in SPLITS["CLEVR"]
             for s in json.load(open(RAW / f"clevr/CLEVR_{split}_scenes.json"))["scenes"]}
    return {"coco": coco_ids, "vaw": vaw, "gqa": gqa, "clevr": clevr}


def check_component(unit, c, idx):
    """True if every annotation ID of the component resolves to the source annotation on this image."""
    src = unit["source"]
    ds, img = src["dataset"], src["image_id"]
    ids = c.get("annotation_ids") or []
    if not ids:
        return False
    if ds == "COCO":
        return all(idx["coco"].get(i) == img for i in ids)
    if ds == "VAW":
        inst = idx["vaw"].get(ids[0])
        return inst is not None and str(inst["image_id"]) == str(img) and c["value"] in inst["positive_attributes"]
    if ds == "GQA":
        g = idx["gqa"].get(str(img))
        if g is None or not all(i in g["objects"] for i in ids):
            return False
        subj, obj = ids
        return any(r["object"] == obj and r["name"] == c["relation"] for r in g["objects"][subj]["relations"])
    if ds == "CLEVR":
        objs = idx["clevr"][img]["objects"]
        for i in ids:
            parts = i.split(":")
            o = int(parts[1][3:])
            if o >= len(objs):
                return False
            if c["type"] == "attribute" and objs[o][parts[2]] != c["value"]:
                return False
            if c["type"] == "count" and objs[o]["shape"] != c["subject"]:
                return False
        return c["type"] != "count" or len(ids) == c["value"]
    return False


# ---------------------------------------------------------------- alias post-processing

# Broad words that would match far more than one component; never valid as an alias.
GENERIC = {
    "object", "objects", "thing", "things", "item", "items", "animal", "animals", "furniture", "clothing", "clothes",
    "garment", "garments", "outfit", "material", "materials", "shape", "shapes", "color", "colour", "vehicle",
    "vehicles", "appliance", "appliances", "food", "device", "equipment", "3d shape", "existence", "count",
    "attribute", "relation", "livestock", "accessory", "accessories", "it", "them", "one of them",
}
COUNT_EXTRA = {1: ["a single", "single", "only one", "just one"], 2: ["a pair of", "a couple of", "both"]}


def clean_aliases(aliases, value):
    out, seen = [], {str(value).lower()}
    for a in aliases or []:
        if not isinstance(a, str):
            continue
        a = a.strip().lower()
        if a and a not in seen and a not in GENERIC and len(a) <= 40 and not any(g in a.split() for g in ("object", "thing", "item")):
            seen.add(a)
            out.append(a)
    return out


def drop_shared_aliases(comps):
    """An alias used by two components (or equal to another component's value) cannot tell them apart; drop it.
    Count values are exempt: two kinds can share a number and are told apart by their subject_aliases."""
    key = "aliases"
    owners = defaultdict(set)
    for c in comps:
        for a in c[key]:
            owners[a].add(c["id"])
    for c in comps:
        if c["type"] == "count":
            continue
        own_value = str(c["subject"] if c["type"] == "existence" else c["value"]).lower()
        c[key] = [a for a in c[key] if len(owners[a]) == 1 or a == own_value]


def base_aliases(c):
    """Deterministic aliases added regardless of the LLM."""
    from b1_ground_truth import plural
    if c["type"] == "existence":
        return [c["subject"], plural(c["subject"])]
    if c["type"] == "count":
        n = c["value"]
        return [str(n)] + ([NUMBER_WORDS[n]] if n < len(NUMBER_WORDS) else []) + COUNT_EXTRA.get(n, [])
    return [c["value"]]


# ---------------------------------------------------------------- human verification export

def export_human_verify(units, rng):
    strata = defaultdict(list)
    for u in units:
        strata[(u["source"]["dataset"], u["branch"])].append(u)
    picked = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda u: u["unit_id"])
        k = max(2, round(len(group) * HUMAN_VERIFY_RATE))
        picked += rng.sample(group, min(k, len(group)))
    out = B2_DIR / "human_verify"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["unit_id", "dataset", "branch", "image", "question", "components",
            "correct_vs_image", "atomic", "nothing_missing", "aliases_ok", "note"]
    with open(out / "human_verify.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for u in picked:
            comps = " | ".join(f'{c["id"]}: {c["text"]} (aliases: {", ".join(c["aliases"])})' for c in u["components"])
            w.writerow([u["unit_id"], u["source"]["dataset"], u["branch"], u["image"], u["question"], comps,
                        "", "", "", "", ""])
    # a simple local gallery so the reviewer can see each image next to its components
    rows = []
    for u in picked:
        comps = "".join(f"<li><b>{c['id']}</b> {html.escape(c['text'])}<br><small>aliases: "
                        f"{html.escape(', '.join(c['aliases']))}</small></li>" for c in u["components"])
        rows.append(f"<tr><td><img src='../../{u['image']}' style='max-width:420px'></td><td><code>{u['unit_id']}</code>"
                    f"<p>{html.escape(u['question'])}</p><ul>{comps}</ul></td></tr>")
    (out / "human_verify.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>B2 human verify</title>"
        "<style>body{font-family:sans-serif}td{vertical-align:top;padding:8px;border-bottom:1px solid #ccc}</style>"
        "<p>Score each row in human_verify.csv: correct_vs_image, atomic, nothing_missing, aliases_ok (1/0).</p>"
        f"<table>{''.join(rows)}</table>")
    return picked


def main():
    from llm import LocalLLM, parse_json

    units = read_jsonl(B1_DIR / "units.jsonl")
    for u in units:
        for k, c in enumerate(u["raw_components"], 1):
            c["id"] = f"C{k}"

    llm = LocalLLM(B2_MODEL)
    prompts = [PROMPT.format(question=u["question"], components=render_components(u["raw_components"])) for u in units]
    replies = llm.chat(prompts)
    # one retry for replies that are not valid JSON
    bad = [i for i, r in enumerate(replies) if parse_json(r) is None]
    if bad:
        for i, r in zip(bad, llm.chat([prompts[i] for i in bad])):
            replies[i] = r

    idx = load_annotation_index()
    kept, stats = [], Counter()
    for u, reply in zip(units, replies):
        parsed = parse_json(reply) or {}
        by_id = {c.get("id"): c for c in parsed.get("components", []) if isinstance(c, dict)}
        comps = []
        for c in u["raw_components"]:
            if not check_component(u, c, idx):
                stats["component_dropped_no_annotation"] += 1
                continue
            llm_c = by_id.get(c["id"], {})
            llm_aliases = [] if c["type"] == "count" else list(llm_c.get("aliases", []))[:5]
            # rule-based aliases (the value itself, plural, number words) are never filtered
            base = [a.lower() for a in base_aliases(c)]
            aliases = list(dict.fromkeys(base + clean_aliases(llm_aliases, "")))
            comp = dict(c)
            comp["aliases"] = aliases
            if c["type"] == "count":
                comp["subject_aliases"] = clean_aliases([c["subject"]] + list(llm_c.get("subject_aliases", [])), "")
            comp["llm_atomic"] = llm_c.get("atomic")
            if llm_c.get("atomic") is False:
                stats["flag_non_atomic"] += 1
            comps.append(comp)
        if len(comps) < MIN_COMPONENTS:
            stats["unit_dropped_lt2_components"] += 1
            continue
        drop_shared_aliases(comps)
        u = {k: v for k, v in u.items() if k != "raw_components"}
        u["components"] = comps
        u["b2"] = {"model": B2_MODEL, "prompt_version": B2_PROMPT_VERSION, "llm_parsed": bool(parsed),
                   "question_ok": parsed.get("question_ok"), "issues": parsed.get("issues", ""),
                   "auto_check": "pass", "raw_reply": reply}
        stats["llm_unparsed"] += not parsed
        kept.append(u)

    rng = random.Random(SEED)
    picked = export_human_verify(kept, rng)
    picked_ids = {u["unit_id"] for u in picked}
    for u in kept:
        u["b2"]["human_verify_selected"] = u["unit_id"] in picked_ids
    write_jsonl(B2_DIR / "units.jsonl", kept)
    print(f"kept {len(kept)}/{len(units)} units; {dict(stats)}; human-verify sample: {len(picked)}")
    print("review files:", (B2_DIR / "human_verify").relative_to(ROOT))


if __name__ == "__main__":
    main()
