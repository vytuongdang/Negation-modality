"""B3 (step 5): pick one component as the negation target and build the text negation instruction.

Input: dataset/b2/units.jsonl. The target is drawn (seeded) uniformly among the unit's components (all of them
VLM-checked in step 2), skipping alias collisions. Output: dataset/out/samples.jsonl (doc schema).

Fixed templates only (7 frames, doc section 6); question first, negation right after, in one text turn.
A seeded subset of units is expanded to all 7 templates to measure sensitivity to wording.
`prompt` holds only the question and `negation_text` only the instruction; the evaluation step joins them
(text: question then negation, per `negation.position`), which keeps the negation swappable for audio later.
"""
import random
import re
import zlib
from collections import Counter

from b1_ground_truth import SLOT_PHRASE, plural, pronoun
from config import B2_DIR, FULL_TEMPLATE_SUBSET_RATE, OUT_DIR, SEED
from utils import read_jsonl, write_jsonl

# id -> (cue, explicit negation word?, frame)
TEMPLATES = {
    "N1": ("not", True, "Do not mention {X}."),
    "N2": ("n't", True, "Don't say anything about {X}."),
    "N3": ("never", True, "Never refer to {X}."),
    "N4": ("no", True, "Make no mention of {X}."),
    "N5": ("without", True, "Answer without mentioning {X}."),
    "N6": ("avoid", False, "Avoid mentioning {X}."),            # verb with negative meaning
    "N7": ("leave_out", False, "Leave {X} out of your answer."),  # exclusion without a negation word (not the "exceptive" form)
}


def target_phrase(unit, c):
    """The {X} slot, which names the slot (not the value) for attribute and relation."""
    if c["type"] == "existence":
        return f"the {c['subject']}"
    if c["type"] == "count":
        return f"the number of {plural(c['subject'])}"
    if c["type"] == "attribute":
        _, _, poss = pronoun(c["subject"])
        return f"{poss} {SLOT_PHRASE.get(c['slot'], c['slot'])}"
    if c["type"] == "relation":
        pron, be, _ = pronoun(c["subject"])
        return f"what {pron} {be} {c['relation']}"
    raise ValueError(c["type"])


def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def collides(target, others):
    """True if the target's surface forms overlap with what another component would legitimately say
    (e.g. target 'dog' vs required 'hot dog'), which would make the forbidden mention undetectable."""
    if target["type"] == "count":
        names = set(target.get("subject_aliases", []))
        return any(names & set(o.get("subject_aliases", [])) for o in others)
    tforms = set(target["aliases"])
    for o in others:
        oforms = set(o["aliases"])
        if tforms & oforms:
            return True
        # multi-word forms: a single-word target form contained in another component's form
        for t in tforms:
            if any(t != f and re.search(rf"\b{re.escape(t)}\b", f) for f in oforms):
                return True
    return False


def distractor_collides(unit):
    """Closed-set existence: an absent option whose name is also an alias of a present component (e.g. distractor
    'bicycle' vs motorcycle alias 'bike' is fine, but distractor 'truck' vs a car alias 'truck' is not) would make
    a correct "no truck" answer look like a mention of the car."""
    forms = {a for c in unit["components"] for a in c["aliases"]}
    return any(d in forms or plural(d) in forms for d in unit.get("distractors", []))


def build_sample(unit, target_id, tid, seed, sweep):
    comps = unit["components"]
    target = next(c for c in comps if c["id"] == target_id)
    cue, explicit, frame = TEMPLATES[tid]
    instruction = frame.format(X=target_phrase(unit, target))
    extra = {k: unit[k] for k in ("supercategory", "subject", "reference_qa", "gqa_original_qa", "options", "distractors",
                                 "protect_boxes", "image_meta", "vlm_checks") if k in unit}
    return {
        "sample_id": None,
        "branch": unit["branch"],
        "source": unit["source"],
        "image": unit["image"],
        "question": unit["question"],
        "question_origin": unit["question_origin"],
        "original_answer": unit["original_answer"],
        "components": comps,
        "negation": {
            "target": target_id,
            "candidates": [c["id"] for c in comps],
            "modality": "text",
            "form": "negative",
            "template_id": tid,
            "cue": cue,
            "explicit_negation_word": explicit,
            "instruction": instruction,
            "position": "after_question",
            "seed": seed,
            "template_sweep": sweep,
        },
        "prompt": unit["question"],
        "negation_text": instruction,
        "expected": {"required": [c["id"] for c in comps if c["id"] != target_id], "forbidden": [target_id]},
        "validation": {
            "unit_id": unit["unit_id"],
            "b2_model": unit["b2"]["model"],
            "b2_prompt_version": unit["b2"]["prompt_version"],
            "b2_question_ok": unit["b2"]["question_ok"],
            "auto_check": "pass",
            "human_verify_selected": unit["b2"]["human_verify_selected"],
            "human_verified": False,
            "baseline": {},
        },
        "extra": extra,
    }


def auto_check(s):
    """Doc B3 step 5: the instruction targets exactly one component and something is still required."""
    return len(s["expected"]["forbidden"]) == 1 and len(s["expected"]["required"]) >= 1 \
        and s["negation"]["target"] in s["negation"]["candidates"]


def main():
    units = read_jsonl(B2_DIR / "units.jsonl")
    rng = random.Random(SEED)
    sweep_ids = set(rng.sample([u["unit_id"] for u in units], round(len(units) * FULL_TEMPLATE_SUBSET_RATE)))

    samples, stats = [], Counter()
    for u in units:
        if distractor_collides(u):
            stats["unit_dropped_distractor_alias"] += 1
            continue
        seed = zlib.crc32(f"{SEED}-{u['unit_id']}".encode())
        urng = random.Random(seed)
        order = [c["id"] for c in u["components"]]
        urng.shuffle(order)
        # first target in seeded order whose surface forms do not collide with a required component
        target_id = None
        for cid in order:
            t = next(c for c in u["components"] if c["id"] == cid)
            if not collides(t, [c for c in u["components"] if c["id"] != cid]):
                target_id = cid
                break
            stats["target_skipped_collision"] += 1
        if target_id is None:
            stats["unit_dropped_collision"] += 1
            continue
        sweep = u["unit_id"] in sweep_ids
        tids = list(TEMPLATES) if sweep else [urng.choice(list(TEMPLATES))]
        for tid in tids:
            s = build_sample(u, target_id, tid, seed, sweep)
            if not auto_check(s):
                stats["sample_failed_auto_check"] += 1
                continue
            samples.append(s)

    counters = Counter()
    for s in samples:
        key = f"{s['source']['dataset'].lower()}-{s['branch']}"
        counters[key] += 1
        s["sample_id"] = f"{key}-{counters[key]:06d}"
    write_jsonl(OUT_DIR / "samples.jsonl", samples)
    print(f"{len(samples)} samples from {len(units)} units; {dict(stats)}")
    print(Counter((s["source"]["dataset"], s["branch"]) for s in samples))
    print(Counter(s["negation"]["template_id"] for s in samples))


if __name__ == "__main__":
    main()
