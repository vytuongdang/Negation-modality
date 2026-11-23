"""H3-B (Idea doc; PLAN.md Stage 3): did the model UNDERSTAND the negative instruction in each channel?

Conditions COMP_T / COMP_A / COMP_V deliver the instruction exactly as T / A / V do (same image, same item slot
after the text), but the text question is replaced by a 3-option multiple-choice question:
    "According to the instruction {where}, which one must not be mentioned? (A) .. (B) .. (C) .. Answer with the
     letter only."
where = "you were given" (T) / "in the audio" (A) / "written in the image" (V)  (wording from PLAN.md).
Options = the forbidden item + 2 others, all phrased as the {X} slot of the negation templates
(create_data/src/b3_negation.target_phrase: "the knife", "the number of cubes", "its material",
"what it is leaning against"). Other components of the same unit come first; units with fewer than 3 components
are filled with same-type items from other units of the SAME SOURCE (so CLEVR gets CLEVR shapes, not "cakes";
attribute / relation: another slot / relation, re-phrased for this unit's subject), never one whose phrase
equals another option. Option order is shuffled with a seed per
sample. Scored by parsing the letter (no judge): comp_correct, comp_choice.
"""
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "create_data" / "src"))
from b3_negation import target_phrase  # noqa: E402

COMP = {"COMP_T": "you were given", "COMP_A": "in the audio", "COMP_V": "written in the image"}
LETTERS = "ABC"
FRAME = ("According to the instruction {where}, which one must not be mentioned? {opts} "
         "Answer with the letter only.")


def _phrase(c):
    return target_phrase(None, c)


def build_pool(data):
    """(type, source) -> list of components over the whole dataset (fillers for small units)."""
    pool = {}
    seen = set()
    for s in data.values():
        for c in s["components"]:
            key = (c["type"], s["source"], c.get("subject"), c.get("slot"), c.get("relation"))
            if key not in seen:
                seen.add(key)
                pool.setdefault((c["type"], s["source"]), []).append(c)
    for v in pool.values():
        v.sort(key=lambda c: (str(c.get("subject")), str(c.get("slot")), str(c.get("relation"))))
    return pool


def options(s, pool):
    """-> (list of 3 phrases in shown order, index of the forbidden one). Deterministic per sample id."""
    rng = random.Random(f"comp-{s['id']}")
    comps = {c["id"]: c for c in s["components"]}
    target = comps[s["target"]]
    chosen = [_phrase(target)]
    others = [c for cid, c in sorted(comps.items()) if cid != s["target"]]
    rng.shuffle(others)
    for c in others:
        p = _phrase(c)
        if p not in chosen and len(chosen) < 3:
            chosen.append(p)
    if len(chosen) < 3:
        subj = target.get("subject")
        own = {c.get("subject") for c in comps.values()} | {c.get("slot") for c in comps.values()} \
            | {c.get("relation") for c in comps.values()}
        cands = []
        for c in pool[(target["type"], s["source"])]:
            if target["type"] in ("attribute", "relation"):
                k = c.get("slot") if target["type"] == "attribute" else c.get("relation")
                if k in own:
                    continue
                c = dict(c, subject=subj)          # another slot / relation of THIS subject
            elif c.get("subject") in own:
                continue
            cands.append(_phrase(c))
        cands = sorted(set(cands) - set(chosen))
        rng.shuffle(cands)
        chosen += cands[:3 - len(chosen)]
    assert len(chosen) == 3, s["id"]
    order = list(range(3))
    rng.shuffle(order)
    shown = [chosen[i] for i in order]
    return shown, order.index(0)


def mcq(s, cond, pool):
    shown, _ = options(s, pool)
    opts = " ".join(f"({L}) {o}" for L, o in zip(LETTERS, shown))
    return FRAME.format(where=COMP[cond], opts=opts)


def parse_choice(answer, shown):
    """Letter A/B/C from the answer; falls back to a unique option phrase in the text. None if unclear."""
    a = answer.strip()
    found = {m.group(1) for m in re.finditer(r"\(([ABC])\)", a)} | \
        {m.group(1) for m in re.finditer(r"^([ABC])(?=[.):]|$)", a)} | \
        {m.group(1) for m in re.finditer(r"\b(?:answer|option)(?: is)?:?\s*\(?([ABC])\b", a, re.I)}
    if len(found) == 1:
        return LETTERS.index(found.pop())
    low = a.lower()
    hits = [i for i, o in enumerate(shown) if o.lower() in low]
    return hits[0] if len(hits) == 1 else None


def score_comp(s, answer, pool):
    shown, gold = options(s, pool)
    choice = parse_choice(answer, shown)
    return {"comp_options": shown, "comp_gold": LETTERS[gold],
            "comp_choice": None if choice is None else LETTERS[choice],
            "comp_correct": choice == gold, "comp_parsed": choice is not None}
