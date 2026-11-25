"""Shared code for H2 (direct conflict) and H3-A (modality dominance): layouts, selection, input builder.

Idea (PLAN.md stage 4): the QUESTION (a positive request) and the INSTRUCTION arrive through DIFFERENT channels.
Three channel pairs, both directions = 6 layouts. A condition is named  <family>_q<Q>_i<I>  with Q = channel of the
question, I = channel of the instruction (T text, A audio, V written in the image):

    layout    input (image first, then the items in order)                                  already in H1 as
    qT_iA     original image | text question, audio instruction                              A
    qA_iT     original image | audio question, text instruction                              new
    qT_iV     image with the instruction written in it | text question                       V
    qV_iT     image with the QUESTION written in it | text instruction                       new
    qA_iV     image with the instruction written in it | audio question                      new
    qV_iA     image with the QUESTION written in it | audio instruction                      new

Families:
    H2   the instruction is the NEGATION of the unit ("Do not mention the dog.")
    H3A  the same 6 layouts with a conflicting POSITIVE instruction instead (no negation word). Default = the
         affirmative form of H4 ("Mention only the cat."); `--h3a_form exceptive` uses "Mention everything except
         the dog." (exists for every branch, names the target). Only units that have that form are used.
Both families are scored like H1 (violation = any mention of the forbidden component, among the units whose own C0
answer states it), so the channel in which the instruction wins can be read per layout and compared across the two
families. qT_iA and qT_iV are identical inputs to H1's A / V (and to H4's affirmative A / V): their answers come from
the answer cache.

New assets (made by create_data/src/h2_assets.py, listed in dataset/h2/questions.jsonl, one row per unit): the question
as audio (CosyVoice2, same 2 voices, Whisper check) and the question written in the image (same box style as the
visual instruction, placed away from the question's objects).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from omnilib import DATA, load_dataset, read_jsonl

HERE = Path(__file__).resolve().parent
H2_DIR = DATA / "h2"
QFILE = H2_DIR / "questions.jsonl"

CHANNELS = ("T", "A", "V")
LAYOUTS = [("T", "A"), ("A", "T"), ("T", "V"), ("V", "T"), ("A", "V"), ("V", "A")]   # (question, instruction)
FAMILIES = ("H2", "H3A")
H3A_FORMS = ("affirmative", "exceptive")


def cond_name(family, q, i):
    return f"{family}_q{q}_i{i}"


ALL_CONDS = [cond_name(f, q, i) for f in FAMILIES for q, i in LAYOUTS]


def parse_cond(cond):
    """'H2_qA_iT' -> ('H2', 'A', 'T')"""
    fam, q, i = cond.split("_")
    return fam, q[1], i[1]


def load_questions():
    return {r["unit_id"]: r for r in read_jsonl(QFILE)} if QFILE.exists() else {}


def select_samples(pilot, all_units, seed=42, pilot_list=None):
    """The same question selection as H1 (run_h1v2.select): `pilot` units per branch, or every unit; or the FIXED list
    of sample ids in `pilot_list` (experiments/h1_v2/pilot1000.json, made by make_pilot.py).
    -> (negative samples, {negative sample id: {form: form sample}})"""
    data_all, data = load_dataset(keep_excluded=True), load_dataset()
    if pilot_list:
        ids = [r["sample_id"] for r in json.loads(Path(pilot_list).read_text())["samples"]]
        missing = [i for i in ids if i not in data]
        if missing:
            raise SystemExit(f"{len(missing)} ids of {pilot_list} are not in the dataset (first: {missing[0]})")
        samples = [data[i] for i in ids]
    else:
        sys.path.insert(0, str(HERE.parent / "h1_v2"))
        import run_h1v2
        samples = [s for s in run_h1v2.select(data_all, None if all_units else pilot, seed) if s["id"] in data]
    forms = defaultdict(dict)
    for s in data.values():
        if s["form"] != "negative":
            forms[s["pair_of"]][s["form"]] = s
    return samples, forms


def list_tag(pilot, all_units, pilot_list):
    """'pilot1000' for a list file named pilot1000.json, 'pilot40' for --pilot 40, 'all' for --all."""
    return Path(pilot_list).stem if pilot_list else ("all" if all_units else f"pilot{pilot}")


def instruction_sample(s, forms, family, h3a_form="affirmative"):
    """The sample whose instruction is delivered: the negative sample itself (H2) or its positive form (H3-A)."""
    return s if family == "H2" else forms.get(s["id"], {}).get(h3a_form)


def needs_assets(q_channel):
    return q_channel in ("A", "V")


def h2_inputs(s, ins, q, Q, I):
    """-> (image relative to dataset/, items). s: the negative sample (question, original image); ins: the sample
    whose instruction is delivered; q: its row in questions.jsonl (question audio / image), None for Q = T."""
    if (Q, I) == ("T", "A"):
        return s["image"], [("text", s["question"]), ("audio", ins["negation_audio"])]
    if (Q, I) == ("A", "T"):
        return s["image"], [("audio", q["q_audio"]), ("text", ins["negation"])]
    if (Q, I) == ("T", "V"):
        return ins["negation_image"], [("text", s["question"])]
    if (Q, I) == ("V", "T"):
        return q["q_image"], [("text", ins["negation"])]
    if (Q, I) == ("A", "V"):
        return ins["negation_image"], [("audio", q["q_audio"])]
    if (Q, I) == ("V", "A"):
        return q["q_image"], [("audio", ins["negation_audio"])]
    raise ValueError((Q, I))
