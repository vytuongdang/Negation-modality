"""H3-B comprehension v2 (Vy approved 2026-11-24 10:44; design in paper/ideas.md).

COMP v1 (comp.py, conds COMP_T/A/V) put the multiple-choice question BEFORE the instruction, so the models applied
the instruction to their MCQ answer (Qwen-7B COMP_T 18.7% < chance): it measured obedience, not comprehension.
v1 results stay as history (comp_pilot200_COMP_T-COMP_A-COMP_V.jsonl).

v2 quotes the instruction as content and delivers it FIRST, then asks about it (same image as T / A / V):
  COMP2_T  image, "Here is an instruction:", <instruction text item>, MCQ
  COMP2_A  image, "Here is a spoken instruction:", <instruction audio item>, MCQ
  COMP2_V  negation image (instruction written in it), MCQ
  MCQ = "Which item does this instruction say must not be mentioned? (A) .. (B) .. (C) .. Answer with the letter only."
  (V: "... the instruction written in the image ..."). Options and their order = v1 (comp.options, seeded per sample).
Verbatim repeat per channel (scored like OCR: normalised exact match + character similarity):
  REP_T    image, "Here is an instruction:", <text>, "Repeat this instruction word for word, and say nothing else."
  REP_A    image, "Here is a spoken instruction:", <audio>, "Repeat the spoken instruction word for word, and say nothing else."
  OCR_V    (exists in run_h1v2) negation image, "What does the text in the white box say? Copy it exactly ..."

This is a WRAPPER around run_h1v2.py (no edit to that file or comp.py, so a running job that reloads them is
untouched): it registers the new conditions in memory and calls run_h1v2.main(). Same flags as run_h1v2.py, e.g.
  python run_comp2.py --model qwen2.5-omni-7b --pilot 200 --conds COMP2_T COMP2_A COMP2_V REP_T REP_A OCR_V
Outputs: results/<model>/comp_<tag>.jsonl (COMP2_*), ocr_<tag>.jsonl (REP_*, OCR_V); tag includes the cond names.
"""
import sys

import comp
import run_h1v2 as R

INTRO = {"T": "Here is an instruction:", "A": "Here is a spoken instruction:"}
Q_TA = "Which item does this instruction say must not be mentioned?"
Q_V = "Which item does the instruction written in the image say must not be mentioned?"
REP = {"REP_T": "Repeat this instruction word for word, and say nothing else.",
       "REP_A": "Repeat the spoken instruction word for word, and say nothing else."}
COMP2 = {"COMP2_T": "v2-T", "COMP2_A": "v2-A", "COMP2_V": "v2-V"}


def mcq2(s, cond):
    shown, _ = comp.options(s, R.POOL)
    opts = " ".join(f"({L}) {o}" for L, o in zip(comp.LETTERS, shown))
    return f"{Q_V if cond == 'COMP2_V' else Q_TA} {opts} Answer with the letter only."


_old_inputs = R.inputs


def inputs(s, cond):
    i = s["negation"]
    if cond == "COMP2_T":
        return s["image"], [("text", INTRO["T"]), ("text", i), ("text", mcq2(s, cond))]
    if cond == "COMP2_A":
        return s["image"], [("text", INTRO["A"]), ("audio", s["negation_audio"]), ("text", mcq2(s, cond))]
    if cond == "COMP2_V":
        return s["negation_image"], [("text", mcq2(s, cond))]
    if cond == "REP_T":
        return s["image"], [("text", INTRO["T"]), ("text", i), ("text", REP[cond])]
    if cond == "REP_A":
        return s["image"], [("text", INTRO["A"]), ("audio", s["negation_audio"]), ("text", REP[cond])]
    return _old_inputs(s, cond)


def register():
    R.inputs = inputs
    R.EXTRA += [c for c in list(COMP2) + list(REP) if c not in R.EXTRA]
    comp.COMP.update(COMP2)          # routed to comp.score_comp (letter parsing, same options as v1)
    R.OCR.update(REP)                # routed to ocr_score against the instruction text


if __name__ == "__main__":
    register()
    sys.exit(R.main())
