"""B3b (H4): affirmative and exceptive versions of each negation, same target and same expected answer.

Run after the baseline filter (reads dataset/out/samples.jsonl). One affirmative and one exceptive record per
unit, paired with the unit's negative sample (`pair_of`); for template-sweep units the N1 sample is the pair.

    form         cue                       example (target = the dog)
    negative     "not" etc. (N1-N7, B3)    Do not mention the dog.
    exceptive    names target, no "not"    Mention everything except the dog.
    affirmative  names only what remains   Mention only the cat and the chicken.

Affirmative is made only where it adds nothing the question does not already say:
    existence_closed  the question lists every option (present + distractors), so "Mention only <the other
                      options>" leaks nothing
    count             the question names every kind (v2 template)
    attribute         the question names every slot ("by its color, material, and state")
    relation          the question names every relation slot
    existence (open)  NO affirmative: naming what to mention would give the answer away
Exceptive is made for every branch; its {X} is the same phrase the negative uses.

Output: dataset/out/samples_forms.jsonl (full records in the samples.jsonl schema, negation.form set).
"""
from collections import Counter

from b1_ground_truth import join_list
from b3_negation import target_phrase
from config import OUT_DIR
from utils import read_jsonl, write_jsonl

EXCEPTIVE = "Mention everything except {X}."
AFFIRMATIVE = "Mention only {Y}."
AFFIRMATIVE_BRANCHES = {"existence_closed", "count", "attribute", "relation"}


def affirmative_phrase(s):
    comps = {c["id"]: c for c in s["components"]}
    target = comps[s["negation"]["target"]]
    if s["branch"] == "existence_closed":
        # every listed option except the target, in the question's order (distractors included)
        return join_list(f"the {o}" for o in s["extra"]["options"] if o != target["subject"])
    return join_list(target_phrase(s, comps[i]) for i in s["expected"]["required"])


def make(s, form, instruction):
    r = {k: v for k, v in s.items() if k not in ("negation", "negation_text", "sample_id")}
    r["sample_id"] = f"{s['sample_id']}-{form[:3]}"
    r["pair_of"] = s["sample_id"]
    r["negation"] = dict(s["negation"], form=form, template_id=f"{form[:3].upper()}1",
                         cue="except" if form == "exceptive" else "only",
                         explicit_negation_word=False, instruction=instruction, template_sweep=False)
    r["negation_text"] = instruction
    return r


def main():
    samples = read_jsonl(OUT_DIR / "samples.jsonl")
    rep = {}  # unit -> its negative sample (N1 for template-sweep units)
    for s in samples:
        u = s["validation"]["unit_id"]
        if u not in rep or s["negation"]["template_id"] == "N1":
            rep[u] = s
    out, stats = [], Counter()
    for s in rep.values():
        comps = {c["id"]: c for c in s["components"]}
        x = target_phrase(s, comps[s["negation"]["target"]])
        out.append(make(s, "exceptive", EXCEPTIVE.format(X=x)))
        stats[(s["branch"], "exceptive")] += 1
        if s["branch"] in AFFIRMATIVE_BRANCHES:
            out.append(make(s, "affirmative", AFFIRMATIVE.format(Y=affirmative_phrase(s))))
            stats[(s["branch"], "affirmative")] += 1
    write_jsonl(OUT_DIR / "samples_forms.jsonl", out)
    print(f"{len(out)} form samples from {len(rep)} units")
    for k in sorted(stats):
        print(" ", k, stats[k])
    for b in sorted({s["branch"] for s in rep.values()}):
        ex = [r for r in out if r["branch"] == b]
        for r in ex[:2]:
            print(f"  [{b}] {r['question']}  ->  {r['negation_text']}")


if __name__ == "__main__":
    main()
