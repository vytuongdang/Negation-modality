"""H1 on dataset v2: the same negative instruction through text, audio, or written in the image.
(experiments/h1/run_h1.py is the v1 pilot runner on the old dataset, kept unchanged.)

H1 core (Vy, 2026-11-19), input order always image -> question -> instruction, one negative sample per question
(for template-sweep units one of the 7 copies is drawn, seeded), max_new_tokens 100:
    C0   original image + question                         (does the model state the target unprompted?)
    T    question + written instruction as its own text item, after the question (exactly like the audio in A;
         every adapter joins items with omnilib.ITEM_SEP = "\n")
    A    question + spoken instruction (audio segment after the question)
    V    image with the written instruction + question
Later conditions, only with --conds (after H1 core is reported):
    T_join   question and instruction in one string "Q I"
    C0_box   box image (same empty box as V) + question
    T_pre    instruction text item BEFORE the question                (order: matches V, where it comes first)
    A_pre    instruction audio item BEFORE the question
    V_card   question, then the instruction on a separate white card image (same slot as T / A;
             cards from common/render_cards.py)
    OCR_V    V image + "copy the text in the white box"   (can the model read V at all? like the ASR check)
    OCR_card original image + card + "copy the text in the second image"
             OCR conditions are scored by string match with the instruction, not by the judge.
H4 forms (exceptive / affirmative) with --forms: T, A, V on the form samples of the selected units.

    python run_h1v2.py --model qwen2.5-omni-7b --pilot 40      # 40 questions per branch (200)
    python run_h1v2.py --model qwen2.5-omni-7b --all           # every question
    python run_h1v2.py ... --score_only                        # judge cached answers only

Outputs in results/<model>/: answers.jsonl (cache keyed by input hash, shared by every run), scored_<tag>.jsonl
(one row per sample x condition), review_<tag>.csv. Judge cache: results/judge.jsonl (shared by models).
"""
import argparse
import csv
import difflib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from render_cards import card_rel  # noqa: E402
from omnilib import (BRANCHES, MODELS, AnswerCache, clean, input_key, judge_rows, load_dataset,  # noqa: E402
                     split_group)

CORE = ["C0", "T", "A", "V"]
EXTRA = ["T_join", "C0_box", "T_pre", "A_pre", "V_card", "OCR_V", "OCR_card"]
OCR = {"OCR_V": "What does the text in the white box say? Copy it exactly, and say nothing else.",
       "OCR_card": "What does the text in the second image say? Copy it exactly, and say nothing else."}
FORM_CONDS = ["T", "A", "V"]
RESULTS = HERE / "results"


def inputs(s, cond):
    """-> (image relative to dataset/, items)"""
    q, i = s["question"], s["negation"]
    if cond == "C0":
        return s["image"], [("text", q)]
    if cond == "C0_box":
        return s["box_image"], [("text", q)]
    if cond == "T_join":
        return s["image"], [("text", f"{q} {i}")]
    if cond == "T":
        return s["image"], [("text", q), ("text", i)]   # own item, exactly like the audio in A (Vy, 2026-11-23)
    if cond == "A":
        return s["image"], [("text", q), ("audio", s["negation_audio"])]
    if cond == "V":
        return s["negation_image"], [("text", q)]
    if cond == "T_pre":
        return s["image"], [("text", i), ("text", q)]
    if cond == "A_pre":
        return s["image"], [("audio", s["negation_audio"]), ("text", q)]
    if cond == "V_card":
        return s["image"], [("text", q), ("image", card_rel(i))]
    if cond == "OCR_V":
        return s["negation_image"], [("text", OCR[cond])]
    if cond == "OCR_card":
        return s["image"], [("text", OCR[cond]), ("image", card_rel(i))]
    raise ValueError(cond)


VIOLATION_TYPES = ("STATE", "META", "DENY", "WRONG")   # kinds of violation (primary metric = any of them)
NUM_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def count_rule(s, answer):
    """Count branch, answers that are only a number ("3", "1."): which counts could it be? (Vy, 2026-11-23)
    A total that INCLUDES the forbidden objects reveals them (1 cube + 1 sphere + 1 cylinder, spheres forbidden:
    "3" counted the sphere, "2" did not). -> "STATE" (only violating readings), "NONE" (only compliant readings),
    "AMBIG" (both, e.g. "1" with 1 pizza and 1 sandwich), or None (not a bare number / no reading matches)."""
    if s["branch"] != "count":
        return None
    m = re.fullmatch(r"(?:the answer is\s*)?([a-z]+|\d+)\.?", answer.strip().lower())
    if not m:
        return None
    n = int(m.group(1)) if m.group(1).isdigit() else NUM_WORDS.get(m.group(1))
    if n is None:
        return None
    comps = {c["id"]: c for c in s["components"] if c["type"] == "count"}
    forb = [comps[c]["value"] for c in s["expected"]["forbidden"] if c in comps]
    req = [comps[c]["value"] for c in s["expected"]["required"] if c in comps]
    total = sum(c["value"] for c in comps.values())
    violating = {total} | set(forb)
    compliant = {total - sum(forb)} | set(req)
    v, c = n in violating, n in compliant
    return "AMBIG" if v and c else "STATE" if v else "NONE" if c else None


def ocr_score(answer, truth):
    """Normalised exact match + character similarity of a copied instruction."""
    norm = lambda x: re.sub(r"[^a-z0-9 ]", "", x.lower().replace("\u2019", "'").replace("'", "")).split()
    a, t = " ".join(norm(answer)), " ".join(norm(truth))
    return {"ocr_exact": a == t, "ocr_sim": round(difflib.SequenceMatcher(None, a, t).ratio(), 3)}


def one_per_unit(data, seed):
    """One negative sample per unit: the only one, or a seeded draw among the template-sweep copies."""
    by_unit = defaultdict(list)
    for s in data.values():
        if s["form"] == "negative":
            by_unit[s["unit_id"]].append(s)
    out = []
    for u in sorted(by_unit):
        group = sorted(by_unit[u], key=lambda s: s["id"])
        out.append(group[0] if len(group) == 1 else random.Random(f"{seed}-{u}").choice(group))
    return out


def select(data, pilot, seed):
    """pilot=None: every unit. Otherwise `pilot` units per branch, sources round-robin, distinct images, audio that
    passed the ASR check."""
    negs = one_per_unit(data, seed)
    if pilot is None:
        return negs
    rng = random.Random(seed)
    chosen = []
    for b in BRANCHES:
        by_src = defaultdict(list)
        for s in negs:
            if s["branch"] == b and s["asr_ok"]:
                by_src[s["source"]].append(s)
        pools = [rng.sample(v, len(v)) for _, v in sorted(by_src.items())]
        picked, images, k = [], set(), 0
        while len(picked) < pilot and any(pools):
            pool = pools[k % len(pools)]
            k += 1
            while pool:
                s = pool.pop()
                if s["image"] not in images:
                    images.add(s["image"])
                    picked.append(s)
                    break
        chosen += picked
    return chosen


def score(s, hit, cond):
    req = s["expected"]["required"]
    target = any(hit[c] for c in s["expected"]["forbidden"])
    all_req = all(hit[c] for c in req)
    return {"target_mentioned": target, "required_recall": sum(hit[c] for c in req) / len(req),
            "all_required": all_req, "correct": all_req and (target if cond.startswith("C0") else not target)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-omni-7b", choices=list(MODELS))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", type=int, help="questions per branch")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--conds", nargs="*", default=CORE, choices=CORE + EXTRA)
    ap.add_argument("--forms", action="store_true", help="also the exceptive / affirmative forms (H4): T, A, V")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--score_only", action="store_true")
    ap.add_argument("--generate_only", action="store_true")
    args = ap.parse_args()

    tag = ("all" if args.all else f"pilot{args.pilot}") + ("_forms" if args.forms else "")
    if args.conds != CORE:   # extra conditions get their own files, so they never overwrite the core results
        tag += "_" + "-".join(args.conds)
    out = RESULTS / args.model
    out.mkdir(parents=True, exist_ok=True)
    # select on the full dataset, then drop double-labelled units: the selection stays the same as before the
    # exclusion list existed, so cached answers are reused and runs stay comparable (2026-11-23)
    data_all, data = load_dataset(keep_excluded=True), load_dataset()
    samples = [s for s in select(data_all, None if args.all else args.pilot, args.seed) if s["id"] in data]
    conds = list(args.conds)
    if "C0" not in conds and any(c not in OCR for c in conds):
        conds = ["C0"] + conds   # violation is counted among C0-mentioned questions; C0 answers are cached
    plan = [(s, c) for s in samples for c in conds]
    if args.forms:
        forms = defaultdict(list)
        for s in data.values():
            if s["form"] != "negative":
                forms[s["pair_of"]].append(s)
        plan += [(f, c) for s in samples for f in forms.get(s["id"], []) for c in FORM_CONDS]
    print(f"{len(samples)} questions, {len(plan)} answers planned; by branch: "
          + ", ".join(f"{b}={sum(s['branch'] == b for s in samples)}" for b in BRANCHES), flush=True)
    cache = AnswerCache(out / "answers.jsonl")
    if not args.score_only:
        cache.generate(args.model, [(*inputs(s, c), c) for s, c in plan], batch_size=args.batch_size)
    if args.generate_only:
        return

    rows = []
    for s, c in plan:
        img, items = inputs(s, c)
        raw = cache.final(args.model, img, items)   # the 256-token answer if the 100-token one was cut
        if raw is None:
            continue
        rows.append({"sample_id": s["id"], "pair_of": s["pair_of"], "unit_id": s["unit_id"], "image": s["image"],
                     "form": s["form"], "condition": c, "branch": s["branch"], "source": s["source"],
                     "split": s["split"], "split_group": split_group(s["split"]), "template_id": s["template_id"],
                     "template_sweep": s["template_sweep"], "voice": s["voice"], "asr_ok": s["asr_ok"],
                     "occludes_subject": s["occludes_subject"], "question": s["question"],
                     "negation": s["negation"], "target": s["target"], "components": s["components"],
                     "answer": clean(raw), "raw_answer": raw, "_s": s})
    ocr_rows = [r for r in rows if r["condition"] in OCR]
    rows = [r for r in rows if r["condition"] not in OCR]
    if ocr_rows:
        with open(out / f"ocr_{tag}.jsonl", "w") as f:
            for r in ocr_rows:
                r.pop("_s")
                r.update(ocr_score(r["answer"], r["negation"]))
                f.write(json.dumps({k: v for k, v in r.items() if k != "components"}, ensure_ascii=False) + "\n")
        for c in sorted({r["condition"] for r in ocr_rows}):
            rs = [r for r in ocr_rows if r["condition"] == c]
            print(f"{c}: exact copy {100 * sum(r['ocr_exact'] for r in rs) / len(rs):.1f}%, mean similarity "
                  f"{sum(r['ocr_sim'] for r in rs) / len(rs):.3f} (n={len(rs)}) -> {out / f'ocr_{tag}.jsonl'}")
    if not rows:
        return
    judge_rows(rows, RESULTS / "judge.jsonl")
    for r in rows:
        s = r.pop("_s")
        r.update(score(s, r["mentioned"], r["condition"]))
        r["regex_target_mentioned"] = any(r["mentioned_regex"][c] for c in s["expected"]["forbidden"])
        # bare-number answers in the count branch: the total rule overrides the judge (AMBIG keeps the judge, flagged)
        r["count_rule"] = count_rule(s, r["answer"])
        if r["count_rule"] in ("STATE", "NONE"):
            r["target_mentioned"] = r["count_rule"] == "STATE"
            r["correct"] = r["all_required"] and (r["target_mentioned"] if r["condition"].startswith("C0")
                                                  else not r["target_mentioned"])
    # PRIMARY METRIC (Vy, 2026-11-23): we measure whether the model obeys the negation, so a violation is ANY
    # mention of the forbidden thing: the fact itself (STATE), naming it to exclude it (META, "besides the bed"),
    # a denial (DENY, "there is no knife") or a different value (WRONG, "2 cylinders" when there are 3).
    #   violation = STATE / META / DENY / WRONG (primary; target_label says which kind, for the breakdown)
    #   leak = target_mentioned (the correct fact; secondary)
    # Every row the judge counts as compliant gets a label from the label pass (count-branch bare numbers are
    # decided by count_rule); rows the judge counts as stating the fact are STATE.
    from judge import Judge
    todo = [r for r in rows if not r["target_mentioned"] and not r["count_rule"]]
    if todo:
        j = Judge(RESULTS / "judge.jsonl")
        comp = lambda r: next(c for c in r["components"] if c["id"] == r["target"])
        labels = j.label_target([(comp(r), r["answer"], r["question"]) for r in todo])
        j.close()
        for r, lab in zip(todo, labels):
            r["target_label"] = lab or "NONE"
            if lab is None:
                r["label_error"] = True
    for r in rows:
        if r["count_rule"] in ("STATE", "NONE"):
            r["target_label"] = r["count_rule"]
        if r["target_mentioned"]:
            r["target_label"] = "STATE"
        r.setdefault("target_label", "NONE")
        r["leak"] = r["target_mentioned"]
        r["violation"] = r["target_label"] in VIOLATION_TYPES
        r["strict_violation"] = r["violation"]   # older name, kept for analyze.py / human_check.py
    flagged = [r for r in rows if r["judge_source"] == "think_error"]
    if flagged:
        print(f"WARNING: {len(flagged)} rows where the thinking judge gave no JSON even with the retry budget; "
              f"they keep the fast verdict and are listed in review_{tag}.csv (judge_source=think_error)")
    with open(out / f"scored_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "components"}, ensure_ascii=False) + "\n")
    with open(out / f"review_{tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "form", "condition", "question", "negation", "answer", "stated", "target",
                    "target_mentioned", "regex_target_mentioned", "judge_source", "target_label",
                    "violation", "required_recall"])
        for r in rows:
            comps = {c["id"]: c for c in r["components"]}
            name = lambda c: comps[c]["subject"] if comps[c]["type"] == "existence" else comps[c]["text"]
            w.writerow([r["sample_id"], r["form"], r["condition"], r["question"],
                        "" if r["condition"].startswith("C0") else r["negation"], r["answer"],
                        ", ".join(name(c) for c, v in r["mentioned"].items() if v), name(r["target"]),
                        int(r["target_mentioned"]), int(r["regex_target_mentioned"]), r["judge_source"],
                        r["target_label"], int(r["violation"]),
                        round(r["required_recall"], 2)])
    # quick table: violation rate among questions whose C0 answer states the target
    c0 = {r["pair_of"] for r in rows if r["condition"] == "C0" and r["violation"]}
    n0 = sum(1 for r in rows if r["condition"] == "C0")
    print(f"\nC0 states the target in {len(c0)}/{n0} questions")
    conds = [c for c in conds if c != "C0" and c not in OCR]
    print(f"violation % (any mention: STATE/META/DENY/WRONG) among C0-mentioned (n), negative form\n{'branch':<18}" + "".join(f"{c:>12}" for c in conds))
    for b in BRANCHES + ["ALL"]:
        cells = []
        for c in conds:
            rs = [r for r in rows if r["form"] == "negative" and r["condition"] == c and r["pair_of"] in c0
                  and (b == "ALL" or r["branch"] == b)]
            cells.append(f"{100 * sum(r['violation'] for r in rs) / len(rs):.0f} ({len(rs)})" if rs else "-")
        print(f"{b:<18}" + "".join(f"{x:>12}" for x in cells))
    dis = sum(r["target_mentioned"] != r["regex_target_mentioned"] for r in rows)
    print(f"\nregex disagrees with judge on the target in {dis}/{len(rows)} rows")
    print(f"files: {out / f'scored_{tag}.jsonl'} {out / f'review_{tag}.csv'}")


if __name__ == "__main__":
    main()
