"""H1 on dataset v2: does the channel carrying a negative instruction (text / audio / written in the image) change
how well an omni-model follows it? (experiments/h1/run_h1.py is the v1 pilot runner, kept unchanged.)

Conditions per NEGATIVE sample (input order always image -> question -> instruction):
    C0      original image + question                          (does the model state the target unprompted?)
    C0_box  box image (same empty box as V) + question         (control for the pixels the text box hides)
    T_join  question and instruction in one string: "Q I"      (as in the v1 pilot)
    T_sep   question, blank line, instruction: "Q\\n\\nI"     (Qwen's chat template joins text blocks with
                                                               nothing in between, so the break is explicit)
    A       question + spoken instruction (audio segment after the question)
    V       image with the written instruction + question
EXCEPTIVE and AFFIRMATIVE samples (H4): T_sep, A, V (their C0 is the C0 of the paired negative sample, same input).

    python run_h1v2.py --model qwen2.5-omni-7b --pilot 40      # 40 units per branch, + their forms
    python run_h1v2.py --model qwen2.5-omni-7b --all           # whole dataset
    python run_h1v2.py ... --score_only                        # judge cached answers only

Outputs in results/<model>/: answers.jsonl (cache, every run), scored_<tag>.jsonl (one row per sample x
condition), review_<tag>.csv. Judge cache: results/judge.jsonl (shared by models).
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from omnilib import (BRANCHES, MODELS, AnswerCache, clean, input_key, judge_rows, load_dataset,  # noqa: E402
                     split_group)

NEG_CONDS = ["C0", "C0_box", "T_join", "T_sep", "A", "V"]
FORM_CONDS = ["T_sep", "A", "V"]
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
    if cond == "T_sep":
        return s["image"], [("text", f"{q}\n\n{i}")]
    if cond == "A":
        return s["image"], [("text", q), ("audio", s["negation_audio"])]
    if cond == "V":
        return s["negation_image"], [("text", q)]
    raise ValueError(cond)


def conditions(s):
    return NEG_CONDS if s["form"] == "negative" else FORM_CONDS


def select(data, pilot, seed):
    """pilot=None: everything. Otherwise `pilot` units per branch (sources round-robin, distinct images, no
    template-sweep units, audio that passed the ASR check), each with its negative sample and its forms."""
    if pilot is None:
        return list(data.values())
    rng = random.Random(seed)
    negs = [s for s in data.values() if s["form"] == "negative" and not s["template_sweep"] and s["asr_ok"]]
    forms = defaultdict(list)
    for s in data.values():
        if s["form"] != "negative":
            forms[s["pair_of"]].append(s)
    chosen = []
    for b in BRANCHES:
        by_src = defaultdict(list)
        for s in negs:
            if s["branch"] == b:
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
        for s in picked:
            chosen += [s] + forms.get(s["id"], [])
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
    g.add_argument("--pilot", type=int, help="units per branch")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--score_only", action="store_true")
    ap.add_argument("--generate_only", action="store_true")
    args = ap.parse_args()

    tag = "all" if args.all else f"pilot{args.pilot}"
    out = RESULTS / args.model
    out.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    samples = select(data, None if args.all else args.pilot, args.seed)
    print(f"{len(samples)} samples ({sum(s['form'] == 'negative' for s in samples)} negative)")
    cache = AnswerCache(out / "answers.jsonl")
    jobs = [(*inputs(s, c), c) for s in samples for c in conditions(s)]
    if not args.score_only:
        cache.generate(args.model, jobs, batch_size=args.batch_size)
    if args.generate_only:
        return

    rows = []
    for s in samples:
        for c in conditions(s):
            img, items = inputs(s, c)
            raw = cache.get(input_key(args.model, img, items))
            if raw is None:
                continue
            rows.append({"sample_id": s["id"], "pair_of": s["pair_of"], "unit_id": s["unit_id"], "form": s["form"],
                         "condition": c, "branch": s["branch"], "source": s["source"], "split": s["split"],
                         "split_group": split_group(s["split"]), "template_id": s["template_id"],
                         "template_sweep": s["template_sweep"], "voice": s["voice"], "asr_ok": s["asr_ok"],
                         "occludes_subject": s["occludes_subject"], "question": s["question"],
                         "negation": s["negation"], "target": s["target"], "components": s["components"],
                         "answer": clean(raw), "raw_answer": raw})
    judge_rows(rows, RESULTS / "judge.jsonl")
    by_id = {s["id"]: s for s in samples}
    for r in rows:
        r.update(score(by_id[r["sample_id"]], r["mentioned"], r["condition"]))
        r["regex_target_mentioned"] = any(r["mentioned_regex"][c] for c in by_id[r["sample_id"]]["expected"]["forbidden"])
    with open(out / f"scored_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "components"}, ensure_ascii=False) + "\n")
    with open(out / f"review_{tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "form", "condition", "question", "negation", "answer", "stated", "target",
                    "target_mentioned", "regex_target_mentioned", "judge_source", "required_recall"])
        for r in rows:
            comps = {c["id"]: c for c in r["components"]}
            stated = ", ".join(comps[c]["text"] if comps[c]["type"] != "existence" else comps[c]["subject"]
                               for c, v in r["mentioned"].items() if v)
            w.writerow([r["sample_id"], r["form"], r["condition"], r["question"],
                        "" if r["condition"].startswith("C0") else r["negation"], r["answer"], stated,
                        comps[r["target"]]["text"], int(r["target_mentioned"]), int(r["regex_target_mentioned"]),
                        r["judge_source"], round(r["required_recall"], 2)])
    # quick table: violation rate among samples whose C0 answer states the target
    c0 = {r["pair_of"] for r in rows if r["condition"] == "C0" and r["target_mentioned"]}
    print(f"\nviolation % among C0-mentioned ({len(c0)} negative samples), by form x condition x branch")
    for form in ("negative", "exceptive", "affirmative"):
        conds = NEG_CONDS[1:] if form == "negative" else FORM_CONDS
        print(f"\n{form:<16}" + "".join(f"{c:>9}" for c in conds))
        for b in BRANCHES + ["ALL"]:
            cells = []
            for c in conds:
                rs = [r for r in rows if r["form"] == form and r["condition"] == c and r["pair_of"] in c0
                      and (b == "ALL" or r["branch"] == b)]
                cells.append(f"{100 * sum(r['target_mentioned'] for r in rs) / len(rs):>6.0f}/{len(rs):<2}"
                             if rs else f"{'-':>9}")
            print(f"{b:<16}" + "".join(f"{x:>9}" for x in cells))
    dis = sum(r["target_mentioned"] != r["regex_target_mentioned"] for r in rows)
    print(f"\nregex disagrees with judge on the target in {dis}/{len(rows)} rows")
    print(f"files: {out / f'scored_{tag}.jsonl'} {out / f'review_{tag}.csv'}")


if __name__ == "__main__":
    main()
