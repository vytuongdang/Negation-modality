"""H2 (direct conflict) and H3-A (modality dominance): the question and the instruction arrive through DIFFERENT channels.

Layouts, conditions and assets: experiments/common/h2_common.py (docstring). Conditions are named
<family>_q<Q>_i<I> (H2_qA_iT = H2, question spoken, instruction written as text). Scoring is the one of run_h1v2.py
(judge v3 + regex, label pass, count rule; violation = any mention of the forbidden component, counted among the units
whose own C0 answer states it), so the numbers are comparable with H1 / H4. C0 is always generated (cache).

Subset or full, chosen on the command line (same selection as run_h1v2.py):
    --pilot N     N units per branch (sources round-robin, distinct images, instruction audio that passed ASR)
    --all         every unit
    --pilot_list  FILE  the fixed list of sample ids (experiments/h1_v2/pilot1000.json); tag h2_pilot1000
Choose what to run:
    --families H2 H3A      (default both)       --h3a_form affirmative|exceptive  (the positive instruction of H3-A)
    --layouts TA AT ...    (default all 6; two letters = question channel, instruction channel)

Prerequisite for layouts with a spoken or written QUESTION (A / V as first letter): the question assets of the same
selection, `h2_assets.py render` (+ `tts` for spoken questions). qT_iA and qT_iV need no new asset (= H1's A and V).

    python run_h2.py --model qwen2.5-omni-7b --pilot 40 --layouts TA TV      # nothing new needed, cache hits after H1
    python run_h2.py --model qwen2.5-omni-7b --pilot 40                      # all 12 conditions
    python run_h2.py --model minicpm-o-2.6 --pilot 40 --generate_only        # then --score_only in env `negation`

Outputs in results/<model>/: answers.jsonl (shared cache), scored_<tag>.jsonl, review_<tag>.csv; tag = h2_pilotN | h2_all
(+ _h3a_exceptive, + _<layouts> when a subset of layouts / families is chosen).
NOTE: the scoring block below mirrors run_h1v2.main(); if the scoring rules change there, change them here too.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import h2_common as H  # noqa: E402
from omnilib import BRANCHES, MODELS, AnswerCache, clean, judge_rows, split_group  # noqa: E402
from run_h1v2 import VIOLATION_TYPES, RESULTS, count_rule, score  # noqa: E402


def build_plan(samples, forms, qrows, families, layouts, h3a_form):
    """-> [(row sample, condition, image, items)]; the row sample is the negative sample for C0 and H2, the positive
    form sample for H3A (its pair_of is the negative sample, which carries the C0 answer)."""
    plan, skipped = [], defaultdict(int)
    for s in samples:
        plan.append((s, "C0", s["image"], [("text", s["question"])]))
        q = qrows.get(s["unit_id"])
        for fam in families:
            ins = H.instruction_sample(s, forms, fam, h3a_form)
            if ins is None:
                skipped[fam] += 1
                continue
            for Q, I in layouts:
                if H.needs_assets(Q) and (q is None or not q.get("q_image" if Q == "V" else "q_audio")):
                    skipped[f"{fam}_q{Q}_i{I} (no question {'image' if Q == 'V' else 'audio'})"] += 1
                    continue
                img, items = H.h2_inputs(s, ins, q, Q, I)
                plan.append((ins, H.cond_name(fam, Q, I), img, items))
    return plan, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-omni-7b", choices=list(MODELS))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", type=int, help="units per branch")
    g.add_argument("--all", action="store_true")
    g.add_argument("--pilot_list", help="fixed list of sample ids, e.g. experiments/h1_v2/pilot1000.json")
    ap.add_argument("--families", nargs="*", default=list(H.FAMILIES), choices=list(H.FAMILIES))
    ap.add_argument("--layouts", nargs="*", default=["".join(l) for l in H.LAYOUTS],
                    choices=["".join(l) for l in H.LAYOUTS], help="question channel + instruction channel, e.g. TA AT")
    ap.add_argument("--h3a_form", default="affirmative", choices=list(H.H3A_FORMS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--score_only", action="store_true")
    ap.add_argument("--generate_only", action="store_true")
    args = ap.parse_args()

    layouts = [tuple(l) for l in args.layouts]
    tag = "h2_" + H.list_tag(args.pilot, args.all, args.pilot_list)
    if args.h3a_form != "affirmative" and "H3A" in args.families:
        tag += f"_h3a_{args.h3a_form}"
    if args.families != list(H.FAMILIES):
        tag += "_" + "-".join(args.families)
    if args.layouts != ["".join(l) for l in H.LAYOUTS]:
        tag += "_" + "-".join(args.layouts)
    out = RESULTS / args.model
    out.mkdir(parents=True, exist_ok=True)

    samples, forms = H.select_samples(args.pilot, args.all, args.seed, args.pilot_list)
    qrows = H.load_questions()
    plan, skipped = build_plan(samples, forms, qrows, args.families, layouts, args.h3a_form)
    conds = sorted({c for _, c, _, _ in plan if c != "C0"})
    print(f"{len(samples)} units, {len(plan)} answers planned, conditions: {conds}", flush=True)
    for k, n in sorted(skipped.items()):
        print(f"  skipped {n} x {k}")
    if any(H.needs_assets(Q) for Q, _ in layouts) and not qrows:
        sys.exit(f"{H.QFILE} is missing: run create_data/src/h2_assets.py render (and tts) for this selection first")
    cache = AnswerCache(out / "answers.jsonl")
    if not args.score_only:
        cache.generate(args.model, [(img, items, c) for _, c, img, items in plan], batch_size=args.batch_size)
    if args.generate_only:
        return

    rows = []
    for s, c, img, items in plan:
        raw = cache.final(args.model, img, items)
        if raw is None:
            continue
        fam, Q, I = ("C0", None, None) if c == "C0" else H.parse_cond(c)
        q = qrows.get(s["unit_id"], {})
        rows.append({"sample_id": s["id"], "pair_of": s["pair_of"], "unit_id": s["unit_id"], "image_id": s["image_id"], "image": s["image"],
                     "form": s["form"], "condition": c, "family": fam, "q_channel": Q, "i_channel": I,
                     "branch": s["branch"], "source": s["source"], "split": s["split"],
                     "split_group": split_group(s["split"]), "template_id": s["template_id"],
                     "template_sweep": s["template_sweep"], "voice": s["voice"], "asr_ok": s["asr_ok"],
                     "q_voice": q.get("q_voice"), "q_asr_ok": q.get("q_asr_ok"),
                     "occludes_subject": s["occludes_subject"], "q_occludes_subject": q.get("q_occludes_subject"),
                     "question": s["question"], "negation": s["negation"], "target": s["target"],
                     "components": s["components"], "answer": clean(raw), "raw_answer": raw, "_s": s})
    if not rows:
        return

    # ---- scoring: the same as run_h1v2.main() -------------------------------------------------------------
    judge_rows(rows, RESULTS / "judge.jsonl")
    for r in rows:
        s = r.pop("_s")
        r.update(score(s, r["mentioned"], r["condition"]))
        r["regex_target_mentioned"] = any(r["mentioned_regex"][c] for c in s["expected"]["forbidden"])
        r["count_rule"] = count_rule(s, r["answer"])
        if r["count_rule"] in ("STATE", "NONE"):
            r["target_mentioned"] = r["count_rule"] == "STATE"
            r["correct"] = r["all_required"] and (r["target_mentioned"] if r["condition"].startswith("C0")
                                                  else not r["target_mentioned"])
    from judge import Judge
    todo = [r for r in rows if not r["target_mentioned"] and not r["count_rule"]]
    if todo:
        j = Judge(RESULTS / "judge.jsonl")
        target_comp = lambda r: next(c for c in r["components"] if c["id"] == r["target"])
        labels = j.label_target([(target_comp(r), r["answer"], r["question"]) for r in todo])
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
        r["strict_violation"] = r["violation"]
    flagged = [r for r in rows if r["judge_source"] == "think_error"]
    if flagged:
        print(f"WARNING: {len(flagged)} rows where the thinking judge gave no JSON; listed in review_{tag}.csv")
    with open(out / f"scored_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "components"}, ensure_ascii=False) + "\n")
    with open(out / f"review_{tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "form", "condition", "question", "negation", "answer", "target", "target_mentioned",
                    "regex_target_mentioned", "judge_source", "target_label", "violation", "required_recall"])
        for r in rows:
            comps = {c["id"]: c for c in r["components"]}
            t = comps[r["target"]]
            w.writerow([r["sample_id"], r["form"], r["condition"], r["question"],
                        "" if r["condition"] == "C0" else r["negation"], r["answer"],
                        t["subject"] if t["type"] == "existence" else t["text"], int(r["target_mentioned"]),
                        int(r["regex_target_mentioned"]), r["judge_source"], r["target_label"], int(r["violation"]),
                        round(r["required_recall"], 2)])
    c0 = {r["pair_of"] for r in rows if r["condition"] == "C0" and r["violation"]}
    print(f"\nC0 states the target in {len(c0)}/{sum(r['condition'] == 'C0' for r in rows)} questions")
    print("violation % among C0-mentioned (n)\n" + f"{'branch':<18}" + "".join(f"{c.replace('_', ' '):>14}" for c in conds))
    for b in BRANCHES + ["ALL"]:
        cells = []
        for c in conds:
            rs = [r for r in rows if r["condition"] == c and r["pair_of"] in c0 and (b == "ALL" or r["branch"] == b)]
            cells.append(f"{100 * sum(r['violation'] for r in rs) / len(rs):.0f} ({len(rs)})" if rs else "-")
        print(f"{b:<18}" + "".join(f"{x:>14}" for x in cells))
    print(f"\nfiles: {out / f'scored_{tag}.jsonl'} {out / f'review_{tag}.csv'}")


if __name__ == "__main__":
    main()
