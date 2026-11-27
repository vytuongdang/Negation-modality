"""NEU control (Vy 2026-11-27): is the channel gap about NEGATION, or about following ANY instruction in that channel?

A neutral instruction "Include the word '<W>' in your answer." (W unrelated to the image, checked automatically, see
create_data/src/neu_assets.py) is delivered exactly like the negation of the same unit: same channel, position, box
placement / font (V), voice (A). Conditions (input order image first, then the items):

    C0         image | question                                   (cache, = H1)
    T / A / V  the negation, as in H1                              (cache, = H1)
    NEU_T      image | question | neutral sentence (own text item)
    NEU_A      image | question | neutral sentence spoken (same voice as the unit's negation audio)
    NEU_V      image with the neutral sentence in the box | question
    V_cue      image with the negation in the box | CUE | question
    NEU_V_cue  image with the neutral sentence in the box | CUE | question
    T_cue      image | CUE | question | negation (text)           (does the cue sentence alone change obedience?)
CUE = "The image contains an instruction for you." (own text item before the question).

Scoring: every row goes through the same judge pipeline as run_h1v2 / run_h2 (target mention, violation kinds), so
the target mention under NEU is measured too. NEU followed = W appears in the answer (whole word, case-insensitive,
plural -s/-es counts); `w_in_answer` is computed for EVERY row (chance rate in C0 / T / A / V). Rows carry image_id,
unit_id, sample_id, W. Analysis: analyze_neu.py.

    python run_neu.py --model qwen2.5-omni-7b --dry_run                    # CPU: plan + asset check, no model
    python run_neu.py --model qwen2.5-omni-7b --pilot 2                    # 2 units per branch (smoke), tag neu_smoke2
    python run_neu.py --model qwen2.5-omni-7b                              # the 1,000 pilot questions, tag neu_pilot1000
    python run_neu.py --model minicpm-o-2.6 --generate_only                # then --score_only in env `negation`
Outputs in results/<model>/: answers.jsonl (shared cache), scored_<tag>.jsonl, review_<tag>.csv.
Does not import-patch or edit run_h1v2 / run_pilot / run_h2 / omnilib / judge; the scoring block mirrors run_h2.py.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import h2_common as H  # noqa: E402
from omnilib import DATA, MODELS, AnswerCache, clean, judge_rows, read_jsonl, split_group  # noqa: E402
from run_h1v2 import VIOLATION_TYPES, RESULTS, count_rule, score  # noqa: E402

PILOT_LIST = HERE / "pilot1000.json"
NEU_DIR = DATA / "neu"
CUE = "The image contains an instruction for you."
NEG_CONDS = ["T", "A", "V", "V_cue", "T_cue"]
NEU_CONDS = ["NEU_T", "NEU_A", "NEU_V", "NEU_V_cue"]
ALL = ["C0"] + NEG_CONDS + NEU_CONDS


def word_re(w):
    return re.compile(rf"\b{re.escape(w.lower())}(s|es)?\b")


def load_neu():
    rows = {r["unit_id"]: r for r in read_jsonl(NEU_DIR / "neu.jsonl")}
    aud = {(a["W"], a["voice"]): a for a in read_jsonl(NEU_DIR / "audio.jsonl")} if (NEU_DIR / "audio.jsonl").exists() else {}
    for r in rows.values():
        a = aud.get((r["W"], r["voice"]))
        if a:
            r.update(neu_audio=a["audio"], neu_asr=a["asr"], neu_asr_ok=a["asr_ok"])
    return rows


def inputs(s, n, cond):
    q, i = s["question"], s["negation"]
    return {
        "C0": (s["image"], [("text", q)]),
        "T": (s["image"], [("text", q), ("text", i)]),            # identical to run_h1v2.inputs -> cache hits
        "A": (s["image"], [("text", q), ("audio", s["negation_audio"])]),
        "V": (s["negation_image"], [("text", q)]),
        "NEU_T": (s["image"], [("text", q), ("text", n["neu_text"])]),
        "NEU_A": (s["image"], [("text", q), ("audio", n.get("neu_audio"))]),
        "NEU_V": (n.get("neu_image"), [("text", q)]),
        "V_cue": (s["negation_image"], [("text", CUE), ("text", q)]),
        "NEU_V_cue": (n.get("neu_image"), [("text", CUE), ("text", q)]),
        "T_cue": (s["image"], [("text", CUE), ("text", q), ("text", i)]),
    }[cond]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-omni-7b", choices=list(MODELS))
    ap.add_argument("--pilot", type=int, help="smoke: N units per branch taken from the pilot list (tag neu_smokeN)")
    ap.add_argument("--conds", nargs="*", default=ALL, choices=ALL)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true", help="CPU only: build the plan, check every file exists")
    ap.add_argument("--score_only", action="store_true")
    ap.add_argument("--generate_only", action="store_true")
    args = ap.parse_args()
    conds = ["C0"] + [c for c in args.conds if c != "C0"]

    samples, _ = H.select_samples(None, False, 42, str(PILOT_LIST))
    if args.pilot:
        per = {}
        samples = [s for s in samples if per.setdefault(s["branch"], []).append(s) or len(per[s["branch"]]) <= args.pilot]
    tag = f"neu_smoke{args.pilot}" if args.pilot else "neu_pilot1000"
    if conds != ALL:
        tag += "_" + "-".join(conds[1:])
    neu = load_neu()
    plan, missing = [], []
    for s in samples:
        n = neu.get(s["unit_id"])
        if n is None:
            missing.append((s["unit_id"], "no NEU row"))
            continue
        for c in conds:
            img, items = inputs(s, n, c)
            files = [img] + [v for t, v in items if t in ("audio", "image")]
            bad = [f for f in files if not f or not (DATA / f).exists()]
            if bad:
                missing.append((s["unit_id"], c, bad))
                continue
            plan.append((s, n, c, img, items))
    print(f"{len(samples)} units, {len(plan)} answers planned ({len(conds)} conditions), missing inputs: {len(missing)}",
          flush=True)
    for m in missing[:10]:
        print("  missing:", m)
    if missing:
        sys.exit("STOP: some inputs are missing (run create_data/src/neu_assets.py words / render / tts first)")
    if args.dry_run:
        for s, n, c, img, items in plan[:len(conds)]:
            print(f"  {c:<10} {img} | {items}")
        return

    out = RESULTS / args.model
    out.mkdir(parents=True, exist_ok=True)
    cache = AnswerCache(out / "answers.jsonl")
    if not args.score_only:
        cache.generate(args.model, [(img, items, c) for _, _, c, img, items in plan], batch_size=args.batch_size)
    if args.generate_only:
        return

    rows = []
    for s, n, c, img, items in plan:
        raw = cache.final(args.model, img, items)
        if raw is None:
            continue
        ans = clean(raw)
        rows.append({"sample_id": s["id"], "pair_of": s["pair_of"], "unit_id": s["unit_id"], "image_id": s["image_id"],
                     "image": img, "form": s["form"], "condition": c,
                     "family": "NEU" if c.startswith("NEU") else ("C0" if c == "C0" else "NEG"),
                     "channel": c.replace("NEU_", "")[0] if c != "C0" else None, "cue": c.endswith("_cue"),
                     "W": n["W"], "w_in_answer": bool(word_re(n["W"]).search(ans.lower())),
                     "neu_asr_ok": n.get("neu_asr_ok"), "neu_occludes_subject": n.get("neu_occludes_subject"),
                     "branch": s["branch"], "source": s["source"], "split": s["split"],
                     "split_group": split_group(s["split"]), "template_id": s["template_id"],
                     "voice": s["voice"], "asr_ok": s["asr_ok"], "occludes_subject": s["occludes_subject"],
                     "question": s["question"], "negation": s["negation"], "neu_text": n["neu_text"],
                     "target": s["target"], "components": s["components"], "answer": ans, "raw_answer": raw, "_s": s})
    if not rows:
        return

    # ---- scoring: the same as run_h2.py / run_h1v2.main() (target mention + violation kinds) ---------------------
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
        r["violation"] = r["target_label"] in VIOLATION_TYPES   # for NEU rows: target mention (baseline), not a violation
        r["strict_violation"] = r["violation"]
    flagged = [r for r in rows if r["judge_source"] == "think_error"]
    if flagged:
        print(f"WARNING: {len(flagged)} rows where the thinking judge gave no JSON")
    with open(out / f"scored_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "components"}, ensure_ascii=False) + "\n")
    with open(out / f"review_{tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "condition", "W", "w_in_answer", "question", "instruction", "answer", "target_label",
                    "violation", "judge_source"])
        for r in rows:
            ins = "" if r["condition"] == "C0" else (r["neu_text"] if r["family"] == "NEU" else r["negation"])
            w.writerow([r["sample_id"], r["condition"], r["W"], int(r["w_in_answer"]), r["question"], ins, r["answer"],
                        r["target_label"], int(r["violation"]), r["judge_source"]])
    c0 = {r["unit_id"] for r in rows if r["condition"] == "C0" and r["violation"]}
    print(f"\nC0 states the target in {len(c0)}/{sum(r['condition'] == 'C0' for r in rows)} units")
    for c in conds:
        rs = [r for r in rows if r["condition"] == c and r["unit_id"] in c0]
        if rs:
            print(f"{c:<10} n={len(rs):>4}  target mentioned {100 * sum(r['violation'] for r in rs) / len(rs):5.1f}%  "
                  f"W in answer {100 * sum(r['w_in_answer'] for r in rs) / len(rs):5.1f}%")
    print(f"\nfiles: {out / f'scored_{tag}.jsonl'} {out / f'review_{tag}.csv'}")


if __name__ == "__main__":
    main()
