"""Method runs on pilot1000 (Qwen2.5-Omni 3B / 7B): generate, score, analyse. See README.md and steerlib.py.

    python run_steer.py gen   --model M --method steer --layers 12 16 20 --alphas 2 4 8 [--v_pos image|box] [--conds A V]
    python run_steer.py gen   --model M --method rand  --layers 16 --alphas 4             (random-direction control)
    python run_steer.py gen   --model M --method cfg   --ws 1 2 4 [--beta 0.1]
    python run_steer.py gen   --model M --method transcribe
    python run_steer.py score --model M [--split val]      # judge every config generated so far (+ unsteered C0/T/A/V)
    python run_steer.py analyze --model M [--split val]    # -> steer/analysis_<model>_<split>.md
Common: --split val (default; evaluation set = VAL images) | train | all; --limit N (first N questions, smoke);
--batch_size; --dir_tag mention (direction file direction_<tag>.pt from direction.py).
Steering: layer specs "16" (one layer) or "12-16" (every layer 12..16, same alpha each). A configuration is one
(method, layers, alpha, positions); A is steered at the audio tokens, V at the image tokens (--v_pos image) or only
those under the written instruction (--v_pos box). T can be added (--conds T: text positions; sanity check only).
"""
import argparse
import json
import sys

from steerlib import (DATA, LONG_NEW_TOKENS, MAX_NEW_TOKENS, RESULTS, TRANSCRIBE_A, SteerCache, cond_inputs, clean,
                      config_id, file_hash, is_cut, load_pilot, steer_key)

POS_OF = {"A": "audio", "T": "text"}   # V: --v_pos
WITHOUT = {"A": "C0", "T": "C0", "V": "C0_box"}   # CFG: the input without the instruction


def pick(args):
    train, val = load_pilot()
    s = {"val": val, "train": train, "all": train + val}[args.split]
    return s[:args.limit] if args.limit else s


def abs_job(img, items):
    return (DATA / img, [(t, str(DATA / v) if t in ("audio", "image") else v) for t, v in items])


def parse_layers(spec):
    if "-" in spec:
        a, b = map(int, spec.split("-"))
        return list(range(a, b + 1))
    return [int(spec)]


def configs(args):
    """-> [(cond, cfg)] for this gen call."""
    out = []
    for c in args.conds:
        if args.method in ("steer", "rand"):
            pos = args.v_pos if c == "V" else POS_OF[c]
            for spec in args.layers:
                for a in args.alphas:
                    out.append((c, {"kind": args.method, "layers": parse_layers(spec), "alpha": a, "pos": pos,
                                    "dir_tag": args.dir_tag}))
        elif args.method == "cfg":
            out += [(c, {"kind": "cfg", "w": w, "beta": args.beta}) for w in args.ws]
        elif args.method == "transcribe":
            out.append((c, {"kind": "transcribe"}))
    return out


def load_vecs(args, cfg):
    import torch
    path = RESULTS / args.model / f"direction_{cfg['dir_tag']}.pt"
    d = torch.load(path)
    vecs = {L: d["dirs"][L] for L in cfg["layers"]}
    if cfg["kind"] == "rand":   # same norm, fixed seed per layer
        g = torch.Generator().manual_seed(1000 + cfg["layers"][0])
        vecs = {L: (lambda r: r / r.norm() * v.norm())(torch.randn(v.shape, generator=g)) for L, v in vecs.items()}
    return vecs, file_hash(path)


def run_batches(model, cache, todo, fn, bs, model_key, cid, extra_rec):
    """todo: [(s, cond, img, items)] not cached. fn(chunk, max_new_tokens) -> answers. Writes MAX_NEW_TOKENS answers,
    then regenerates the cut ones with LONG_NEW_TOKENS (same rule as omnilib.AnswerCache)."""
    for budget in (MAX_NEW_TOKENS, LONG_NEW_TOKENS):
        if budget == LONG_NEW_TOKENS:
            todo = [j for j in todo if is_cut(model.n_tokens(cache.get(steer_key(model_key, cid, j[2], j[3])) or ""))
                    and cache.get(steer_key(model_key, cid, j[2], j[3], LONG_NEW_TOKENS)) is None]
        else:
            todo = [j for j in todo if cache.get(steer_key(model_key, cid, j[2], j[3])) is None]
        print(f"  {cid}: {len(todo)} answers to generate ({budget} tokens)", flush=True)
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            answers = fn(chunk, budget)
            cache.put_many([{"key": steer_key(model_key, cid, img, items, budget), "model": model_key, "config": cid,
                             "sample_id": s["id"], "unit_id": s["unit_id"], "image_id": s["image_id"], "condition": c,
                             "image": str(img), "items": [list(x) for x in items], "max_new_tokens": budget,
                             "answer": a, **extra_rec} for (s, c, img, items), a in zip(chunk, answers)])
            if (i // bs) % 25 == 0:
                print(f"    [{i + len(chunk)}/{len(todo)}] {chunk[0][1]}: {answers[0][:90]!r}", flush=True)


def final(cache, model_key, cid, img, items):
    a = cache.get(steer_key(model_key, cid, img, items, LONG_NEW_TOKENS))
    return a if a is not None else cache.get(steer_key(model_key, cid, img, items))


def transcript_of(raw):
    """First non-empty line of a copy answer, quotes stripped."""
    line = next((x for x in clean(raw).splitlines() if x.strip()), "").strip().strip('"“”\'')
    return line or "(no instruction)"


def gen(args):
    from PIL import Image
    from steerlib import SteerModel
    from omnilib import AnswerCache
    samples = pick(args)
    cache = SteerCache(RESULTS / args.model / "answers.jsonl")
    h1 = AnswerCache(DATA.parents[1] / "experiments" / "h1_v2" / "results" / args.model / "answers.jsonl")  # READ ONLY
    model = SteerModel(args.model)
    for cond, cfg in configs(args):
        cid = config_id(cfg)
        print(f"== {cond} {cid} on {len(samples)} {args.split} questions", flush=True)
        if cfg["kind"] in ("steer", "rand"):
            vecs, dh = load_vecs(args, cfg)
            todo = [(s, cond, *cond_inputs(s, cond)) for s in samples]

            def fn(chunk, budget, vecs=vecs, cfg=cfg):
                sizes = [Image.open(DATA / s["negation_image"]).size for s, *_ in chunk]
                return model.steer([abs_job(img, items) for _, _, img, items in chunk], vecs, cfg["alpha"], cfg["pos"],
                                   budget, sizes=sizes, boxes=[s["text_box"] for s, *_ in chunk],
                                   texts=[s["negation"] for s, *_ in chunk])
            run_batches(model, cache, todo, fn, args.batch_size, args.model, cid, {"dir_hash": dh})
        elif cfg["kind"] == "cfg":
            todo = [(s, cond, *cond_inputs(s, cond)) for s in samples]

            def fn(chunk, budget, cfg=cfg):
                pairs = [(abs_job(img, items), abs_job(*cond_inputs(s, WITHOUT[c]))) for s, c, img, items in chunk]
                return model.cfg(pairs, cfg["w"], cfg["beta"], budget)
            run_batches(model, cache, todo, fn, max(1, args.batch_size // 2), args.model, cid, {})
        else:   # transcribe-then-obey
            step1 = {}
            need = []
            for s in samples:
                if cond == "V":   # = H1's OCR_V input: reuse the H1 answer if cached
                    img, items = cond_inputs(s, "OCR_V")
                    a = h1.final(args.model, img, items)
                else:
                    img, items = s["image"], [("text", TRANSCRIBE_A), ("audio", s["negation_audio"])]
                    a = None
                if a is None:
                    a = final(cache, args.model, "transcribe_step1", img, items)
                if a is None:
                    need.append((s, cond, img, items))
                step1[s["id"]] = (img, items)
            run_batches(model, cache, need, lambda ch, b: model.plain([abs_job(i, it) for _, _, i, it in ch], b),
                        args.batch_size, args.model, "transcribe_step1", {})
            todo = []
            for s in samples:
                img, items = step1[s["id"]]
                a = (h1.final(args.model, img, items) if cond == "V" else None) or final(cache, args.model, "transcribe_step1", img, items)
                base_img = cond_inputs(s, cond)[0]
                todo.append((s, cond, base_img, [("text", s["question"]), ("text", transcript_of(a))]))
            run_batches(model, cache, todo, lambda ch, b: model.plain([abs_job(i, it) for _, _, i, it in ch], b),
                        args.batch_size, args.model, cid, {})
    model.close()


# ---------------------------------------------------------------- scoring (the H1 judge path, unchanged)

def score_rows(rows, judge_cache):
    """Same steps as run_h1v2.main after generation: judge (fast + thinking, regex second opinion), score(),
    count_rule override, label pass for compliant rows, violation = any of STATE / META / DENY / WRONG."""
    from omnilib import judge_rows
    from run_h1v2 import VIOLATION_TYPES, count_rule, score
    from judge import Judge
    judge_rows(rows, judge_cache)
    for r in rows:
        s = r.pop("_s")
        r["_s2"] = s
        r.update(score(s, r["mentioned"], r["condition"]))
        r["regex_target_mentioned"] = any(r["mentioned_regex"][c] for c in s["expected"]["forbidden"])
        r["count_rule"] = count_rule(s, r["answer"])
        if r["count_rule"] in ("STATE", "NONE"):
            r["target_mentioned"] = r["count_rule"] == "STATE"
            r["correct"] = r["all_required"] and (r["target_mentioned"] if r["condition"].startswith("C0")
                                                  else not r["target_mentioned"])
    todo = [r for r in rows if not r["target_mentioned"] and not r["count_rule"]]
    if todo:
        j = Judge(judge_cache)
        target_comp = lambda r: next(c for c in r["components"] if c["id"] == r["target"])
        labels = j.label_target([(target_comp(r), r["answer"], r["question"]) for r in todo])
        j.close()
        for r, lab in zip(todo, labels):
            r["target_label"] = lab or "NONE"
            if lab is None:
                r["label_error"] = True
    for r in rows:
        r.pop("_s2")
        if r["count_rule"] in ("STATE", "NONE"):
            r["target_label"] = r["count_rule"]
        if r["target_mentioned"]:
            r["target_label"] = "STATE"
        r.setdefault("target_label", "NONE")
        r["leak"] = r["target_mentioned"]
        r["violation"] = r["target_label"] in VIOLATION_TYPES
    return rows


def make_row(s, cond, cid, raw):
    return {"config": cid, "sample_id": s["id"], "unit_id": s["unit_id"], "image_id": s["image_id"],
            "image": s["image"], "condition": cond, "branch": s["branch"], "source": s["source"], "split": s["split"],
            "template_id": s["template_id"], "voice": s["voice"], "asr_ok": s["asr_ok"],
            "occludes_subject": s["occludes_subject"], "question": s["question"], "negation": s["negation"],
            "target": s["target"], "components": s["components"], "answer": clean(raw), "raw_answer": raw, "_s": s}


def score(args):
    from omnilib import AnswerCache
    samples = pick(args)
    by_id = {s["id"]: s for s in samples}
    h1 = AnswerCache(DATA.parents[1] / "experiments" / "h1_v2" / "results" / args.model / "answers.jsonl")  # READ ONLY
    rows, missing = [], 0
    for s in samples:   # unsteered baselines from the H1 cache
        for c in ("C0", "T", "A", "V", "C0_box"):
            img, items = cond_inputs(s, c)
            raw = h1.final(args.model, img, items)
            if raw is None:
                missing += 1
                continue
            rows.append(make_row(s, c, "none", raw))
    cache = SteerCache(RESULTS / args.model / "answers.jsonl")
    seen = set()
    for r in cache.d.values():   # every steered answer of these questions: the final (long if cut) one per input
        if r["config"] == "transcribe_step1" or r["sample_id"] not in by_id:
            continue
        k = (r["config"], r["sample_id"], r["condition"])
        if k in seen:
            continue
        seen.add(k)
        items = [tuple(x) for x in r["items"]]
        rows.append(make_row(by_id[r["sample_id"]], r["condition"], r["config"],
                             final(cache, args.model, r["config"], r["image"], items)))
    print(f"{len(rows)} rows to score ({missing} unsteered answers not in the H1 cache)", flush=True)
    score_rows(rows, args.judge_cache)
    out = RESULTS / args.model / f"scored_{args.split}{'_lim' + str(args.limit) if args.limit else ''}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "components"}, ensure_ascii=False) + "\n")
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["gen", "score", "analyze"])
    ap.add_argument("--model", required=True, choices=["qwen2.5-omni-3b", "qwen2.5-omni-7b"])
    ap.add_argument("--method", choices=["steer", "rand", "cfg", "transcribe"])
    ap.add_argument("--conds", nargs="*", default=["A", "V"], choices=["A", "V", "T"])
    ap.add_argument("--layers", nargs="*", default=["16"])
    ap.add_argument("--alphas", nargs="*", type=float, default=[4.0])
    ap.add_argument("--v_pos", default="box", choices=["box", "image"])
    ap.add_argument("--dir_tag", default="mention")
    ap.add_argument("--ws", nargs="*", type=float, default=[1.0])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--judge_cache", default=str(RESULTS.parents[1] / "results" / "judge.jsonl"),
                    help="shared H1 judge cache by default (only one judge writer at a time)")
    args = ap.parse_args()
    if args.step == "gen":
        if not args.method:
            sys.exit("--method is required for gen")
        gen(args)
    elif args.step == "score":
        score(args)
    else:
        from analyze_steer import analyze
        analyze(args.model, args.split, args.limit)


if __name__ == "__main__":
    main()
