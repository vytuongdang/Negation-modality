"""H3-C (activation order, Vy OK 2026-11-27): the instruction BEFORE the image vs after it.

    T_first   instruction (text) | image | question
    A_first   instruction (spoken) | image | question
compared with the existing T (image | question | instruction text) and A (image | question | instruction audio), which
come from the answer cache, and C0 (image | question) for the C0-mentioned conditioning. T_pre / A_pre (instruction
between image and question) are NOT H3-C. V is skipped: a V_card placed before the image would need new rendering
(no instruction cards exist in dataset/visual), so it is left out (Vy: "if it needs new rendering, skip V").

The adapters in experiments/common/omnilib.py always put the main image first. omnilib.py is NOT edited (other jobs
import it). This script wraps the adapters IN ITS OWN PROCESS ONLY: subclasses that, when the main image path is
marked NOLEAD, put no leading image and take the image from the items list, where it is an ordinary ("image", path)
item placed after the instruction. Every item is joined with omnilib.ITEM_SEP ("\\n") as in the existing prompts. Known
formatting difference: in T / A the question follows the image directly; in T_first / A_first the image is preceded and
followed by ITEM_SEP. Prompts for every other condition are built by the original adapter code (unchanged).

    python run_h3c.py --model qwen2.5-omni-7b --pilot 2          # smoke: 2 units per branch from pilot1000 (tag h3c_smoke2)
    python run_h3c.py --model qwen2.5-omni-7b --n 200            # first 40 per branch of pilot1000 (tag h3c_pilot200)
    python run_h3c.py --model qwen2.5-omni-7b --n 1000           # all of pilot1000 (tag h3c_pilot1000)
    python run_h3c.py --model minicpm-o-2.6 --n 200 --generate_only   # then --score_only in env `negation`
--score_only calls the judge (results/judge.jsonl is shared): never run it while another judge writer runs.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import h2_common as H  # noqa: E402
import omnilib as O  # noqa: E402
from omnilib import DATA, MODELS, AnswerCache, clean, judge_rows, split_group  # noqa: E402
from run_h1v2 import VIOLATION_TYPES, RESULTS, count_rule, score  # noqa: E402

PILOT_LIST = HERE / "pilot1000.json"
NOLEAD = "NOLEAD__"          # prefix of the main-image path: no leading image, the image is in the items
CONDS = ["C0", "T", "A", "T_first", "A_first"]


def nolead(p):
    return NOLEAD in str(p)


# ------------------------------------------------------------------ adapter wrappers (this process only)
class OmniH3C(O.Omni):
    def prompt_nolead(self, items):
        content = []
        for k, (t, v) in enumerate(items):
            if k:
                content.append({"type": "text", "text": O.ITEM_SEP})
            content.append({"type": t, **({"text": v} if t == "text" else {})})
        messages = [{"role": "system", "content": [{"type": "text", "text": O.QWEN_SYSTEM}]},
                    {"role": "user", "content": content}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def answer_batch(self, jobs, max_new_tokens=O.MAX_NEW_TOKENS):
        if not any(nolead(p) for p, _ in jobs):
            return super().answer_batch(jobs, max_new_tokens)
        if not all(nolead(p) for p, _ in jobs):
            return [self.answer_batch([j], max_new_tokens)[0] for j in jobs]
        texts = [self.prompt_nolead(items) for _, items in jobs]
        images = [O.load_image(v) for _, items in jobs for t, v in items if t == "image"]
        audios = [O.load_audio(v) for _, items in jobs for t, v in items if t == "audio"]
        kw = {"audio": audios, "sampling_rate": 16000} if audios else {}
        enc = self.processor(text=texts, images=images, return_tensors="pt", padding=True, **kw).to("cuda")
        enc = {k: (v.half() if v.dtype == self.torch.float32 else v) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      eos_token_id=O.IM_END, pad_token_id=O.IM_END)
        return [a.strip() for a in self.processor.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                                               skip_special_tokens=True)]


class MiniCPMoH3C(O.MiniCPMo):
    def answer(self, image_path, items, max_new_tokens=O.MAX_NEW_TOKENS):
        if not nolead(image_path):
            return super().answer(image_path, items, max_new_tokens)
        media = {"text": lambda v: v, "audio": O.load_audio, "image": O.load_image}
        content = [media[t](v) for t, v in items]
        with self.torch.no_grad():
            return self.model.chat(msgs=[{"role": "user", "content": content}], tokenizer=self.tok, sampling=False,
                                   max_new_tokens=max_new_tokens).strip()


class Phi4MMH3C(O.Phi4MM):
    def answer(self, image_path, items, max_new_tokens=O.MAX_NEW_TOKENS):
        if not nolead(image_path):
            return super().answer(image_path, items, max_new_tokens)
        parts, audios, images = [], [], []
        for t, v in items:
            if t == "text":
                parts.append(v)
            elif t == "image":
                images.append(O.load_image(v))
                parts.append(f"<|image_{len(images)}|>")
            else:
                audios.append((O.load_audio(v), 16000))
                parts.append(f"<|audio_{len(audios)}|>")
        prompt = f"<|user|>{O.ITEM_SEP.join(parts)}<|end|><|assistant|>"
        kw = {"audios": audios} if audios else {}
        enc = self.processor(text=prompt, images=images, return_tensors="pt", **kw).to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      generation_config=self.gen_cfg)
        return self.processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


class InternOmniH3C(O.InternOmni):
    def answer(self, image_path, items, max_new_tokens=O.MAX_NEW_TOKENS):
        if not nolead(image_path):
            return super().answer(image_path, items, max_new_tokens)
        torch = self.torch
        pix, parts, audio = [], [], None
        for t, v in items:
            if t == "text":
                parts.append(v)
            elif t == "image":
                pix.append(self._pixels(v))
                parts.append("<image>")
            else:
                assert audio is None, "InternOmni adapter: one audio item per prompt"
                audio = self._audio(v)
                parts.append("<audio>")
        conv = self.conv.copy()
        conv.system_message = self.model.system_message
        conv.append_message(conv.roles[0], O.ITEM_SEP.join(parts))
        conv.append_message(conv.roles[1], None)
        query = conv.get_prompt()
        for p in pix:
            query = query.replace("<image>", "<img>" + "<IMG_CONTEXT>" * (self.model.num_image_token * len(p))
                                  + "</img>", 1)
        kw = {}
        if audio is not None:
            feats, after_cnn, n_tok = audio
            query = query.replace("<audio>", "<audio>" + "<AUDIO_CONTEXT>" * n_tok + "</audio>", 1)
            kw = dict(audio_values=feats.to(self.device, dtype=torch.float16),
                      audio_len_after_cnn=torch.tensor([after_cnn]), audio_token_num=torch.tensor([n_tok]))
        enc = self.tok(query, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(pixel_values=torch.cat(pix).to(self.device, dtype=torch.float16),
                                      input_ids=enc["input_ids"].to(self.device), attention_mask=enc["attention_mask"].to(self.device),
                                      max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=self.eos, **kw)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0].split(self.conv.sep.strip())[0].strip()


def load_model_h3c(key):
    if key.startswith("qwen2.5-omni"):
        return OmniH3C(MODELS[key])
    if key.startswith("minicpm-o"):
        return MiniCPMoH3C(MODELS[key])
    if key.startswith("phi-4"):
        return Phi4MMH3C(MODELS[key])
    if key == "internomni":
        return InternOmniH3C(MODELS[key])
    raise NotImplementedError(key)


O.load_model = load_model_h3c   # in this process only: AnswerCache.generate looks it up in omnilib's namespace


def inputs(s, cond):
    q, i, img = s["question"], s["negation"], s["image"]
    return {"C0": (img, [("text", q)]),
            "T": (img, [("text", q), ("text", i)]),                    # = run_h1v2.inputs -> cache hits
            "A": (img, [("text", q), ("audio", s["negation_audio"])]),
            "T_first": (NOLEAD + img, [("text", i), ("image", img), ("text", q)]),
            "A_first": (NOLEAD + img, [("audio", s["negation_audio"]), ("image", img), ("text", q)])}[cond]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-omni-7b", choices=list(MODELS))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", type=int, help="smoke: N units per branch from pilot1000")
    g.add_argument("--n", type=int, choices=[200, 1000], help="200 = first 40 per branch of pilot1000; 1000 = all")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--score_only", action="store_true")
    ap.add_argument("--generate_only", action="store_true")
    args = ap.parse_args()

    samples, _ = H.select_samples(None, False, 42, str(PILOT_LIST))
    k = args.pilot or (args.n // 5)
    per = {}
    samples = [s for s in samples if per.setdefault(s["branch"], []).append(s) or len(per[s["branch"]]) <= k]
    tag = f"h3c_smoke{args.pilot}" if args.pilot else f"h3c_pilot{args.n}"
    plan = [(s, c) + inputs(s, c) for s in samples for c in CONDS]
    for s, c, img, items in plan:
        for f in [img.replace(NOLEAD, "")] + [v for t, v in items if t in ("audio", "image")]:
            if not (DATA / f).exists():
                sys.exit(f"STOP: missing input {f} ({s['id']}, {c})")
    print(f"{len(samples)} units, {len(plan)} answers planned ({CONDS})", flush=True)
    out = RESULTS / args.model
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
        rows.append({"sample_id": s["id"], "pair_of": s["pair_of"], "unit_id": s["unit_id"], "image_id": s["image_id"],
                     "image": s["image"], "form": s["form"], "condition": c, "position": "first" if c.endswith("_first") else "after",
                     "branch": s["branch"], "source": s["source"], "split": s["split"], "split_group": split_group(s["split"]),
                     "template_id": s["template_id"], "voice": s["voice"], "asr_ok": s["asr_ok"],
                     "question": s["question"], "negation": s["negation"], "target": s["target"],
                     "components": s["components"], "answer": clean(raw), "raw_answer": raw, "_s": s})
    # ---- scoring: the same as run_h2.py / run_neu.py / run_h1v2.main() -----------------------------------------
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
    with open(out / f"scored_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "components"}, ensure_ascii=False) + "\n")
    with open(out / f"review_{tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "condition", "question", "negation", "answer", "target_label", "violation",
                    "required_recall", "judge_source"])
        for r in rows:
            w.writerow([r["sample_id"], r["condition"], r["question"], r["negation"], r["answer"], r["target_label"],
                        int(r["violation"]), round(r["required_recall"], 2), r["judge_source"]])
    c0 = {r["unit_id"] for r in rows if r["condition"] == "C0" and r["violation"]}
    print(f"\nC0 states the target in {len(c0)}/{sum(r['condition'] == 'C0' for r in rows)} units")
    for c in CONDS[1:]:
        rs = [r for r in rows if r["condition"] == c and r["unit_id"] in c0]
        if rs:
            print(f"{c:<8} n={len(rs):>4} violation {100 * sum(r['violation'] for r in rs) / len(rs):5.1f}%  "
                  f"recall {100 * sum(r['required_recall'] for r in rs) / len(rs):5.1f}%")
    print(f"files: {out / f'scored_{tag}.jsonl'}")


if __name__ == "__main__":
    main()
