"""H1 (text vs audio): does the modality carrying the negation change negation faithfulness?

Three conditions per sample, same image and question:
    C0  no negation        image + question                    (baseline: does the model name the target?)
    C1  text negation      image + question + negation (text)  -> NF_text
    C2  audio negation     image + question + negation (audio) -> NF_audio

Models: --model qwen2.5-omni-7b (default) | qwen2.5-omni-3b | gemma-3n-e4b | minicpm-o-2.6 | phi-4-mm, greedy, text output.
Results go to results/<model>/. Runs in the `negation` env:

    cd experiments/h1 && python run_h1.py --per_branch 5        # pilot: 5 samples x 4 branches
    python run_h1.py --per_branch 5 --score_only              # no new answers, re-score cached ones
    python run_h1.py --all                                     # whole dataset, template-sweep copies included

Steps: (1) Omni answers, cached in results/answers_<tag>.jsonl (resumable); (2) which components each answer
states: LLM judge (create_data/src/judge.py, Qwen3-8B, closed set, cached in results/judge.jsonl) is the score;
the regex matcher (mention.py) is kept as a second opinion and where the two disagree a thinking-mode judge
decides; (3) summary table + results/review_<tag>.csv.

Main number: target% = how often the forbidden component is stated. Reported on all samples and on the samples
where the model states the target without negation (C0), i.e. where there was something to suppress.
"""
import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "create_data" / "dataset"
sys.path.insert(0, str(ROOT / "create_data" / "src"))
from judge import Judge  # noqa: E402
from mention import mentions  # noqa: E402

MODELS = {  # --model key -> HF id (adapter class chosen in load_model)
    "qwen2.5-omni-7b": "Qwen/Qwen2.5-Omni-7B",
    "qwen2.5-omni-3b": "Qwen/Qwen2.5-Omni-3B",
    "gemma-3n-e4b": "google/gemma-3n-E4B-it",
    "minicpm-o-2.6": "openbmb/MiniCPM-o-2_6",
    "phi-4-mm": "microsoft/Phi-4-multimodal-instruct",
}
SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving "
          "auditory and visual inputs, as well as generating text and speech.")
BRANCHES = ["existence", "count", "attribute", "relation"]
CONDITIONS = ["C0_none", "C1_text", "C2_audio"]
MAX_SIDE = 1024
IM_END = 151645  # <|im_end|>: the thinker loaded alone does not stop on it unless told
RESULTS = Path(__file__).resolve().parent / "results"  # results/<model key>/..., judge cache shared in results/


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def select(per_branch, seed, everything=False):
    """per_branch samples per branch, sources mixed round-robin, distinct images, no template-sweep copies,
    no audio flagged by the TTS ASR check. everything=True: every sample that has audio (template-sweep copies
    included; audio flagged by the ASR check is kept and marked with asr_ok=False)."""
    full = {s["sample_id"]: s for s in read_jsonl(DATA / "out" / "samples.jsonl")}
    audio = {s["id"]: s for s in read_jsonl(DATA / "out" / "samples_compact_audio.jsonl")}
    bad_audio = {sid for m in read_jsonl(DATA / "audio" / "negation_sentences" / "manifest.jsonl")
                 if not m["asr_ok"] for sid in m["sample_ids"]}
    if everything:
        return [dict(full[sid], negation_audio=audio[sid]["negation_audio"], negation_voice=audio[sid]["negation_voice"],
                     asr_ok=sid not in bad_audio) for sid in full if sid in audio]
    rng = random.Random(seed)
    chosen = []
    for b in BRANCHES:
        by_src = defaultdict(list)
        for sid, s in full.items():
            if s["branch"] == b and sid in audio and sid not in bad_audio and not s["negation"]["template_sweep"]:
                by_src[s["source"]["dataset"]].append(sid)
        pools = [rng.sample(v, len(v)) for _, v in sorted(by_src.items())]
        picked, images, k = [], set(), 0
        while len(picked) < per_branch and any(pools):
            pool = pools[k % len(pools)]
            k += 1
            while pool:
                sid = pool.pop()
                if full[sid]["image"] not in images:
                    images.add(full[sid]["image"])
                    picked.append(sid)
                    break
        chosen += [dict(full[sid], negation_audio=audio[sid]["negation_audio"],
                        negation_voice=audio[sid]["negation_voice"]) for sid in picked]
    return chosen


class Omni:
    """Qwen2.5-Omni (3B / 7B), thinker only."""
    def __init__(self, MODEL):
        import torch
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        self.torch = torch
        self.processor = Qwen2_5OmniProcessor.from_pretrained(MODEL)
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            MODEL, dtype=torch.float16).to("cuda").eval()

    @staticmethod
    def image(path):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        scale = MAX_SIDE / max(img.size)
        return img.resize((round(img.width * scale), round(img.height * scale))) if scale < 1 else img

    @staticmethod
    def audio(path, sr=16000):
        import soundfile as sf
        from math import gcd
        from scipy.signal import resample_poly
        wav, orig = sf.read(path, dtype="float32")
        g = gcd(sr, orig)
        return resample_poly(wav, sr // g, orig // g).astype("float32")

    def answer(self, image_path, text, audio_path=None, max_new_tokens=200):
        content = [{"type": "image"}, {"type": "text", "text": text}]
        if audio_path:
            content.append({"type": "audio"})
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        kw = {"audio": [self.audio(audio_path)], "sampling_rate": 16000} if audio_path else {}
        enc = self.processor(text=[prompt], images=[self.image(image_path)], return_tensors="pt", padding=True, **kw)
        enc = enc.to("cuda")
        enc = {k: (v.half() if v.dtype == self.torch.float32 else v) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      eos_token_id=IM_END, pad_token_id=IM_END)
        return self.processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


def comp_label(c):
    return c["subject"] if c["type"] == "existence" else c["text"]


def load_audio(path, sr=16000):
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    wav, orig = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    g = gcd(sr, orig)
    return resample_poly(wav, sr // g, orig // g).astype("float32")


class Gemma3n:
    """Gemma 3n (image + audio + text in, text out). bf16: fp16 overflows in Gemma; Turing emulates bf16 (slower)."""
    def __init__(self, name):
        import torch
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = Gemma3nForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()

    def answer(self, image_path, text, audio_path=None, max_new_tokens=200):
        content = [{"type": "image", "image": Omni.image(image_path)}, {"type": "text", "text": text}]
        if audio_path:
            content.append({"type": "audio", "audio": load_audio(audio_path)})
        enc = self.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                 tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        enc = {k: (v.to(self.torch.bfloat16) if v.is_floating_point() else v) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


class MiniCPMo:
    """MiniCPM-o 2.6 via its remote code (needs transformers 4.44 -> own env; run with --generate_only, then score
    in the `negation` env with --score_only). TTS head not loaded."""
    def __init__(self, name):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True, attn_implementation="sdpa",
                                               torch_dtype=torch.float16, init_vision=True, init_audio=True,
                                               init_tts=False).eval().cuda()
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    def answer(self, image_path, text, audio_path=None, max_new_tokens=200):
        content = [Omni.image(image_path), text] + ([load_audio(audio_path)] if audio_path else [])
        with self.torch.no_grad():
            return self.model.chat(msgs=[{"role": "user", "content": content}], tokenizer=self.tok, sampling=False,
                                   max_new_tokens=max_new_tokens).strip()

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


class Phi4MM:
    """Phi-4-multimodal-instruct via its remote code (pins transformers 4.48 -> same separate env as MiniCPM-o).
    Order as for the other models: image, question (+ text negation), audio."""
    def __init__(self, name):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.float16,
                                                          _attn_implementation="sdpa").cuda().eval()
        self.gen_cfg = GenerationConfig.from_pretrained(name)

    def answer(self, image_path, text, audio_path=None, max_new_tokens=200):
        prompt = f"<|user|><|image_1|>{text}{'<|audio_1|>' if audio_path else ''}<|end|><|assistant|>"
        kw = {"audios": [(load_audio(audio_path), 16000)]} if audio_path else {}
        enc = self.processor(text=prompt, images=[Omni.image(image_path)], return_tensors="pt", **kw).to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      generation_config=self.gen_cfg)
        return self.processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


def load_model(key):
    name = MODELS[key]
    if key.startswith("qwen2.5-omni"):
        return Omni(name)
    if key.startswith("gemma-3n"):
        return Gemma3n(name)
    if key.startswith("minicpm-o"):
        return MiniCPMo(name)
    if key.startswith("phi-4"):
        return Phi4MM(name)
    raise NotImplementedError(key)


def inputs(s, cond):
    if cond == "C0_none":
        return s["question"], None
    if cond == "C1_text":
        return f"{s['question']} {s['negation_text']}", None
    return s["question"], DATA / s["negation_audio"]


def clean(answer):
    """Cut a run-on answer at the first invented follow-up turn (answers cached before the eos fix)."""
    return re.split(r"\n\s*(?:Human|Assistant|User)\b", answer)[0].strip()


def regex_hits(s, answer):
    comps = s["components"]
    return {c["id"]: mentions(answer, c, [o for o in comps if o is not c]) for c in comps}


def score(s, hit, cond):
    """correct: with a negation, every required component and no forbidden one; without (C0), every component."""
    req = s["expected"]["required"]
    target = any(hit[c] for c in s["expected"]["forbidden"])
    all_req = all(hit[c] for c in req)
    return {"correct": all_req and (target if cond == "C0_none" else not target),
            "target_mentioned": target, "required_recall": sum(hit[c] for c in req) / len(req), "mentioned": hit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_branch", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="qwen2.5-omni-7b", choices=list(MODELS))
    ap.add_argument("--all", action="store_true", help="every sample in the dataset (template-sweep copies too)")
    ap.add_argument("--score_only", action="store_true")
    ap.add_argument("--generate_only", action="store_true", help="answers only (e.g. in an env without the judge)")
    args = ap.parse_args()

    tag = "all" if args.all else f"pilot{args.per_branch * len(BRANCHES)}"
    OUT = RESULTS / args.model
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / f"answers_{tag}.jsonl"
    samples = select(args.per_branch, args.seed, everything=args.all)
    print(f"{len(samples)} samples: " + ", ".join(f"{b}={sum(s['branch'] == b for s in samples)}" for b in BRANCHES))

    cache = {(r["sample_id"], r["condition"]): r for r in read_jsonl(cache_path)} if cache_path.exists() else {}
    for other in OUT.glob("answers_*.jsonl"):  # greedy decoding: an answer from another run with the same input is reused
        if other != cache_path:
            for r in read_jsonl(other):
                cache.setdefault((r["sample_id"], r["condition"]), dict(r, reused_from=other.name))

    def same_input(s, cond):
        r = cache.get((s["sample_id"], cond))
        text, audio = inputs(s, cond)
        return r is not None and r["prompt_text"] == text and r["audio"] == (str(audio.relative_to(DATA)) if audio else None)

    for s in samples:  # drop cached answers whose prompt changed since (e.g. the its -> their fix)
        for c in CONDITIONS:
            if (s["sample_id"], c) in cache and not same_input(s, c):
                del cache[(s["sample_id"], c)]
    reused = [cache[(s["sample_id"], c)] for s in samples for c in CONDITIONS
              if cache.get((s["sample_id"], c), {}).get("reused_from")]
    if reused:
        with open(cache_path, "a") as f:
            for r in reused:
                r = {k: v for k, v in r.items() if k != "reused_from"}
                cache[(r["sample_id"], r["condition"])] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"reused {len(reused)} answers from earlier runs")
    todo = [(s, c) for s in samples for c in CONDITIONS if (s["sample_id"], c) not in cache]
    print(f"{len(todo)} answers to generate")
    if todo and not args.score_only:
        model = load_model(args.model)
        with open(cache_path, "a") as f:
            for k, (s, cond) in enumerate(todo, 1):
                text, audio = inputs(s, cond)
                r = {"sample_id": s["sample_id"], "condition": cond, "prompt_text": text,
                     "audio": str(audio.relative_to(DATA)) if audio else None,
                     "answer": model.answer(DATA / s["image"], text, audio)}
                cache[(s["sample_id"], cond)] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{k}/{len(todo)}] {s['sample_id']} {cond}: {r['answer'][:90]!r}", flush=True)
        model.close()

    if args.generate_only:
        print(f"answers in {cache_path}; score with --score_only in the `negation` env")
        return
    rows = []
    for s in samples:
        for cond in CONDITIONS:
            if (s["sample_id"], cond) not in cache:
                continue
            raw = cache[(s["sample_id"], cond)]["answer"]
            a = clean(raw)
            rows.append({"sample_id": s["sample_id"], "branch": s["branch"], "source": s["source"]["dataset"],
                         "condition": cond, "template_id": s["negation"]["template_id"],
                         "voice": s["negation_voice"], "question": s["question"], "negation": s["negation_text"],
                         "answer": a, "raw_answer": raw})
    by_id = {s["sample_id"]: s for s in samples}
    regex = [regex_hits(by_id[r["sample_id"]], r["answer"]) for r in rows]
    judge = Judge(RESULTS / "judge.jsonl")
    llm = judge.judge([(by_id[r["sample_id"]]["components"], r["answer"]) for r in rows], regex=regex)
    sources = judge.last_source
    judge.close()
    for r, rg, lm, src in zip(rows, regex, llm, sources):
        s = by_id[r["sample_id"]]
        hit = lm if lm is not None else rg
        reg = score(s, rg, r["condition"])
        r.update(score(s, hit, r["condition"]), judge_source=src if lm is not None else "regex_fallback",
                 regex_mentioned=rg, regex_target_mentioned=reg["target_mentioned"],
                 llm_components=", ".join(comp_label(c) for c in s["components"] if hit[c["id"]]))
    print("judge passes:", {k: sources.count(k) for k in set(sources)})

    with open(OUT / f"scored_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    neg_tag = {"C0_none": "", "C1_text": "[text] ", "C2_audio": "[audio] "}
    with open(OUT / f"review_{tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "question", "negation", "model_answer", "stated_components", "target_mentioned",
                    "regex_target_mentioned", "judge_source", "correct"])
        for r in rows:
            neg = "" if r["condition"] == "C0_none" else neg_tag[r["condition"]] + r["negation"]
            w.writerow([r["sample_id"], r["question"], neg, r["answer"], r["llm_components"],
                        int(r["target_mentioned"]), int(r["regex_target_mentioned"]), r["judge_source"],
                        int(r["correct"])])

    # correct% (all allowed components named, forbidden one not; C0: all components named),
    # target% (forbidden component named; lower = more faithful under negation)
    print(f"\n{'branch':<10} " + "  ".join(f"{c:>20}" for c in CONDITIONS))
    print(f"{'':<10} " + "  ".join(f"{'correct% / target%':>20}" for _ in CONDITIONS))
    for b in BRANCHES + ["ALL"]:
        cells = []
        for c in CONDITIONS:
            rs = [r for r in rows if r["condition"] == c and (b == "ALL" or r["branch"] == b)]
            if not rs:
                cells.append(f"{'-':>20}")
                continue
            ok = 100 * sum(r["correct"] for r in rs) / len(rs)
            t = 100 * sum(r["target_mentioned"] for r in rs) / len(rs)
            cells.append(f"{f'{ok:.0f} / {t:.0f}  (n={len(rs)})':>20}")
        print(f"{b:<10} " + "  ".join(cells))
    # conditional on the model stating the target at C0: violation rate under text vs audio negation
    c0 = {r["sample_id"] for r in rows if r["condition"] == "C0_none" and r["target_mentioned"]}
    print(f"\ntarget% among samples whose C0 answer states the target ({len(c0)}/{len(samples)} samples)")
    print(f"{'branch':<10} {'C1_text':>12} {'C2_audio':>12} {'audio-text':>12}")
    for b in BRANCHES + ["ALL"]:
        ids = {i for i in c0 if b == "ALL" or by_id[i]["branch"] == b}
        if not ids:
            continue
        t = {c: 100 * sum(r["target_mentioned"] for r in rows if r["condition"] == c and r["sample_id"] in ids)
             / len(ids) for c in ("C1_text", "C2_audio")}
        print(f"{b:<10} {t['C1_text']:>11.0f}% {t['C2_audio']:>11.0f}% {t['C2_audio'] - t['C1_text']:>+11.0f}  (n={len(ids)})")
    for field, name in (("template_id", "template"), ("voice", "voice"), ("source", "source")):
        groups = sorted({r[field] for r in rows})
        if len(groups) < 2:
            continue
        print(f"\nsame, by {name}")
        for g in groups:
            ids = {i for i in c0 if next(r[field] for r in rows if r["sample_id"] == i) == g}
            if not ids:
                continue
            t = {c: 100 * sum(r["target_mentioned"] for r in rows if r["condition"] == c and r["sample_id"] in ids)
                 / len(ids) for c in ("C1_text", "C2_audio")}
            print(f"{g:<10} {t['C1_text']:>11.0f}% {t['C2_audio']:>11.0f}% {t['C2_audio'] - t['C1_text']:>+11.0f}  (n={len(ids)})")
    dis = sum(r["target_mentioned"] != r["regex_target_mentioned"] for r in rows)
    print(f"\nregex disagrees with the judge on the target in {dis}/{len(rows)} answers")
    print(f"\nfiles: {OUT / f'review_{tag}.csv'} (to check by hand)\n       {cache_path} {OUT / f'scored_{tag}.jsonl'}")


if __name__ == "__main__":
    main()
