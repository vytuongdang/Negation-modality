"""Shared code for the dataset-v2 experiments (H1 v2, H2): data loading, model adapters, answer cache, scoring.

A model input is an image path plus an ordered list of items after it: ("text", str), ("audio", wav path) or
("image", path) (a second image, e.g. the V_card instruction card).
Input order is always image first, then the items in the given order. Answers are cached in a JSONL file keyed by
sha1(model, image, items, max_new_tokens), so identical inputs (e.g. C0 shared by template-sweep copies and by
the exceptive / affirmative forms of one unit) are generated once and every run is resumable.
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "create_data" / "dataset"
OUT = DATA / "out"
sys.path.insert(0, str(ROOT / "create_data" / "src"))

MODELS = {
    "qwen2.5-omni-7b": "Qwen/Qwen2.5-Omni-7B",
    "qwen2.5-omni-3b": "Qwen/Qwen2.5-Omni-3B",
    "gemma-3n-e4b": "google/gemma-3n-E4B-it",
    "minicpm-o-2.6": "openbmb/MiniCPM-o-2_6",
    "phi-4-mm": "microsoft/Phi-4-multimodal-instruct",
}
QWEN_SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving "
               "auditory and visual inputs, as well as generating text and speech.")
MAX_SIDE = 1024
MAX_NEW_TOKENS = 100  # answers are short lists (Vy, 2026-11-19)
LONG_NEW_TOKENS = 256  # answers that hit MAX_NEW_TOKENS are generated again with this budget (Vy, 2026-11-23)
CUT_MARGIN = 3         # an answer this close to MAX_NEW_TOKENS (re-tokenized) counts as cut
ITEM_SEP = "\n"   # between the items after the image, in every model: Qwen / Phi join with it here, MiniCPM-o's
                   # chat() already joins content strings with "\n" (2026-11-23, Vy: T and A must be set up the same)
IM_END = 151645  # <|im_end|>
BRANCHES = ["existence", "existence_closed", "count", "attribute", "relation"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- data

def load_dataset(keep_double_labels=False):
    """Every sample of dataset v2 with audio: compact row merged with the full record (components, expected,
    split) and the ASR flag of its audio. -> {sample_id: dict}
    Units listed in out/excluded_double_label.jsonl (create_data/src/flag_double_labels.py) are left out."""
    full = {s["sample_id"]: s for s in read_jsonl(OUT / "samples.jsonl")}
    forms = OUT / "samples_forms.jsonl"
    if forms.exists():
        full.update({s["sample_id"]: s for s in read_jsonl(forms)})
    asr = {m["text"]: m for m in read_jsonl(DATA / "audio" / "negation_sentences" / "manifest.jsonl")}
    excl = OUT / "excluded_double_label.jsonl"   # COCO units with one object labelled twice (Vy, 2026-11-23)
    dropped = {r["unit_id"] for r in read_jsonl(excl)} if excl.exists() and not keep_double_labels else set()
    out = {}
    for r in read_jsonl(OUT / "samples_compact_all_audio.jsonl"):
        s = full[r["id"]]
        if s["validation"]["unit_id"] in dropped:
            continue
        m = asr.get(r["negation"], {})
        out[r["id"]] = {
            "id": r["id"], "image": r["image"], "question": r["question"], "negation": r["negation"],
            "form": r["form"], "pair_of": r["pair_of"], "branch": s["branch"], "source": s["source"]["dataset"],
            "split": s["source"]["split"], "unit_id": s["validation"]["unit_id"],
            "template_id": s["negation"]["template_id"], "template_sweep": s["negation"].get("template_sweep", False),
            "components": s["components"], "expected": s["expected"], "target": s["negation"]["target"],
            "negation_image": r["negation_image"], "box_image": r["box_image"],
            "occludes_subject": r["occludes_subject"], "subject_frac_covered": r.get("subject_frac_covered"),
            "negation_audio": r["negation_audio"], "voice": r["negation_voice"],
            "asr_ok": m.get("asr_ok"), "audio_duration": m.get("duration"),
        }
    return out


def split_group(split):
    return "val" if split in ("val", "val2017", "test") else "train"


# ---------------------------------------------------------------- media

def load_image(path):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    scale = MAX_SIDE / max(img.size)
    return img.resize((round(img.width * scale), round(img.height * scale))) if scale < 1 else img


def load_audio(path, sr=16000):
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    wav, orig = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    g = gcd(sr, orig)
    return resample_poly(wav, sr // g, orig // g).astype("float32")


# ---------------------------------------------------------------- models

class Omni:
    """Qwen2.5-Omni (3B / 7B), thinker only, fp16, greedy, batched with left padding."""
    def __init__(self, name):
        import torch
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        self.torch = torch
        self.processor = Qwen2_5OmniProcessor.from_pretrained(name)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            name, dtype=torch.float16).to("cuda").eval()

    def prompt(self, items):
        content = [{"type": "image"}]
        for k, (t, v) in enumerate(items):   # items joined by exactly "\n", the same in every adapter (ITEM_SEP)
            if k:
                content.append({"type": "text", "text": ITEM_SEP})
            content.append({"type": t, **({"text": v} if t == "text" else {})})   # "image" / "audio": placeholders
        messages = [{"role": "system", "content": [{"type": "text", "text": QWEN_SYSTEM}]},
                    {"role": "user", "content": content}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def tokenizer(self):
        return self.processor.tokenizer

    def answer_batch(self, jobs, max_new_tokens=MAX_NEW_TOKENS):
        """jobs: [(image_path, items)] -> [answer]"""
        texts = [self.prompt(items) for _, items in jobs]
        images = [load_image(x) for p, items in jobs for x in [p] + [v for t, v in items if t == "image"]]
        audios = [load_audio(v) for _, items in jobs for t, v in items if t == "audio"]
        kw = {"audio": audios, "sampling_rate": 16000} if audios else {}
        enc = self.processor(text=texts, images=images, return_tensors="pt", padding=True, **kw).to("cuda")
        enc = {k: (v.half() if v.dtype == self.torch.float32 else v) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      eos_token_id=IM_END, pad_token_id=IM_END)
        return [a.strip() for a in self.processor.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                                               skip_special_tokens=True)]

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


class _Single:
    """Base for models without batched generation: answer_batch loops over jobs."""
    def answer_batch(self, jobs, max_new_tokens=MAX_NEW_TOKENS):
        return [self.answer(img, items, max_new_tokens) for img, items in jobs]

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


class MiniCPMo(_Single):
    """MiniCPM-o 2.6, remote code (needs transformers 4.44: env `omni_neg`, run with --generate_only, score in
    `negation` with --score_only). TTS head not loaded. Content: image, then items in order."""

    def tokenizer(self):
        return self.tok

    def __init__(self, name):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True, attn_implementation="sdpa",
                                               torch_dtype=torch.float16, init_vision=True, init_audio=True,
                                               init_tts=False).eval().cuda()
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    def answer(self, image_path, items, max_new_tokens=MAX_NEW_TOKENS):
        media = {"text": lambda v: v, "audio": load_audio, "image": load_image}
        content = [load_image(image_path)] + [media[t](v) for t, v in items]
        with self.torch.no_grad():
            return self.model.chat(msgs=[{"role": "user", "content": content}], tokenizer=self.tok, sampling=False,
                                   max_new_tokens=max_new_tokens).strip()


class Phi4MM(_Single):
    """Phi-4-multimodal-instruct, remote code (env `omni_neg`, as MiniCPM-o). Prompt: image, then items in order."""

    def tokenizer(self):
        return self.processor.tokenizer

    def __init__(self, name):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.float16,
                                                          _attn_implementation="sdpa").cuda().eval()
        self.gen_cfg = GenerationConfig.from_pretrained(name)

    def answer(self, image_path, items, max_new_tokens=MAX_NEW_TOKENS):
        parts, audios, images = [], [], [load_image(image_path)]
        for t, v in items:
            if t == "text":
                parts.append(v)
            elif t == "image":
                images.append(load_image(v))
                parts.append(f"<|image_{len(images)}|>")
            else:
                audios.append((load_audio(v), 16000))
                parts.append(f"<|audio_{len(audios)}|>")
        prompt = f"<|user|><|image_1|>{ITEM_SEP.join(parts)}<|end|><|assistant|>"
        kw = {"audios": audios} if audios else {}
        enc = self.processor(text=prompt, images=images, return_tensors="pt", **kw).to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      generation_config=self.gen_cfg)
        return self.processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def load_tokenizer(key):
    """Tokenizer only (no weights): to count answer tokens when nothing has to be generated."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODELS[key], trust_remote_code=True)


def load_model(key):
    if key.startswith("qwen2.5-omni"):
        return Omni(MODELS[key])
    if key.startswith("minicpm-o"):
        return MiniCPMo(MODELS[key])
    if key.startswith("phi-4"):
        return Phi4MM(MODELS[key])
    raise NotImplementedError(f"{key}: adapter not ported yet (see experiments/h1/run_h1.py)")


# ---------------------------------------------------------------- answer cache

def input_key(model, image, items, max_new_tokens=MAX_NEW_TOKENS):
    key = [model, str(image), [list(i) for i in items], max_new_tokens]
    if len(items) > 1:   # prompts with several items render differently since ITEM_SEP (2026-11-23): new cache keys
        key.append("sep-v2")
    payload = json.dumps(key, ensure_ascii=False)
    return hashlib.sha1(payload.encode()).hexdigest()


class AnswerCache:
    """Answers keyed by input_key. Every job is first generated with MAX_NEW_TOKENS; answers that hit that limit
    (cut mid-sentence, mostly MiniCPM-o's step-by-step answers) are generated again with LONG_NEW_TOKENS and
    stored under their own key; final() returns the long answer when there is one (Vy, 2026-11-23). Greedy
    decoding, so the long answer starts with the same text as the cut one."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.d = {r["key"]: r for r in read_jsonl(self.path)} if self.path.exists() else {}

    def get(self, k):
        r = self.d.get(k)
        return None if r is None else r["answer"]

    def final(self, model_key, img, items):
        """The answer to use: the LONG_NEW_TOKENS one if the first answer was cut, else the first one."""
        long = self.get(input_key(model_key, img, items, LONG_NEW_TOKENS))
        return long if long is not None else self.get(input_key(model_key, img, items))

    def _run(self, model, model_key, todo, max_new_tokens, batch_size, log_every):
        t0, n_done = time.time(), 0
        with open(self.path, "a") as f:
            for i in range(0, len(todo), batch_size):
                chunk = todo[i:i + batch_size]
                abs_jobs = [(DATA / img, [(t, str(DATA / v) if t in ("audio", "image") else v) for t, v in items])
                            for _, img, items, _ in chunk]
                try:
                    answers = model.answer_batch(abs_jobs, max_new_tokens)
                except model.torch.cuda.OutOfMemoryError:
                    model.torch.cuda.empty_cache()
                    answers = [model.answer_batch([j], max_new_tokens)[0] for j in abs_jobs]
                for (k, img, items, tag), a in zip(chunk, answers):
                    r = {"key": k, "model": model_key, "image": str(img), "items": [list(x) for x in items],
                         "tag": tag, "max_new_tokens": max_new_tokens, "answer": a}
                    self.d[k] = r
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                n_done += len(chunk)
                if (i // batch_size) % max(1, log_every // batch_size) == 0:
                    sec = (time.time() - t0) / n_done
                    print(f"[{i + len(chunk)}/{len(todo)}] {sec:.2f} s/answer, ETA {sec * (len(todo) - n_done) / 3600:.1f} h"
                          f" | {chunk[0][3]}: {answers[0][:100]!r}", flush=True)
        print(f"generated {n_done} answers ({max_new_tokens} tokens max) in {time.time() - t0:.0f} s "
              f"({(time.time() - t0) / max(1, n_done):.2f} s/answer)", flush=True)

    def generate(self, model_key, jobs, batch_size=4, log_every=20):
        """jobs: [(image_rel, items, tag)] with image_rel relative to DATA and audio paths relative to DATA.
        Generates every uncached job (grouped by tag so a batch has the same input shape), then regenerates the
        ones that hit MAX_NEW_TOKENS with LONG_NEW_TOKENS."""
        todo, seen = [], set()
        for img, items, tag in jobs:
            k = input_key(model_key, img, items)
            if k not in self.d and k not in seen:
                seen.add(k)
                todo.append((k, img, items, tag))
        print(f"{len(todo)} answers to generate ({len(jobs)} jobs)", flush=True)
        model = None
        if todo:
            todo.sort(key=lambda j: (j[3], j[1]))
            model = load_model(model_key)
            self._run(model, model_key, todo, MAX_NEW_TOKENS, batch_size, log_every)
        # answers cut at MAX_NEW_TOKENS: count output tokens with the model's own tokenizer
        tok = model.tokenizer() if model is not None else load_tokenizer(model_key)
        cut, seen = [], set()
        for img, items, tag in jobs:
            k = input_key(model_key, img, items)
            kl = input_key(model_key, img, items, LONG_NEW_TOKENS)
            a = self.get(k)
            if a is None or kl in self.d or kl in seen:
                continue
            if len(tok(a, add_special_tokens=False)["input_ids"]) >= MAX_NEW_TOKENS - CUT_MARGIN:
                seen.add(kl)
                cut.append((kl, img, items, tag))
        print(f"{len(cut)} answers hit {MAX_NEW_TOKENS} tokens -> regenerate with {LONG_NEW_TOKENS}", flush=True)
        if cut:
            cut.sort(key=lambda j: (j[3], j[1]))
            model = model or load_model(model_key)
            self._run(model, model_key, cut, LONG_NEW_TOKENS, batch_size, log_every)
        if model is not None:
            model.close()


# ---------------------------------------------------------------- scoring

def clean(answer):
    """Cut a run-on answer at the first invented follow-up turn."""
    return re.split(r"\n\s*(?:Human|Assistant|User)\b", answer)[0].strip()


def regex_hits(components, answer):
    from mention import mentions
    return {c["id"]: mentions(answer, c, [o for o in components if o is not c]) for c in components}


def judge_rows(rows, judge_cache):
    """rows: dicts with 'components' and 'answer'. Adds 'mentioned' (judge, regex fallback), 'mentioned_regex',
    'judge_source'."""
    from judge import Judge
    regex = [regex_hits(r["components"], r["answer"]) for r in rows]
    judge = Judge(judge_cache)
    llm = judge.judge([(r["components"], r["answer"], r["question"]) for r in rows], regex=regex)   # v3 prompt
    src = judge.last_source
    judge.close()
    for r, rg, lm, s in zip(rows, regex, llm, src):
        r["mentioned"] = lm if lm is not None else rg
        r["mentioned_regex"] = rg
        r["judge_source"] = s if lm is not None else "regex_fallback"
    return rows
