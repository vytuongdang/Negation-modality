"""Shared code for the dataset-v2 experiments (H1 v2, H2): data loading, model adapters, answer cache, scoring.

A model input is an image path plus an ordered list of items after it: ("text", str) or ("audio", wav path).
Input order is always image first, then the items in the given order. Answers are cached in a JSONL file keyed by
sha1(model, image, items, max_new_tokens), so identical inputs (e.g. C0 shared by template-sweep copies and by
the exceptive / affirmative forms of one unit) are generated once and every run is resumable.
"""
import hashlib
import json
import re
import sys
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
MAX_NEW_TOKENS = 256
IM_END = 151645  # <|im_end|>
BRANCHES = ["existence", "existence_closed", "count", "attribute", "relation"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- data

def load_dataset():
    """Every sample of dataset v2 with audio: compact row merged with the full record (components, expected,
    split) and the ASR flag of its audio. -> {sample_id: dict}"""
    full = {s["sample_id"]: s for s in read_jsonl(OUT / "samples.jsonl")}
    forms = OUT / "samples_forms.jsonl"
    if forms.exists():
        full.update({s["sample_id"]: s for s in read_jsonl(forms)})
    asr = {m["text"]: m for m in read_jsonl(DATA / "audio" / "negation_sentences" / "manifest.jsonl")}
    out = {}
    for r in read_jsonl(OUT / "samples_compact_all_audio.jsonl"):
        s = full[r["id"]]
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
        content = [{"type": "image"}] + [{"type": t, **({"text": v} if t == "text" else {})} for t, v in items]
        messages = [{"role": "system", "content": [{"type": "text", "text": QWEN_SYSTEM}]},
                    {"role": "user", "content": content}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def answer_batch(self, jobs, max_new_tokens=MAX_NEW_TOKENS):
        """jobs: [(image_path, items)] -> [answer]"""
        texts = [self.prompt(items) for _, items in jobs]
        images = [load_image(p) for p, _ in jobs]
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


def load_model(key):
    if key.startswith("qwen2.5-omni"):
        return Omni(MODELS[key])
    raise NotImplementedError(f"{key}: adapter not ported yet (see experiments/h1/run_h1.py)")


# ---------------------------------------------------------------- answer cache

def input_key(model, image, items, max_new_tokens=MAX_NEW_TOKENS):
    payload = json.dumps([model, str(image), [list(i) for i in items], max_new_tokens], ensure_ascii=False)
    return hashlib.sha1(payload.encode()).hexdigest()


class AnswerCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.d = {r["key"]: r for r in read_jsonl(self.path)} if self.path.exists() else {}

    def get(self, k):
        r = self.d.get(k)
        return None if r is None else r["answer"]

    def generate(self, model_key, jobs, batch_size=4, log_every=20):
        """jobs: [(image_rel, items, tag)] with image_rel relative to DATA and audio paths relative to DATA.
        Generates every uncached job (grouped by tag so a batch has the same input shape)."""
        todo, seen = [], set()
        for img, items, tag in jobs:
            k = input_key(model_key, img, items)
            if k not in self.d and k not in seen:
                seen.add(k)
                todo.append((k, img, items, tag))
        print(f"{len(todo)} answers to generate ({len(jobs)} jobs)", flush=True)
        if not todo:
            return
        todo.sort(key=lambda j: (j[3], j[1]))
        model = load_model(model_key)
        with open(self.path, "a") as f:
            for i in range(0, len(todo), batch_size):
                chunk = todo[i:i + batch_size]
                abs_jobs = [(DATA / img, [(t, str(DATA / v) if t == "audio" else v) for t, v in items])
                            for _, img, items, _ in chunk]
                try:
                    answers = model.answer_batch(abs_jobs)
                except model.torch.cuda.OutOfMemoryError:
                    model.torch.cuda.empty_cache()
                    answers = [model.answer_batch([j])[0] for j in abs_jobs]
                for (k, img, items, tag), a in zip(chunk, answers):
                    r = {"key": k, "model": model_key, "image": str(img), "items": [list(x) for x in items],
                         "tag": tag, "answer": a}
                    self.d[k] = r
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                if (i // batch_size) % max(1, log_every // batch_size) == 0:
                    print(f"[{i + len(chunk)}/{len(todo)}] {chunk[0][3]}: {answers[0][:100]!r}", flush=True)
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
    llm = judge.judge([(r["components"], r["answer"]) for r in rows], regex=regex)
    src = judge.last_source
    judge.close()
    for r, rg, lm, s in zip(rows, regex, llm, src):
        r["mentioned"] = lm if lm is not None else rg
        r["mentioned_regex"] = rg
        r["judge_source"] = s if lm is not None else "regex_fallback"
    return rows
