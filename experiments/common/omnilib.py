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
    "internomni": "OpenGVLab/InternOmni",
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

def image_id(path):
    """Stable image id (Vy 2026-11-24): '<folder>/<file stem>', e.g. 'coco/000000123456', 'vg/2345',
    'clevr/CLEVR_val_008260' (VAW and GQA share Visual Genome images: same image, same id)."""
    p = Path(str(path))
    return f"{p.parent.name}/{p.stem}"


def removal_set(paths):
    """Ids to leave out at analysis time (Vy: extra filtering later by id, nothing regenerated). Each file: jsonl
    rows with image_id / unit_id / sample_id (or image), or plain text, one id per line. -> (images, units, samples)"""
    import json as _json
    imgs, units, samples = set(), set(), set()
    for path in paths or []:
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                r = _json.loads(line)
                if r.get("image_id") or r.get("image"):
                    imgs.add(r.get("image_id") or image_id(r["image"]))
                elif r.get("unit_id"):
                    units.add(r["unit_id"])
                elif r.get("sample_id"):
                    samples.add(r["sample_id"])
            elif "/" in line:
                imgs.add(line if line.count("/") == 1 and "." not in line else image_id(line))
            else:   # unit ids (…-00000) and sample ids (…-000000[-aff|-exc]) never collide
                units.add(line)
                samples.add(line)
    return imgs, units, samples


def removed(r, rm):
    """True if a result row is covered by removal_set(...)."""
    imgs, units, samples = rm
    return (r.get("image_id") or image_id(r["image"])) in imgs or r.get("unit_id") in units \
        or r.get("sample_id") in samples or r.get("pair_of") in samples


def load_dataset(keep_excluded=False):
    """Every sample of dataset v2 with audio: compact row merged with the full record (components, expected,
    split) and the ASR flag of its audio. -> {sample_id: dict}
    Units listed in out/excluded_units.jsonl (create_data/src/flag_excluded_units.py) are left out."""
    full = {s["sample_id"]: s for s in read_jsonl(OUT / "samples.jsonl")}
    forms = OUT / "samples_forms.jsonl"
    if forms.exists():
        full.update({s["sample_id"]: s for s in read_jsonl(forms)})
    asr = {m["text"]: m for m in read_jsonl(DATA / "audio" / "negation_sentences" / "manifest.jsonl")}
    excl = OUT / "excluded_units.jsonl"   # double labels, part-whole, attribute overlap (create_data/src/flag_excluded_units.py)
    dropped = {r["unit_id"] for r in read_jsonl(excl)} if excl.exists() and not keep_excluded else set()
    out = {}
    for r in read_jsonl(OUT / "samples_compact_all_audio.jsonl"):
        s = full[r["id"]]
        if s["validation"]["unit_id"] in dropped:
            continue
        m = asr.get(r["negation"], {})
        out[r["id"]] = {
            "id": r["id"], "image": r["image"], "image_id": image_id(r["image"]), "question": r["question"], "negation": r["negation"],
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


def internomni_local_dir(repo):
    """The InternOmni repo code does `from .flash_attention import FlashAttention` in a try block, but the file is
    not in the repo, so transformers' remote-code loader fails. Load from a project folder of symlinks to the HF
    snapshot plus a stub flash_attention.py that raises ImportError -> has_flash_attn = False (eager attention,
    which TITAN RTX / Turing needs anyway). Folder: experiments/common/models/internomni (git-ignored)."""
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(repo, local_files_only=True))
    local = ROOT / "experiments" / "common" / "models" / "internomni"
    local.mkdir(parents=True, exist_ok=True)
    import shutil
    for f in snap.iterdir():   # weights as symlinks; code/config copied (symlinked code breaks relative imports)
        link = local / f.name
        if f.suffix == ".safetensors":
            if not link.exists():
                link.symlink_to(f.resolve())
        else:
            shutil.copyfile(f, link)
    (local / "flash_attention.py").write_text(
        "raise ImportError('flash-attention disabled for InternOmni (stub, see omnilib.internomni_local_dir)')\n")
    # transformers 4.44 copies only the direct relative imports of a local module (not nested ones such as
    # configuration_internlm2): put every code file into the dynamic-module folder up front
    from transformers.utils import HF_MODULES_CACHE
    mods = Path(HF_MODULES_CACHE) / "transformers_modules" / local.name
    mods.mkdir(parents=True, exist_ok=True)
    (mods / "__init__.py").touch()
    for f in local.glob("*.py"):
        shutil.copyfile(f, mods / f.name)
    return str(local)


class InternOmni(_Single):
    """OpenGVLab/InternOmni (InternViT + Whisper-large-v3 encoder + InternLM2.5-7B), remote code. Env `omni_intern`
    (clone of omni_neg + sentencepiece, timm), run with --generate_only, score in `negation`. Added 2026-11-23 (Vy).
    Prompt in the model's own internlm2-chat template: "<image>\n" then the items joined with ITEM_SEP, where an
    audio item is "<audio>" and a second image "<image>" (each expanded to its context tokens). fp16 and no
    flash-attention (TITAN RTX = Turing). Tiles: max_dynamic_patch from config (6) + thumbnail, 448 px."""

    device = "cuda"

    def tokenizer(self):
        return self.tok

    def __init__(self, name):
        import sys as _sys
        import torch
        from transformers import AutoConfig, AutoTokenizer, WhisperFeatureExtractor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        self.torch = torch
        name = internomni_local_dir(name)
        cls = get_class_from_dynamic_module("modeling_internvl_audio.InternVLChatAudioModel", name)
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        self.model = cls.from_pretrained(name, config=cfg, torch_dtype=torch.float16,
                                         low_cpu_mem_usage=True).eval().cuda()
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=False)
        self.fe = WhisperFeatureExtractor.from_pretrained(name)
        self.max_num = cfg.max_dynamic_patch
        conv = _sys.modules[type(self.model).__module__].get_conv_template(self.model.template)
        self.conv = conv
        self.eos = self.tok.convert_tokens_to_ids(conv.sep.strip())
        self.img_ctx = self.tok.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.aud_ctx = self.tok.convert_tokens_to_ids("<AUDIO_CONTEXT>")
        self.model.img_context_token_id, self.model.audio_context_token_id = self.img_ctx, self.aud_ctx

    def _pixels(self, path):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        img, size = load_image(path), 448
        w, h = img.size
        ratios = sorted({(i, j) for n in range(1, self.max_num + 1) for i in range(1, n + 1)
                         for j in range(1, n + 1) if i * j <= self.max_num}, key=lambda r: r[0] * r[1])
        best, best_diff = (1, 1), float("inf")    # = InternVL find_closest_aspect_ratio
        for r in ratios:
            d = abs(w / h - r[0] / r[1])
            if d < best_diff:
                best, best_diff = r, d
            elif d == best_diff and w * h > 0.5 * size * size * r[0] * r[1]:
                best = r
        tw, th = size * best[0], size * best[1]
        big = img.resize((tw, th))
        tiles = [big.crop((x * size, y * size, (x + 1) * size, (y + 1) * size))
                 for y in range(best[1]) for x in range(best[0])]
        if len(tiles) != 1:
            tiles.append(img.resize((size, size)))
        tf = T.Compose([T.Resize((size, size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        return self.torch.stack([tf(t) for t in tiles])

    def _audio(self, path):
        wav = load_audio(path)[:480000]
        mel_len = len(wav) // 160
        after_cnn = (mel_len + 2 - 2 - 1) // 1 + 1          # conv1: pad 1, kernel 3, stride 1
        after_cnn = (after_cnn + 2 - 2 - 1) // 2 + 1        # conv2: pad 1, kernel 3, stride 2
        n_tok = (after_cnn - 2) // 2 + 1
        feats = self.fe(wav, sampling_rate=16000, return_tensors="pt")["input_features"]
        return feats, after_cnn, n_tok

    def answer(self, image_path, items, max_new_tokens=MAX_NEW_TOKENS):
        torch = self.torch
        pix, parts, audio = [self._pixels(image_path)], [], None
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
        conv.append_message(conv.roles[0], "<image>\n" + ITEM_SEP.join(parts))
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
    if key == "internomni":
        return InternOmni(MODELS[key])
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
