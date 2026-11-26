"""Shared code for the METHOD part (Vy, 2026-11-26): reduce violations of audio / visual negative instructions on
Qwen2.5-Omni-3B / 7B, pilot1000 list only. New files only: nothing here is imported by run_h1v2.py, run_pilot.py,
run_h2.py, omnilib.py or judge.py, and nothing here edits them (they are imported read-only).

Methods (one "config" = one method with its parameters; config_id names it in caches and outputs):
    none                  unsteered answers, read from the shared H1 answer cache (results/<model>/answers.jsonl,
                          read-only): C0, T, A, V (and C0_box for the CFG baseline of V)
    steer  L, alpha, pos  activation steering: a "prohibition" direction d_L (mean hidden state at the TEXT
                          instruction tokens of "Do not mention X." minus that of "Mention X.", layer L, TRAIN images
                          only; direction.py) is added as h += alpha * d_L at the instruction positions of the AUDIO
                          (audio tokens) or IMAGE (all image tokens, or only the tokens under the text box) input,
                          during the prompt pass only (the KV cache carries it into generation)
    rand  L, alpha, pos   the same with a random direction of the same norm (control: is it the direction or the push?)
    cfg  w                contrastive / CFG decoding: next token = argmax of
                          log p(y|with instruction) + w * (log p(y|with) - log p(y|without)), restricted to tokens with
                          p(y|with) >= beta * max p(y|with) (plausibility). "without" = C0 (A, T) or C0_box (V).
    transcribe            transcribe-then-obey: step 1 the model copies the instruction (audio: transcription request;
                          image: the OCR_V input of H1), step 2 the same image + question + the copied text as its own
                          text item (= condition T with the model's own transcript).
Every answer is cached in steer/results/<model>/answers.jsonl, keyed by omnilib.input_key(model|config_id, ...),
so runs resume. Scoring = the H1 judge path unchanged (omnilib.judge_rows + run_h1v2.score / count_rule + the
judge's label pass), judge cache results/judge.jsonl (shared; only one judge writer at a time).
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
H1 = HERE.parent
sys.path.insert(0, str(H1.parent / "common"))
sys.path.insert(0, str(H1))
from omnilib import (DATA, IM_END, LONG_NEW_TOKENS, MAX_NEW_TOKENS, CUT_MARGIN, OUT, clean, input_key,  # noqa: E402
                     load_audio, load_dataset, load_image, read_jsonl, split_group)

RESULTS = HERE / "results"
PILOT = H1 / "pilot1000.json"
AUDIO_TOKEN, IMAGE_TOKEN = 151646, 151655        # Qwen2.5-Omni thinker config: audio_token_index, image_token_index
MERGE = 2                                        # vision spatial_merge_size: one LLM token = 2x2 patches of 14 px
PATCH = 14
TRANSCRIBE_A = "What does the speaker say in the audio? Copy it exactly, and say nothing else."
OCR_V = "What does the text in the white box say? Copy it exactly, and say nothing else."   # = run_h1v2.OCR["OCR_V"]
TEMPLATES = {   # create_data/src/b3_negation.TEMPLATES (frames only), to recover the {X} slot of an instruction
    "N1": "Do not mention {X}.", "N2": "Don't say anything about {X}.", "N3": "Never refer to {X}.",
    "N4": "Make no mention of {X}.", "N5": "Answer without mentioning {X}.", "N6": "Avoid mentioning {X}.",
    "N7": "Leave {X} out of your answer.",
}


# ---------------------------------------------------------------- data

def load_pilot(pilot_list=PILOT):
    """-> (train samples, val samples): the pilot1000 negative samples, split by IMAGE through the source split
    (split_group: val/val2017/test -> val). Asserts that no image is in both parts."""
    ids = [r["sample_id"] for r in json.loads(Path(pilot_list).read_text())["samples"]]
    data = load_dataset()
    missing = [i for i in ids if i not in data]
    if missing:
        raise SystemExit(f"{len(missing)} pilot ids not in the dataset (first {missing[0]})")
    boxes = {r["id"]: r.get("text_box") for r in read_jsonl(OUT / "samples_compact_all_audio.jsonl")}
    samples = [dict(data[i], text_box=boxes.get(i)) for i in ids]
    train = [s for s in samples if split_group(s["split"]) == "train"]
    val = [s for s in samples if split_group(s["split"]) == "val"]
    both = {s["image_id"] for s in train} & {s["image_id"] for s in val}
    assert not both, f"{len(both)} images in train and val"
    return train, val


def slot_x(s):
    """The {X} of the sample's instruction, from its template frame ("Make no mention of its size." -> "its size")."""
    frame = TEMPLATES[s["template_id"]]
    pat = "^" + re.escape(frame).replace(re.escape("{X}"), "(.+)") + "$"
    m = re.match(pat, s["negation"].strip(), flags=re.I)
    if not m:
        raise ValueError(f"{s['id']}: {s['negation']!r} does not match {frame!r}")
    return m.group(1)


def contrast_pair(s, contrast="mention"):
    """(negative text, positive text) for the direction. 'mention': "Do not mention X." vs "Mention X." (a minimal
    pair: same slot, only the prohibition differs). 'own': the sample's own template vs "Mention X."."""
    x = slot_x(s)
    neg = f"Do not mention {x}." if contrast == "mention" else s["negation"]
    return neg, f"Mention {x}."


def cond_inputs(s, cond):
    """Same inputs as run_h1v2.inputs for C0 / C0_box / T / A / V / OCR_V (image relative to dataset/, items)."""
    q, i = s["question"], s["negation"]
    return {"C0": (s["image"], [("text", q)]),
            "C0_box": (s["box_image"], [("text", q)]),
            "T": (s["image"], [("text", q), ("text", i)]),
            "A": (s["image"], [("text", q), ("audio", s["negation_audio"])]),
            "V": (s["negation_image"], [("text", q)]),
            "OCR_V": (s["negation_image"], [("text", OCR_V)])}[cond]


# ---------------------------------------------------------------- configs

def config_id(cfg):
    """Readable, stable name of a method config (used in cache keys and file names)."""
    k = cfg["kind"]
    if k == "none":
        return "none"
    if k in ("steer", "rand"):
        return f"{k}_L{'-'.join(map(str, cfg['layers']))}_a{cfg['alpha']:g}_{cfg['pos']}_{cfg['dir_tag']}"
    if k == "cfg":
        return f"cfg_w{cfg['w']:g}_b{cfg['beta']:g}"
    if k == "transcribe":
        return "transcribe"
    raise ValueError(k)


def steer_key(model_key, cfg_id, img, items, max_new_tokens=MAX_NEW_TOKENS):
    return input_key(f"{model_key}|{cfg_id}", img, items, max_new_tokens)


# ---------------------------------------------------------------- token positions

def instruction_span(ids, tok, text):
    """[start, end) of the instruction text item in one row of input_ids: the last item before the user turn's
    <|im_end|> (T inputs end with the instruction). Checked by decoding."""
    ids = list(ids)
    end = len(ids) - 1 - ids[::-1].index(IM_END)
    n = len(tok(text, add_special_tokens=False)["input_ids"])
    start = end - n
    got = tok.decode(ids[start:end])
    if got.strip() != text.strip():
        raise ValueError(f"instruction span mismatch: {got!r} != {text!r}")
    return start, end


def box_token_mask(grid_hw, img_size, box):
    """Image tokens (row-major over the merged grid) that overlap the text box. grid_hw = (h, w) in 14-px patches
    from image_grid_thw; img_size = (W, H) of the image file; box = [x, y, w, h] in file pixels (render_visual.py text_box). The processor
    resizes the image to exactly (w*14, h*14) (after omnilib.load_image's MAX_SIDE resize, same aspect)."""
    gh, gw = grid_hw[0] // MERGE, grid_hw[1] // MERGE
    sx, sy = grid_hw[1] * PATCH / img_size[0], grid_hw[0] * PATCH / img_size[1]
    x0, y0, x1, y1 = box[0] * sx, box[1] * sy, (box[0] + box[2]) * sx, (box[1] + box[3]) * sy
    cell = PATCH * MERGE
    return [(r * cell < y1 and (r + 1) * cell > y0 and c * cell < x1 and (c + 1) * cell > x0)
            for r in range(gh) for c in range(gw)]


def position_mask(input_ids, pos, grids=None, sizes=None, boxes=None, texts=None, tok=None):
    """Bool mask (B, S) of the positions to steer. pos: 'audio' (audio tokens), 'image' (all image tokens of the
    FIRST image of each row), 'box' (image tokens under the text box), 'text' (the instruction text span).
    grids: per row (h, w) of its first image; sizes / boxes / texts: per row."""
    import torch
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for b in range(input_ids.shape[0]):
        row = input_ids[b]
        if pos == "audio":
            mask[b] = row == AUDIO_TOKEN
        elif pos in ("image", "box"):
            idx = (row == IMAGE_TOKEN).nonzero().flatten()
            n = (grids[b][0] // MERGE) * (grids[b][1] // MERGE)
            idx = idx[:n]   # first image of the row
            if pos == "box":
                keep = torch.tensor(box_token_mask(grids[b], sizes[b], boxes[b]), dtype=torch.bool)
                assert len(keep) == len(idx), (len(keep), len(idx))
                idx = idx[keep]
            mask[b, idx] = True
        elif pos == "text":
            s, e = instruction_span(row.tolist(), tok, texts[b])
            mask[b, s:e] = True
        else:
            raise ValueError(pos)
    return mask


# ---------------------------------------------------------------- hooks

class Steerer:
    """Adds alpha * vec[L] at masked positions of the output of decoder layers L, only on the prompt pass (output
    length == mask length). Use as a context manager around model.generate."""
    def __init__(self, layers_module, vecs, alpha, mask):
        self.layers, self.vecs, self.alpha, self.mask, self.handles = layers_module, vecs, alpha, mask, []

    def _hook(self, L):
        def f(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] != self.mask.shape[1]:
                return out   # decoding step (1 new token): no steering
            v = self.vecs[L].to(h.device, h.dtype)
            h = h + self.alpha * self.mask.to(h.device).unsqueeze(-1).to(h.dtype) * v
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return f

    def __enter__(self):
        self.handles = [self.layers[L].register_forward_hook(self._hook(L)) for L in self.vecs]
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


class CFGProcessor:
    """Rows come in pairs (2k = with instruction, 2k+1 = without). Both rows of a pair get the same next token:
    argmax over plausible tokens of lw + w * (lw - lwo) (log-softmax), plausible = p_with >= beta * max p_with."""
    def __init__(self, w, beta):
        self.w, self.beta = w, beta

    def __call__(self, input_ids, scores):
        import math
        import torch
        lp = torch.log_softmax(scores.float(), dim=-1)
        lw, lwo = lp[0::2], lp[1::2]
        comb = lw + self.w * (lw - lwo)
        comb = comb.masked_fill(lw < lw.max(-1, keepdim=True).values + math.log(self.beta), float("-inf"))
        nxt = comb.argmax(-1)
        out = torch.full_like(scores, float("-inf"))
        rows = torch.arange(nxt.shape[0], device=scores.device)
        out[2 * rows, nxt] = 0.0
        out[2 * rows + 1, nxt] = 0.0
        return out


# ---------------------------------------------------------------- model runner

class SteerModel:
    """Qwen2.5-Omni thinker (omnilib.Omni, unchanged) with steering / CFG / plain batched greedy generation."""
    def __init__(self, model_key):
        from omnilib import load_model
        self.key = model_key
        self.omni = load_model(model_key)
        self.torch = self.omni.torch
        self.model, self.proc = self.omni.model, self.omni.processor
        self.tok = self.proc.tokenizer
        self.layers = self.model.model.layers
        self.device = "cuda"

    def encode(self, jobs):
        """jobs: [(abs image path, [(type, abs path or text)])] -> processor batch on self.device (fp16 floats), grids."""
        texts = [self.omni.prompt(items) for _, items in jobs]
        images = [load_image(x) for p, items in jobs for x in [p] + [v for t, v in items if t == "image"]]
        audios = [load_audio(v) for _, items in jobs for t, v in items if t == "audio"]
        kw = {"audio": audios, "sampling_rate": 16000} if audios else {}
        enc = self.proc(text=texts, images=images, return_tensors="pt", padding=True, **kw).to(self.device)
        enc = {k: (v.half() if v.dtype == self.torch.float32 else v) for k, v in enc.items()}
        # first image of each row: rows have 1 + (#extra images) images, in order
        grids, k = [], 0
        for _, items in jobs:
            g = enc["image_grid_thw"][k].tolist()
            grids.append((g[1], g[2]))
            k += 1 + sum(t == "image" for t, _ in items)
        return enc, grids

    def _gen(self, enc, max_new_tokens, **kw):
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      eos_token_id=IM_END, pad_token_id=IM_END, **kw)
        return [a.strip() for a in self.proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]

    def plain(self, jobs, max_new_tokens):
        enc, _ = self.encode(jobs)
        return self._gen(enc, max_new_tokens)

    def steer(self, jobs, vecs, alpha, pos, max_new_tokens, sizes=None, boxes=None, texts=None):
        enc, grids = self.encode(jobs)
        mask = position_mask(enc["input_ids"], pos, grids, sizes, boxes, texts, self.tok)
        if not mask.any(dim=1).all():
            raise RuntimeError(f"no {pos} positions found in some rows")
        with Steerer(self.layers, vecs, alpha, mask):
            return self._gen(enc, max_new_tokens)

    def cfg(self, pairs, w, beta, max_new_tokens):
        """pairs: [(job with instruction, job without)] -> answers (one per pair)."""
        from transformers import LogitsProcessorList
        enc, _ = self.encode([j for p in pairs for j in p])
        ans = self._gen(enc, max_new_tokens, logits_processor=LogitsProcessorList([CFGProcessor(w, beta)]))
        return ans[0::2]

    def span_means(self, jobs, texts, layers):
        """Mean hidden state (output of each decoder layer in `layers`) over the instruction text span of each row.
        -> {L: tensor (B, H) float32}"""
        enc, _ = self.encode(jobs)
        mask = position_mask(enc["input_ids"], "text", texts=texts, tok=self.tok)
        got = {}

        def mk(L):
            def f(_m, _i, out):
                h = (out[0] if isinstance(out, tuple) else out).float()
                m = mask.to(h.device).unsqueeze(-1).float()
                got[L] = ((h * m).sum(1) / m.sum(1)).cpu()
            return f
        hs = [self.layers[L].register_forward_hook(mk(L)) for L in layers]
        try:
            with self.torch.no_grad():
                self.model(**enc, use_cache=False)
        finally:
            for h in hs:
                h.remove()
        return got

    def n_tokens(self, text):
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def close(self):
        self.omni.close()


# ---------------------------------------------------------------- cache

class SteerCache:
    """Append-only answer cache of the steering runs (never the shared H1 cache)."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.d = {r["key"]: r for r in read_jsonl(self.path)} if self.path.exists() else {}

    def get(self, k):
        r = self.d.get(k)
        return None if r is None else r["answer"]

    def put_many(self, recs):
        with open(self.path, "a") as f:
            for r in recs:
                self.d[r["key"]] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_cut(n_tokens):
    return n_tokens >= MAX_NEW_TOKENS - CUT_MARGIN


def file_hash(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:10]


__all__ = ["DATA", "LONG_NEW_TOKENS", "MAX_NEW_TOKENS", "clean", "load_pilot", "cond_inputs", "contrast_pair",
           "config_id", "steer_key", "SteerModel", "SteerCache", "Steerer", "CFGProcessor", "position_mask",
           "box_token_mask", "instruction_span", "is_cut", "file_hash", "RESULTS", "TRANSCRIBE_A", "slot_x"]
