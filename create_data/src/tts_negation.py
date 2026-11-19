"""Audio version of the negation instructions (CosyVoice2-0.5B, zero-shot voice cloning).

Runs in the `cosyvoice` conda env (CosyVoice pins torch 2.3 / transformers 4.51), not in `negation`:

    PYTHONNOUSERSITE=1 ~/anaconda3/envs/cosyvoice/bin/python tts_negation.py

Each unique negation sentence is synthesised once, in one voice: the sentences are shuffled with SEED and
split half female, half male. Re-running reuses every sentence already in the manifest (same file, same voice)
and only synthesises new ones, assigned to keep the voices balanced. Every audio file is transcribed with Whisper; a file whose transcript does not match the text is regenerated with a new seed
(up to MAX_TRIES), and the closest attempt is kept and flagged if none matches.

Outputs (paths in the JSON files are relative to dataset/):
    dataset/audio/prompts/{voice}.wav                reference clips (LibriSpeech dev-clean)
    dataset/audio/negation_sentences/{neg_id}.wav    one file per unique sentence, 24 kHz mono
    dataset/audio/negation_sentences/manifest.jsonl  one row per sentence: text, voice, path, duration, asr, asr_ok, sample_ids
    dataset/audio/negation/{sample_id}.wav           copy of its sentence's file, one per sample (easy to look up)
    dataset/out/samples_compact_audio.jsonl          samples_compact + "negation_audio" (path), "negation_voice"
"""
import argparse
import difflib
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
import sys
from pathlib import Path

import soundfile as sf
import torch

from config import DATA, OUT_DIR, SEED
from utils import read_jsonl, write_jsonl

TTS_ROOT = DATA.parent.parent / "tts"
COSYVOICE_REPO = TTS_ROOT / "CosyVoice"
MODEL_DIR = TTS_ROOT / "pretrained_models" / "CosyVoice2-0.5B"
ASR_MODEL = "openai/whisper-large-v3-turbo"
AUDIO_DIR = DATA / "audio"
SENT_DIR = AUDIO_DIR / "negation_sentences"
SAMPLE_DIR = AUDIO_DIR / "negation"
MAX_TRIES = 3

# LibriSpeech dev-clean clips; gender picked by median F0 (female ~244 Hz, male ~108 Hz)
VOICES = {
    "female": {"clip": "1673-143396-0002",
               "text": "Yet the most charitable criticism must refuse these sectaries any knowledge "
                       "of the pure and proper divinity of Christ."},
    "male": {"clip": "1272-128104-0000",
             "text": "Mister Quilter is the apostle of the middle classes and we are glad to welcome his gospel."},
}


def neg_id(text):
    return "neg_" + hashlib.sha1(text.encode()).hexdigest()[:10]


def norm(text):
    text = text.lower().replace("’", "'").replace("'", "")
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text).split())


def load_tts():
    sys.path[:0] = [str(COSYVOICE_REPO), str(COSYVOICE_REPO / "third_party" / "Matcha-TTS")]
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.common import set_all_random_seed
    return CosyVoice2(str(MODEL_DIR), fp16=True), set_all_random_seed


def load_asr():
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model=ASR_MODEL, torch_dtype=torch.float16, device="cuda:0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="samples_compact.jsonl")
    ap.add_argument("--dst", default="samples_compact_audio.jsonl")
    ap.add_argument("--limit", type=int, help="only the first N unique sentences (smoke test)")
    args = ap.parse_args()

    samples = read_jsonl(OUT_DIR / args.src)
    texts = list(dict.fromkeys(s["negation"] for s in samples))[:args.limit]

    # resumable: sentences in the partial manifest (interrupted run) or the final manifest (earlier run) are
    # reused with their voice; only new sentences are synthesised
    part = SENT_DIR / "manifest.partial.jsonl"
    final = SENT_DIR / "manifest.jsonl"
    prev = {}
    for src in (final, part):
        if src.exists():
            for m in read_jsonl(src):
                if (DATA / m["path"]).exists():
                    prev[m["text"]] = {k: v for k, v in m.items() if k != "sample_ids"}
    manifest = [prev[t] for t in texts if t in prev]
    done = {m["text"] for m in manifest}
    voice_of = {m["text"]: m["voice"] for m in manifest}
    new = random.Random(SEED).sample([t for t in texts if t not in done], len(texts) - len(done))
    n_voice = {v: sum(x == v for x in voice_of.values()) for v in VOICES}
    for t in new:  # keep the two voices balanced
        voice_of[t] = min(VOICES, key=lambda v: n_voice[v])
        n_voice[voice_of[t]] += 1
    print(f"{len(samples)} samples, {len(texts)} unique negations ({len(done)} reused, {len(new)} new), "
          f"voices: {list(VOICES)}", flush=True)
    if new:
        with open(part, "w") as f:
            for m in manifest:
                f.write(json.dumps(m) + "\n")
    failed = sum(not m["asr_ok"] for m in manifest)

    tts, set_seed = load_tts() if new else (None, None)
    asr = load_asr() if new else None

    prompt_dir = AUDIO_DIR / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for v, cfg in VOICES.items():
        shutil.copy(TTS_ROOT / "ref_candidates" / f"{cfg['clip']}.wav", prompt_dir / f"{v}.wav")
        if tts is not None:
            tts.add_zero_shot_spk(cfg["text"], str(prompt_dir / f"{v}.wav"), v)

    SENT_DIR.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(texts):
        if text in done:
            continue
        v = voice_of[text]
        path = SENT_DIR / f"{neg_id(text)}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        best = None
        for attempt in range(MAX_TRIES):
            set_seed(SEED + attempt)
            wav = torch.cat([o["tts_speech"] for o in tts.inference_zero_shot(
                text, "", "", zero_shot_spk_id=v, stream=False)], dim=1).squeeze(0).float().cpu().numpy()
            heard = asr({"raw": wav, "sampling_rate": tts.sample_rate}, generate_kwargs={"language": "en"})["text"].strip()
            score = difflib.SequenceMatcher(None, norm(text), norm(heard)).ratio()
            if best is None or score > best[0]:
                best = (score, wav, heard, attempt)
            if norm(heard) == norm(text):
                break
        score, wav, heard, attempt = best
        sf.write(path, wav, tts.sample_rate)
        ok = norm(heard) == norm(text)
        failed += not ok
        manifest.append({"neg_id": neg_id(text), "voice": v, "text": text,
                         "path": str(path.relative_to(DATA)), "sample_rate": tts.sample_rate,
                         "duration": round(len(wav) / tts.sample_rate, 2), "seed": SEED + attempt,
                         "asr": heard, "asr_ok": ok, "asr_similarity": round(score, 3)})
        with open(part, "a") as f:
            f.write(json.dumps(manifest[-1]) + "\n")
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(texts)} sentences, asr mismatches so far: {failed}", flush=True)

    by_text = {m["text"]: m for m in manifest}
    sample_ids = defaultdict(list)
    rows = []
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for s in samples:
        if s["negation"] not in by_text:
            continue
        m = by_text[s["negation"]]
        dst = SAMPLE_DIR / f"{s['id']}.wav"
        shutil.copy(DATA / m["path"], dst)
        sample_ids[m["text"]].append(s["id"])
        rows.append(dict(s, negation_audio=str(dst.relative_to(DATA)), negation_voice=m["voice"]))
    write_jsonl(final, [dict(m, sample_ids=sample_ids[m["text"]]) for m in manifest])
    part.unlink(missing_ok=True)
    # per-sample copies of samples no longer in the dataset
    keep = {f"{s['id']}.wav" for s in samples}
    for f in SAMPLE_DIR.glob("*.wav"):
        if f.name not in keep:
            f.unlink()
    write_jsonl(OUT_DIR / args.dst, rows)
    print(f"{len(manifest)} audio files ({sum(m['voice'] == 'female' for m in manifest)} female), "
          f"{len(rows)} samples ({sum(r['negation_voice'] == 'female' for r in rows)} female), "
          f"{failed} still mismatched after {MAX_TRIES} tries")
    for m in manifest:
        if not m["asr_ok"]:
            print(f"  [{m['voice']}] {m['text']!r} -> heard {m['asr']!r}")
    print(f"files: {SAMPLE_DIR} {SENT_DIR} {OUT_DIR / args.dst}")


if __name__ == "__main__":
    main()
