"""Synthesise a list of sentences into a SEPARATE folder (for H2 / H3 experiments), never touching dataset audio.

Same TTS setup as create_data/src/tts_negation.py (CosyVoice2-0.5B, the same two LibriSpeech reference voices,
Whisper check, up to MAX_TRIES seeds), but:
  - input: a JSONL file of {"text": ..., "voice": "female"|"male"} rows (one per wanted audio file);
  - output: <out_dir>/<voice>_<sha1(text)[:10]>.wav + <out_dir>/manifest.jsonl; nothing is ever deleted;
  - resumable: rows whose wav already exists in the manifest are skipped.

    PYTHONNOUSERSITE=1 ~/anaconda3/envs/cosyvoice/bin/python tts_sentences.py --src sentences.jsonl --out_dir DIR
"""
import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "create_data" / "src"))
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from tts_negation import AUDIO_DIR, MAX_TRIES, SEED, VOICES, load_asr, load_tts, norm  # noqa: E402


def wav_name(text, voice):
    return f"{voice}_{hashlib.sha1(text.encode()).hexdigest()[:10]}.wav"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.jsonl"
    done = {}
    if man_path.exists():
        for l in open(man_path):
            if l.strip():
                m = json.loads(l)
                if (out / m["file"]).exists():
                    done[(m["text"], m["voice"])] = m
    want = []
    for l in open(args.src):
        if l.strip():
            r = json.loads(l)
            if (r["text"], r["voice"]) not in done and (r["text"], r["voice"]) not in want:
                want.append((r["text"], r["voice"]))
    want = want[:args.limit]
    print(f"{len(done)} done, {len(want)} to synthesise -> {out}", flush=True)
    if not want:
        return
    tts, set_seed = load_tts()
    asr = load_asr()
    for v, cfg in VOICES.items():  # reference clips already copied there by tts_negation.py
        tts.add_zero_shot_spk(cfg["text"], str(AUDIO_DIR / "prompts" / f"{v}.wav"), v)
    failed = 0
    with open(man_path, "a") as f:
        for i, (text, v) in enumerate(want, 1):
            best = None
            for attempt in range(MAX_TRIES):
                set_seed(SEED + attempt)
                wav = torch.cat([o["tts_speech"] for o in tts.inference_zero_shot(
                    text, "", "", zero_shot_spk_id=v, stream=False)], dim=1).squeeze(0).float().cpu().numpy()
                heard = asr({"raw": wav, "sampling_rate": tts.sample_rate},
                            generate_kwargs={"language": "en"})["text"].strip()
                score = difflib.SequenceMatcher(None, norm(text), norm(heard)).ratio()
                if best is None or score > best[0]:
                    best = (score, wav, heard, attempt)
                if norm(heard) == norm(text):
                    break
            score, wav, heard, attempt = best
            name = wav_name(text, v)
            sf.write(out / name, wav, tts.sample_rate)
            ok = norm(heard) == norm(text)
            failed += not ok
            f.write(json.dumps({"text": text, "voice": v, "file": name, "sample_rate": tts.sample_rate,
                                "duration": round(len(wav) / tts.sample_rate, 2), "seed": SEED + attempt,
                                "asr": heard, "asr_ok": ok, "asr_similarity": round(score, 3)}) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"{i}/{len(want)}, asr mismatches: {failed}", flush=True)
    print(f"done: {len(want)} files, {failed} ASR mismatches")


if __name__ == "__main__":
    main()
