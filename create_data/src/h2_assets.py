"""Assets for H2 / H3-A: each question as audio and written inside the image (dataset/h2/).

The selection is the same one run_h2.py uses (same --pilot / --all and seed as run_h1v2.py), so a small pilot can be
extended later: rows and files that already exist are reused, only new units are processed.

    # 1. images (CPU, `negation` env): the question written in the image, same box style as the visual instruction
    PYTHONNOUSERSITE=1 ~/anaconda3/envs/negation/bin/python h2_assets.py render --pilot 40      # 40 units per branch
    PYTHONNOUSERSITE=1 ~/anaconda3/envs/negation/bin/python h2_assets.py render --all
    # 2. audio (GPU, `cosyvoice` env): CosyVoice2, same two voices 50/50, Whisper check (asr_ok)
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=2 ~/anaconda3/envs/cosyvoice/bin/python h2_assets.py tts --pilot 40
    python h2_assets.py status --pilot 40                          # what exists, what is missing (no GPU, `negation` env)
    (`tts` takes no selection: it makes the audio for every row of questions.jsonl that has none)

Outputs (paths relative to dataset/):
    h2/question_image/{unit_id}.{jpg|png}   original image + the question in a white box (placed away from the objects)
    h2/question_audio/q_<sha1>.wav          one file per unique question text, 24 kHz
    h2/questions.jsonl                      one row per unit: question, q_image, q_text_box, q_occludes_subject,
                                            q_audio, q_voice, q_asr, q_asr_ok, q_asr_similarity
`render` and `tts` only need the units' questions; the instruction audio / images of H2 and H3-A are the ones the
dataset already has (negative form and the H4 forms).
"""
import argparse
import difflib
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from config import DATA, OUT_DIR, SEED
from utils import read_jsonl, write_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "common"))
import h2_common as H  # noqa: E402

IMG_DIR = H.H2_DIR / "question_image"
AUD_DIR = H.H2_DIR / "question_audio"


def save_rows(rows):
    H.H2_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(H.QFILE, sorted(rows.values(), key=lambda r: r["unit_id"]))


def cmd_render(args, samples):
    from PIL import Image

    import render_visual as rv
    full = {r["sample_id"]: r for r in read_jsonl(OUT_DIR / "samples.jsonl")}
    rows = H.load_questions()
    new = 0
    for k, s in enumerate(samples, 1):
        u = s["unit_id"]
        if u in rows and (DATA / rows[u]["q_image"]).exists() and rows[u]["question"] == s["question"]:
            continue
        src = DATA / s["image"]
        img = Image.open(src).convert("RGB")
        p = rv.place(img, s["question"], full[s["id"]]["extra"]["protect_boxes"])
        dst = IMG_DIR / (re.sub(r"[^A-Za-z0-9_.-]", "_", u) + src.suffix.lower())
        rv.save(rv.draw_box(img, p, True), dst)
        old = rows.get(u, {})
        keep = {k_: v for k_, v in old.items() if k_.startswith("q_a") or k_ == "q_voice"} \
            if old.get("question") == s["question"] else {}
        rows[u] = dict(keep, unit_id=u, sample_id=s["id"], question=s["question"], q_image=str(dst.relative_to(DATA)),
                       q_text_box=p["box"], q_font_px=p["size"], q_occludes_subject=p["subject_px_covered"] > 0)
        new += 1
        if new % 500 == 0:
            print(f"{new} question images rendered", flush=True)
    save_rows(rows)
    print(f"{len(samples)} units selected, {new} question images rendered, {len(rows)} rows in {H.QFILE}; "
          f"question occludes an object of the question in {sum(r['q_occludes_subject'] for r in rows.values())} rows")


def cmd_tts(args, samples=None):
    """Audio for every row of questions.jsonl that has none (rows exist for the units `render` was run on)."""
    import tts_negation as T   # cosyvoice env: torch, soundfile, CosyVoice
    rows = H.load_questions()
    if not rows:
        sys.exit(f"{H.QFILE} is empty: run `h2_assets.py render` (pilot or all) first")
    samples = list(rows.values())
    known = {}   # question text -> audio info already made (same text = same file, same voice)
    for r in rows.values():
        if r.get("q_audio") and (DATA / r["q_audio"]).exists():
            known[r["question"]] = {k: v for k, v in r.items() if k.startswith("q_a") or k == "q_voice"}
    texts = list(dict.fromkeys(s["question"] for s in samples))
    new = random.Random(f"{SEED}-h2-voices").sample([t for t in texts if t not in known], len(texts) - sum(t in known for t in texts))
    n_voice = Counter(v["q_voice"] for v in known.values())
    voice_of = {}
    for t in new:   # keep the two voices balanced
        voice_of[t] = min(T.VOICES, key=lambda v: n_voice[v])
        n_voice[voice_of[t]] += 1
    print(f"{len(samples)} units, {len(texts)} unique questions ({len(texts) - len(new)} reused, {len(new)} new)", flush=True)
    if new:
        AUD_DIR.mkdir(parents=True, exist_ok=True)
        tts, set_seed = T.load_tts()
        asr = T.load_asr()
        prompt_dir = T.AUDIO_DIR / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for v, cfg in T.VOICES.items():
            shutil.copy(T.TTS_ROOT / "ref_candidates" / f"{cfg['clip']}.wav", prompt_dir / f"{v}.wav")
            tts.add_zero_shot_spk(cfg["text"], str(prompt_dir / f"{v}.wav"), v)
        import soundfile as sf
        import torch
        failed = 0
        for i, text in enumerate(new):
            v = voice_of[text]
            path = AUD_DIR / ("q_" + hashlib.sha1(text.encode()).hexdigest()[:10] + ".wav")
            best = None
            for attempt in range(T.MAX_TRIES):   # same retry rule as the instruction audio
                set_seed(T.SEED + attempt)
                wav = torch.cat([o["tts_speech"] for o in tts.inference_zero_shot(
                    text, "", "", zero_shot_spk_id=v, stream=False)], dim=1).squeeze(0).float().cpu().numpy()
                heard = asr({"raw": wav, "sampling_rate": tts.sample_rate}, generate_kwargs={"language": "en"})["text"].strip()
                score = difflib.SequenceMatcher(None, T.norm(text), T.norm(heard)).ratio()
                if best is None or score > best[0]:
                    best = (score, wav, heard)
                if T.norm(heard) == T.norm(text):
                    break
            score, wav, heard = best
            sf.write(path, wav, tts.sample_rate)
            ok = T.norm(heard) == T.norm(text)
            failed += not ok
            known[text] = {"q_audio": str(path.relative_to(DATA)), "q_voice": v, "q_asr": heard, "q_asr_ok": ok,
                           "q_asr_similarity": round(score, 3), "q_audio_duration": round(len(wav) / tts.sample_rate, 2)}
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(new)} questions, asr mismatches so far: {failed}", flush=True)
                for u, r in rows.items():   # partial save: a crash keeps the finished audio
                    if r["question"] in known:
                        r.update(known[r["question"]])
                save_rows(rows)
        print(f"{len(new)} new question audios, {failed} still mismatched after {T.MAX_TRIES} tries")
    for r in rows.values():
        if r["question"] in known:
            r.update(known[r["question"]])
    save_rows(rows)
    done = [r for r in rows.values() if r.get("q_audio")]
    print(f"{len(done)}/{len(rows)} rows have audio ({sum(r['q_voice'] == 'female' for r in done)} female), "
          f"asr_ok false: {sum(not r['q_asr_ok'] for r in done)} -> {H.QFILE}")
    for r in done:
        if not r["q_asr_ok"]:
            print(f"  [{r['q_voice']}] {r['question']!r} -> heard {r['q_asr']!r}")


def cmd_status(args, samples):
    rows = H.load_questions()
    have = [s for s in samples if s["unit_id"] in rows]
    img = sum((DATA / rows[s["unit_id"]]["q_image"]).exists() for s in have)
    aud = sum(bool(rows[s["unit_id"]].get("q_audio")) and (DATA / rows[s["unit_id"]]["q_audio"]).exists() for s in have)
    print(f"{len(samples)} units selected: {len(have)} with a row, {img} question images, {aud} question audios")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "tts", "status"])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pilot", type=int, help="units per branch (same selection as run_h1v2.py --pilot)")
    g.add_argument("--all", action="store_true", help="every unit")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.cmd == "tts":   # works on questions.jsonl only: no import of the H1 runner in the cosyvoice env
        return cmd_tts(args)
    if args.pilot is None and not args.all:
        ap.error("render / status need --pilot N or --all")
    samples, _ = H.select_samples(args.pilot, args.all, args.seed)
    print(f"{len(samples)} units selected ({'all' if args.all else f'pilot {args.pilot} per branch'})", flush=True)
    {"render": cmd_render, "status": cmd_status}[args.cmd](args, samples)


if __name__ == "__main__":
    main()
