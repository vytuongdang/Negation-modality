"""Instruction cards for the V_card condition: each negation sentence written alone on a plain white card.

V puts the instruction INSIDE the scene image, so it arrives before the question and looks like scene text. V_card
gives the same words as a SEPARATE image item after the question, the same slot as the text item in T and the audio
item in A, so the three channels differ only in the modality of the instruction (Vy, 2026-11-23).

Style matches the V box: black DejaVu Sans on white, centred, wrapped. One card per distinct sentence, named by
the sentence's sha1, so every sample with the same words shares a file. Never overwrites, never deletes.

    dataset/visual/card/<sha1(text)[:12]>.png     + dataset/visual/card/manifest.jsonl  {text, file}

    python3 render_cards.py            # every negation sentence of the dataset
    python3 render_cards.py --show 3   # print 3 paths to look at
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from omnilib import DATA, OUT, read_jsonl  # noqa: E402

CARD_DIR = DATA / "visual" / "card"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
WIDTH = 640           # COCO-like width; V's font is 4% of image width (~26 px at 640)
FONT_PX = 26
PAD = 24
LINE_GAP = 1.3


def card_rel(text):
    """Path of the card for this sentence, relative to DATA (as the answer cache stores it)."""
    return f"visual/card/{hashlib.sha1(text.encode()).hexdigest()[:12]}.png"


def wrap(draw, text, font, max_w):
    lines, cur = [], ""
    for w in text.split():
        t = f"{cur} {w}".strip()
        if draw.textlength(t, font=font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


def render(text, path):
    font = ImageFont.truetype(FONT, FONT_PX)
    probe = ImageDraw.Draw(Image.new("RGB", (WIDTH, 10)))
    lines = wrap(probe, text, font, WIDTH - 2 * PAD)
    line_h = round(FONT_PX * LINE_GAP)
    img = Image.new("RGB", (WIDTH, 2 * PAD + line_h * len(lines)), "white")
    d = ImageDraw.Draw(img)
    for k, ln in enumerate(lines):
        d.text(((WIDTH - d.textlength(ln, font=font)) / 2, PAD + k * line_h), ln, fill="black", font=font)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    texts = sorted({r["negation"] for r in read_jsonl(OUT / "samples_compact_all_audio.jsonl")})
    made = 0
    for t in texts:
        p = DATA / card_rel(t)
        if not p.exists():
            render(t, p)
            made += 1
    with open(CARD_DIR / "manifest.jsonl", "w") as f:
        for t in texts:
            f.write(json.dumps({"text": t, "file": card_rel(t)}, ensure_ascii=False) + "\n")
    print(f"{len(texts)} sentences, {made} new cards in {CARD_DIR}")
    for t in texts[:args.show]:
        print(DATA / card_rel(t), "|", t)


if __name__ == "__main__":
    main()
