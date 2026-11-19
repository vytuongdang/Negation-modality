"""Visual channel: the instruction written INSIDE the original image, in a spot that does not cover the subject.

Where the text goes: every annotated object of the image is a "protected" box, stored by B1 in the unit's
protect_boxes (COCO instances, VAW / GQA scene-graph objects, CLEVR object positions). Boxes of the question's
own objects (components, the queried subject) have weight 1, every other object weight OTHER_WEIGHT. A text box is slid over
the image and placed where it covers the least protected area; if it would touch a weight-1 box, a smaller font /
narrower wrapping is tried first. Samples where the subject still had to be covered are flagged
(`occludes_subject`), so analyses can drop them.

Text box: white, slightly transparent background, black DejaVu Sans. Each sample also gets a control image with
the SAME box at the SAME place but no text, so "visual instruction" vs "no instruction" differs only in the words,
not in which pixels are hidden.

    dataset/visual/instr/{sample_id}.{jpg|png}   image with the instruction box
    dataset/visual/box/{sample_id}.{jpg|png}     same box, empty (control)
    dataset/out/samples_compact_all.jsonl        compact rows for every form + form, pair_of, negation_image,
                                                 box_image, text_box [x, y, w, h], font px, occlusion stats

    python3 render_visual.py                     # after b3_forms.py
    python3 render_visual.py --preview 30        # B1 units, a dummy instruction, 30 images -> visual/preview/
"""
import argparse
import random
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import B1_DIR, DATA, OUT_DIR, SEED, VIS
from utils import read_jsonl, write_jsonl

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_FRAC = 0.04          # base font size as a share of image width
MIN_FONT = 12
WRAP_FRACS = (0.9, 0.7, 0.5)   # max text width as a share of image width, tried in this order
SHRINK = (1.0, 0.85, 0.7)      # font scale steps
BOX_ALPHA = 215           # 0-255 opacity of the white box
JPEG_QUALITY = 95


# ---------------------------------------------------------------- placement

def cost_maps(W, H, boxes):
    """Integral images of the weight-1 mask and of the weighted mask (max weight per pixel)."""
    main = np.zeros((H, W), np.float32)
    weighted = np.zeros((H, W), np.float32)
    for x, y, w, h, wt in boxes:
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, int(np.ceil(x + w))), min(H, int(np.ceil(y + h)))
        if x1 <= x0 or y1 <= y0:
            continue
        weighted[y0:y1, x0:x1] = np.maximum(weighted[y0:y1, x0:x1], wt)
        if wt >= 1.0:
            main[y0:y1, x0:x1] = 1.0
    integ = lambda m: np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return integ(main), integ(weighted), main.sum()


def box_sum(I, x, y, w, h):
    return I[y + h, x + w] - I[y, x + w] - I[y + h, x] + I[y, x]


def wrap(draw, text, font, max_w):
    lines, cur = [], ""
    for word in text.split():
        t = f"{cur} {word}".strip()
        if draw.textlength(t, font=font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = word
    return lines + ([cur] if cur else [])


def layout(draw, text, W, size, wrap_frac):
    font = ImageFont.truetype(FONT, size)
    lines = wrap(draw, text, font, W * wrap_frac)
    pad = round(size * 0.45)
    line_h = round(size * 1.25)
    tw = max(draw.textlength(l, font=font) for l in lines)
    return font, lines, int(tw + 2 * pad), int(line_h * len(lines) + 2 * pad), pad, line_h


def place(img, text, boxes):
    W, H = img.size
    I_main, I_w, main_area = cost_maps(W, H, boxes)
    draw = ImageDraw.Draw(img)
    base = max(MIN_FONT, round(W * FONT_FRAC))
    best = None
    for scale in SHRINK:
        size = max(MIN_FONT, round(base * scale))
        for wf in WRAP_FRACS:
            font, lines, bw, bh, pad, line_h = layout(draw, text, W, size, wf)
            if bw > W or bh > H:
                continue
            step = max(4, min(W, H) // 60)
            xs = list(range(0, W - bw + 1, step)) + [W - bw]
            ys = list(range(0, H - bh + 1, step)) + [H - bh]
            for y in ys:
                for x in xs:
                    m = box_sum(I_main, x, y, bw, bh)
                    c = box_sum(I_w, x, y, bw, bh)
                    edge = min(x, y, W - bw - x, H - bh - y) / max(W, H)  # tie-break: hug an edge
                    key = (m > 0, m, -size if m == 0 else 0, c, edge)
                    if best is None or key < best[0]:
                        best = (key, (x, y, bw, bh), font, lines, pad, line_h, size, m)
        if best is not None and best[7] == 0:
            break  # the largest font that avoids the subject
    _, (x, y, bw, bh), font, lines, pad, line_h, size, m = best
    return {"box": [x, y, bw, bh], "font": font, "lines": lines, "pad": pad, "line_h": line_h, "size": size,
            "subject_px_covered": int(m), "subject_frac_covered": round(float(m / main_area), 4) if main_area else 0.0}


def draw_box(img, p, text=True):
    x, y, bw, bh = p["box"]
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    d.rectangle([x, y, x + bw - 1, y + bh - 1], fill=(255, 255, 255, BOX_ALPHA))
    out = Image.alpha_composite(img.convert("RGBA"), over)
    if text:
        d = ImageDraw.Draw(out)
        ty = y + p["pad"]
        for ln in p["lines"]:
            tx = x + (bw - d.textlength(ln, font=p["font"])) / 2
            d.text((tx, ty), ln, fill="black", font=p["font"])
            ty += p["line_h"]
    return out.convert("RGB")


def save(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path) if path.suffix == ".png" else img.save(path, quality=JPEG_QUALITY)


# ---------------------------------------------------------------- main

def render_all(samples, out_dir):
    rows, stats = [], Counter()
    for k, s in enumerate(samples, 1):
        src = DATA / s["image"]
        img = Image.open(src).convert("RGB")
        p = place(img, s["negation_text"], s["extra"]["protect_boxes"])
        ext = src.suffix.lower()
        instr = out_dir / "instr" / f"{s['sample_id']}{ext}"
        boxed = out_dir / "box" / f"{s['sample_id']}{ext}"
        save(draw_box(img, p, True), instr)
        save(draw_box(img, p, False), boxed)
        stats["occludes_subject"] += p["subject_px_covered"] > 0
        stats["font_shrunk"] += p["size"] < max(MIN_FONT, round(img.width * FONT_FRAC))
        rows.append((s, {"negation_image": str(instr.relative_to(DATA)), "box_image": str(boxed.relative_to(DATA)),
                         "text_box": p["box"], "negation_font_px": p["size"],
                         "occludes_subject": p["subject_px_covered"] > 0,
                         "subject_frac_covered": p["subject_frac_covered"]}))
        if k % 500 == 0:
            print(f"{k}/{len(samples)} rendered", flush=True)
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=int, help="render N B1 units with a dummy instruction into visual/preview/")
    args = ap.parse_args()

    if args.preview:
        units = read_jsonl(B1_DIR / "units.jsonl")  # selected units (after select_images.py)
        rng = random.Random(SEED)
        by_branch = defaultdict(list)
        for u in units:
            by_branch[u["branch"]].append(u)
        pick = [u for b in sorted(by_branch) for u in rng.sample(by_branch[b], min(len(by_branch[b]),
                                                                              -(-args.preview // len(by_branch))))]
        samples = []
        for u in pick:
            c = u["raw_components"][0]
            samples.append(dict(u, sample_id=u["unit_id"], extra={"protect_boxes": u["protect_boxes"]},
                                negation_text=f"Do not mention the {c['subject']}." if c["type"] != "attribute"
                                else f"Do not mention its {c['slot']}."))
        rows, stats = render_all(samples, VIS / "preview")
        print(f"preview: {len(rows)} images in {VIS / 'preview'}; {dict(stats)}")
        return

    from export_compact import compact
    samples = read_jsonl(OUT_DIR / "samples.jsonl")
    forms = OUT_DIR / "samples_forms.jsonl"
    samples += read_jsonl(forms) if forms.exists() else []
    rows, stats = render_all(samples, VIS)
    out = []
    for s, v in rows:
        r = compact(s)
        r.update(form=s["negation"]["form"], pair_of=s.get("pair_of", s["sample_id"]), **v)
        r["meta"]["form"] = s["negation"]["form"]
        out.append(r)
    write_jsonl(OUT_DIR / "samples_compact_all.jsonl", out)
    print(f"{len(out)} samples rendered; {dict(stats)}; forms: {dict(Counter(r['form'] for r in out))}")
    print(f"files: {VIS} {OUT_DIR / 'samples_compact_all.jsonl'}")


if __name__ == "__main__":
    main()
