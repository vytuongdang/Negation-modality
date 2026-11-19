"""Step 2: pick the final images from the B1 candidates, val first, train only to fill what val cannot.

For each source (COCO, CLEVR, VAW, GQA) the candidates are ordered by split (config.SPLITS: val, then test /
train) and shuffled with SEED inside a split. They are downloaded and checked in batches until the source
reaches config.TARGET_IMAGES or runs out of candidates. Checks, in order:

  pixels (not CLEVR)   sharpness >= MIN_SHARPNESS (variance of the Laplacian at 512 px width);
                       mean brightness in BRIGHTNESS_RANGE; clipped (<=5 or >=250) pixels <= MAX_CLIPPED;
                       colourful (not black-and-white); no near-duplicate of an accepted image (dHash)
  text space           the longest instruction this unit can get (any target, any template or form) fits in
                       the image covering at most MAX_SUBJECT_COVER (5%) of the question's object boxes
                       (render_visual.place; boxes are loose rectangles, so 5% is mostly their empty corners)
  VLM (not CLEVR)      Qwen3-VL-8B, yes/no per fact: every component holds (count: the number matches), every
                       closed-set distractor is absent, and the image has no large readable text
A unit failing a check is dropped; an image is kept if at least one unit survives.
Cross-source de-duplication: a VG image whose coco_id is already selected from COCO is skipped, and an image
used by VAW is not used by GQA.

Outputs (dataset/b1/):
    units.jsonl              the selected units (B2 input), with image metadata and the check scores
    selection_report.jsonl   every candidate looked at, with the reason it was rejected
    selection_summary.json   per source: images taken from each split, rejections by reason
    review/review.html       random accepted and rejected images for a human look before B2
"""
import html
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from b1_ground_truth import join_list, plural
from b3_negation import TEMPLATES, target_phrase
from config import (B1_DIR, BRIGHTNESS_RANGE, DATA, DHASH_MAX_DIST, MAX_CLIPPED, MAX_SUBJECT_COVER, MIN_COLORFULNESS,
                    MIN_SHARPNESS,
                    REJECTED, SEED, SPLITS, TARGET_IMAGES, URLS, VERIFY_MODEL)
from utils import RemoteZip, download, read_jsonl, write_jsonl

SOURCE_ORDER = ["COCO", "CLEVR", "VAW", "GQA"]
BATCH = 48
NUMBER_WORDS = {w: i for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven",
                                             "eight", "nine", "ten"])}


# ---------------------------------------------------------------- pixel checks

def pixel_scores(path):
    img = Image.open(path).convert("RGB")
    g = img.convert("L").resize((512, max(1, round(512 * img.height / img.width))))
    a = np.asarray(g, dtype=np.float32)
    lap = a[1:-1, 1:-1] * -4 + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    rgb = np.asarray(img.resize((256, max(1, round(256 * img.height / img.width)))), dtype=np.float32)
    colorful = float(np.mean(np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(rgb[..., 1] - rgb[..., 2])))
    small = np.asarray(img.convert("L").resize((9, 8)), dtype=np.int16)
    dh = int("".join("1" if b else "0" for b in (small[:, 1:] > small[:, :-1]).flatten()), 2)
    return {"sharpness": round(float(lap.var()), 1), "brightness": round(float(a.mean()), 1),
            "clipped": round(float(((a <= 5) | (a >= 250)).mean()), 3), "colorfulness": round(colorful, 2),
            "dhash": dh}


def pixel_reject(sc):
    if sc["sharpness"] < MIN_SHARPNESS:
        return "blurry"
    if not BRIGHTNESS_RANGE[0] <= sc["brightness"] <= BRIGHTNESS_RANGE[1]:
        return "too_dark_or_bright"
    if sc["clipped"] > MAX_CLIPPED:
        return "clipped"
    if sc["colorfulness"] < MIN_COLORFULNESS:
        return "black_and_white"
    return None


# ---------------------------------------------------------------- text space

def instructions_for(unit):
    """Every instruction the unit can get later (each target x 7 negative templates, exceptive, affirmative)."""
    from b3_forms import AFFIRMATIVE, AFFIRMATIVE_BRANCHES, EXCEPTIVE
    comps = unit["raw_components"]
    texts = []
    for t in comps:
        x = target_phrase(unit, t)
        texts += [frame.format(X=x) for _, _, frame in TEMPLATES.values()] + [EXCEPTIVE.format(X=x)]
        if unit["branch"] in AFFIRMATIVE_BRANCHES:
            if unit["branch"] == "existence_closed":
                y = join_list(f"the {o}" for o in unit["options"] if o != t["subject"])
            else:
                y = join_list(target_phrase(unit, c) for c in comps if c is not t)
            texts.append(AFFIRMATIVE.format(Y=y))
    return texts


def text_space_ok(path, unit):
    from render_visual import place
    img = Image.open(path).convert("RGB")
    longest = max(instructions_for(unit), key=len)
    p = place(img, longest, unit["protect_boxes"])
    return p["subject_frac_covered"] <= MAX_SUBJECT_COVER, longest


# ---------------------------------------------------------------- VLM checks

def article(noun):
    return "an" if noun[0] in "aeiou" else "a"


def parse_yes(ans):
    a = ans.strip().lower()
    return True if a.startswith("yes") else False if a.startswith("no") else None


def parse_number(ans):
    a = ans.strip().lower()
    m = re.search(r"\d+", a)
    if m:
        return int(m.group())
    for w, n in NUMBER_WORDS.items():
        if re.search(rf"\b{w}\b", a):
            return n
    return None


def checks_for(unit):
    """[(kind, question, expected)] for one unit; kind 'yes' expects yes/no, 'num' expects a number."""
    out = []
    for c in unit["raw_components"]:
        t = c["type"]
        if t == "existence":
            out.append(("yes", f"Is there {article(c['subject'])} {c['subject']} in this image? Answer yes or no.", True))
        elif t == "count":
            out.append(("num", f"How many {plural(c['subject'])} are in this image? Answer with a single number.",
                        c["value"]))
        elif t == "attribute":
            out.append(("yes", f"Is the {c['subject']} {c['value']}? Answer yes or no.", True))
        elif t == "relation":
            out.append(("yes", f"Is the {c['subject']} {c['relation']} the {c['value']}? Answer yes or no.", True))
    for d in unit.get("distractors", []):
        out.append(("yes", f"Is there {article(d)} {d} in this image? Answer yes or no.", False))
    return out


BIG_TEXT_Q = ("Does this image contain large, clearly readable text, such as a big sign, banner, caption or "
              "watermark? Answer yes or no.")


class Verifier:
    CACHE = B1_DIR / "vlm_cache.jsonl"  # answers survive a restart of this step

    def __init__(self):
        from vlm import LocalVLM
        self.vlm = LocalVLM(VERIFY_MODEL)
        self.cache = {}
        if self.CACHE.exists():
            for r in read_jsonl(self.CACHE):
                self.cache[(r["image"], r["q"])] = r["answer"]

    def ask(self, path, q):
        key = (str(path.relative_to(DATA)), q)
        if key not in self.cache:
            self.cache[key] = self.vlm.answer(path, q, max_new_tokens=8)
            self.CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.CACHE, "a") as f:
                f.write(json.dumps({"image": key[0], "q": q, "answer": self.cache[key]}) + "\n")
        return self.cache[key]

    def unit_ok(self, path, unit):
        results = []
        for kind, q, exp in checks_for(unit):
            a = self.ask(path, q)
            got = parse_yes(a) if kind == "yes" else parse_number(a)
            results.append({"q": q, "answer": a, "ok": got == exp})
        return all(r["ok"] for r in results), results

    def big_text(self, path):
        a = self.ask(path, BIG_TEXT_Q)
        return parse_yes(a) is True, a

    def close(self):
        self.vlm.close()


# ---------------------------------------------------------------- download

_clevr_zip = None


def fetch(c):
    dst = DATA / c["image"]
    if dst.exists():
        return dst
    old = REJECTED / dst.relative_to(DATA / "images")  # rejected in an earlier run: re-check it, no re-download
    if old.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), dst)
        return dst
    if c["source"] == "CLEVR":
        global _clevr_zip
        if _clevr_zip is None:
            _clevr_zip = RemoteZip(URLS["clevr_zip"])
        return _clevr_zip.extract(c["zip_member"], dst)
    return download(c["url"], dst)


def fetch_batch(batch, workers):
    def one(c):
        try:
            return c, fetch(c), None
        except Exception as e:  # noqa: BLE001  (dead link, timeout after retries)
            return c, None, repr(e)[:200]
    if batch and batch[0]["source"] == "CLEVR":
        return [one(c) for c in batch]  # one remote-zip reader, sequential
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, batch))


def reject_file(path):
    dst = REJECTED / path.relative_to(DATA / "images")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), dst)
    return dst


# ---------------------------------------------------------------- main loop

def main():
    cands = read_jsonl(B1_DIR / "candidates.jsonl")
    by_source = defaultdict(list)
    for c in cands:
        by_source[c["source"]].append(c)
    verifier = Verifier()
    units_out, report = [], []
    summary = {}
    used_vg, used_coco, hashes = set(), set(), []
    for source in SOURCE_ORDER:
        rng = random.Random(f"{SEED}-{source}")
        ordered = []
        for split in SPLITS[source]:
            pool = sorted((c for c in by_source[source] if c["split"] == split), key=lambda c: str(c["image_id"]))
            rng.shuffle(pool)
            ordered += pool
        target = TARGET_IMAGES[source]
        taken, reasons, per_split = 0, Counter(), Counter()
        i = 0
        workers = 8 if source == "COCO" else 3
        while taken < target and i < len(ordered):
            batch = []
            while len(batch) < BATCH and i < len(ordered):
                c = ordered[i]
                i += 1
                img_key = str(c["image_id"])
                if source in ("VAW", "GQA") and (img_key in used_vg or (c.get("coco_id") and c["coco_id"] in used_coco)):
                    reasons["duplicate_across_sources"] += 1
                    report.append({"source": source, "split": c["split"], "image": c["image"],
                                   "reason": "duplicate_across_sources"})
                    continue
                batch.append(c)
            for c, path, err in fetch_batch(batch, workers):
                if taken >= target:
                    break
                rec = {"source": source, "split": c["split"], "image": c["image"]}
                if err:
                    rec["reason"] = "download_failed"
                    reasons["download_failed"] += 1
                    report.append(rec)
                    continue
                meta = {"split": c["split"], "width": c["width"], "height": c["height"], "license": c["license"],
                        "n_objects": c["n_objects"], "flickr_url": c.get("flickr_url"), "url": c.get("url")}
                reason = None
                if source != "CLEVR":
                    sc = pixel_scores(path)
                    meta.update({k: v for k, v in sc.items() if k != "dhash"})
                    reason = pixel_reject(sc)
                    if not reason and any(bin(sc["dhash"] ^ h).count("1") <= DHASH_MAX_DIST for h in hashes):
                        reason = "near_duplicate"
                    if not reason:
                        big, ans = verifier.big_text(path)
                        meta["big_text_answer"] = ans
                        if big:
                            reason = "large_text_in_image"
                kept_units, unit_notes = [], []
                if not reason:
                    for u in c["units"]:
                        ok_space, longest = text_space_ok(path, u)
                        if not ok_space:
                            unit_notes.append({"branch": u["branch"], "reason": "no_text_space", "longest": longest})
                            continue
                        if source != "CLEVR":
                            ok, res = verifier.unit_ok(path, u)
                            if not ok:
                                unit_notes.append({"branch": u["branch"], "reason": "vlm_check_failed",
                                                   "checks": res})
                                continue
                            u = dict(u, vlm_checks=res)
                        kept_units.append(u)
                    if not kept_units:
                        reason = "no_unit_left"
                rec.update(meta=meta, units=unit_notes)
                if reason:
                    rec["reason"] = reason
                    rec["rejected_copy"] = str(reject_file(path).relative_to(DATA))
                    reasons[reason] += 1
                    report.append(rec)
                    continue
                taken += 1
                per_split[c["split"]] += 1
                if source != "CLEVR":
                    hashes.append(sc["dhash"])
                if source == "COCO":
                    used_coco.add(c["image_id"])
                else:
                    used_vg.add(str(c["image_id"]))
                for u in kept_units:
                    src = {"dataset": source, "split": c["split"], "image_id": c["image_id"]}
                    for k in ("instance_id", "subject_id"):
                        if k in u:
                            src[k] = u[k]
                    units_out.append(dict(u, source=src, image=c["image"], image_meta=meta))
                rec["reason"] = "accepted"
                report.append(rec)
            print(f"{source}: {taken}/{target} images after {i}/{len(ordered)} candidates; "
                  f"splits {dict(per_split)}; rejected {dict(reasons)}", flush=True)
        summary[source] = {"target": target, "taken": taken, "candidates_seen": i, "candidates_total": len(ordered),
                           "by_split": dict(per_split), "rejected": dict(reasons)}
    verifier.close()

    counters = Counter()
    prefix = {"COCO": "coco", "CLEVR": "clevr", "VAW": "vaw", "GQA": "gqa"}
    for u in units_out:
        key = f"{prefix[u['source']['dataset']]}-{u['branch']}"
        u["unit_id"] = f"{key}-{counters[key]:05d}"
        counters[key] += 1
    write_jsonl(B1_DIR / "units.jsonl", units_out)
    write_jsonl(B1_DIR / "selection_report.jsonl", report)
    summary["units_by_branch"] = dict(Counter(f"{u['source']['dataset']}/{u['branch']}" for u in units_out))
    (B1_DIR / "selection_summary.json").write_text(json.dumps(summary, indent=1))
    review_page(units_out, report)
    print(json.dumps(summary, indent=1))


def review_page(units, report, n_ok=100, n_bad=60):
    rng = random.Random(SEED)
    ok = rng.sample(units, min(n_ok, len(units)))
    bad = [r for r in report if r.get("rejected_copy")]
    bad = rng.sample(bad, min(n_bad, len(bad)))
    out = B1_DIR / "review"
    out.mkdir(parents=True, exist_ok=True)

    def meta_line(m):
        keys = ["split", "n_objects", "sharpness", "brightness", "colorfulness", "license"]
        return " · ".join(f"{k}={m[k]}" for k in keys if k in m)

    rows = []
    for u in ok:
        comps = "".join(f"<li>{html.escape(c['text'])}</li>" for c in u["raw_components"])
        rows.append(f"<div class=card><img src='../../{u['image']}'><p><b>{u['unit_id']}</b><br>"
                    f"{html.escape(u['question'])}</p><ul>{comps}</ul><small>{meta_line(u['image_meta'])}</small></div>")
    bad_rows = []
    for r in bad:
        notes = "; ".join(f"{n['branch']}: {n['reason']}" for n in r.get("units", []))
        bad_rows.append(f"<div class=card><img src='../../{r['rejected_copy']}'><p><b>{r['reason']}</b> "
                        f"({r['source']}/{r['split']})</p><small>{html.escape(notes)} "
                        f"{meta_line(r.get('meta', {}))}</small></div>")
    (out / "review.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Image selection review</title><style>"
        "body{font-family:sans-serif;margin:16px}.grid{display:grid;grid-template-columns:repeat(auto-fill,"
        "minmax(320px,1fr));gap:12px}.card{border:1px solid #ccc;padding:8px;border-radius:6px}"
        "img{width:100%}small{color:#555}</style>"
        f"<h1>Accepted ({len(ok)} random of {len(units)} units)</h1><div class=grid>{''.join(rows)}</div>"
        f"<h1>Rejected ({len(bad)} random)</h1><div class=grid>{''.join(bad_rows)}</div>")
    print(f"review page: {out / 'review.html'}")


if __name__ == "__main__":
    main()
