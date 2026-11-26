"""Method analysis: for every config x channel, violation AND required recall side by side, each with a 95% bootstrap
CI over images, and the paired difference to the unsteered answer of the same question (config 'none').

Population = questions whose own unsteered C0 answer states the target (C0-mentioned), as in analyze.py;
required recall (over-suppression check) also on all questions. -> steer/analysis_<model>_<split>.md

    python analyze_steer.py --model qwen2.5-omni-3b [--split val]
"""
import argparse
import json
import random
from collections import defaultdict

from steerlib import HERE, RESULTS

B = 2000


def boot(by_img, f, seed=0):
    """by_img: {image_id: [values]}; f(list of values) -> statistic. -> (point, lo, hi)"""
    imgs = sorted(by_img)
    pt = f([v for i in imgs for v in by_img[i]])
    rng = random.Random(seed)
    st = sorted(f([v for i in (rng.choice(imgs) for _ in imgs) for v in by_img[i]]) for _ in range(B))
    return pt, st[int(0.025 * B)], st[int(0.975 * B) - 1]


def fmt(t, sign=False):
    p, lo, hi = t
    s = "+" if sign else ""
    return f"{p:{s}.1f} [{lo:{s}.1f}, {hi:{s}.1f}]"


mean = lambda xs: 100 * sum(xs) / len(xs) if xs else float("nan")


def analyze(model, split="val", limit=None):
    path = RESULTS / model / f"scored_{split}{'_lim' + str(limit) if limit else ''}.jsonl"
    rows = [json.loads(line) for line in open(path)]
    by = {(r["config"], r["sample_id"], r["condition"]): r for r in rows}
    c0m = {r["sample_id"] for r in rows if r["config"] == "none" and r["condition"] == "C0" and r["violation"]}
    n_q = len({r["sample_id"] for r in rows if r["config"] == "none" and r["condition"] == "C0"})
    out = [f"# Method results: {model}, {split} questions\n",
           f"From {path.name}. Violation = any mention of the forbidden component (judge, H1 path unchanged), among "
           f"C0-mentioned questions ({len(c0m)} of {n_q}). Required recall = share of required components stated "
           f"(over-suppression check). 95% CIs: bootstrap over images ({B}). Diff = config minus the unsteered answer "
           f"of the same question (paired).\n"]
    configs = sorted({(r["config"], r["condition"]) for r in rows if r["config"] != "none"})
    out.append("| channel | config | n | violation % | diff | recall % (C0-m.) | diff | recall % (all q) | diff |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for cond in ("T", "A", "V"):
        base = [(sid, by.get(("none", sid, cond))) for sid in c0m]
        base = {sid: r for sid, r in base if r}
        if base:
            g = defaultdict(list)
            ga = defaultdict(list)
            for sid, r in base.items():
                g[r["image_id"]].append(r)
            for r in rows:
                if r["config"] == "none" and r["condition"] == cond:
                    ga[r["image_id"]].append(r)
            out.append(f"| {cond} | none | {len(base)} | {fmt(boot(g, lambda x: mean([r['violation'] for r in x])))} | | "
                       f"{fmt(boot(g, lambda x: mean([r['required_recall'] for r in x])))} | | "
                       f"{fmt(boot(ga, lambda x: mean([r['required_recall'] for r in x])))} | |")
        for cfg, c in configs:
            if c != cond:
                continue
            pairs, pairs_all = defaultdict(list), defaultdict(list)
            for r in rows:
                if r["config"] != cfg or r["condition"] != cond:
                    continue
                b = by.get(("none", r["sample_id"], cond))
                if b is None:
                    continue
                pairs_all[r["image_id"]].append((r, b))
                if r["sample_id"] in c0m:
                    pairs[r["image_id"]].append((r, b))
            if not pairs:
                continue
            n = sum(map(len, pairs.values()))
            v = boot(pairs, lambda x: mean([r["violation"] for r, _ in x]))
            dv = boot(pairs, lambda x: mean([r["violation"] for r, _ in x]) - mean([b["violation"] for _, b in x]))
            rc = boot(pairs, lambda x: mean([r["required_recall"] for r, _ in x]))
            drc = boot(pairs, lambda x: mean([r["required_recall"] for r, _ in x]) - mean([b["required_recall"] for _, b in x]))
            ra = boot(pairs_all, lambda x: mean([r["required_recall"] for r, _ in x]))
            dra = boot(pairs_all, lambda x: mean([r["required_recall"] for r, _ in x]) - mean([b["required_recall"] for _, b in x]))
            out.append(f"| {cond} | {cfg} | {n} | {fmt(v)} | {fmt(dv, True)} | {fmt(rc)} | {fmt(drc, True)} | "
                       f"{fmt(ra)} | {fmt(dra, True)} |")
    dest = HERE / f"analysis_{model}_{split}{'_lim' + str(limit) if limit else ''}.md"
    dest.write_text("\n".join(out) + "\n")
    print("\n".join(out[-12:]))
    print(f"-> {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    analyze(a.model, a.split, a.limit)
