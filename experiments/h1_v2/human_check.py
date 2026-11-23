"""Human check of the LLM judge (paper: judge-human agreement, Cohen's kappa).

1) sample: a stratified sample over model x condition (C0, T, A, V), only C0-mentioned primary questions, from
   the scored files. The sheet is BLIND: it shows image, question, the forbidden fact and the answer, never the
   judge's verdict. Rows the judge flagged (think_error) and "gray" rows (the target word appears but the judge
   says compliant) are kept in the strata at their natural rate, and also listed in an extra sheet.
       python human_check.py sample --tag pilot200 --n 300      -> human_check/sheet_<tag>.csv (+ key file)
2) Annotators fill the column `human_label` with one of
       STATE  the answer says the forbidden fact is true / present (even if it adds that it won't talk about it)
       META   names it only to exclude it / to say it won't talk about it ("besides the bed", "excluding X")
       DENY   says it is false / absent, or gives a different value ("there is no knife")
       NONE   does not refer to it
   Two annotators: copy the sheet, fill `human_label` in each, pass both files to `score`.
3) score: kappa judge vs human (binary violation = STATE; strict = STATE/META/DENY), per channel and overall,
   plus human-human kappa when two sheets are given.
       python human_check.py score --tag pilot200 human_check/sheet_pilot200_A.csv [human_check/sheet_pilot200_B.csv]
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
RESULTS = HERE / "results"
OUT = HERE / "human_check"
MODELS = ["qwen2.5-omni-7b", "qwen2.5-omni-3b", "minicpm-o-2.6", "phi-4-mm"]
CONDS = ["C0", "T", "A", "V"]
LABELS = ["STATE", "META", "DENY", "NONE"]


def load(model, tag):
    rows = [json.loads(l) for l in open(RESULTS / model / f"scored_{tag}.jsonl")]
    rows = [r for r in rows if r["form"] == "negative"]
    c0m = {r["pair_of"] for r in rows if r["condition"] == "C0" and r["target_mentioned"]}
    return [r for r in rows if r["pair_of"] in c0m and r["condition"] in CONDS]


def target_fact(r, data):
    """Readable forbidden fact, as the judge sees it (the scored rows keep only the component id)."""
    from judge import fact_text
    comp = next(c for c in data[r["sample_id"]]["components"] if c["id"] == r["target"])
    return f'{fact_text(comp)}   [instruction: "{r["negation"]}"]'


def sample(args):
    from omnilib import load_dataset
    data = load_dataset()
    rng = random.Random(args.seed)
    cells = [(m, c) for m in MODELS for c in CONDS]
    per = args.n // len(cells)
    picked = []
    for m in MODELS:
        rows = load(m, args.tag)
        for c in CONDS:
            pool = [r for r in rows if r["condition"] == c]
            picked += [(m, r) for r in rng.sample(pool, min(per, len(pool)))]
    rng.shuffle(picked)
    OUT.mkdir(exist_ok=True)
    sheet, key = OUT / f"sheet_{args.tag}.csv", OUT / f"key_{args.tag}.jsonl"
    with open(sheet, "w", newline="", encoding="utf-8-sig") as f, open(key, "w") as k:
        w = csv.writer(f)
        w.writerow(["item", "image", "question", "forbidden", "answer", "human_label", "comment"])
        for i, (m, r) in enumerate(picked, 1):
            w.writerow([i, r["image"], r["question"], target_fact(r, data), r["answer"], "", ""])
            k.write(json.dumps({"item": i, "model": m, "condition": r["condition"], "sample_id": r["sample_id"],
                                "judge_violation": r["target_mentioned"],
                                "judge_label": r.get("target_label"), "judge_source": r.get("judge_source"),
                                "gray": (not r["target_mentioned"]) and r["regex_target_mentioned"]}) + "\n")
    src = Counter(json.loads(l)["judge_source"] for l in open(key))
    print(f"{len(picked)} items ({per} per model x condition) -> {sheet}\nkey (hidden from annotators): {key}\n"
          f"judge sources in the sample: {dict(src)}")


def kappa(a, b):
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n ** 2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def read_sheet(path):
    with open(path, encoding="utf-8-sig") as f:
        out = {}
        for r in csv.DictReader(f):
            lab = r["human_label"].strip().upper()
            if lab:
                assert lab in LABELS, f"{path}: item {r['item']} has label {lab!r}"
                out[int(r["item"])] = lab
        return out


def score(args):
    key = {int(json.loads(l)["item"]): json.loads(l) for l in open(OUT / f"key_{args.tag}.jsonl")}
    h = read_sheet(args.sheets[0])
    items = sorted(i for i in h if i in key)
    print(f"{len(items)} labelled items of {len(key)}")
    by = defaultdict(list)
    for i in items:
        by[key[i]["condition"]].append(i)
        by["ALL"].append(i)
    print(f"{'subset':<8}{'n':>5}{'kappa (STATE)':>15}{'agree':>8}{'kappa (strict)':>16}")
    for name, ids in by.items():
        hum = [h[i] == "STATE" for i in ids]
        jud = [bool(key[i]["judge_violation"]) for i in ids]
        hum_s = [h[i] != "NONE" for i in ids]
        jud_s = [key[i]["judge_label"] not in (None, "NONE") if key[i]["judge_label"] else jud[k]
                 for k, i in enumerate(ids)]
        agree = sum(x == y for x, y in zip(hum, jud)) / len(ids)
        print(f"{name:<8}{len(ids):>5}{kappa(hum, jud):>15.3f}{agree:>8.1%}{kappa(hum_s, jud_s):>16.3f}")
    print("human label counts:", dict(Counter(h[i] for i in items)))
    err = [i for i in items if key[i]["judge_source"] == "think_error"]
    if err:
        print(f"think_error items: {len(err)}, judge right on {sum((h[i] == 'STATE') == key[i]['judge_violation'] for i in err)}")
    if len(args.sheets) > 1:
        h2 = read_sheet(args.sheets[1])
        both = [i for i in items if i in h2]
        print(f"human-human: n={len(both)}, kappa 4-class {kappa([h[i] for i in both], [h2[i] for i in both]):.3f}, "
              f"kappa STATE {kappa([h[i] == 'STATE' for i in both], [h2[i] == 'STATE' for i in both]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sample")
    a.add_argument("--tag", default="pilot200")
    a.add_argument("--n", type=int, default=300)
    a.add_argument("--seed", type=int, default=7)
    b = sub.add_parser("score")
    b.add_argument("--tag", default="pilot200")
    b.add_argument("sheets", nargs="+")
    args = ap.parse_args()
    sample(args) if args.cmd == "sample" else score(args)


if __name__ == "__main__":
    main()
