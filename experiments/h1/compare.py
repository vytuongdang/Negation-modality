"""Compare H1 across models: violation rate under text vs audio negation, on the samples whose C0 answer states
the target (per model). Reads results/<model>/scored_<tag>.jsonl.

    python compare.py            # tag "all"
    python compare.py --tag pilot20
"""
import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
BRANCHES = ["existence", "count", "attribute", "relation"]


def mcnemar(only_text, only_audio):
    n = only_text + only_audio
    if n == 0:
        return 1.0
    k = min(only_text, only_audio)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="all")
    args = ap.parse_args()

    files = sorted(RESULTS.glob(f"*/scored_{args.tag}.jsonl"))
    if not files:
        print(f"no results/*/scored_{args.tag}.jsonl yet")
        return
    print(f"violation rate (target stated) among samples whose C0 answer states the target\n")
    head = f"{'model':<18} {'branch':<10} {'n':>4} {'C0 tgt%':>8} {'text':>6} {'audio':>6} {'a-t':>5} " \
           f"{'only_t':>6} {'only_a':>6} {'p':>6}"
    print(head)
    print("-" * len(head))
    for f in files:
        by = defaultdict(dict)
        for line in open(f):
            r = json.loads(line)
            by[r["sample_id"]][r["condition"]] = r
        by = {k: v for k, v in by.items() if len(v) == 3}
        for b in BRANCHES + ["ALL"]:
            ids = [k for k, v in by.items() if b == "ALL" or v["C0_none"]["branch"] == b]
            if not ids:
                continue
            c0 = [k for k in ids if by[k]["C0_none"]["target_mentioned"]]
            t = [by[k]["C1_text"]["target_mentioned"] for k in c0]
            a = [by[k]["C2_audio"]["target_mentioned"] for k in c0]
            ot = sum(x and not y for x, y in zip(t, a))
            oa = sum(y and not x for x, y in zip(t, a))
            n = len(c0) or 1
            name = f.parent.name if b == BRANCHES[0] else ""
            print(f"{name:<18} {b:<10} {len(c0):>4} {100 * len(c0) / len(ids):>7.0f}% {100 * sum(t) / n:>5.0f}% "
                  f"{100 * sum(a) / n:>5.0f}% {100 * (sum(a) - sum(t)) / n:>+5.0f} {ot:>6} {oa:>6} {mcnemar(ot, oa):>6.3f}")
        print()
    print("a-t < 0: audio negation obeyed better than text; only_t / only_a: pairs where only text / only audio "
          "violated; p: exact McNemar (treats template-sweep copies as independent, so optimistic)")


if __name__ == "__main__":
    main()
