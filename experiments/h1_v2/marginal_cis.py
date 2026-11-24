"""Marginal rates with 95% image-bootstrap CIs for the paper (same selection as analyze.py).
Usage: ~/anaconda3/bin/python marginal_cis.py --tag pilot200"""
import argparse, random
import analyze as A

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="pilot200"); args = ap.parse_args()
excl = A.HERE.parents[1] / "create_data" / "dataset" / "out" / "excluded_units.jsonl"
dropped = {r["unit_id"] for r in A.read_jsonl(excl)}
print("| model | C0 states target % [CI] (n) | T viol. % [CI] | A viol. % [CI] | V viol. % [CI] | n C0-mentioned |")
print("|---|---|---|---|---|---|")
for f in sorted(A.RESULTS.glob(f"*/scored_{args.tag}.jsonl")):
    md = A.ModelData([r for r in A.read_jsonl(f) if r["unit_id"] not in dropped])
    rng = random.Random(0)
    def ci(sids, cond):
        st = lambda s: md.rate(s, cond)[0]
        lo, hi = A.bootstrap(sids, md.image, st, rng)
        return f"{100*st(sids):.1f} [{100*lo:.1f}, {100*hi:.1f}]"
    c0 = lambda s: A.mean(md.neg[x]["target_mentioned"] for x in s)
    lo, hi = A.bootstrap(md.primary, md.image, c0, rng)
    cells = [f"{100*c0(md.primary):.1f} [{100*lo:.1f}, {100*hi:.1f}] ({len(md.primary)})"] + \
            [ci(md.c0m, c) for c in "TAV"]
    print(f"| {f.parent.name} | " + " | ".join(cells) + f" | {len(md.c0m)} |")
