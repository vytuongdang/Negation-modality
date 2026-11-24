"""Analysis of H2 (direct conflict) and H3-A (modality dominance) from results/<model>/scored_<tag>.jsonl (run_h2.py).

    python analyze_h2.py --tag h2_pilot40                 # every model with that tag
    python analyze_h2.py --tag h2_all --models qwen2.5-omni-7b --asr_ok_only

Unit of analysis = the question (unit); violation = any mention of the forbidden component (as in analyze.py), counted
among the units whose own C0 answer states it. CIs: bootstrap over IMAGES (units of one image are not independent);
paired tests: McNemar on the same units. Per model:
  1. violation per condition (H2 and H3A, 6 layouts each) and required recall (over-suppression)
  2. per channel pair, the two directions of the same pair (e.g. qT_iA vs qA_iT), per family: paired difference + CI + McNemar.
     Reading: if the instruction wins equally whichever channel carries the question, the direction does not matter.
  3. H2 minus H3A: the direction effect with the negation minus the direction effect without it, on the units that
     have both (negation-specific vs general channel dominance, PLAN stage 4).
No direction is assumed (all two-sided); nothing here says WHICH channel is stronger before the numbers are in.
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import h2_common as H  # noqa: E402
from omnilib import removal_set, removed  # noqa: E402
from analyze import bootstrap, mcnemar, mean, pct, read_jsonl  # noqa: E402

RESULTS = HERE / "results"
PAIRS = [("T", "A"), ("T", "V"), ("A", "V")]


def load(path, asr_ok_only, rm=(set(), set(), set())):
    rows = [r for r in read_jsonl(path) if not removed(r, rm)]
    dropped = {r["unit_id"] for r in read_jsonl(HERE.parents[1] / "create_data" / "dataset" / "out" / "excluded_units.jsonl")} \
        if (HERE.parents[1] / "create_data" / "dataset" / "out" / "excluded_units.jsonl").exists() else set()
    rows = [r for r in rows if r["unit_id"] not in dropped]
    if asr_ok_only:   # leave out units whose spoken instruction or spoken question failed the Whisper check
        bad = {r["unit_id"] for r in rows if r["asr_ok"] is False or r.get("q_asr_ok") is False}
        rows = [r for r in rows if r["unit_id"] not in bad]
    return rows


def ci(vals, rng, group, f):
    lo, hi = bootstrap(vals, group, f, rng)
    return "-" if lo is None else f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def section(name, rows, out, rng):
    by = {(r["unit_id"], r["condition"]): r for r in rows}
    c0 = {r["unit_id"]: r for r in rows if r["condition"] == "C0"}
    units = sorted(u for u, r in c0.items() if r["violation"])
    image = {u: c0[u]["image"] for u in c0}
    conds = [c for c in H.ALL_CONDS if any(r["condition"] == c for r in rows)]
    out.append(f"\n## {name}\n")
    out.append(f"C0 states the target in {len(units)}/{len(c0)} units (violation is counted among these).\n")
    if not units:
        return

    def viol(u, c):
        r = by.get((u, c))
        return None if r is None else int(r["violation"])

    out.append("| condition | n | violation % | 95% CI | required recall % |\n|---|---|---|---|---|")
    for c in conds:
        us = [u for u in units if viol(u, c) is not None]
        f = lambda sids, c=c: mean(viol(u, c) for u in sids if viol(u, c) is not None)
        rec = mean(by[(u, c)]["required_recall"] for u in us)
        out.append(f"| {c} | {len(us)} | {pct(f(us))} | {ci(us, rng, image, f)} | {pct(rec)} |")

    out.append("\n### Direction within a channel pair (violation of q<first>_i<second> minus q<second>_i<first>)\n")
    out.append("| family | pair | n | dir 1 % | dir 2 % | diff (pp) | 95% CI | McNemar p |\n|---|---|---|---|---|---|---|---|")
    diffs = {}
    for fam in H.FAMILIES:
        for a, b in PAIRS:
            c1, c2 = H.cond_name(fam, a, b), H.cond_name(fam, b, a)
            us = [u for u in units if viol(u, c1) is not None and viol(u, c2) is not None]
            if not us:
                continue
            f = lambda sids, c1=c1, c2=c2: mean(viol(u, c1) - viol(u, c2) for u in sids)
            n10 = sum(viol(u, c1) == 1 and viol(u, c2) == 0 for u in us)
            n01 = sum(viol(u, c1) == 0 and viol(u, c2) == 1 for u in us)
            diffs[(fam, a, b)] = (us, f)
            out.append(f"| {fam} | q{a}_i{b} vs q{b}_i{a} | {len(us)} | {pct(mean(viol(u, c1) for u in us))} | "
                       f"{pct(mean(viol(u, c2) for u in us))} | {100 * f(us):+.1f} | {ci(us, rng, image, f)} | "
                       f"{mcnemar(n10, n01):.3g} |")

    out.append("\n### Direction effect with negation (H2) minus without (H3A), units that have both\n")
    out.append("| pair | n | H2 diff (pp) | H3A diff (pp) | H2 - H3A (pp) | 95% CI |\n|---|---|---|---|---|---|")
    for a, b in PAIRS:
        if ("H2", a, b) not in diffs or ("H3A", a, b) not in diffs:
            continue
        us = sorted(set(diffs[("H2", a, b)][0]) & set(diffs[("H3A", a, b)][0]))
        if not us:
            continue
        f2, f3 = diffs[("H2", a, b)][1], diffs[("H3A", a, b)][1]
        g = lambda sids, f2=f2, f3=f3: None if not sids else f2(sids) - f3(sids)
        out.append(f"| {a}-{b} | {len(us)} | {100 * f2(us):+.1f} | {100 * f3(us):+.1f} | {100 * g(us):+.1f} | {ci(us, rng, image, g)} |")

    out.append("\n### By branch (violation %, n) among C0-mentioned units\n")
    br = sorted({c0[u]["branch"] for u in units})
    out.append("| condition | " + " | ".join(br) + " |\n|---|" + "---|" * len(br))
    for c in conds:
        cells = []
        for b in br:
            us = [u for u in units if c0[u]["branch"] == b and viol(u, c) is not None]
            cells.append(f"{pct(mean(viol(u, c) for u in us))} ({len(us)})" if us else "-")
        out.append(f"| {c} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--asr_ok_only", action="store_true", help="leave out units whose spoken question / instruction failed the Whisper check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--remove", nargs="*", default=[], help="optional removal lists by id (see analyze.py); output gets _rm")
    args = ap.parse_args()
    rm = removal_set(args.remove)
    files = sorted(RESULTS.glob(f"*/scored_{args.tag}.jsonl"))
    if args.models:
        files = [f for f in files if f.parent.name in args.models]
    rng = random.Random(args.seed)
    out = [f"# H2 / H3-A analysis ({args.tag}{', asr_ok only' if args.asr_ok_only else ''})", "",
           "Generated by experiments/h1_v2/analyze_h2.py (definitions in its docstring). Conditions: "
           "<family>_q<question channel>_i<instruction channel>; T text, A audio, V written in the image."]
    for f in files:
        section(f.parent.name, load(f, args.asr_ok_only, rm), out, rng)
    path = HERE / f"analysis_{args.tag}{'_asrok' if args.asr_ok_only else ''}{'_rm' if args.remove else ''}.md"
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
