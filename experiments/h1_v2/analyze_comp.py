"""H3-B analysis (PLAN.md Stage 3): comprehension of the negative instruction per channel, and violation among
questions the model understood vs did not understand.

    python analyze_comp.py --tag pilot200            # reads results/<model>/comp_<tag>_COMP_T-COMP_A-COMP_V.jsonl
                                                    #   and results/<model>/scored_<tag>.jsonl (C0, T, A, V)
Writes analysis_comp_<tag>.md. NC_m = share of questions whose forbidden item was picked (3 options, chance 33%).
Violation (primary metric) is counted among C0-mentioned questions, as in analyze.py; CIs = 95% bootstrap over
images (analyze.bootstrap). Descriptive only: understanding is not randomised, so understood-vs-not differences
are associations (different questions), not causal effects.
"""
import argparse
import json
import random
from pathlib import Path

from analyze import bootstrap, read_jsonl

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
CH = {"T": "COMP_T", "A": "COMP_A", "V": "COMP_V"}
# v2 (run_comp2.py, Vy 2026-11-24): instruction quoted first, then the MCQ; plus verbatim repeat per channel
V2_CONDS = "COMP2_T-COMP2_A-COMP2_V-REP_T-REP_A-OCR_V"
CH2 = {"T": "COMP2_T", "A": "COMP2_A", "V": "COMP2_V"}
REP2 = {"T": "REP_T", "A": "REP_A", "V": "OCR_V"}


def rate(rows, f):
    return None if not rows else 100 * sum(f(r) for r in rows) / len(rows)


def fmt(v, lo=None, hi=None, n=None):
    if v is None:
        return "n/a"
    s = f"{v:.1f}"
    if lo is not None:
        s += f" [{lo:.1f}, {hi:.1f}]"
    return s + (f" (n={n})" if n is not None else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot200")
    ap.add_argument("--models", nargs="*", default=["qwen2.5-omni-7b", "qwen2.5-omni-3b", "minicpm-o-2.6",
                                                     "phi-4-mm", "internomni"])
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    args = ap.parse_args()
    v2 = args.version == "v2"
    ch_map = CH2 if v2 else CH
    rng = random.Random(0)
    out = [f"# H3-B comprehension {args.version} ({args.tag})\n",
           "NC = forbidden item picked among 3 options (chance 33.3%). Violation among C0-mentioned questions. "
           "95% CIs: bootstrap over images. Understood vs not = association only.\n"]
    for m in args.models:
        cf = RES / m / (f"comp_{args.tag}_{V2_CONDS}.jsonl" if v2 else f"comp_{args.tag}_COMP_T-COMP_A-COMP_V.jsonl")
        sf = RES / m / f"scored_{args.tag}.jsonl"
        if not cf.exists() or not sf.exists():
            out.append(f"\n## {m}\nmissing {cf.name if not cf.exists() else sf.name}\n")
            continue
        comp = {(r["sample_id"], r["condition"]): r for r in read_jsonl(cf)}
        scored = [r for r in read_jsonl(sf) if r["form"] == "negative"]
        c0 = {r["sample_id"]: r["target_mentioned"] for r in scored if r["condition"] == "C0"}
        image = {r["sample_id"]: r["image"] for r in scored}
        out.append(f"\n## {m}\n")
        out.append("| channel | NC % [95% CI] | parsed % | violation if understood | violation if not understood |")
        out.append("|---|---|---|---|---|")
        for ch, cc in ch_map.items():
            rows = [r for (sid, c), r in comp.items() if c == cc]
            sids = [r["sample_id"] for r in rows]
            ok = {r["sample_id"]: r["comp_correct"] for r in rows}
            nc = rate(rows, lambda r: r["comp_correct"])
            lo, hi = bootstrap(sids, image, lambda p: rate(p, lambda s: ok[s]), rng)
            parsed = rate(rows, lambda r: r["comp_parsed"])
            viol = {r["sample_id"]: r["violation"] for r in scored if r["condition"] == ch and c0.get(r["sample_id"])}
            cells = []
            for want in (True, False):
                sub = [s for s in viol if s in ok and ok[s] == want]
                v = rate(sub, lambda s: viol[s])
                l2, h2 = bootstrap(sub, image, lambda p: rate(p, lambda s: viol[s]), rng) if sub else (None, None)
                cells.append(fmt(v, l2, h2, len(sub)))
            out.append(f"| {ch} | {fmt(nc, lo, hi, len(rows))} | {fmt(parsed)} | {cells[0]} | {cells[1]} |")
        of = RES / m / f"ocr_{args.tag}_{V2_CONDS}.jsonl"
        if v2 and of.exists():
            ocr = list(read_jsonl(of))
            out.append("\n| channel | verbatim repeat: exact % [95% CI] | mean similarity |")
            out.append("|---|---|---|")
            for ch, cc in REP2.items():
                rows = [r for r in ocr if r["condition"] == cc]
                if not rows:
                    continue
                ex = {r["sample_id"]: r["ocr_exact"] for r in rows}
                lo, hi = bootstrap(list(ex), image, lambda p: rate(p, lambda s: ex[s]), rng)
                sim = sum(r["ocr_sim"] for r in rows) / len(rows)
                out.append(f"| {ch} ({cc}) | {fmt(rate(rows, lambda r: r['ocr_exact']), lo, hi, len(rows))} | {sim:.3f} |")
    path = HERE / (f"analysis_comp2_{args.tag}.md" if v2 else f"analysis_comp_{args.tag}.md")
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print("->", path)


if __name__ == "__main__":
    main()
