"""Analysis of the NEU control (run_neu.py; Vy 2026-11-27).

    python analyze_neu.py                       # every model with scored_neu_pilot1000.jsonl -> analysis_neu_pilot1000.md
    python analyze_neu.py --tag neu_smoke2      # the smoke run

Per model, on units whose own C0 answer states the target (C0-mentioned, the same units as every violation number):
  NEG follow_c  = 1 - violation under the negation in channel c (T, A, V, V_cue, T_cue)
  NEU follow_c  = the word W appears in the answer under the neutral instruction (NEU_T, NEU_A, NEU_V, NEU_V_cue)
  D_c           = NEU follow_c - NEG follow_c, paired on the same units (main metric, Vy)
  D_A - D_T, D_V - D_T: does the channel gap of the negation exceed the channel gap of a neutral instruction?
  2x2 {V, V_cue} x {NEG, NEU}: the cue effect for each instruction and its interaction.
  T_cue - T (NEG follow): does the cue sentence alone change obedience?
  Baselines: W in the C0 answer (chance), target mention under NEU (should stay near C0).
95% CIs: percentile bootstrap over IMAGES (2,000 resamples, seed 0); McNemar exact for paired binary contrasts.
Comparisons are within a model only (Vy: never compare raw rates across models).
"""
import argparse
import random
from collections import defaultdict
from math import comb
from pathlib import Path

from analyze import read_jsonl

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODELS = ["qwen2.5-omni-7b", "qwen2.5-omni-3b", "minicpm-o-2.6", "phi-4-mm", "internomni"]
N_BOOT = 2000


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def boot(units, img_of, f):
    """f(list of units) -> value; CI by resampling images."""
    by = defaultdict(list)
    for u in units:
        by[img_of[u]].append(u)
    imgs = list(by)
    rng = random.Random(0)
    vals = []
    for _ in range(N_BOOT):
        us = [u for i in rng.choices(imgs, k=len(imgs)) for u in by[i]]
        vals.append(f(us))
    vals.sort()
    return vals[int(0.025 * N_BOOT)], vals[int(0.975 * N_BOOT) - 1]


def fmt(x):
    return f"{100 * x:+.1f}"


def analyse(model, tag):
    p = RESULTS / model / f"scored_{tag}.jsonl"
    if not p.exists():
        return None
    rows = read_jsonl(p)
    B = lambda v: v is True or v == "True"
    by = defaultdict(dict)
    img_of = {}
    for r in rows:
        by[r["unit_id"]][r["condition"]] = r
        img_of[r["unit_id"]] = r["image_id"]
    asr_bad = {u for u, d in by.items() if any(not B(r.get("asr_ok")) for r in d.values())
               or ("NEU_A" in d and not B(d["NEU_A"].get("neu_asr_ok")))}
    units = [u for u in by if u not in asr_bad and "C0" in by[u]]
    U = [u for u in units if B(by[u]["C0"]["violation"])]
    have = lambda c: [u for u in U if c in by[u]]

    def neg(u, c):
        return 0.0 if B(by[u][c]["violation"]) else 1.0

    def neu(u, c):
        return 1.0 if B(by[u][c]["w_in_answer"]) else 0.0

    def rate(us, g):
        return sum(g(u) for u in us) / len(us) if us else float("nan")

    out = [f"## {model}", "",
           f"Units: {len(units)} (asr_ok, {len(asr_bad)} left out); C0 states the target in {len(U)} "
           f"({100 * len(U) / max(1, len(units)):.1f}%). All rates below are on these C0-mentioned units.", ""]
    # follow rates
    out += ["| channel | NEG follow % (1 - violation) | NEU follow % (W in answer) | D = NEU - NEG (pp) [95% CI] | "
            "only NEU followed / only NEG followed | McNemar p | n |", "|---|---|---|---|---|---|---|"]
    D = {}
    for ch, nc, uc in [("T", "T", "NEU_T"), ("A", "A", "NEU_A"), ("V", "V", "NEU_V"), ("V_cue", "V_cue", "NEU_V_cue")]:
        us = [u for u in U if nc in by[u] and uc in by[u]]
        if not us:
            continue
        d = lambda xs, nc=nc, uc=uc: rate(xs, lambda u: neu(u, uc) - neg(u, nc))
        lo, hi = boot(us, img_of, d)
        b = sum(neu(u, uc) == 1 and neg(u, nc) == 0 for u in us)
        c = sum(neu(u, uc) == 0 and neg(u, nc) == 1 for u in us)
        D[ch] = (us, d)
        out.append(f"| {ch} | {100 * rate(us, lambda u: neg(u, nc)):.1f} | {100 * rate(us, lambda u: neu(u, uc)):.1f} | "
                   f"{fmt(d(us))} [{fmt(lo)}, {fmt(hi)}] | {b} / {c} | {mcnemar(b, c):.3g} | {len(us)} |")
    # interactions across channels
    out += ["", "Channel contrasts of D (is the channel gap of the negation different from that of a neutral instruction?):",
            "", "| contrast | value (pp) | 95% CI |", "|---|---|---|"]
    for a, b_ in [("A", "T"), ("V", "T"), ("V", "A")]:
        if a in D and b_ in D:
            us = sorted(set(D[a][0]) & set(D[b_][0]))
            f = lambda xs, a=a, b_=b_: D[a][1](xs) - D[b_][1](xs)
            lo, hi = boot(us, img_of, f)
            out.append(f"| D_{a} - D_{b_} | {fmt(f(us))} | [{fmt(lo)}, {fmt(hi)}] |")
    # NEU channel gaps alone and NEG channel gaps alone (for reading)
    out += ["", "Channel gaps within each instruction type (follow rate, pp):", "",
            "| instruction | A - T [95% CI] | V - T [95% CI] |", "|---|---|---|"]
    for name, g, cs in [("NEG", neg, ("T", "A", "V")), ("NEU", neu, ("NEU_T", "NEU_A", "NEU_V"))]:
        cells = []
        for x in cs[1:]:
            us = [u for u in U if cs[0] in by[u] and x in by[u]]
            f = lambda xs, x=x: rate(xs, lambda u: g(u, x) - g(u, cs[0]))
            lo, hi = boot(us, img_of, f) if us else (float("nan"),) * 2
            cells.append(f"{fmt(f(us))} [{fmt(lo)}, {fmt(hi)}]" if us else "-")
        out.append(f"| {name} | {cells[0]} | {cells[1]} |")
    # 2x2 cue table
    us = [u for u in U if all(c in by[u] for c in ("V", "V_cue", "NEU_V", "NEU_V_cue"))]
    if us:
        out += ["", "2x2 written instruction x cue (follow %, same units):", "",
                "| | no cue | cue ('The image contains an instruction for you.') | cue effect (pp) [95% CI] |", "|---|---|---|---|"]
        effs = {}
        for name, g, c1, c2 in [("NEG (do not mention X)", neg, "V", "V_cue"), ("NEU (include W)", neu, "NEU_V", "NEU_V_cue")]:
            f = lambda xs, g=g, c1=c1, c2=c2: rate(xs, lambda u: g(u, c2) - g(u, c1))
            effs[name] = f
            lo, hi = boot(us, img_of, f)
            out.append(f"| {name} | {100 * rate(us, lambda u: g(u, c1)):.1f} | {100 * rate(us, lambda u: g(u, c2)):.1f} | "
                       f"{fmt(f(us))} [{fmt(lo)}, {fmt(hi)}] |")
        f = lambda xs: effs["NEU (include W)"](xs) - effs["NEG (do not mention X)"](xs)
        lo, hi = boot(us, img_of, f)
        out.append(f"| interaction (NEU cue effect - NEG cue effect) | | | {fmt(f(us))} [{fmt(lo)}, {fmt(hi)}] |")
    us = [u for u in U if "T" in by[u] and "T_cue" in by[u]]
    if us:
        f = lambda xs: rate(xs, lambda u: neg(u, "T_cue") - neg(u, "T"))
        lo, hi = boot(us, img_of, f)
        b = sum(neg(u, "T_cue") == 1 and neg(u, "T") == 0 for u in us)
        c = sum(neg(u, "T_cue") == 0 and neg(u, "T") == 1 for u in us)
        out += ["", f"T_cue vs T (NEG follow; the cue sentence with a typed negation, n={len(us)}): "
                    f"{100 * rate(us, lambda u: neg(u, 'T')):.1f} -> {100 * rate(us, lambda u: neg(u, 'T_cue')):.1f}, "
                    f"{fmt(f(us))} pp [{fmt(lo)}, {fmt(hi)}], McNemar p={mcnemar(b, c):.3g}"]
    # baselines
    out += ["", "Baselines (C0-mentioned units): W in answer by chance, target mention under each condition:", "",
            "| condition | n | W in answer % | target mentioned % |", "|---|---|---|---|"]
    for c in ["C0", "T", "A", "V", "V_cue", "T_cue", "NEU_T", "NEU_A", "NEU_V", "NEU_V_cue"]:
        us = have(c)
        if us:
            out.append(f"| {c} | {len(us)} | {100 * rate(us, lambda u: neu(u, c)):.1f} | "
                       f"{100 * rate(us, lambda u: 0.0 if neg(u, c) else 1.0):.1f} |")
    # NEU follow on ALL units and per word
    out += ["", "NEU follow % on all units (not only C0-mentioned) and per word:", ""]
    for c in ["NEU_T", "NEU_A", "NEU_V", "NEU_V_cue"]:
        us = [u for u in units if c in by[u]]
        if not us:
            continue
        per = defaultdict(list)
        for u in us:
            per[by[u][c]["W"]].append(neu(u, c))
        out.append(f"- {c}: {100 * rate(us, lambda u: neu(u, c)):.1f}% (n={len(us)}); " +
                   ", ".join(f"{w} {100 * sum(v) / len(v):.0f}%" for w, v in sorted(per.items())))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="neu_pilot1000")
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()
    parts = [f"# NEU control ({args.tag})", "", "Generated by experiments/h1_v2/analyze_neu.py (definitions in its docstring).",
             "Follow = obeys the instruction: NEG = target not mentioned (any mention counts as violation), "
             "NEU = the word W appears (plural counts). Within-model comparisons only.", ""]
    for m in args.models:
        s = analyse(m, args.tag)
        if s:
            parts.append(s)
    path = HERE / f"analysis_{args.tag}.md"
    path.write_text("\n".join(parts))
    print(f"written {path}")


if __name__ == "__main__":
    main()
